package com.example.medicalaiguidance

import com.example.medicalaiguidance.model.VisitPlan
import com.example.medicalaiguidance.navigation.Route
import com.example.medicalaiguidance.network.QuickSearchRequest
import com.example.medicalaiguidance.network.RecommendationItemDto
import com.example.medicalaiguidance.network.ReferenceDepartmentDto
import com.example.medicalaiguidance.network.ScriptRequest
import com.example.medicalaiguidance.network.parseQuickSearchResult
import com.example.medicalaiguidance.network.quickSearchPath
import com.example.medicalaiguidance.network.toJson
import com.example.medicalaiguidance.repository.MedicalRepository
import com.example.medicalaiguidance.repository.resolveRecommendationVisitType
import com.example.medicalaiguidance.screen.shouldShowAllSelectionOptions
import com.example.medicalaiguidance.screen.sortQuickSearchDepartments
import com.example.medicalaiguidance.viewmodel.QUICK_SEARCH_PERIODS
import com.example.medicalaiguidance.viewmodel.QuickSearchForm
import com.example.medicalaiguidance.viewmodel.normalizeQuickSearchDate
import com.example.medicalaiguidance.viewmodel.quickSearchSelectionError
import com.example.medicalaiguidance.viewmodel.quickSearchValidationError
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Test

class QuickSearchIntegrationUnitTest {
    @Test
    fun quickSearchHasIndependentVisitTypeAndRouteWhileLegacyReturnVisitRemains() {
        assertEquals("quick_search", VisitPlan.QUICK_SEARCH.apiValue)
        assertEquals("quick_search", VisitPlan.QUICK_SEARCH.routeValue)
        assertEquals("快速查詢", VisitPlan.QUICK_SEARCH.displayName)
        assertEquals("quick_search", Route.QUICK_SEARCH)

        assertEquals("return_visit", VisitPlan.RETURN_VISIT.apiValue)
        assertEquals("return_visit", Route.RETURN_VISIT)
    }

    @Test
    fun quickSearchUsesIndependentScheduleEndpoint() {
        val path = quickSearchPath(
            QuickSearchRequest(
                deptId = 7,
                date = "2099-09-20",
                period = "evening"
            )
        )

        assertEquals(
            "/schedules/search?dept_id=7&date=2099-09-20&period=evening",
            path
        )
    }

    @Test
    fun quickSearchResponseReusesScheduleDisplayModel() {
        val parsed = parseQuickSearchResult(
            """
            {
              "case_id": "case_qs",
              "dept_id": 7,
              "parentDept": "外科系",
              "childDept": "一般骨科",
              "date": "2099-09-20",
              "period": "morning",
              "results": [{
                "recommendation_id": "qs_case_qs_9001",
                "parentDept": "外科系",
                "childDept": "一般骨科",
                "doctor": "測試醫師",
                "doctor_id": "101",
                "schedule_id": "9001",
                "dept_id": 7,
                "date": "2099-09-20",
                "session": "上午",
                "session_time": "08:30-12:00",
                "room": "3201診",
                "slot": "3201診",
                "visit_type": "初診",
                "score": 0.0,
                "reasons": ["符合指定條件"]
              }],
              "total_count": 1
            }
            """.trimIndent()
        )

        assertEquals("case_qs", parsed.caseId)
        assertEquals(7, parsed.deptId)
        assertEquals(1, parsed.results.size)
        assertEquals("9001", parsed.results.single().scheduleId)
        assertEquals("101", parsed.results.single().doctorId)
        assertEquals(7, parsed.results.single().deptId)
        assertEquals("2099-09-20", parsed.results.single().date)
        assertEquals("上午", parsed.results.single().session)
        assertEquals("測試醫師", parsed.results.single().doctor)
        assertNull(quickSearchSelectionError(parsed.results.single()))
    }

    @Test
    fun selectedScheduleIsSentToExistingScriptFlowWithAllDatabaseIds() {
        val selected = quickSchedule()
        val json = JSONObject(
            ScriptRequest(
                caseId = "quick_abcd1234",
                recommendationId = selected.recommendationId,
                recommendation = selected
            ).toJson()
        )
        val payload = json.getJSONObject("recommendation")

        assertEquals("confirm_need", Route.CONFIRM_NEED)
        assertEquals("quick_abcd1234", json.getString("case_id"))
        assertEquals("qs_quick_abcd1234_9001", json.getString("recommendation_id"))
        assertEquals("9001", payload.getString("schedule_id"))
        assertEquals("101", payload.getString("doctor_id"))
        assertEquals(7, payload.getInt("dept_id"))
        assertEquals("2099-09-20", payload.getString("date"))
        assertEquals("上午", payload.getString("session"))
    }

    @Test
    fun selectedQuickScheduleBuildsExistingConfirmationAppointmentWithoutLosingIds() {
        val repository = MedicalRepository()
        val selected = quickSchedule()
        repository.clearRecommendationFlow()
        repository.setActiveCaseId("quick_abcd1234")
        repository.selectRecommendation(selected)

        val retained = repository.getSelectedRecommendation()
        val appointment = repository.getConfirmedAppointment()

        assertEquals("9001", retained?.scheduleId)
        assertEquals("101", retained?.doctorId)
        assertEquals(7, retained?.deptId)
        assertEquals("測試醫師", appointment.doctor.name)
        assertEquals("2099-09-20", appointment.date)
        assertEquals("08:30-12:00", appointment.timeSlot)
    }

    @Test
    fun formRequiresCanonicalDepartmentDateAndOneExistingPeriod() {
        val valid = QuickSearchForm(
            department = "一般骨科",
            departmentId = "7",
            parentDepartment = "外科系",
            availableDate = "2099-09-20",
            period = "afternoon"
        )

        assertEquals(setOf("morning", "afternoon", "evening"), QUICK_SEARCH_PERIODS)
        assertEquals("2099-09-20", normalizeQuickSearchDate(valid.availableDate))
        assertNull(quickSearchValidationError(valid, departmentIsValid = true))
        assertEquals(
            "請從正式科別清單選擇就診科別。",
            quickSearchValidationError(valid.copy(departmentId = ""), departmentIsValid = false)
        )
        assertEquals(
            "請選擇有效的可看診日期。",
            quickSearchValidationError(valid.copy(availableDate = "not-a-date"), departmentIsValid = true)
        )
        assertEquals(
            "請選擇上午診、下午診或夜診。",
            quickSearchValidationError(valid.copy(period = "night"), departmentIsValid = true)
        )
    }

    @Test
    fun selectedDepartmentDoesNotRemainAsDropdownSearchFilter() {
        assertEquals(true, shouldShowAllSelectionOptions("一般骨科", "一般骨科"))
        assertEquals(true, shouldShowAllSelectionOptions("", "一般骨科"))
        assertEquals(false, shouldShowAllSelectionOptions("睡眠", "一般骨科"))
    }

    @Test
    fun departmentDropdownUsesSpecifiedGroupOrderAndKeepsOriginalOrderWithinGroup() {
        val departments = listOf(
            referenceDepartment("81", "AI輔助門診", "AI 門診"),
            referenceDepartment("21", "外科系", "一般骨科"),
            referenceDepartment("12", "一般內科", "胸腔內科"),
            referenceDepartment("51", "其他科", "皮膚科"),
            referenceDepartment("11", "內科系", "心臟內科"),
            referenceDepartment("31", "婦幼", "兒童內科"),
            referenceDepartment("41", "五官科", "眼科"),
            referenceDepartment("61", "大我門診", "大我內科"),
            referenceDepartment("71", "整合門診", "整合門診"),
            referenceDepartment("91", "未分類", "測試科")
        )

        assertEquals(
            listOf("12", "11", "21", "31", "41", "51", "61", "71", "81", "91"),
            sortQuickSearchDepartments(departments).map { it.deptId }
        )
    }

    @Test
    fun quickSearchCannotEnterRecommendationVisitTypeResolver() {
        assertThrows(IllegalStateException::class.java) {
            resolveRecommendationVisitType(null, VisitPlan.QUICK_SEARCH.apiValue)
        }
        assertEquals(
            "return_visit",
            resolveRecommendationVisitType(null, VisitPlan.RETURN_VISIT.apiValue)
        )
    }

    private fun quickSchedule() = RecommendationItemDto(
        recommendationId = "qs_quick_abcd1234_9001",
        parentDept = "外科系",
        childDept = "一般骨科",
        doctor = "測試醫師",
        date = "2099-09-20",
        session = "上午",
        slot = "3201診",
        score = 0.0,
        doctorId = "101",
        scheduleId = "9001",
        sessionTime = "08:30-12:00",
        room = "3201診",
        visitType = "初診",
        deptId = 7
    )

    private fun referenceDepartment(
        id: String,
        parent: String,
        child: String
    ) = ReferenceDepartmentDto(deptId = id, parentDept = parent, childDept = child)
}
