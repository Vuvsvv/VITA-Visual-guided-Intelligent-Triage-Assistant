package com.example.medicalaiguidance

import com.example.medicalaiguidance.network.parseRecommendationResult
import com.example.medicalaiguidance.network.parseScriptResponse
import com.example.medicalaiguidance.network.RecommendationItemDto
import com.example.medicalaiguidance.network.RecommendRequest
import com.example.medicalaiguidance.network.ScriptResponseDto
import com.example.medicalaiguidance.network.toJson
import com.example.medicalaiguidance.model.VisitPlan
import com.example.medicalaiguidance.repository.MedicalRepository
import com.example.medicalaiguidance.screen.compactCardReason
import com.example.medicalaiguidance.screen.SPECIALTY_MATCH_TITLE
import com.example.medicalaiguidance.screen.TIME_MATCH_TITLE
import com.example.medicalaiguidance.screen.specialtyDescription
import com.example.medicalaiguidance.screen.timeDescription
import com.example.medicalaiguidance.viewmodel.revalidationError
import org.json.JSONException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class RecommendationIntegrationUnitTest {
    @Test
    fun recommendRequestIncludesSelectedVisitType() {
        val json = org.json.JSONObject(
            RecommendRequest(caseId = "case_visit_plan", visitType = "followup").toJson()
        )

        assertEquals("case_visit_plan", json.getString("case_id"))
        assertEquals("followup", json.getString("visit_type"))
    }

    @Test
    fun uncertainVisitPlanDoesNotSilentlyFallbackToInitial() {
        assertEquals("unknown", VisitPlan.UNKNOWN.routeValue)
        assertEquals("unknown", VisitPlan.UNKNOWN.apiValue)
    }

    @Test
    fun repositoryDoesNotCreateLocalFallbackAppointmentWithoutBackendRecommendation() {
        val repository = MedicalRepository()
        repository.clearRecommendationFlow()

        val error = assertThrows(IllegalStateException::class.java) {
            repository.getConfirmedAppointment()
        }

        assertTrue(error.message.orEmpty().contains("後端推薦結果"))
    }

    @Test
    fun repositoryBuildsAppointmentFromSelectedBackendRecommendation() {
        val repository = MedicalRepository()
        repository.clearRecommendationFlow()
        repository.selectRecommendation(
            RecommendationItemDto(
                recommendationId = "rec_sql_001",
                parentDept = "內科系",
                childDept = "一般內科",
                doctor = "王醫師",
                date = "2026-07-27",
                session = "上午",
                slot = "320診",
                score = 91.0,
                doctorId = "D001",
                scheduleId = "S001",
                room = "320"
            )
        )

        val appointment = repository.getConfirmedAppointment()

        assertEquals("rec_sql_001", appointment.id)
        assertEquals("內科系", appointment.department.name)
        assertEquals("一般內科", appointment.department.clinicName)
        assertEquals("王醫師", appointment.doctor.name)
        assertEquals("2026-07-27", appointment.date)
        assertEquals("上午", appointment.timeSlot)
    }

    @Test
    fun scriptResponseParsesRevalidatedRecommendationAndAllowsEmptySteps() {
        val response = parseScriptResponse(
            """
            {
              "isSuccess": true,
              "recommendation_id": "rec_sql_001",
              "recommendation": {
                "recommendation_id": "rec_sql_001",
                "parentDept": "內科系",
                "childDept": "一般內科",
                "doctor": "王醫師",
                "doctor_id": "D001",
                "schedule_id": "S001",
                "dept_id": 7,
                "date": "2099-07-27",
                "session": "上午",
                "room": "B診",
                "score": 91.0
              },
              "steps": [],
              "step_count": 0
            }
            """.trimIndent()
        )

        assertTrue(response.isSuccess)
        assertTrue(response.steps.isEmpty())
        assertEquals("S001", response.recommendation?.scheduleId)
        assertEquals("B診", response.recommendation?.room)
        assertNull(response.revalidationError())
    }

    @Test
    fun scriptResponseWithoutRecommendationRemainsCompatible() {
        val response = parseScriptResponse(
            """{"isSuccess":true,"recommendation_id":"rec_legacy","steps":[]}"""
        )

        assertNull(response.recommendation)
        assertNull(response.revalidationError())
    }

    @Test
    fun repositoryUsesBackendRevalidatedRecommendationForConfirmation() {
        val repository = MedicalRepository()
        repository.clearRecommendationFlow()
        repository.selectRecommendation(
            RecommendationItemDto(
                recommendationId = "rec_sql_001",
                parentDept = "內科系",
                childDept = "一般內科",
                doctor = "王醫師",
                date = "2099-07-27",
                session = "上午",
                slot = "A診",
                room = "A診",
                score = 91.0,
                doctorId = "D001",
                scheduleId = "S001",
                deptId = 7
            )
        )
        val revalidated = repository.getSelectedRecommendation()!!.copy(
            slot = "B診",
            room = "B診",
            sessionTime = "09:00-12:00"
        )

        repository.consumeScriptResponse(
            ScriptResponseDto(
                isSuccess = true,
                recommendationId = revalidated.recommendationId,
                recommendation = revalidated,
                steps = emptyList()
            )
        )

        assertEquals("B診", repository.getSelectedRecommendation()?.room)
        assertEquals("B診", repository.getConfirmedAppointment().doctor.title)
        assertEquals("上午", repository.getConfirmedAppointment().timeSlot)
        assertEquals("09:00-12:00", repository.getConfirmedAppointment().sessionTime)
    }

    @Test
    fun recommendationParserReadsSpecialtyAndTimeColumnsWithOptionalFields() {
        val result = parseRecommendationResult(
            """
            {
              "case_id": "case_123",
              "recommendations": {
                "specialty_first": [
                  {
                    "recommendation_id": "rec_s_001",
                    "parentDept": "外科系",
                    "childDept": "一般骨科",
                    "doctor": "王醫師",
                    "date": "2026-07-20",
                    "session": "上午",
                    "slot": "320診",
                    "score": 91.5,
                    "reasons": ["膝蓋疼痛與骨科專長相符"],
                    "rank": 1,
                    "is_best_match": true,
                    "doctor_id": "D001",
                    "schedule_id": "S001",
                    "session_time": "09:00-12:00",
                    "room": "320",
                    "specialty_tags": ["膝關節", "運動傷害"],
                    "specialty_score": 0.94,
                    "time_score": 0.7,
                    "match_reason": "專長符合"
                  }
                ],
                "time_first": [
                  {
                    "recommendation_id": "rec_t_001",
                    "parentDept": "外科系",
                    "childDept": "一般骨科",
                    "doctor": "李醫師",
                    "date": "2026-07-17",
                    "session": "下午",
                    "score": 82.0
                  }
                ]
              },
              "fallback_departments": [],
              "total_count": 2
            }
            """.trimIndent()
        )

        val specialty = result.recommendations.specialtyFirst.first()
        val time = result.recommendations.timeFirst.first()

        assertEquals("case_123", result.caseId)
        assertEquals(2, result.totalCount)
        assertEquals("rec_s_001", specialty.recommendationId)
        assertTrue(specialty.isBestMatch)
        assertEquals("D001", specialty.doctorId)
        assertEquals("09:00-12:00", specialty.sessionTime)
        assertEquals(listOf("膝關節", "運動傷害"), specialty.specialtyTags)
        assertEquals(0.94, specialty.specialtyScore!!, 0.001)
        assertEquals("rec_t_001", time.recommendationId)
        assertTrue(time.specialtyTags.isEmpty())
    }

    @Test
    fun recommendationParserReadsStringSpecialtyTagsFromBackend() {
        val result = parseRecommendationResult(
            """
            {
              "case_id": "case_string_tags",
              "recommendations": {
                "specialty_first": [
                  {
                    "recommendation_id": "rec_s_001",
                    "parentDept": "內科系",
                    "childDept": "一般內科",
                    "doctor": "王醫師",
                    "date": "2026-07-20",
                    "session": "上午",
                    "score": 80.0,
                    "specialty_tags": "頭暈、慢性病, 內科"
                  }
                ],
                "time_first": []
              },
              "fallback_departments": [],
              "total_count": 1
            }
            """.trimIndent()
        )

        assertEquals(listOf("頭暈", "慢性病", "內科"), result.recommendations.specialtyFirst.first().specialtyTags)
    }

    @Test
    fun legacyReasonWithoutStructuredBasisDoesNotBecomeMedicalClaim() {
        val recommendation = RecommendationItemDto(
            recommendationId = "rec_reason",
            parentDept = "五官科",
            childDept = "鼻科",
            doctor = "測試醫師",
            date = "2026-09-15",
            session = "上午",
            score = 0.9,
            reasons = listOf(
                "推薦理由：醫師專長相符：過敏性鼻炎診療、鼻塞",
                "科別依據：鼻部症狀對應鼻科",
                "時間依據：符合偏好",
            ),
        )

        assertNull(recommendation.compactCardReason())
    }

    private fun displayRecommendation(score: Double? = 0.5, time: Double? = null) = RecommendationItemDto(
        recommendationId = "rec_facts", parentDept = "五官科", childDept = "耳科",
        doctor = "測試醫師", date = "2026-10-20", session = "下午", score = 99.0,
        specialtyScore = score, timeScore = time, specialtyTags = listOf("聽力評估"),
        matchReason = "舊版高度符合症狀", reasons = listOf("排序依據：70/30")
    )

    @Test fun neutralScoreIsNotInterpretedAsClinicalRating() {
        val item = displayRecommendation()
        assertEquals("目前建議您看耳科門診。", item.specialtyDescription())
        assertFalse(item.specialtyDescription().contains("高度符合"))
        assertFalse(item.specialtyDescription().contains("星"))
        assertEquals("SQL 專長資料：聽力評估", item.compactCardReason())
    }

    @Test fun noAiScoreStillShowsOnlySqlSpecialties() {
        assertEquals(displayRecommendation().specialtyDescription(), displayRecommendation(null).specialtyDescription())
    }

    @Test fun semanticScoreDoesNotInventPatientMedicalRelationship() {
        assertEquals(displayRecommendation(0.5).specialtyDescription(), displayRecommendation(0.99).specialtyDescription())
    }

    @Test fun noTimePreferenceDoesNotClaimTimeMatch() {
        assertFalse(displayRecommendation().timeDescription().contains("符合"))
        assertTrue(displayRecommendation().timeDescription().contains("下午"))
    }

    @Test fun legacyTimeScoreDoesNotBecomeMatchClaim() {
        assertEquals(displayRecommendation(time = 0.0).timeDescription(), displayRecommendation(time = 1.0).timeDescription())
    }

    @Test fun presentationTitlesDescribeFactsNotAccuracy() {
        assertEquals("科別建議", SPECIALTY_MATCH_TITLE)
        assertEquals("看診時間", TIME_MATCH_TITLE)
    }

    @Test fun missingSqlSpecialtiesDoesNotInferFromDepartmentOrSymptom() {
        val item = displayRecommendation().copy(specialtyTags = emptyList(), matchReason = null)
        assertEquals("目前建議您看耳科門診。", item.specialtyDescription())
    }

    @Test fun patientFacingReasonsUseBackendSentencesWithoutInventingMatch() {
        val item = displayRecommendation().copy(
            departmentReason = "  依您描述的眼睛紅，建議眼科。  ",
            timeReason = "這個門診在11月10日（週二）下午，符合您說的「星期二」和「下午」。"
        )
        assertEquals("依您描述的眼睛紅，建議眼科。", item.specialtyDescription())
        assertEquals(item.timeReason, item.timeDescription())
        assertEquals(item.timeReason, item.copy(timeScore = 0.0).timeDescription())
    }

    @Test fun whitespaceAndLegacyReasonsDoNotBecomePatientFacingReasons() {
        val item = displayRecommendation().copy(departmentReason = " \n ", timeReason = " ")
        assertEquals("目前建議您看耳科門診。", item.specialtyDescription())
        assertTrue(item.timeDescription().contains("下午"))
        assertFalse(item.timeDescription().contains("符合"))
        val empty = item.copy(childDept = "", parentDept = "", date = "", session = "")
        assertEquals("", empty.specialtyDescription())
        assertEquals("", empty.timeDescription())
    }

    @Test fun recommendationParserReadsBothPatientFacingReasons() {
        val result = parseRecommendationResult("""{
            "case_id":"reason_test",
            "recommendations":{"specialty_first":[{
                "recommendation_id":"rec_s_reason_001", "doctor":"測試醫師",
                "department_reason":"依您描述的症狀，建議眼科。",
                "time_reason":"此門診為週二下午，符合您的需求。"
            }],"time_first":[]}
        }""")
        val item = result.recommendations.specialtyFirst.single()
        assertEquals("依您描述的症狀，建議眼科。", item.departmentReason)
        assertEquals("此門診為週二下午，符合您的需求。", item.timeReason)
    }

    @Test
    fun recommendationParserHandlesMissingOptionalFieldsAndEmptyResult() {
        val result = parseRecommendationResult(
            """
            {
              "case_id": "case_empty",
              "recommendations": {
                "specialty_first": [],
                "time_first": []
              },
              "fallback_departments": [
                {"parentDept": "內科系", "childDept": "胸腔內科", "reason": "胸痛與呼吸困難需優先評估"}
              ],
              "total_count": 0
            }
            """.trimIndent()
        )

        assertEquals("case_empty", result.caseId)
        assertTrue(result.recommendations.specialtyFirst.isEmpty())
        assertTrue(result.recommendations.timeFirst.isEmpty())
        assertEquals("胸腔內科", result.fallbackDepartments.first().childDept)
    }

    @Test
    fun recommendationParserRejectsMalformedResponse() {
        assertThrows(JSONException::class.java) {
            parseRecommendationResult("""{"case_id": "case_1", "recommendations": """)
        }
    }

}
