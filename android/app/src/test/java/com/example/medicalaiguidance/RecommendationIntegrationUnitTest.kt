package com.example.medicalaiguidance

import com.example.medicalaiguidance.network.parseRecommendationResult
import com.example.medicalaiguidance.network.RecommendationItemDto
import com.example.medicalaiguidance.network.RecommendRequest
import com.example.medicalaiguidance.network.toJson
import com.example.medicalaiguidance.model.VisitPlan
import com.example.medicalaiguidance.repository.MedicalRepository
import com.example.medicalaiguidance.screen.compactCardReason
import com.example.medicalaiguidance.screen.SPECIALTY_MATCH_TITLE
import com.example.medicalaiguidance.screen.TIME_MATCH_TITLE
import com.example.medicalaiguidance.screen.specialtyDescription
import com.example.medicalaiguidance.screen.scoreToStars
import com.example.medicalaiguidance.screen.timeDescription
import org.json.JSONException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
    fun doctorCardDisplaysOnlyBackendCompactRecommendationReason() {
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

        assertEquals(
            "醫師專長相符：過敏性鼻炎診療、鼻塞",
            recommendation.compactCardReason(),
        )
    }

    @Test
    fun doctorDetailUsesNaturalSpecialtyAndTimeReasonsWithoutInternalScores() {
        val recommendation = RecommendationItemDto(
            recommendationId = "rec_detail",
            parentDept = "外科系",
            childDept = "一般骨科",
            doctor = "測試醫師",
            date = "2026-09-14",
            session = "下午",
            score = 90.2,
            specialtyScore = 0.86,
            timeScore = 1.0,
            specialtyTags = listOf("骨科復健", "膝關節診療", "兒童骨科"),
            matchReason = "症狀與膝、膝關節/骨科或復健相關；醫師專長包含骨科復健",
            reasons = listOf(
                "科別依據：使用者症狀「膝蓋怪怪的」對應到 一般骨科",
                "時間依據：使用者偏好 週一/週二/週四/週五/週六/週日 下午；Schedule 為 2026-09-14 週一 下午，符合偏好，時間分數 1.00",
                "排序依據：specialty_score*70 + time_score*30，總分=90.20",
            ),
        )

        val specialty = recommendation.specialtyDescription()
        val time = recommendation.timeDescription()

        assertEquals("症狀與專長相符", SPECIALTY_MATCH_TITLE)
        assertEquals("看診時間符合", TIME_MATCH_TITLE)
        assertEquals(
            "症狀與膝、膝關節/骨科或復健相關；醫師專長包含骨科復健",
            specialty,
        )
        assertEquals(
            "您偏好週三以外的下午時段，此門診為 9 月 14 日週一下午，符合您的需求。",
            time,
        )
        listOf(specialty, time).forEach { text ->
            assertFalse(text.contains("時間分數"))
            assertFalse(text.contains("score", ignoreCase = true))
            assertFalse(text.contains("90.20"))
            assertFalse(text.contains("1.00"))
        }
    }

    @Test
    fun specialtyReasonPrefersBackendMatchReason() {
        val backendReason = "飯後胃痛與反胃屬消化道不適方向；此醫師具胃鏡相關診療專長，與目前就診需求較相符。"
        val recommendation = RecommendationItemDto(
            recommendationId = "rec_backend_reason",
            parentDept = "內科系",
            childDept = "胃腸肝膽科",
            doctor = "測試醫師",
            date = "2026-09-20",
            session = "下午",
            score = 0.9,
            specialtyTags = listOf("胃鏡、大腸鏡診斷及治療"),
            matchReason = backendReason,
            reasons = listOf("推薦理由：Android 舊版固定理由"),
        )

        assertEquals(backendReason, recommendation.specialtyDescription())
        assertEquals(backendReason.take(55), recommendation.compactCardReason())
    }

    @Test
    fun specialtyReasonFallsBackWhenBackendMatchReasonIsNull() {
        val recommendation = RecommendationItemDto(
            recommendationId = "rec_reason_fallback",
            parentDept = "五官科",
            childDept = "鼻科",
            doctor = "測試醫師",
            date = "2026-09-20",
            session = "上午",
            score = 0.8,
            specialtyTags = listOf("過敏性鼻炎診斷與治療"),
            reasons = listOf(
                "推薦理由：既有卡片 fallback",
                "科別依據：使用者症狀「鼻塞」對應到鼻科",
            ),
        )

        assertEquals("既有卡片 fallback", recommendation.compactCardReason())
        assertTrue(recommendation.specialtyDescription().contains("過敏性鼻炎"))
    }

    @Test
    fun backendReasonIsBoundedForCompactCardAndDetail() {
        val longReason = "長".repeat(100)
        val recommendation = RecommendationItemDto(
            recommendationId = "rec_long_reason",
            parentDept = "內科系",
            childDept = "胃腸肝膽科",
            doctor = "測試醫師",
            date = "2026-09-20",
            session = "下午",
            score = 0.9,
            matchReason = longReason,
        )

        assertEquals(55, recommendation.compactCardReason()?.length)
        assertEquals(68, recommendation.specialtyDescription().length)
    }

    @Test
    fun doctorDetailUsesExactDateAndSessionFromBackendReason() {
        val recommendation = RecommendationItemDto(
            recommendationId = "rec_exact_date",
            parentDept = "內科系",
            childDept = "神經內科",
            doctor = "測試醫師",
            date = "2026-09-20",
            session = "下午",
            score = 0.9,
            specialtyScore = 0.86,
            timeScore = 1.0,
            specialtyTags = listOf("腦血管疾病", "腦血管剝離"),
            reasons = listOf(
                "科別依據：使用者症狀「頭部不舒服」，目前建議科別為 神經內科",
                "時間依據：使用者偏好 2026-09-20 下午；Schedule 為 2026-09-20 週日 下午，符合偏好，時間分數 1.00",
            ),
        )

        assertEquals(
            "您先前提到「頭部不舒服」，目前建議科別為神經內科。這位醫師的專長包含「腦血管疾病」、「腦血管剝離」，與目前需要由神經內科評估的症狀方向相關，因此專長匹配度較高。",
            recommendation.specialtyDescription(),
        )
        assertEquals(
            "您偏好 9 月 20 日下午，此門診為 9 月 20 日週日下午，符合您的需求。",
            recommendation.timeDescription(),
        )
    }

    @Test
    fun starRatingReflectsActualBackendScoreWithoutArtificialMinimum() {
        assertEquals(5.0, scoreToStars(1.0), 0.001)
        assertEquals(4.5, scoreToStars(0.86), 0.001)
        assertEquals(2.5, scoreToStars(0.50), 0.001)
        assertEquals(0.0, scoreToStars(0.0), 0.001)
    }

    @Test
    fun doctorDetailFallsBackToCompactVisitTimeWithoutListingWeekdayArray() {
        val recommendation = RecommendationItemDto(
            recommendationId = "rec_time_fallback",
            parentDept = "五官科",
            childDept = "耳科",
            doctor = "測試醫師",
            date = "2026-09-15",
            session = "上午",
            score = 0.8,
            reasons = listOf("時間依據：符合偏好，時間分數 0.80"),
        )

        val description = recommendation.timeDescription()

        assertEquals(
            "此門診為 9 月 15 日週二上午，符合您的看診時間偏好。",
            description,
        )
        assertFalse(description.contains("週一/"))
        assertFalse(description.contains("分數"))
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
