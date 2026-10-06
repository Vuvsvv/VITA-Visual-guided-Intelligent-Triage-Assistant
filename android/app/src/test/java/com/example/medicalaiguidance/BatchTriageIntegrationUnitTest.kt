package com.example.medicalaiguidance

import com.example.medicalaiguidance.model.VisitPlan
import com.example.medicalaiguidance.navigation.Route
import com.example.medicalaiguidance.network.AvailabilityDto
import com.example.medicalaiguidance.network.BatchAnswerDto
import com.example.medicalaiguidance.network.ChatRequest
import com.example.medicalaiguidance.network.FollowupRecommendRequest
import com.example.medicalaiguidance.network.QuestionItemDto
import com.example.medicalaiguidance.network.medicalApiErrorMessage
import com.example.medicalaiguidance.network.MedicalApiException
import com.example.medicalaiguidance.viewmodel.intentionalNoSlotsMessage
import com.example.medicalaiguidance.network.canonicalVoiceVisitType
import com.example.medicalaiguidance.network.parseFollowupRecommendationResult
import com.example.medicalaiguidance.network.parseReferenceDepartments
import com.example.medicalaiguidance.network.parseReferenceDoctors
import com.example.medicalaiguidance.network.parseTriageResult
import com.example.medicalaiguidance.network.toJson
import com.example.medicalaiguidance.repository.parseHistoryJson
import com.example.medicalaiguidance.repository.resolveRecommendationVisitType
import com.example.medicalaiguidance.viewmodel.missingRequiredQuestionKeys
import com.example.medicalaiguidance.viewmodel.currentQuestionAnswer
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class BatchTriageIntegrationUnitTest {
    @Test
    fun homeEntryAndVisitTypeRoutesAreCanonical() {
        assertEquals("visit_type_selection", Route.VISIT_TYPE_SELECTION)
        assertEquals("chat/start/initial", Route.chat(VisitPlan.INITIAL))
        assertEquals("chat/start/followup", Route.chat(VisitPlan.FOLLOW_UP))
        assertEquals("return_visit", Route.RETURN_VISIT)
    }

    @Test
    fun visitPlansExposeThreeCanonicalApiValues() {
        assertEquals("initial", VisitPlan.INITIAL.apiValue)
        assertEquals("followup", VisitPlan.FOLLOW_UP.apiValue)
        assertEquals("return_visit", VisitPlan.RETURN_VISIT.apiValue)
        assertEquals("unknown", VisitPlan.UNKNOWN.apiValue)
    }

    @Test
    fun questionBatchAndVisitTypeAreParsedAdditively() {
        val result = parseTriageResult(batchResponse("symptom", "請描述主要症狀"))

        assertEquals("initial", result.triageCase?.visitType)
        assertEquals(1, result.questionBatch.size)
        assertEquals("symptom", result.questionBatch.single().key)
        assertEquals("symptom", result.questionBatch.single().questionId)
        assertEquals("symptom", result.questionBatch.single().stateField)
        assertTrue(result.questionBatch.single().required)
    }

    @Test
    fun malformedOrEmptyQuestionBatchDoesNotCrashParser() {
        val malformed = parseTriageResult(baseResponse(questionBatch = "{}"))
        val missing = parseTriageResult(baseResponse(questionBatch = null))

        assertTrue(malformed.questionBatch.isEmpty())
        assertTrue(missing.questionBatch.isEmpty())
    }

    @Test
    fun batchAnswersUseQuestionKeysAndAreSentInOneRequest() {
        val json = JSONObject(
            ChatRequest(
                caseId = "case_batch",
                visitType = "initial",
                answers = listOf(
                    BatchAnswerDto("symptom", "膝痛"),
                    BatchAnswerDto("red_flags", "以上都沒有")
                )
            ).toJson()
        )

        assertEquals("initial", json.getString("visit_type"))
        assertEquals(2, json.getJSONArray("answers").length())
        assertEquals("symptom", json.getJSONArray("answers").getJSONObject(0).getString("key"))
        assertEquals("red_flags", json.getJSONArray("answers").getJSONObject(1).getString("key"))
    }

    @Test
    fun requiredAnswerValidationUsesKeysNotIndexes() {
        val questions = listOf(
            QuestionItemDto("symptom", "症狀", required = true),
            QuestionItemDto("original_note", "備註", required = false)
        )

        assertEquals(listOf("symptom"), missingRequiredQuestionKeys(questions, emptyMap()))
        assertTrue(missingRequiredQuestionKeys(questions, mapOf("symptom" to "膝痛")).isEmpty())
    }

    @Test
    fun nextBatchContainsOnlyBackendReplacementQuestions() {
        val first = parseTriageResult(batchResponse("symptom", "症狀"))
        val next = parseTriageResult(batchResponse("preferred_sessions", "偏好時段"))

        assertEquals(listOf("symptom"), first.questionBatch.map { it.key })
        assertEquals(listOf("preferred_sessions"), next.questionBatch.map { it.key })
    }

    @Test
    fun currentQuestionIsSubmittedImmediatelyWithoutPreAdvancing() {
        val questions = listOf(
            QuestionItemDto("symptom", "哪裡不舒服？"),
            QuestionItemDto("duration", "持續多久？"),
            QuestionItemDto("severity", "嚴重程度？")
        )

        val answer = currentQuestionAnswer(questions.first(), "膝蓋痛")

        assertEquals("symptom", answer?.key)
        assertEquals("膝蓋痛", answer?.answer)
    }

    @Test
    fun blankCurrentAnswerIsNotSubmitted() {
        val question = QuestionItemDto("duration", "持續多久？")

        assertNull(currentQuestionAnswer(question, "   "))
    }

    @Test
    fun chineseAsrTranscriptBuildsOneKeyedImmediateAnswer() {
        val question = QuestionItemDto("red_flags", "是否有危險症狀？")

        val answer = currentQuestionAnswer(question, "以上都沒有")

        assertEquals("red_flags", answer?.key)
        assertEquals("以上都沒有", answer?.answer)
    }

    @Test
    fun confirmationRequestKeepsOriginalVisitType() {
        val json = JSONObject(
            ChatRequest(caseId = "case_confirm", visitType = "followup", confirmed = true).toJson()
        )

        assertEquals("followup", json.getString("visit_type"))
        assertTrue(json.getBoolean("confirmed"))
        assertFalse(json.has("answers"))
    }

    @Test
    fun doctorSelectionKeepsCaseVisitTypeInsteadOfRouteOverride() {
        assertEquals("followup", resolveRecommendationVisitType("followup", "initial"))
        assertEquals("initial", resolveRecommendationVisitType(null, "initial"))
    }

    @Test
    fun doctorNavigationRouteExplicitlyPersistsVisitPlan() {
        assertEquals("select_doctor/initial", Route.selectDoctor(VisitPlan.INITIAL))
        assertEquals("select_doctor/followup", Route.selectDoctor(VisitPlan.FOLLOW_UP))
        assertEquals("select_doctor/return_visit", Route.selectDoctor(VisitPlan.RETURN_VISIT))
        assertEquals(VisitPlan.FOLLOW_UP, VisitPlan.fromRoute("followup"))
    }

    @Test
    fun doctorVisitPlanCanBeRecoveredAfterInMemoryStateIsLost() {
        assertEquals("followup", resolveRecommendationVisitType(null, VisitPlan.FOLLOW_UP.apiValue))
        assertEquals("return_visit", resolveRecommendationVisitType(null, VisitPlan.RETURN_VISIT.apiValue))
    }

    @Test
    fun unknownVisitTypeIsRejectedInsteadOfBecomingInitial() {
        assertThrows(IllegalStateException::class.java) {
            resolveRecommendationVisitType(null, VisitPlan.UNKNOWN.apiValue)
        }
    }

    @Test
    fun oldSingleMessageRequestAndResponseRemainCompatible() {
        val request = JSONObject(ChatRequest(message = "膝蓋痛").toJson())
        val response = parseTriageResult(baseResponse(questionBatch = null))

        assertEquals("膝蓋痛", request.getString("message"))
        assertFalse(request.has("visit_type"))
        assertFalse(request.has("answers"))
        assertTrue(response.questionBatch.isEmpty())
    }

    @Test
    fun voiceOptionalVisitTypePassthroughRemainsBackwardCompatible() {
        assertEquals("initial", canonicalVoiceVisitType("initial"))
        assertEquals("followup", canonicalVoiceVisitType("followup"))
        assertNull(canonicalVoiceVisitType(null))
        assertNull(canonicalVoiceVisitType("unknown"))
    }

    @Test
    fun noSlotsAndConflictHaveClearMessages() {
        assertEquals(
            "目前找不到 9 月 20 日下午符合條件的一般骨科班表。",
            medicalApiErrorMessage(503, "目前找不到 9 月 20 日下午符合條件的一般骨科班表。")
        )
        assertTrue(medicalApiErrorMessage(409, "conflict").contains("就診類型"))
    }

    @Test
    fun onlyIntentionalEmptyScheduleResponseUsesNoSlotsState() {
        val noSlots = MedicalApiException(
            message = "查無班表",
            statusCode = 503,
            detail = "目前找不到 9 月 20 日下午符合條件的一般骨科班表。"
        )
        val serviceFailure = MedicalApiException(
            message = "正式班表目前無法查詢，請稍後重試。",
            statusCode = 503,
            detail = "正式班表目前無法查詢，請稍後重試。"
        )

        assertEquals(
            "目前找不到 9 月 20 日下午符合條件的一般骨科班表。",
            intentionalNoSlotsMessage(noSlots)
        )
        assertNull(intentionalNoSlotsMessage(serviceFailure))
    }

    @Test
    fun oldHistoryWithoutVisitTypeRemainsReadable() {
        val history = parseHistoryJson(
            """[{"id":"old_case","date":"2026/08/01","typeTitle":"一般內科","summaryText":"舊紀錄","status":"COMPLETED"}]"""
        ).single()

        assertNull(history.visitType)
        assertEquals("一般內科", history.typeTitle)
    }

    @Test
    fun returnVisitRequestMapsToExistingFollowupSchema() {
        val json = JSONObject(
            FollowupRecommendRequest(
                parentDept = "外科系",
                childDept = "一般骨科",
                deptId = "7",
                originalDoctor = "王醫師",
                originalDoctorId = "101",
                availability = AvailabilityDto(
                    preferredDates = listOf("2026-08-24"),
                    preferredSessions = listOf("上午")
                )
            ).toJson()
        )

        assertEquals("return_visit", json.getString("visit_type"))
        assertEquals("一般骨科", json.getString("childDept"))
        assertEquals("7", json.getString("dept_id"))
        assertEquals("王醫師", json.getString("original_doctor"))
        assertEquals("101", json.getString("original_doctor_id"))
        assertEquals(
            "2026-08-24",
            json.getJSONObject("availability").getJSONArray("preferred_dates").getString(0)
        )
        assertEquals(0, json.getJSONObject("availability").getJSONArray("preferred_days").length())
        assertEquals("上午", json.getJSONObject("availability").getJSONArray("preferred_sessions").getString(0))
    }

    @Test
    fun referenceDepartmentAndDoctorResponsesUseCanonicalApiValues() {
        val departments = parseReferenceDepartments(
            """{"departments":[{"dept_id":"7","parentDept":"外科系","childDept":"一般骨科"}]}"""
        )
        val doctors = parseReferenceDoctors(
            """{"department":"一般骨科","doctors":[{"doctor_id":"12","name":"骨科醫師甲"}]}"""
        )

        assertEquals("一般骨科", departments.single().childDept)
        assertEquals("7", departments.single().deptId)
        assertEquals("外科系", departments.single().parentDept)
        assertEquals("骨科醫師甲", doctors.single().name)
        assertEquals("12", doctors.single().doctorId)
    }

    @Test
    fun emptyReferenceResponsesNeverCreateFakeOptions() {
        assertTrue(parseReferenceDepartments("""{"departments":[]}""").isEmpty())
        assertTrue(parseReferenceDoctors("""{"department":"一般骨科","doctors":[]}""").isEmpty())
        assertTrue(parseReferenceDepartments("""{}""").isEmpty())
        assertTrue(parseReferenceDoctors("""{}""").isEmpty())
    }

    @Test
    fun followupResponseParsesRecommendationsAndEmptyState() {
        val populated = parseFollowupRecommendationResult(
            """{"case_id":"followup","recommendations":[{"recommendation_id":"fu_1","parentDept":"外科系","childDept":"一般骨科","doctor":"王醫師","date":"2026-08-24","session":"上午","score":95.0,"visit_type":"複診"}],"fallback_departments":[],"total_count":1}"""
        )
        val empty = parseFollowupRecommendationResult(
            """{"case_id":"followup","recommendations":[],"fallback_departments":[],"total_count":0}"""
        )

        assertEquals("複診", populated.recommendations.single().visitType)
        assertTrue(empty.recommendations.isEmpty())
    }

    private fun batchResponse(key: String, question: String): String = baseResponse(
        questionBatch = """[{"key":"$key","question":"$question","question_id":"$key","state_field":"$key","required":true,"input_type":"text"}]"""
    )

    private fun baseResponse(questionBatch: String?): String {
        val batchField = questionBatch?.let { ",\"question_batch\":$it" }.orEmpty()
        return """{"case_id":"case_batch","triage_case":{"case_id":"case_batch","visit_type":"initial"},"conversation_state":{"stage":"collecting"},"triage":{"need_more_info":true},"reply":"請回答","needMoreInfo":true$batchField}"""
    }
}
