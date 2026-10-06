package com.example.medicalaiguidance.network

import org.json.JSONArray
import org.json.JSONObject

data class ChatRequest(
    val caseId: String? = null,
    val message: String? = null,
    val visitType: String? = null,
    val answers: List<BatchAnswerDto> = emptyList(),
    val confirmed: Boolean = false,
    val revisionRequested: Boolean = false
)

data class QuestionItemDto(
    val key: String,
    val question: String,
    val questionId: String? = null,
    val stateField: String? = null,
    val required: Boolean = true,
    val inputType: String? = null
)

data class BatchAnswerDto(
    val key: String,
    val answer: String
)

data class RecommendRequest(
    val caseId: String,
    val confirmed: Boolean = false,
    val visitType: String
)

data class ScriptRequest(
    val caseId: String,
    val recommendationId: String,
    val recommendation: RecommendationItemDto? = null
)

data class FollowupRecommendRequest(
    val caseId: String? = null,
    val visitType: String = "return_visit",
    val parentDept: String? = null,
    val childDept: String,
    val deptId: String? = null,
    val originalDoctor: String,
    val originalDoctorId: String? = null,
    val followupReason: String? = null,
    val availability: AvailabilityDto = AvailabilityDto()
)

data class QuickSearchRequest(
    val deptId: Int,
    val date: String,
    val period: String
)

data class ReferenceDepartmentDto(
    val deptId: String,
    val parentDept: String,
    val childDept: String
)

data class ReferenceDoctorDto(
    val doctorId: String,
    val name: String
)

data class MessageDto(
    val role: String,
    val content: String
)

data class PatientInputDto(
    val symptom: String = "",
    val bodyPart: String? = null,
    val duration: String? = null,
    val severity: String? = null,
    val onset: String? = null,
    val accompanyingSymptoms: List<String> = emptyList(),
    val redFlags: List<String> = emptyList(),
    val collectedFields: List<String> = emptyList(),
    val redFlagsChecked: Boolean = false
)

data class AvailabilityDto(
    val preferredDates: List<String> = emptyList(),
    val preferredDays: List<String> = emptyList(),
    val preferredSessions: List<String> = emptyList(),
    val canTakeLeave: Boolean = false,
    val semanticStatus: Map<String, String> = emptyMap(),
    val confidence: Map<String, Double> = emptyMap()
)

data class PreferencesDto(
    val specialtyPriority: Boolean = true,
    val doctorPreference: String = "不限",
    val hospitalPreference: String = "台北榮總"
)

data class ConversationStateDto(
    val stage: String = "collecting",
    val isComplete: Boolean = false,
    val awaitingConfirmation: Boolean = false,
    val confirmed: Boolean = false,
    val askedFields: List<String> = emptyList(),
    val consumedFields: List<String> = emptyList(),
    val lastQuestionKey: String? = null,
    val questionAttempts: Map<String, Int> = emptyMap(),
    val fieldStatuses: Map<String, String> = emptyMap(),
    val fieldConfidence: Map<String, Double> = emptyMap(),
    val clarificationReasons: Map<String, String> = emptyMap()
)

data class UrgencyResultDto(
    val urgencyScore: Int? = null,
    val urgencyLevel: String? = null,
    val warningRequired: Boolean = false,
    val warningMessage: String? = null,
    val needMoreInfo: Boolean = true,
    val nextQuestion: String? = null,
    val reasons: List<String> = emptyList(),
    val isFinal: Boolean = false
)

data class DepartmentResultDto(
    val deptId: Int? = null,
    val parentDept: String = "",
    val childDept: String = "",
    val confidence: Double = 0.0,
    val reason: List<String> = emptyList()
)

data class TriageCaseDto(
    val caseId: String,
    val visitType: String? = null,
    val historyRecords: List<MessageDto> = emptyList(),
    val patientInput: PatientInputDto = PatientInputDto(),
    val availability: AvailabilityDto = AvailabilityDto(),
    val preferences: PreferencesDto = PreferencesDto(),
    val triage: UrgencyResultDto = UrgencyResultDto(),
    val conversationState: ConversationStateDto = ConversationStateDto(),
    val departmentResult: DepartmentResultDto? = null,
    val confirmed: Boolean = false,
    val recommendationGenerated: Boolean = false,
    val scriptGenerated: Boolean = false
)

data class TriageResultDto(
    val caseId: String,
    val triageCase: TriageCaseDto?,
    val conversationState: ConversationStateDto,
    val triage: UrgencyResultDto,
    val departmentResult: DepartmentResultDto? = null,
    val nextQuestion: String? = null,
    val reply: String? = null,
    val needMoreInfo: Boolean = true,
    val triageReasons: List<String> = emptyList(),
    val questionBatch: List<QuestionItemDto> = emptyList()
)

data class RecommendationItemDto(
    val recommendationId: String,
    val parentDept: String,
    val childDept: String,
    val doctor: String,
    val date: String,
    val session: String,
    val slot: String = "",
    val score: Double,
    val reasons: List<String> = emptyList(),
    val rank: Int? = null,
    val isBestMatch: Boolean = false,
    val doctorId: String? = null,
    val scheduleId: String? = null,
    val sessionTime: String? = null,
    val room: String? = null,
    val visitType: String? = null,
    val specialtyTags: List<String> = emptyList(),
    val specialtyScore: Double? = null,
    val timeScore: Double? = null,
    val matchReason: String? = null,
    val deptId: Int? = null
)

data class RecommendationColumnsDto(
    val specialtyFirst: List<RecommendationItemDto> = emptyList(),
    val timeFirst: List<RecommendationItemDto> = emptyList()
)

data class FallbackDepartmentDto(
    val parentDept: String,
    val childDept: String,
    val reason: String
)

data class RecommendationResultDto(
    val caseId: String,
    val recommendations: RecommendationColumnsDto,
    val fallbackDepartments: List<FallbackDepartmentDto> = emptyList(),
    val totalCount: Int = 0
)

data class FollowupRecommendationResultDto(
    val caseId: String,
    val department: DepartmentResultDto? = null,
    val recommendations: List<RecommendationItemDto> = emptyList(),
    val fallbackDepartments: List<FallbackDepartmentDto> = emptyList(),
    val totalCount: Int = 0
)

data class QuickSearchResultDto(
    val caseId: String,
    val deptId: Int,
    val parentDept: String,
    val childDept: String,
    val date: String,
    val period: String,
    val results: List<RecommendationItemDto> = emptyList(),
    val totalCount: Int = 0
)

data class ScriptStepDto(
    val action: String,
    val target: String,
    val resourceId: String? = null,
    val text: String? = null,
    val className: String? = null,
    val description: String? = null,
    val delayMs: Int = 0,
    val retry: Int = 0
)

data class ScriptResponseDto(
    val isSuccess: Boolean,
    val scriptId: String = "vgh_booking_001",
    val recommendationId: String? = null,
    val steps: List<ScriptStepDto> = emptyList(),
    val message: String? = null,
    val stepCount: Int = 0
)

data class TtsRequest(
    val text: String,
    val lang: String = "chinese",
    val speed: Double = 1.0,
    val sessionId: String? = null
)

data class VoiceTtsResponseDto(
    val audioBase64: String = "",
    val audioFormat: String = "wav",
    val ttsFailed: Boolean = false,
    val error: String? = null
)

data class VoiceChatResponseDto(
    val caseId: String,
    val userText: String = "",
    val replyText: String = "",
    val replyAudioBase64: String = "",
    val audioFormat: String = "wav",
    val needMoreInfo: Boolean = true,
    val stage: String = "",
    val departmentResult: DepartmentResultDto? = null,
    val ttsFailed: Boolean = false,
    val error: String? = null
)

data class VoiceAsrResponseDto(
    val text: String = "",
    val lang: String = "taiwanese",
    val asrFailed: Boolean = false,
    val error: String? = null
)

fun ChatRequest.toJson(): String = JSONObject().apply {
    caseId?.let { put("case_id", it) }
    message?.let { put("message", it) }
    visitType?.let { put("visit_type", it) }
    if (answers.isNotEmpty()) {
        put("answers", JSONArray().also { array ->
            answers.forEach { answer ->
                array.put(JSONObject().apply {
                    put("key", answer.key)
                    put("answer", answer.answer)
                })
            }
        })
    }
    if (confirmed) put("confirmed", true)
    if (revisionRequested) put("revision_requested", true)
}.toString()

fun RecommendRequest.toJson(): String = JSONObject().apply {
    put("case_id", caseId)
    if (confirmed) put("confirmed", true)
    put("visit_type", visitType)
}.toString()

fun ScriptRequest.toJson(): String = JSONObject().apply {
    put("case_id", caseId)
    put("recommendation_id", recommendationId)
    recommendation?.let { put("recommendation", it.toJsonObject()) }
}.toString()

fun FollowupRecommendRequest.toJson(): String = JSONObject().apply {
    caseId?.let { put("case_id", it) }
    put("visit_type", visitType)
    parentDept?.takeIf { it.isNotBlank() }?.let { put("parentDept", it) }
    put("childDept", childDept)
    deptId?.takeIf { it.isNotBlank() }?.let { put("dept_id", it) }
    put("original_doctor", originalDoctor)
    originalDoctorId?.takeIf { it.isNotBlank() }?.let { put("original_doctor_id", it) }
    followupReason?.takeIf { it.isNotBlank() }?.let { put("followup_reason", it) }
    put("availability", availability.toJsonObject())
}.toString()

fun TtsRequest.toJson(): String = JSONObject().apply {
    put("text", text)
    put("lang", lang)
    put("speed", speed)
    sessionId?.let { put("session_id", it) }
}.toString()

fun parseTriageResult(json: String): TriageResultDto {
    val obj = JSONObject(json)
    return TriageResultDto(
        caseId = obj.optString("case_id"),
        triageCase = obj.optObject("triage_case")?.toTriageCaseDto(),
        conversationState = obj.optObject("conversation_state")?.toConversationStateDto() ?: ConversationStateDto(),
        triage = obj.optObject("triage")?.toUrgencyResultDto() ?: UrgencyResultDto(),
        departmentResult = obj.optObject("department_result")?.toDepartmentResultDto(),
        nextQuestion = obj.optNullableString("next_question"),
        reply = obj.optNullableString("reply"),
        needMoreInfo = obj.optBoolean("needMoreInfo", true),
        triageReasons = obj.optArray("triage_reasons").toStringList(),
        questionBatch = obj.optArray("question_batch").toQuestionList()
    )
}

fun parseRecommendationResult(json: String): RecommendationResultDto {
    val obj = JSONObject(json)
    val recommendations = obj.optObject("recommendations")
    return RecommendationResultDto(
        caseId = obj.optString("case_id"),
        recommendations = RecommendationColumnsDto(
            specialtyFirst = recommendations?.optArray("specialty_first").toRecommendationList(),
            timeFirst = recommendations?.optArray("time_first").toRecommendationList()
        ),
        fallbackDepartments = obj.optArray("fallback_departments").toFallbackDepartmentList(),
        totalCount = obj.optInt("total_count", 0)
    )
}

fun parseFollowupRecommendationResult(json: String): FollowupRecommendationResultDto {
    val obj = JSONObject(json)
    return FollowupRecommendationResultDto(
        caseId = obj.optString("case_id", "followup"),
        department = obj.optObject("department")?.toDepartmentResultDto(),
        recommendations = obj.optArray("recommendations").toRecommendationList(),
        fallbackDepartments = obj.optArray("fallback_departments").toFallbackDepartmentList(),
        totalCount = obj.optInt("total_count", 0)
    )
}

fun parseQuickSearchResult(json: String): QuickSearchResultDto {
    val obj = JSONObject(json)
    return QuickSearchResultDto(
        caseId = obj.optString("case_id"),
        deptId = obj.optInt("dept_id"),
        parentDept = obj.optString("parentDept"),
        childDept = obj.optString("childDept"),
        date = obj.optString("date"),
        period = obj.optString("period"),
        results = obj.optArray("results").toRecommendationList(),
        totalCount = obj.optInt("total_count", 0)
    )
}

fun parseReferenceDepartments(json: String): List<ReferenceDepartmentDto> {
    val array = JSONObject(json).optJSONArray("departments") ?: return emptyList()
    return (0 until array.length()).mapNotNull { index ->
        val item = array.optJSONObject(index) ?: return@mapNotNull null
        val childDept = item.optString("childDept").trim()
        if (childDept.isBlank()) return@mapNotNull null
        ReferenceDepartmentDto(
            deptId = item.optString("dept_id").trim(),
            parentDept = item.optString("parentDept").trim(),
            childDept = childDept
        )
    }
}

fun parseReferenceDoctors(json: String): List<ReferenceDoctorDto> {
    val array = JSONObject(json).optJSONArray("doctors") ?: return emptyList()
    return (0 until array.length()).mapNotNull { index ->
        val item = array.optJSONObject(index) ?: return@mapNotNull null
        val name = item.optString("name").trim()
        if (name.isBlank()) return@mapNotNull null
        ReferenceDoctorDto(
            doctorId = item.optString("doctor_id").trim(),
            name = name
        )
    }
}

fun parseScriptResponse(json: String): ScriptResponseDto {
    val obj = JSONObject(json)
    return ScriptResponseDto(
        isSuccess = obj.optBoolean("isSuccess", false),
        scriptId = obj.optString("script_id", "vgh_booking_001"),
        recommendationId = obj.optNullableString("recommendation_id"),
        steps = obj.optArray("steps").toScriptStepList(),
        message = obj.optNullableString("message"),
        stepCount = obj.optInt("step_count", 0)
    )
}

fun parseVoiceTtsResponse(json: String): VoiceTtsResponseDto {
    val obj = JSONObject(json)
    return VoiceTtsResponseDto(
        audioBase64 = obj.optString("audio_base64", ""),
        audioFormat = obj.optString("audio_format", "wav"),
        ttsFailed = obj.optBoolean("tts_failed", false),
        error = obj.optNullableString("error")
    )
}

fun parseVoiceChatResponse(json: String): VoiceChatResponseDto {
    val obj = JSONObject(json)
    return VoiceChatResponseDto(
        caseId = obj.optString("case_id"),
        userText = obj.optString("user_text", ""),
        replyText = obj.optString("reply_text", ""),
        replyAudioBase64 = obj.optString("reply_audio_base64", ""),
        audioFormat = obj.optString("audio_format", "wav"),
        needMoreInfo = obj.optBoolean("needMoreInfo", true),
        stage = obj.optString("stage", ""),
        departmentResult = obj.optObject("department_result")?.toDepartmentResultDto(),
        ttsFailed = obj.optBoolean("tts_failed", false),
        error = obj.optNullableString("error")
    )
}

fun parseVoiceAsrResponse(json: String): VoiceAsrResponseDto {
    val obj = JSONObject(json)
    return VoiceAsrResponseDto(
        text = obj.optString("text", ""),
        lang = obj.optString("lang", "taiwanese"),
        asrFailed = obj.optBoolean("asr_failed", false),
        error = obj.optNullableString("error")
    )
}

private fun JSONObject.toTriageCaseDto(): TriageCaseDto = TriageCaseDto(
    caseId = optString("case_id"),
    visitType = optNullableString("visit_type"),
    historyRecords = optArray("history_records").toMessageList(),
    patientInput = optObject("patient_input")?.toPatientInputDto() ?: PatientInputDto(),
    availability = optObject("availability")?.toAvailabilityDto() ?: AvailabilityDto(),
    preferences = optObject("preferences")?.toPreferencesDto() ?: PreferencesDto(),
    triage = optObject("triage")?.toUrgencyResultDto() ?: UrgencyResultDto(),
    conversationState = optObject("conversation_state")?.toConversationStateDto() ?: ConversationStateDto(),
    departmentResult = optObject("department_result")?.toDepartmentResultDto(),
    confirmed = optBoolean("confirmed", false),
    recommendationGenerated = optBoolean("recommendation_generated", false),
    scriptGenerated = optBoolean("script_generated", false)
)

private fun AvailabilityDto.toJsonObject(): JSONObject = JSONObject().apply {
    put("preferred_dates", JSONArray(preferredDates))
    put("preferred_days", JSONArray(preferredDays))
    put("preferred_sessions", JSONArray(preferredSessions))
    put("can_take_leave", canTakeLeave)
}

private fun JSONObject.toPatientInputDto(): PatientInputDto = PatientInputDto(
    symptom = optString("symptom", ""),
    bodyPart = optNullableString("body_part"),
    duration = optNullableString("duration"),
    severity = optNullableString("severity"),
    onset = optNullableString("onset"),
    accompanyingSymptoms = optArray("accompanying_symptoms").toStringList(),
    redFlags = optArray("red_flags").toStringList(),
    collectedFields = optArray("collected_fields").toStringList(),
    redFlagsChecked = optBoolean("red_flags_checked", false)
)

private fun JSONObject.toAvailabilityDto(): AvailabilityDto = AvailabilityDto(
    preferredDates = optArray("preferred_dates").toStringList(),
    preferredDays = optArray("preferred_days").toStringList(),
    preferredSessions = optArray("preferred_sessions").toStringList(),
    canTakeLeave = optBoolean("can_take_leave", false),
    semanticStatus = optObject("semantic_status").toStringMap(),
    confidence = optObject("confidence").toDoubleMap()
)

private fun JSONObject.toPreferencesDto(): PreferencesDto = PreferencesDto(
    specialtyPriority = optBoolean("specialty_priority", true),
    doctorPreference = optString("doctor_preference", "不限"),
    hospitalPreference = optString("hospital_preference", "台北榮總")
)

private fun JSONObject.toConversationStateDto(): ConversationStateDto = ConversationStateDto(
    stage = optString("stage", "collecting"),
    isComplete = optBoolean("is_complete", false),
    awaitingConfirmation = optBoolean("awaiting_confirmation", false),
    confirmed = optBoolean("confirmed", false),
    askedFields = optArray("asked_fields").toStringList(),
    consumedFields = optArray("consumed_fields").toStringList(),
    lastQuestionKey = optNullableString("last_question_key"),
    questionAttempts = optObject("question_attempts").toIntMap(),
    fieldStatuses = optObject("field_statuses").toStringMap(),
    fieldConfidence = optObject("field_confidence").toDoubleMap(),
    clarificationReasons = optObject("clarification_reasons").toStringMap()
)

private fun JSONObject.toUrgencyResultDto(): UrgencyResultDto = UrgencyResultDto(
    urgencyScore = optNullableInt("urgency_score"),
    urgencyLevel = optNullableString("urgency_level"),
    warningRequired = optBoolean("warning_required", false),
    warningMessage = optNullableString("warning_message"),
    needMoreInfo = optBoolean("need_more_info", true),
    nextQuestion = optNullableString("next_question"),
    reasons = optArray("reasons").toStringList(),
    isFinal = optBoolean("is_final", false)
)

private fun JSONObject.toDepartmentResultDto(): DepartmentResultDto = DepartmentResultDto(
    deptId = optNullableInt("dept_id"),
    parentDept = optString("parentDept", ""),
    childDept = optString("childDept", ""),
    confidence = optDouble("confidence", 0.0),
    reason = optArray("reason").toStringList()
)

private fun JSONArray?.toMessageList(): List<MessageDto> {
    if (this == null) return emptyList()
    return (0 until length()).mapNotNull { index ->
        optJSONObject(index)?.let {
            MessageDto(
                role = it.optString("role"),
                content = it.optString("content")
            )
        }
    }
}

private fun JSONArray?.toQuestionList(): List<QuestionItemDto> {
    if (this == null) return emptyList()
    return (0 until length()).mapNotNull { index ->
        optJSONObject(index)?.let { item ->
            val key = item.optString("key").trim()
            val question = item.optString("question").trim()
            if (key.isBlank() || question.isBlank()) return@let null
            QuestionItemDto(
                key = key,
                question = question,
                questionId = item.optNullableString("question_id"),
                stateField = item.optNullableString("state_field"),
                required = item.optBoolean("required", true),
                inputType = item.optNullableString("input_type")
            )
        }
    }
}

private fun JSONArray?.toRecommendationList(): List<RecommendationItemDto> {
    if (this == null) return emptyList()
    return (0 until length()).mapNotNull { index ->
        optJSONObject(index)?.let {
            RecommendationItemDto(
                recommendationId = it.optString("recommendation_id"),
                parentDept = it.optString("parentDept"),
                childDept = it.optString("childDept"),
                doctor = it.optString("doctor"),
                date = it.optString("date"),
                session = it.optString("session"),
                slot = it.optString("slot", ""),
                score = it.optDouble("score", 0.0),
                reasons = it.optArray("reasons").toStringList(),
                rank = it.optNullableInt("rank"),
                isBestMatch = it.optBoolean("is_best_match", false),
                doctorId = it.optNullableString("doctor_id"),
                scheduleId = it.optNullableString("schedule_id"),
                sessionTime = it.optNullableString("session_time"),
                room = it.optNullableString("room"),
                visitType = it.optNullableString("visit_type"),
                specialtyTags = it.optStringList("specialty_tags"),
                specialtyScore = it.optNullableDouble("specialty_score"),
                timeScore = it.optNullableDouble("time_score"),
                matchReason = it.optNullableString("match_reason"),
                deptId = it.optNullableInt("dept_id")
            )
        }
    }
}

private fun RecommendationItemDto.toJsonObject(): JSONObject = JSONObject().apply {
    put("recommendation_id", recommendationId)
    put("parentDept", parentDept)
    put("childDept", childDept)
    put("doctor", doctor)
    put("date", date)
    put("session", session)
    put("slot", slot)
    put("score", score)
    put("reasons", JSONArray(reasons))
    put("is_best_match", isBestMatch)
    rank?.let { put("rank", it) }
    doctorId?.let { put("doctor_id", it) }
    scheduleId?.let { put("schedule_id", it) }
    sessionTime?.let { put("session_time", it) }
    room?.let { put("room", it) }
    visitType?.let { put("visit_type", it) }
    if (specialtyTags.isNotEmpty()) put("specialty_tags", specialtyTags.joinToString(","))
    specialtyScore?.let { put("specialty_score", it) }
    timeScore?.let { put("time_score", it) }
    matchReason?.let { put("match_reason", it) }
    deptId?.let { put("dept_id", it) }
}

private fun JSONArray?.toFallbackDepartmentList(): List<FallbackDepartmentDto> {
    if (this == null) return emptyList()
    return (0 until length()).mapNotNull { index ->
        optJSONObject(index)?.let {
            FallbackDepartmentDto(
                parentDept = it.optString("parentDept"),
                childDept = it.optString("childDept"),
                reason = it.optString("reason")
            )
        }
    }
}

private fun JSONArray?.toScriptStepList(): List<ScriptStepDto> {
    if (this == null) return emptyList()
    return (0 until length()).mapNotNull { index ->
        optJSONObject(index)?.let {
            ScriptStepDto(
                action = it.optString("action"),
                target = it.optString("target"),
                resourceId = it.optNullableString("resource_id"),
                text = it.optNullableString("text"),
                className = it.optNullableString("class_name"),
                description = it.optNullableString("description"),
                delayMs = it.optInt("delay_ms", 0),
                retry = it.optInt("retry", 0)
            )
        }
    }
}

private fun JSONArray?.toStringList(): List<String> {
    if (this == null) return emptyList()
    return (0 until length()).mapNotNull { index -> opt(index)?.toString() }
}

private fun JSONObject.optStringList(name: String): List<String> {
    optJSONArray(name)?.let { return it.toStringList() }
    val raw = optNullableString(name)?.trim().orEmpty()
    if (raw.isBlank()) return emptyList()
    return raw
        .split("、", ",", ";", "；")
        .map { it.trim() }
        .filter { it.isNotBlank() }
}

private fun JSONObject?.toStringMap(): Map<String, String> {
    if (this == null) return emptyMap()
    return keys().asSequence().associateWith { key -> optString(key) }
}

private fun JSONObject?.toIntMap(): Map<String, Int> {
    if (this == null) return emptyMap()
    return keys().asSequence().associateWith { key -> optInt(key, 0) }
}

private fun JSONObject?.toDoubleMap(): Map<String, Double> {
    if (this == null) return emptyMap()
    return keys().asSequence().associateWith { key -> optDouble(key, 0.0) }
}

private fun JSONObject.optObject(name: String): JSONObject? =
    if (isNull(name)) null else optJSONObject(name)

private fun JSONObject.optArray(name: String): JSONArray? =
    if (isNull(name)) null else optJSONArray(name)

private fun JSONObject.optNullableString(name: String): String? =
    if (isNull(name)) null else optString(name)

private fun JSONObject.optNullableInt(name: String): Int? =
    if (isNull(name)) null else optInt(name)

private fun JSONObject.optNullableDouble(name: String): Double? =
    if (isNull(name)) null else optDouble(name)
