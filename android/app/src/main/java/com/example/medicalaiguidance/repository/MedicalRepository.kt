package com.example.medicalaiguidance.repository

import android.content.Context
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import com.example.medicalaiguidance.model.Appointment
import com.example.medicalaiguidance.model.ChatMessage
import com.example.medicalaiguidance.model.Department
import com.example.medicalaiguidance.model.Doctor
import com.example.medicalaiguidance.model.History
import com.example.medicalaiguidance.model.HistoryRecommendation
import com.example.medicalaiguidance.model.resolveHistorySummary
import com.example.medicalaiguidance.model.HistoryStatus
import com.example.medicalaiguidance.model.MessageSender
import com.example.medicalaiguidance.network.ChatRequest
import com.example.medicalaiguidance.network.CaseResumeDto
import com.example.medicalaiguidance.network.BatchAnswerDto
import com.example.medicalaiguidance.network.FollowupRecommendRequest
import com.example.medicalaiguidance.network.FollowupRecommendationResultDto
import com.example.medicalaiguidance.network.MedicalApiClient
import com.example.medicalaiguidance.network.QuickSearchRequest
import com.example.medicalaiguidance.network.QuickSearchResultDto
import com.example.medicalaiguidance.network.RecommendRequest
import com.example.medicalaiguidance.network.RecommendationItemDto
import com.example.medicalaiguidance.network.RecommendationResultDto
import com.example.medicalaiguidance.network.ScriptRequest
import com.example.medicalaiguidance.network.ScriptResponseDto
import com.example.medicalaiguidance.network.TriageResultDto
import com.example.medicalaiguidance.network.TtsRequest
import com.example.medicalaiguidance.network.VoiceAsrResponseDto
import com.example.medicalaiguidance.network.VoiceChatResponseDto
import com.example.medicalaiguidance.network.VoiceTtsResponseDto
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONArray
import org.json.JSONObject

class MedicalRepository(
    private val apiClient: MedicalApiClient = MedicalApiClient()
) {
    companion object {
        private const val PREFS_NAME = "medical_guidance_history"
        private const val HISTORY_KEY = "history_items"

        private var appContext: Context? = null
        private var currentDepartment: Department? = null
        private var currentCaseId: String? = null
        private var currentVisitType: String? = null
        private var currentRecommendation: RecommendationItemDto? = null
        private var currentScript: ScriptResponseDto? = null
        // Memory only: persisted history never proves that a Backend case still exists.
        private val liveTriageResults = mutableMapOf<String, TriageResultDto>()
        private val chatMessages = mutableListOf<ChatMessage>()
        private val historyItems = mutableListOf<History>()
        private val _historyFlow = MutableStateFlow<List<History>>(emptyList())
        private val _historyLoadError = MutableStateFlow<String?>(null)
        private var historyLoaded = false
        private val voiceCleanupScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
        private var currentTtsSession: TtsSession? = null

        fun initialize(context: Context) {
            appContext = context.applicationContext
            loadHistoryIfNeeded()
        }

        private fun loadHistoryIfNeeded() {
            if (historyLoaded) return
            val stored = appContext
                ?.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                ?.getString(HISTORY_KEY, null)

            if (stored.isNullOrBlank()) {
                historyItems.clear()
                _historyFlow.value = emptyList()
                _historyLoadError.value = null
                historyLoaded = true
                return
            }

            runCatching { parseHistoryJson(stored) }
                .onSuccess { loadedHistory ->
                    historyItems.clear()
                    historyItems.addAll(loadedHistory)
                    _historyFlow.value = historyItems.toList()
                    _historyLoadError.value = null
                }
                .onFailure { error ->
                    historyItems.clear()
                    _historyFlow.value = emptyList()
                    _historyLoadError.value =
                        "歷史紀錄格式損壞，無法載入（${error::class.simpleName ?: "ParseError"}）"
                }
            historyLoaded = true
        }

        private fun persistHistory() {
            val context = appContext ?: return
            val array = JSONArray()
            historyItems.forEach { history -> array.put(history.toJson()) }
            context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .putString(HISTORY_KEY, array.toString())
                .apply()
        }

        private fun buildVghDepartment(parentDept: String, childDept: String): Department {
            val clinicName = childDept.ifBlank { parentDept }
            val parentName = findVghParentDepartment(clinicName)
                ?: normalizeBackendParentDepartment(parentDept)
                ?: clinicName
            return Department(
                id = clinicName,
                name = parentName,
                clinicName = clinicName
            )
        }

        private fun findVghParentDepartment(clinicName: String): String? {
            if (clinicName.isBlank()) return null
            val context = appContext ?: return null
            return runCatching {
                val json = context.assets.open("vgh_departments.json")
                    .bufferedReader(Charsets.UTF_8)
                    .use { JSONObject(it.readText()) }
                json.keys().asSequence().firstOrNull { parent ->
                    parent == clinicName ||
                        json.optJSONArray(parent)?.let { children ->
                            (0 until children.length()).any { index -> children.optString(index) == clinicName }
                        } == true
                }
            }.getOrNull()
        }

        private fun normalizeBackendParentDepartment(parentDept: String): String? {
            val text = parentDept.trim()
            if (text.isBlank()) return null
            return when (text) {
                "外科部" -> "外科系"
                "內科部" -> "內科系"
                "婦女醫學部" -> "婦女醫學部"
                else -> text
            }
        }
    }

    suspend fun chat(caseId: String?, message: String, visitType: String? = currentVisitType): TriageResultDto =
        writable(caseId) { apiClient.chat(ChatRequest(caseId = caseId, message = message, visitType = visitType)) }
            .also(::consumeTriageResult)

    suspend fun resumeCase(caseId: String): CaseResumeDto = apiClient.resumeCase(caseId)

    fun acceptResumedCase(result: CaseResumeDto) {
        require(result.canContinue) { "無法安全續接原問診，請重新開始。" }
        ensureWritable(result.caseId)
        setActiveVisitType(result.visitType)
        consumeTriageResult(result.toTriageResult())
    }

    private fun ensureWritable(caseId: String?) {
        check(caseId == null || getHistoryById(caseId)?.status != HistoryStatus.COMPLETED) {
            "此導引紀錄已完成，只能查看；請建立新的問診。"
        }
    }

    private suspend fun <T> writable(caseId: String?, action: suspend () -> T): T {
        ensureWritable(caseId)
        val result = action()
        ensureWritable(caseId)
        return result
    }

    suspend fun startBatchTriage(visitType: String): TriageResultDto =
        apiClient.chat(ChatRequest(visitType = visitType, legacyBatch = true)).also(::consumeTriageResult)

    suspend fun startNaturalTriage(visitType: String): TriageResultDto =
        apiClient.chat(ChatRequest(visitType = visitType)).also(::consumeTriageResult)

    suspend fun submitBatchAnswers(
        caseId: String,
        visitType: String,
        answers: List<BatchAnswerDto>
    ): TriageResultDto = writable(caseId) { apiClient.chat(
        ChatRequest(caseId = caseId, visitType = visitType, answers = answers)
    ) }.also(::consumeTriageResult)

    suspend fun confirmTriage(caseId: String): TriageResultDto =
        writable(caseId) { apiClient.chat(ChatRequest(caseId = caseId, visitType = currentVisitType, confirmed = true)) }
            .also { result ->
                if (result.caseId != caseId) liveTriageResults.remove(caseId)
                consumeTriageResult(result)
            }

    suspend fun requestRevision(caseId: String): TriageResultDto =
        writable(caseId) { apiClient.chat(ChatRequest(caseId = caseId, visitType = currentVisitType, revisionRequested = true)) }
            .also(::consumeTriageResult)

    suspend fun recommend(caseId: String, visitType: String): RecommendationResultDto {
        ensureWritable(caseId)
        val effectiveVisitType = resolveRecommendationVisitType(currentVisitType, visitType)
        return apiClient.recommend(RecommendRequest(caseId = caseId, visitType = effectiveVisitType)).also { result ->
            saveRecommendationSnapshot(caseId, result)
        }
    }

    suspend fun followupRecommend(request: FollowupRecommendRequest): FollowupRecommendationResultDto =
        apiClient.followupRecommend(request)

    suspend fun quickSearch(request: QuickSearchRequest): QuickSearchResultDto =
        apiClient.quickSearch(request)

    suspend fun referenceDepartments() = apiClient.referenceDepartments()

    suspend fun referenceDoctors(department: String) = apiClient.referenceDoctors(department)

    suspend fun generateScript(
        caseId: String,
        recommendationId: String,
        recommendation: RecommendationItemDto? = null
    ): ScriptResponseDto =
        writable(caseId) { apiClient.generateScript(
            ScriptRequest(
                caseId = caseId,
                recommendationId = recommendationId,
                recommendation = recommendation
            )
        ) }.also { response ->
            if (response.isSuccess) {
                require(currentCaseId == caseId) { "目前問診案件已變更，請重新選擇班表。" }
                require(response.recommendationId == recommendationId &&
                    (response.recommendation == null || response.recommendation.recommendationId == recommendationId)) {
                    "重新確認的班表識別資料不一致，請重新選擇。"
                }
                if (recommendation != null && response.recommendation != null) {
                    val updated = response.recommendation
                    require(updated.scheduleId == recommendation.scheduleId &&
                        updated.doctorId == recommendation.doctorId &&
                        updated.deptId == recommendation.deptId && updated.date == recommendation.date &&
                        accessibilitySession(updated.session) == accessibilitySession(recommendation.session)) {
                        "重新確認的班表與所選診次不一致，請重新選擇。"
                    }
                }
                if (response.recommendation == null) recommendation?.let(::selectRecommendation)
            }
            consumeScriptResponse(response)
        }

    internal fun consumeScriptResponse(response: ScriptResponseDto) {
        if (response.isSuccess) {
            ensureWritable(currentCaseId)
            response.recommendation?.let(::selectRecommendation)
            currentScript = response
            currentRecommendation?.takeIf { it.recommendationId == response.recommendationId }?.let { item ->
                currentCaseId?.let { id ->
                    updateHistory(id) { history ->
                        val snapshot = item.toHistoryRecommendation()
                        history.copy(
                            recommendations = if (history.recommendations.any { it.recommendationId == snapshot.recommendationId }) {
                                history.recommendations
                            } else history.recommendations + snapshot,
                            selectedRecommendationId = snapshot.recommendationId
                        )
                    }
                }
            }
        }
    }

    fun validatedGuidanceSelection(): Pair<String, String> {
        val id = requireNotNull(currentCaseId) { "缺少有效掛號案件，請重新選擇。" }
        ensureWritable(id)
        val item = requireNotNull(currentRecommendation) { "尚未選擇正式班表。" }
        check(!item.doctorId.isNullOrBlank() && !item.scheduleId.isNullOrBlank() &&
            (item.deptId ?: 0) > 0 && item.date.isNotBlank() && item.recommendationId.isNotBlank()) {
            "正式班表識別資料不完整，請重新選擇。"
        }
        accessibilitySession(item.session)
        check(currentVisitType in setOf("initial", "followup", "quick_search")) { "無法確認就診類型。" }
        check(currentScript?.isSuccess == true && currentScript?.recommendationId == item.recommendationId) {
            "班表尚未重新驗證，請重新選擇。"
        }
        return id to item.recommendationId
    }

    fun completeGuidanceAfterLaunch(selection: Pair<String, String>) {
        check(validatedGuidanceSelection() == selection) { "掛號案件已變更，無法更新歷史。" }
        updateHistory(selection.first) { history ->
            history.copy(status = HistoryStatus.COMPLETED, completedAt = completionTimeText(),
                typeTitle = currentDepartment?.clinicName ?: history.typeTitle)
        }
    }

    fun beginTtsSession(): TtsSession {
        currentTtsSession?.let { old ->
            old.close()
            voiceCleanupScope.launch { cleanupRemoteTts(old.id) }
        }
        return TtsSession(CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)) { id, text, lang ->
            apiClient.tts(TtsRequest(text = text, lang = lang, sessionId = id))
        }.also { currentTtsSession = it }
    }

    suspend fun cleanupTtsBeforeHospitalLaunch() {
        val session = currentTtsSession ?: return
        currentTtsSession = null
        session.close()
        // Independent scope lets the Intent proceed after 1.5s even if blocking HTTP is slow.
        val cleanup = voiceCleanupScope.async { cleanupRemoteTts(session.id) }
        if (withTimeoutOrNull(1500) { cleanup.await(); true } == null) {
            Log.w("MedicalRepository", "TTS cleanup timed out; continuing hospital launch")
        }
    }

    private suspend fun cleanupRemoteTts(sessionId: String) {
        try {
            apiClient.cleanupTts(sessionId)
        } catch (error: Exception) {
            Log.w("MedicalRepository", "TTS cleanup failed; backend TTL remains active", error)
        }
    }

    suspend fun transcribeVoice(audioBytes: ByteArray, lang: String): VoiceAsrResponseDto =
        apiClient.voiceAsr(audioBytes = audioBytes, lang = lang)

    suspend fun voiceChat(
        audioBytes: ByteArray,
        caseId: String?,
        lang: String,
        confirmed: Boolean = false
    ): VoiceChatResponseDto =
        writable(caseId) { apiClient.voiceChat(
            audioBytes = audioBytes,
            caseId = caseId,
            lang = lang,
            confirmed = confirmed,
            visitType = currentVisitType
        ) }

    fun getAllHistory(): StateFlow<List<History>> {
        loadHistoryIfNeeded()
        return _historyFlow.asStateFlow()
    }

    fun getHistoryLoadError(): StateFlow<String?> {
        loadHistoryIfNeeded()
        return _historyLoadError.asStateFlow()
    }

    fun getHistoryById(id: String): History? {
        loadHistoryIfNeeded()
        return historyItems.find { it.id == id }
    }

    fun loadHistoryIntoCurrentChat(id: String): History? {
        chatMessages.clear()
        currentCaseId = null
        currentVisitType = null
        currentRecommendation = null
        currentScript = null
        currentDepartment = null

        val history = getHistoryById(id) ?: return null
        currentVisitType = history.visitType?.takeIf { it in CANONICAL_VISIT_TYPES }
        currentDepartment = history.typeTitle
            .takeIf {
                history.status == HistoryStatus.COMPLETED &&
                    it !in setOf("掛號導引", "AI 問診", "症狀評估中")
            }
            ?.let { buildVghDepartment(parentDept = "", childDept = it) }
        liveTriageResults[id]?.let { result ->
            (result.departmentResult ?: result.triageCase?.departmentResult)?.let {
                currentDepartment = buildVghDepartment(it.parentDept, it.childDept)
            }
        }
        chatMessages.addAll(history.chatMessages)
        return history
    }

    fun getLiveTriageResult(caseId: String): TriageResultDto? = liveTriageResults[caseId]

    fun saveToHistory(history: History) {
        loadHistoryIfNeeded()
        if (historyItems.any { it.id == history.id && it.status == HistoryStatus.COMPLETED }) return
        historyItems.removeAll { it.id == history.id }
        historyItems.add(0, history)
        persistHistory()
        _historyFlow.value = historyItems.toList()
        _historyLoadError.value = null
    }

    fun deleteHistory(id: String) {
        loadHistoryIfNeeded()
        historyItems.removeAll { it.id == id }
        liveTriageResults.remove(id)
        persistHistory()
        _historyFlow.value = historyItems.toList()
    }

    fun saveCurrentChatToHistory(
        historyId: String,
        summaryText: String,
        completed: Boolean,
        doctorName: String? = null,
        allowReopen: Boolean = false
    ) {
        val existing = getHistoryById(historyId)
        if (existing?.status == HistoryStatus.COMPLETED) return
        // Backend workflow completion never completes local guidance history.
        val typeTitle = currentDepartment?.clinicName ?: existing?.typeTitle ?: "AI 問診"
        val recommendations = existing?.recommendations.orEmpty()
        val history = History(
            id = historyId,
            date = existing?.date ?: todayText(),
            typeTitle = typeTitle,
            summaryText = resolveHistorySummary(summaryText, chatMessages),
            status = HistoryStatus.UNCOMPLETED,
            chatMessages = chatMessages.toList(),
            completedAt = null,
            recommendations = recommendations,
            selectedRecommendationId = existing?.selectedRecommendationId
                ?.takeIf { selectedId -> recommendations.any { it.recommendationId == selectedId } },
            visitType = existing?.visitType ?: currentVisitType
        )
        saveToHistory(history)
    }

    fun getChatMessages(): List<ChatMessage> = chatMessages

    fun addMessage(message: ChatMessage) {
        chatMessages.add(message)
    }

    fun clearChatMessages() {
        chatMessages.clear()
    }

    fun setActiveCaseId(caseId: String?) {
        currentCaseId = caseId
    }

    fun getActiveCaseId(): String? = currentCaseId

    fun setActiveVisitType(visitType: String?) {
        val normalized = visitType?.takeIf { it in CANONICAL_VISIT_TYPES }
        if (currentVisitType != null && normalized != null && currentVisitType != normalized) {
            error("同一問診案件不可變更 visit_type。")
        }
        currentVisitType = normalized
    }

    fun getActiveVisitType(): String? = currentVisitType

    fun clearRecommendationFlow() {
        currentCaseId = null
        currentVisitType = null
        currentRecommendation = null
        currentScript = null
        currentDepartment = null
    }

    fun selectRecommendation(item: RecommendationItemDto) {
        ensureWritable(currentCaseId)
        currentRecommendation = item
        currentScript = null
        currentDepartment = buildVghDepartment(
            parentDept = item.parentDept,
            childDept = item.childDept
        )
    }

    fun getSelectedRecommendation(): RecommendationItemDto? = currentRecommendation

    fun getCurrentGuidanceScript(): ScriptResponseDto? = currentScript

    fun getConfirmedAppointment(): Appointment {
        val item = currentRecommendation
            ?: error("尚未取得後端推薦結果，無法建立掛號資料。請先完成 /recommend 並選擇一筆推薦。")
        val department = currentDepartment ?: buildVghDepartment(
            parentDept = item.parentDept,
            childDept = item.childDept
        )
        return Appointment(
            id = item.recommendationId,
            date = item.date,
            dayOfWeek = item.date,
            timeSlot = accessibilitySession(item.session),
            sessionTime = item.sessionTime,
            department = department,
            doctor = item.toDoctor()
        )
    }

    internal fun saveRecommendationSnapshot(caseId: String, result: RecommendationResultDto) {
        require(result.caseId == caseId) { "推薦案件與目前問診不一致，請重新取得推薦。" }
        val recommendationSnapshots = (
            result.recommendations.specialtyFirst.map { it.toHistoryRecommendation().copy(category = "specialty") } +
                result.recommendations.timeFirst.map { it.toHistoryRecommendation().copy(category = "time") }
        ).distinctBy { it.recommendationId }
        updateHistory(caseId) { history ->
            val selected = history.recommendations.find { it.recommendationId == history.selectedRecommendationId }
            history.copy(
                recommendations = (recommendationSnapshots + listOfNotNull(selected)).distinctBy { it.recommendationId },
                selectedRecommendationId = history.selectedRecommendationId
            )
        }
    }

    private fun updateHistory(id: String, transform: (History) -> History) {
        loadHistoryIfNeeded()
        val index = historyItems.indexOfFirst { it.id == id }
        if (index < 0) return
        if (historyItems[index].status == HistoryStatus.COMPLETED) return
        historyItems[index] = transform(historyItems[index])
        persistHistory()
        _historyFlow.value = historyItems.toList()
    }

    internal fun consumeTriageResult(result: TriageResultDto) {
        liveTriageResults[result.caseId] = result
        setActiveCaseId(result.caseId)
        val responseVisitType = result.triageCase?.visitType
        if (responseVisitType in CANONICAL_VISIT_TYPES) {
            setActiveVisitType(responseVisitType)
        }
        val departmentResult = result.departmentResult ?: result.triageCase?.departmentResult
        if (departmentResult != null && departmentResult.childDept.isNotBlank()) {
            currentDepartment = buildVghDepartment(
                parentDept = departmentResult.parentDept,
                childDept = departmentResult.childDept
            )
        }
    }
}

private fun RecommendationItemDto.toDoctor(): Doctor =
    Doctor(
        id = doctorId ?: recommendationId,
        name = doctor.ifBlank { "Recommended doctor" },
        departmentId = childDept.ifBlank { parentDept },
        title = room?.takeIf { it.isNotBlank() } ?: matchReason?.takeIf { it.isNotBlank() } ?: "Backend recommendation",
        specialties = specialtyTags.ifEmpty { reasons },
        imageUrl = null,
        availableSlots = listOf(date to (sessionTime ?: session))
    )

internal fun RecommendationItemDto.toHistoryRecommendation(): HistoryRecommendation =
    HistoryRecommendation(
        recommendationId = recommendationId,
        parentDepartment = parentDept,
        department = childDept.ifBlank { parentDept },
        doctor = doctor,
        date = date,
        session = session,
        sessionTime = sessionTime,
        room = room ?: slot.takeIf { it.isNotBlank() }
    )

internal fun accessibilitySession(session: String): String = when (session.trim().lowercase(Locale.ROOT)) {
    "上午", "早上", "morning" -> "上午"
    "下午", "午診", "afternoon" -> "下午"
    "晚上", "晚診", "夜診", "夜間", "evening" -> "晚上"
    else -> error("無法辨識正式看診時段，請重新選擇班表。")
}

private fun todayText(): String =
    SimpleDateFormat("yyyy/MM/dd", Locale.TAIWAN).format(Date())

private fun completionTimeText(): String =
    SimpleDateFormat("yyyy/MM/dd HH:mm", Locale.TAIWAN).format(Date())

internal fun History.toJson(): JSONObject = JSONObject().apply {
    put("id", id)
    put("date", date)
    put("typeTitle", typeTitle)
    put("summaryText", summaryText)
    put("status", status.name)
    visitType?.let { put("visitType", it) }
    completedAt?.let { put("completedAt", it) }
    selectedRecommendationId?.let { put("selectedRecommendationId", it) }
    put("chatMessages", JSONArray().also { array ->
        chatMessages.forEach { message ->
            array.put(JSONObject().apply {
                put("id", message.id)
                put("content", message.content)
                put("sender", message.sender.name)
                put("timestamp", message.timestamp)
            })
        }
    })
    put("recommendations", JSONArray().also { array ->
        recommendations.forEach { recommendation ->
            array.put(JSONObject().apply {
                put("recommendationId", recommendation.recommendationId)
                put("parentDepartment", recommendation.parentDepartment)
                put("department", recommendation.department)
                put("doctor", recommendation.doctor)
                put("date", recommendation.date)
                put("session", recommendation.session)
                recommendation.sessionTime?.let { put("sessionTime", it) }
                recommendation.room?.let { put("room", it) }
                recommendation.category?.let { put("category", it) }
            })
        }
    })
}

internal fun parseHistoryJson(json: String): List<History> {
    val array = JSONArray(json)
    return (0 until array.length()).map { index ->
        array.getJSONObject(index).toHistory()
    }
}

private fun JSONObject.toHistory(): History {
    val messages = optJSONArray("chatMessages")?.let { array ->
        (0 until array.length()).mapNotNull { index ->
            array.optJSONObject(index)?.let { item ->
                ChatMessage(
                    id = item.optString("id"),
                    content = item.optString("content"),
                    sender = runCatching {
                        MessageSender.valueOf(item.optString("sender"))
                    }.getOrDefault(MessageSender.AI),
                    timestamp = item.optLong("timestamp", System.currentTimeMillis())
                )
            }
        }
    } ?: emptyList()
    val recommendations = optJSONArray("recommendations")?.let { array ->
        (0 until array.length()).mapNotNull { index ->
            array.optJSONObject(index)?.let { item ->
                HistoryRecommendation(
                    recommendationId = item.optString("recommendationId"),
                    parentDepartment = item.optString("parentDepartment"),
                    department = item.optString("department"),
                    doctor = item.optString("doctor"),
                    date = item.optString("date"),
                    session = item.optString("session"),
                    sessionTime = item.optString("sessionTime").takeIf { it.isNotBlank() },
                    room = item.optString("room").takeIf { it.isNotBlank() },
                    category = item.optString("category").takeIf { it in setOf("specialty", "time") }
                )
            }
        }
    } ?: emptyList()

    val typeTitle = optString("typeTitle", "AI 問診")
    val status = runCatching {
        HistoryStatus.valueOf(optString("status"))
    }.getOrDefault(HistoryStatus.UNCOMPLETED)
    val compatibleRecommendations = recommendations
    val selectedRecommendationId = optString("selectedRecommendationId")
        .takeIf { selectedId ->
            selectedId.isNotBlank() &&
                compatibleRecommendations.any { it.recommendationId == selectedId }
        }

    return History(
        id = optString("id"),
        date = optString("date"),
        typeTitle = typeTitle,
        summaryText = optString("summaryText", "問診紀錄"),
        status = status,
        chatMessages = messages,
        completedAt = optString("completedAt").takeIf { it.isNotBlank() },
        recommendations = compatibleRecommendations,
        selectedRecommendationId = selectedRecommendationId,
        visitType = optString("visitType").takeIf { it in CANONICAL_VISIT_TYPES }
    )
}

private val CANONICAL_VISIT_TYPES = setOf("initial", "followup", "quick_search", "return_visit")
private val RECOMMENDATION_VISIT_TYPES = setOf("initial", "followup", "return_visit")

internal fun resolveRecommendationVisitType(storedVisitType: String?, requestedVisitType: String): String =
    storedVisitType?.takeIf { it in RECOMMENDATION_VISIT_TYPES }
        ?: requestedVisitType.takeIf { it in RECOMMENDATION_VISIT_TYPES }
        ?: throw IllegalStateException("缺少合法 visit_type，請重新選擇就診類型。")
