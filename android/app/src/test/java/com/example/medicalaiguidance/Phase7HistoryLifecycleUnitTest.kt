package com.example.medicalaiguidance

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.medicalaiguidance.model.*
import com.example.medicalaiguidance.network.*
import com.example.medicalaiguidance.repository.*
import com.example.medicalaiguidance.viewmodel.ChatViewModel
import com.example.medicalaiguidance.viewmodel.ConfirmViewModel
import kotlinx.coroutines.*
import kotlinx.coroutines.test.*
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.Assert.*

@OptIn(ExperimentalCoroutinesApi::class)
class Phase7HistoryLifecycleUnitTest {
    private val dispatcher = StandardTestDispatcher()
    private val client = FakeClient()
    private val repository = MedicalRepository(client)
    private val models = mutableListOf<ViewModel>()
    private val session = TtsSession(CoroutineScope(SupervisorJob() + dispatcher)) { _, _, _ -> error("no TTS") }
    private val id = "case_history_${System.nanoTime()}"
    private val other = "${id}_other"

    @Before fun setUp() {
        Dispatchers.setMain(dispatcher)
        repository.clearRecommendationFlow()
        repository.clearChatMessages()
    }

    @After fun tearDown() {
        models.forEach { it.viewModelScope.cancel() }
        session.close()
        repository.deleteHistory(id)
        repository.deleteHistory(other)
        repository.clearRecommendationFlow()
        repository.clearChatMessages()
        Dispatchers.resetMain()
    }

    private fun history(caseId: String = id) = History(caseId, "2026/10/09", "AI 問診", "summary",
        HistoryStatus.UNCOMPLETED, chatMessages = listOf(ChatMessage(content = "原對話", sender = MessageSender.USER)),
        visitType = "initial")

    private fun item() = RecommendationItemDto("rec_$id", "五官科", "耳科", "測試醫師", "2099-10-13", "下午",
        score = 50.0, doctorId = "D1", scheduleId = "S1", deptId = 1333, room = "B")

    private fun readySelection(withHistory: Boolean = true) {
        if (withHistory) repository.saveToHistory(history())
        repository.setActiveCaseId(id)
        repository.setActiveVisitType(if (withHistory) "initial" else "quick_search")
        repository.consumeScriptResponse(ScriptResponseDto(true, recommendationId = item().recommendationId,
            recommendation = item(), steps = emptyList()))
    }

    private fun chatModel() = ChatViewModel(repository, ttsSessionFactory = { session }).also(models::add)
    private fun confirmModel() = ConfirmViewModel(repository, voiceCleanup = {}).also(models::add)
    private fun resume(caseId: String = id, stage: String = "collecting", key: String? = "duration") = CaseResumeDto(
        caseId, "initial", ConversationStateDto(stage = stage, confirmed = stage != "collecting" && stage != "waiting_confirmation",
            lastQuestionKey = key, departmentStatus = if (stage == "collecting") "ambiguous" else "resolved"),
        true, "negative", false, null, DepartmentResultDto(1333, "五官科", "耳科"),
        if (stage == "collecting") "請問持續多久？" else null,
        if (stage == "collecting" && key != null) listOf(QuestionItemDto(key, "請問持續多久？")) else emptyList(), true
    )

    @Test fun backendCompletionAndScriptSuccessNeverCompleteLocalHistory() {
        repository.setActiveCaseId(id)
        repository.addMessage(ChatMessage(content = "原對話", sender = MessageSender.USER))
        for (completed in listOf(false, true)) repository.saveCurrentChatToHistory(id, "summary", completed)
        assertEquals(HistoryStatus.UNCOMPLETED, repository.getHistoryById(id)?.status)
        readySelection()
        repository.saveCurrentChatToHistory(id, "返回聊天", completed = true, allowReopen = true)
        assertEquals(HistoryStatus.UNCOMPLETED, repository.getHistoryById(id)?.status)
        assertEquals(item().recommendationId, repository.getHistoryById(id)?.selectedRecommendationId)
        assertNull(repository.getHistoryById(id)?.completedAt)
    }

    @Test fun cancelFinalDialogDoesNotCompleteHistory() {
        readySelection()
        confirmModel() // Cancel never invokes the final-launch operation.
        assertEquals(HistoryStatus.UNCOMPLETED, repository.getHistoryById(id)?.status)
    }

    @Test fun launchFailureKeepsHistoryUncompletedAndCanRetry() = runTest(dispatcher) {
        readySelection()
        val model = confirmModel()
        model.launchHospitalAfterVoiceCleanup { error("ActivityNotFound") }
        runCurrent()
        assertEquals(HistoryStatus.UNCOMPLETED, repository.getHistoryById(id)?.status)
        assertTrue(model.errorMessage.value.orEmpty().contains("無法完成前往"))
        model.launchHospitalAfterVoiceCleanup { }
        runCurrent()
        assertEquals(HistoryStatus.COMPLETED, repository.getHistoryById(id)?.status)
    }

    @Test fun successfulExternalLaunchCompletesSameCaseOnlyAfterLaunchAndPreservesSnapshot() = runTest(dispatcher) {
        readySelection()
        val before = repository.getHistoryById(id)!!
        val model = confirmModel()
        var calls = 0
        model.launchHospitalAfterVoiceCleanup {
            calls++
            assertEquals(HistoryStatus.UNCOMPLETED, repository.getHistoryById(id)?.status)
        }
        model.launchHospitalAfterVoiceCleanup { fail("duplicate launch") }
        runCurrent()
        val saved = repository.getHistoryById(id)!!
        assertEquals(1, calls)
        assertEquals(HistoryStatus.COMPLETED, saved.status)
        assertEquals(before.chatMessages, saved.chatMessages)
        assertEquals(before.recommendations, saved.recommendations)
        assertEquals(item().recommendationId, saved.selectedRecommendationId)
        assertEquals("耳科", saved.typeTitle)
        assertNotNull(saved.completedAt)
        assertEquals(1, repository.getAllHistory().value.count { it.id == id })
        repository.saveCurrentChatToHistory(id, "不能覆蓋", false, allowReopen = true)
        assertEquals(saved, repository.getHistoryById(id))
    }

    @Test fun invalidVisitOrUnvalidatedSelectionCannotLaunch() = runTest(dispatcher) {
        repository.saveToHistory(history())
        repository.setActiveCaseId(id)
        repository.selectRecommendation(item())
        repository.setActiveVisitType("initial")
        confirmModel().launchHospitalAfterVoiceCleanup { fail("no validated script") }
        runCurrent()
        repository.consumeScriptResponse(ScriptResponseDto(true, recommendationId = item().recommendationId, recommendation = item()))
        repository.setActiveVisitType(null)
        confirmModel().launchHospitalAfterVoiceCleanup { fail("invalid visit") }
        runCurrent()
        assertEquals(HistoryStatus.UNCOMPLETED, repository.getHistoryById(id)?.status)
    }

    @Test fun quickSearchLaunchDoesNotInventConsultationHistory() = runTest(dispatcher) {
        readySelection(withHistory = false)
        var launched = false
        confirmModel().launchHospitalAfterVoiceCleanup { launched = true }
        runCurrent()
        assertTrue(launched)
        assertNull(repository.getHistoryById(id))
    }

    @Test fun completedHistoryIsPermanentlyReadOnlyEvenWithLiveBackendSnapshot() = runTest(dispatcher) {
        readySelection()
        repository.completeGuidanceAfterLaunch(repository.validatedGuidanceSelection())
        repository.consumeTriageResult(resume(stage = "recommending", key = null).toTriageResult())
        val model = chatModel()
        model.openHistory(id)
        runCurrent()
        assertTrue(model.isHistoryReadOnly.value)
        assertTrue(client.resumes.isEmpty())
        model.onInputTextChanged("試圖修改")
        model.acceptVoiceTranscript("試圖語音輸入")
        assertFalse(model.beginSystemVoiceInput())
        assertFalse(model.startVoiceRecording())
        model.sendMessage({ fail("navigate") })
        model.continueEditing()
        model.chooseRecommendation { fail("navigate") }
        runCurrent()
        assertTrue(client.chats.isEmpty())
        assertTrue(model.inputText.value.isEmpty())
        assertEquals("原對話", model.messages.value.single().content)
        assertEquals(item().recommendationId, model.openedHistory.value?.selectedRecommendationId)
        repository.setActiveCaseId(id)
        try { repository.recommend(id, "initial"); fail("completed recommend") } catch (_: IllegalStateException) { }
        try { repository.generateScript(id, item().recommendationId, item()); fail("completed script") } catch (_: IllegalStateException) { }
        assertEquals(0, client.recommendCalls)
        assertEquals(0, client.scriptCalls)
    }

    @Test fun uncompletedSelectionSurvivesPersistenceAndNeverRestoresUnvalidatedScript() {
        readySelection()
        repository.saveCurrentChatToHistory(id, "返回聊天", false)
        val saved = repository.getHistoryById(id)!!
        val parsed = parseHistoryJson("[${saved.toJson()}]").single()
        assertEquals(saved, parsed)
        repository.loadHistoryIntoCurrentChat(id)
        assertNull(repository.getCurrentGuidanceScript())
        assertNull(repository.getSelectedRecommendation())
        assertEquals(item().recommendationId, repository.getHistoryById(id)?.selectedRecommendationId)
    }

    @Test fun restartedAndroidRestoresBackendQuestionWithoutForcingKeyedBatch() = runTest(dispatcher) {
        repository.saveToHistory(history())
        assertNull(repository.getLiveTriageResult(id))
        client.resumeResponse = { resume() }
        val model = chatModel()
        model.openHistory(id)
        assertTrue(model.isHistoryReadOnly.value)
        assertEquals("原對話", model.messages.value.single().content)
        runCurrent()
        assertFalse(model.isHistoryReadOnly.value)
        assertNull(model.currentBatchQuestion.value)
        assertEquals("請問持續多久？", model.messages.value.last().content)
        model.onInputTextChanged("三天")
        model.sendMessage({})
        runCurrent()
        assertTrue(client.chats.single().answers.isEmpty())
        assertEquals("三天", client.chats.single().message)
        assertEquals(id, client.chats.single().caseId)
    }

    @Test fun backendMissingOverridesStaleMemoryAndPreservesLocalHistory() = runTest(dispatcher) {
        for (code in listOf(404, 410)) {
            repository.saveToHistory(history())
            repository.consumeTriageResult(resume(stage = "recommending").toTriageResult())
            client.resumeResponse = { throw MedicalApiException("gone", statusCode = code) }
            val model = chatModel()
            model.openHistory(id)
            runCurrent()
            assertTrue(model.isHistoryReadOnly.value)
            assertFalse(model.canRetryHistoryResume.value)
            assertTrue(model.historyResumeMessage.value.orEmpty().contains("已失效"))
            assertNull(repository.getActiveCaseId())
            assertNotNull(repository.getHistoryById(id))
        }
        assertTrue(client.chats.isEmpty())
    }

    @Test fun networkFailureIsRetryableNotExpired() = runTest(dispatcher) {
        repository.saveToHistory(history())
        client.resumeResponse = { throw MedicalApiException("network") }
        val model = chatModel()
        model.openHistory(id)
        runCurrent()
        assertTrue(model.isHistoryReadOnly.value)
        assertTrue(model.canRetryHistoryResume.value)
        assertEquals("目前無法確認案件狀態，可稍後重試。", model.historyResumeMessage.value)
        client.resumeResponse = { resume(stage = "waiting_confirmation", key = null) }
        model.retryHistoryResume()
        runCurrent()
        assertFalse(model.isHistoryReadOnly.value)
        assertTrue(model.showDecisionButtons.value)
        assertEquals(id, repository.getActiveCaseId())
    }

    @Test fun lateResumeCannotOverwriteCurrentlyViewedHistory() = runTest(dispatcher) {
        repository.saveToHistory(history())
        repository.saveToHistory(history(other))
        val pending = CompletableDeferred<CaseResumeDto>()
        client.resumeResponse = { case ->
            if (case == id) withContext(NonCancellable) { pending.await() }
            else resume(caseId = other, stage = "waiting_confirmation", key = null)
        }
        val model = chatModel()
        model.openHistory(id)
        runCurrent()
        model.openHistory(other)
        runCurrent()
        pending.complete(resume())
        runCurrent()
        assertEquals(other, model.openedHistory.value?.id)
        assertEquals(other, repository.getActiveCaseId())
        assertNull(model.currentBatchQuestion.value)
        assertTrue(model.showDecisionButtons.value)
    }

    @Test fun differentCaseCannotReceiveSelectedDoctorSnapshot() = runTest(dispatcher) {
        repository.saveToHistory(history())
        repository.saveToHistory(history(other))
        repository.setActiveCaseId(id)
        val pending = CompletableDeferred<ScriptResponseDto>()
        client.scriptResponse = { pending.await() }
        val call = async { runCatching { repository.generateScript(id, item().recommendationId, item()) } }
        runCurrent()
        repository.loadHistoryIntoCurrentChat(other)
        pending.complete(ScriptResponseDto(true, recommendationId = item().recommendationId, recommendation = item()))
        runCurrent()
        assertTrue(call.await().isFailure)
        assertNull(repository.getHistoryById(other)?.selectedRecommendationId)
        assertNull(repository.getHistoryById(id)?.selectedRecommendationId)
    }

    @Test fun revisionFailureNeverClaimsBackendAcceptedEditing() = runTest(dispatcher) {
        repository.saveToHistory(history())
        client.resumeResponse = { resume(stage = "waiting_confirmation", key = null) }
        val model = chatModel()
        model.openHistory(id)
        runCurrent()
        model.continueEditing()
        runCurrent()
        assertTrue(model.messages.value.last().content.contains("尚未完成切換"))
        assertFalse(model.messages.value.any { it.content.contains("已切回可修改") })
        assertEquals(id, repository.getActiveCaseId())
        assertTrue(model.showDecisionButtons.value)
    }

    @Test fun resumeParserUsesIndependentMinimalContract() {
        val parsed = parseCaseResume("""{"case_id":"$id","visit_type":"initial","stage":"collecting",
            "can_continue":true,"last_question_key":"severity","next_question":"程度？",
            "question_batch":[{"key":"severity","question":"程度？"}]}""")
        assertEquals("severity", parsed.toTriageResult().questionBatch.single().key)
        assertNull(parsed.toTriageResult().triageCase)
        assertFalse(parsed.state.confirmed)
    }

    @Test fun continuationRequestRequiresExistingCaseAndExpiryDisablesFurtherInput() = runTest(dispatcher) {
        repository.saveToHistory(history())
        client.resumeResponse = { resume() }
        client.chatResponse = { throw MedicalApiException("expired", statusCode = 404) }
        val model = chatModel()
        model.openHistory(id)
        runCurrent()
        model.onInputTextChanged("三天")
        model.sendMessage({ fail("expired case navigated") })
        runCurrent()
        assertTrue(org.json.JSONObject(client.chats.single().toJson()).getBoolean("require_existing_case"))
        assertTrue(model.isHistoryReadOnly.value)
        assertFalse(model.showDecisionButtons.value)
        assertNotNull(repository.getHistoryById(id))
        assertNull(repository.getActiveCaseId())
    }

    @Test fun confirmationResponseArrivingAfterFinalLaunchCannotReopenCompletedHistory() = runTest(dispatcher) {
        readySelection()
        val selection = repository.validatedGuidanceSelection()
        val pending = CompletableDeferred<TriageResultDto>()
        client.chatResponse = { pending.await() }
        val request = async { runCatching { repository.confirmTriage(id) } }
        runCurrent()
        repository.completeGuidanceAfterLaunch(selection)
        val completed = repository.getHistoryById(id)
        pending.complete(resume(stage = "recommending", key = null).toTriageResult())
        runCurrent()
        assertTrue(request.await().isFailure)
        assertEquals(completed, repository.getHistoryById(id))
        assertEquals(HistoryStatus.COMPLETED, repository.getHistoryById(id)?.status)
    }

    private class FakeClient : MedicalApiClient() {
        val resumes = mutableListOf<String>()
        val chats = mutableListOf<ChatRequest>()
        var recommendCalls = 0
        var scriptCalls = 0
        var resumeResponse: suspend (String) -> CaseResumeDto = { error("no resume") }
        var scriptResponse: suspend (ScriptRequest) -> ScriptResponseDto = { error("no script") }
        var chatResponse: suspend (ChatRequest) -> TriageResultDto = { error("offline test") }
        override suspend fun resumeCase(caseId: String): CaseResumeDto { resumes.add(caseId); return resumeResponse(caseId) }
        override suspend fun chat(request: ChatRequest): TriageResultDto { chats.add(request); return chatResponse(request) }
        override suspend fun recommend(request: RecommendRequest): RecommendationResultDto { recommendCalls++; error("not allowed") }
        override suspend fun generateScript(request: ScriptRequest): ScriptResponseDto { scriptCalls++; return scriptResponse(request) }
    }
}
