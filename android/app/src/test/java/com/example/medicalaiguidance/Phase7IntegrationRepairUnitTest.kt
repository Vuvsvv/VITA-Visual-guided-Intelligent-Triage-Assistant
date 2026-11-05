package com.example.medicalaiguidance

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.medicalaiguidance.model.History
import com.example.medicalaiguidance.model.HistoryStatus
import com.example.medicalaiguidance.network.*
import com.example.medicalaiguidance.repository.MedicalRepository
import com.example.medicalaiguidance.repository.TtsSession
import com.example.medicalaiguidance.repository.accessibilitySession
import com.example.medicalaiguidance.repository.parseHistoryJson
import com.example.medicalaiguidance.viewmodel.ChatViewModel
import com.example.medicalaiguidance.viewmodel.DoctorViewModel
import com.example.medicalaiguidance.viewmodel.canNavigateToRecommendations
import com.example.medicalaiguidance.viewmodel.shouldAppendAssistantReply
import com.example.medicalaiguidance.model.VisitPlan
import com.example.medicalaiguidance.model.ChatMessage
import com.example.medicalaiguidance.model.MessageSender
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class Phase7IntegrationRepairUnitTest {
    private val dispatcher = StandardTestDispatcher()
    private val client = FakeClient()
    private val repository = MedicalRepository(client)
    private val ttsSession = TtsSession(CoroutineScope(dispatcher + SupervisorJob())) { _, _, _ ->
        error("unexpected TTS call")
    }
    private val models = mutableListOf<ViewModel>()
    private val caseId = "case_phase7_${System.nanoTime()}"

    @Before
    fun setUp() {
        Dispatchers.setMain(dispatcher)
        repository.clearRecommendationFlow()
        repository.clearChatMessages()
    }

    @After
    fun tearDown() {
        models.forEach { it.viewModelScope.cancel() }
        ttsSession.close()
        repository.deleteHistory(caseId)
        repository.deleteHistory("${caseId}_new")
        repository.clearRecommendationFlow()
        repository.clearChatMessages()
        Dispatchers.resetMain()
    }

    @Test
    fun naturalStartAndFreeTextIgnoreLegacyBatchAuthority() = runTest(dispatcher) {
        val question = "頭暈時有沒有伴隨耳鳴？"
        client.chatResponse = { result(stage = "collecting").copy(
            needMoreInfo = true, reply = question, nextQuestion = question,
            questionBatch = listOf(QuestionItemDto(key = "duration", question = "舊固定題"))
        ) }
        val model = ChatViewModel(repository, { ttsSession }).also(models::add)
        model.startNewConversation(VisitPlan.INITIAL)
        runCurrent()
        assertFalse(client.chatRequests.single().legacyBatch)
        assertTrue(client.chatRequests.single().answers.isEmpty())
        assertEquals(question, model.messages.value.last().content)
        assertNull(model.currentBatchQuestion.value)
        model.onInputTextChanged("有耳鳴")
        model.sendMessage({ fail("collecting cannot navigate") })
        runCurrent()
        assertEquals("有耳鳴", client.chatRequests.last().message)
        assertTrue(client.chatRequests.last().answers.isEmpty())
        assertEquals(HistoryStatus.UNCOMPLETED, repository.getHistoryById(caseId)?.status)
    }

    @Test
    fun identicalAssistantQuestionIsNotInsertedTwice() {
        val question = "頭暈是在什麼時候開始的？"
        assertFalse(shouldAppendAssistantReply(ChatMessage(content = question, sender = MessageSender.AI), question))
        assertTrue(shouldAppendAssistantReply(ChatMessage(content = question, sender = MessageSender.USER), question))
        assertTrue(shouldAppendAssistantReply(null, question))
    }

    @Test
    fun officialSessionAlwaysWinsOverDisplayTimeForAllFormalEntrances() {
        for (visitType in listOf("initial", "followup", "quick_search")) {
            repository.clearRecommendationFlow()
            repository.setActiveVisitType(visitType)
            for ((session, canonical) in listOf(
                "上午" to "上午", "下午" to "下午", "晚上" to "晚上",
                "morning" to "上午", "afternoon" to "下午", "evening" to "晚上",
                "早上" to "上午", "午診" to "下午", "夜診" to "晚上",
                "晚診" to "晚上", "夜間" to "晚上"
            )) {
                val item = recommendation().copy(session = session, sessionTime = "13:30-17:00")
                repository.selectRecommendation(item)
                val appointment = repository.getConfirmedAppointment()
                assertEquals(canonical, appointment.timeSlot)
                assertEquals(item.sessionTime, appointment.sessionTime)
                assertEquals(item.recommendationId, appointment.id)
                assertEquals(item.scheduleId, repository.getSelectedRecommendation()?.scheduleId)
                assertEquals(item.doctorId, appointment.doctor.id)
            }
        }
    }

    @Test
    fun unknownSessionDoesNotGuessFromClockTime() {
        assertThrows(IllegalStateException::class.java) { accessibilitySession("13:30-17:00") }
        assertThrows(IllegalStateException::class.java) { accessibilitySession("unknown") }
    }

    @Test
    fun confirmedFalseNeverNavigatesAndHistoryStaysUncompleted() = runTest(dispatcher) {
        val model = liveChat()
        client.chatResponse = { result() }
        var navigations = 0
        model.chooseRecommendation { navigations++ }
        runCurrent()
        assertEquals(0, navigations)
        assertTrue(model.showDecisionButtons.value)
        assertEquals(HistoryStatus.UNCOMPLETED, repository.getHistoryById(caseId)?.status)
        assertNull(repository.getHistoryById(caseId)?.completedAt)
        assertTrue(model.messages.value.any { it.content == "後端回覆" })
        assertFalse(model.isConfirmingRecommendation.value)
    }

    @Test
    fun collectingResponseKeepsChatAnswerableEvenWhenConfirmedIsTrue() = runTest(dispatcher) {
        val model = liveChat()
        client.chatResponse = {
            result(stage = "collecting", confirmed = true).copy(
                needMoreInfo = true, nextQuestion = "請補充時間"
            )
        }
        var navigated = false
        model.chooseRecommendation { navigated = true }
        runCurrent()
        assertFalse(navigated)
        assertFalse(model.showDecisionButtons.value)
        assertFalse(model.isHistoryReadOnly.value)
        model.onInputTextChanged("明天下午")
        assertEquals("明天下午", model.inputText.value)
        assertEquals(HistoryStatus.UNCOMPLETED, repository.getHistoryById(caseId)?.status)
    }

    @Test
    fun onlyConfirmedBackendRecommendationStateAllowsNavigation() {
        val ready = result(stage = "recommending", confirmed = true)
        assertTrue(ready.canNavigateToRecommendations())
        assertFalse(ready.copy(conversationState = ready.conversationState.copy(confirmed = false)).canNavigateToRecommendations())
        for (stage in listOf("collecting", "waiting_confirmation", "done")) {
            assertFalse(ready.copy(conversationState = ready.conversationState.copy(stage = stage)).canNavigateToRecommendations())
        }
        for (status in listOf("ambiguous", "unresolved")) {
            assertFalse(ready.copy(conversationState = ready.conversationState.copy(departmentStatus = status)).canNavigateToRecommendations())
        }
        assertFalse(ready.copy(conversationState = ready.conversationState.copy(clarificationStatus = "safety_check")).canNavigateToRecommendations())
        assertFalse(ready.copy(triage = ready.triage.copy(warningRequired = true)).canNavigateToRecommendations())
        assertFalse(ready.copy(triageCase = TriageCaseDto(caseId, patientInput = PatientInputDto(redFlagsChecked = false))).canNavigateToRecommendations())
        assertFalse(ready.copy(departmentResult = null).canNavigateToRecommendations())
    }

    @Test
    fun backendSafetyAndDepartmentStatesAreParsedForNavigationGate() {
        val parsed = parseTriageResult("""{
          "case_id":"$caseId", "needMoreInfo":false,
          "conversation_state":{"stage":"recommending","confirmed":true,
            "department_status":"ambiguous","clarification_status":"safety_check"},
          "triage":{"need_more_info":false},
          "department_result":{"childDept":"一般內科"}
        }""")
        assertEquals("ambiguous", parsed.conversationState.departmentStatus)
        assertEquals("safety_check", parsed.conversationState.clarificationStatus)
        assertFalse(parsed.canNavigateToRecommendations())
    }

    @Test
    fun confirmationUpdatesSameCaseHistoryAndPreventsConcurrentRequests() = runTest(dispatcher) {
        val model = liveChat()
        val response = CompletableDeferred<TriageResultDto>()
        client.chatResponse = { response.await() }
        var navigations = 0
        model.chooseRecommendation { navigations++ }
        model.chooseRecommendation { navigations++ }
        assertTrue(model.isConfirmingRecommendation.value)
        runCurrent()
        model.chooseRecommendation { navigations++ }
        assertEquals(1, client.chatRequests.size)
        response.complete(result(stage = "recommending", confirmed = true))
        runCurrent()
        assertEquals(1, navigations)
        val history = repository.getHistoryById(caseId)!!
        assertEquals(HistoryStatus.UNCOMPLETED, history.status)
        assertEquals("一般內科", history.typeTitle)
        assertNull(history.completedAt)
        assertEquals(1, repository.getAllHistory().value.count { it.id == caseId })
        assertEquals(caseId, client.chatRequests.single().caseId)
        assertTrue(client.chatRequests.single().confirmed)
    }

    @Test
    fun confirmationFailurePreservesExistingCaseHistoryAndAllowsRetry() = runTest(dispatcher) {
        val model = liveChat(confirmed = true)
        val original = repository.getHistoryById(caseId)!!
        client.chatResponse = { error("connection unavailable") }
        var navigated = false
        model.chooseRecommendation { navigated = true }
        runCurrent()
        assertFalse(navigated)
        assertEquals(caseId, repository.getActiveCaseId())
        assertEquals(original, repository.getHistoryById(caseId))
        assertFalse(model.isConfirmingRecommendation.value)
        assertTrue(model.showDecisionButtons.value)
        client.chatResponse = { result(stage = "recommending", confirmed = true) }
        model.chooseRecommendation { navigated = true }
        runCurrent()
        assertTrue(navigated)
    }

    @Test
    fun changedBackendCaseCannotNavigateOrPretendOldCaseWasRestored() = runTest(dispatcher) {
        val model = liveChat()
        client.chatResponse = { result(stage = "collecting").copy(caseId = "${caseId}_new", needMoreInfo = true) }
        var navigated = false
        model.chooseRecommendation { navigated = true }
        runCurrent()
        assertFalse(navigated)
        assertNull(repository.getLiveTriageResult(caseId))
        assertNotNull(repository.getHistoryById(caseId))
        assertTrue(model.messages.value.any { it.content.contains("原問診案件可能已失效") })
    }

    @Test
    fun unfinishedLiveHistoryRestoresButtonsThroughReadOnlyResumeWithoutChatProbe() = runTest(dispatcher) {
        val model = liveChat()
        assertFalse(model.isHistoryReadOnly.value)
        assertTrue(model.showDecisionButtons.value)
        assertEquals(0, client.chatRequests.size)
        assertEquals(listOf(caseId), client.resumeRequests)
        assertEquals(HistoryStatus.UNCOMPLETED, repository.getHistoryById(caseId)?.status)
    }

    @Test
    fun oldUnfinishedHistoryIsReadOnlyAndCanStartFreshWithoutChatProbe() = runTest(dispatcher) {
        repository.saveToHistory(history())
        val model = ChatViewModel(repository, ttsSessionFactory = { ttsSession }).also(models::add)
        model.openHistory(caseId)
        runCurrent()
        assertTrue(model.isHistoryReadOnly.value)
        assertNull(repository.getActiveCaseId())
        model.chooseRecommendation { fail("old history must not navigate") }
        assertEquals(0, client.chatRequests.size)
        model.restartFromHistory()
        assertFalse(model.isHistoryReadOnly.value)
        assertNull(repository.getActiveCaseId())
    }

    @Test
    fun snapshotsUseCaseAndRecommendationIdRatherThanDisplayTitleOrDoctorName() {
        repository.saveToHistory(history())
        val first = recommendation()
        val second = first.copy(recommendationId = "rec_second", scheduleId = "S002", session = "下午")
        repository.saveRecommendationSnapshot(caseId, RecommendationResultDto(
            caseId, RecommendationColumnsDto(listOf(first, second), listOf(first)), totalCount = 3
        ))
        repository.setActiveCaseId(caseId)
        repository.consumeScriptResponse(ScriptResponseDto(true, recommendationId = second.recommendationId, recommendation = second))
        val saved = repository.getHistoryById(caseId)!!
        assertEquals("AI 問診", saved.typeTitle)
        assertEquals(2, saved.recommendations.size)
        assertEquals("rec_second", saved.selectedRecommendationId)
        assertEquals("下午", saved.recommendations.single { it.recommendationId == saved.selectedRecommendationId }.session)
        assertEquals("S002", repository.getSelectedRecommendation()?.scheduleId)
        assertEquals("D001", repository.getSelectedRecommendation()?.doctorId)
    }

    @Test
    fun recommendFailureDoesNotEraseConfirmationOrConversation() = runTest(dispatcher) {
        liveChat(confirmed = true)
        val before = repository.getHistoryById(caseId)
        repository.setActiveVisitType("initial")
        client.recommendResponse = { error("recommend unavailable") }
        try {
            repository.recommend(caseId, "initial")
            fail("expected failure")
        } catch (_: IllegalStateException) { }
        assertEquals(before, repository.getHistoryById(caseId))
    }

    @Test
    fun transientSaveCannotDemoteConfirmedHistoryOrLoseSnapshot() {
        repository.saveToHistory(history().copy(status = HistoryStatus.COMPLETED, completedAt = "2026/10/09 09:00",
            recommendations = listOf(com.example.medicalaiguidance.model.HistoryRecommendation(
                recommendationId = "rec_first", parentDepartment = "內科系", department = "一般內科",
                doctor = "測試醫師", date = "2099-10-13", session = "下午"))))
        repository.saveCurrentChatToHistory(caseId, "retry", completed = false)
        val saved = repository.getHistoryById(caseId)!!
        assertEquals(HistoryStatus.COMPLETED, saved.status)
        assertEquals("2026/10/09 09:00", saved.completedAt)
        assertEquals(1, saved.recommendations.size)
    }

    @Test
    fun repeatedSavesHaveOnlyOneHistoryPerCaseAndOldJsonRemainsReadable() {
        repeat(3) { repository.saveCurrentChatToHistory(caseId, "summary", completed = false) }
        assertEquals(1, repository.getAllHistory().value.count { it.id == caseId })
        val old = parseHistoryJson("""[{"id":"old","date":"2026/01/01","typeTitle":"AI 問診","status":"UNCOMPLETED"}]""").single()
        assertNull(old.completedAt)
        assertTrue(old.recommendations.isEmpty())
        assertTrue(old.chatMessages.isEmpty())
    }

    @Test
    fun concurrentDoctorCardSelectionsOnlyRevalidateFirstAndUseReturnedSlot() = runTest(dispatcher) {
        repository.setActiveCaseId(caseId)
        val model = DoctorViewModel(repository).also(models::add)
        val first = recommendation()
        val second = first.copy(recommendationId = "rec_other", doctorId = "D002", scheduleId = "S002")
        val response = CompletableDeferred<ScriptResponseDto>()
        client.scriptResponse = { response.await() }
        var navigations = 0
        model.selectRecommendationAndNavigate(first) { navigations++ }
        model.selectRecommendationAndNavigate(second) { navigations++ }
        runCurrent()
        model.selectRecommendationAndNavigate(second) { navigations++ }
        assertEquals(1, client.scriptRequests.size)
        assertNull(repository.getSelectedRecommendation())
        val revalidated = first.copy(room = "B診", sessionTime = "13:30-17:00")
        response.complete(ScriptResponseDto(true, recommendationId = first.recommendationId, recommendation = revalidated, steps = emptyList()))
        runCurrent()
        assertEquals(1, navigations)
        assertEquals(first, client.scriptRequests.single().recommendation)
        assertEquals(revalidated, repository.getSelectedRecommendation())
        assertEquals("B診", repository.getConfirmedAppointment().doctor.title)
        assertEquals("下午", repository.getConfirmedAppointment().timeSlot)
        assertNull(model.selectingRecommendationId.value)
    }

    @Test
    fun failedRevalidationCannotReplacePreviousSuccessAndSelectionCanRetry() = runTest(dispatcher) {
        repository.setActiveCaseId(caseId)
        val previous = recommendation()
        repository.consumeScriptResponse(ScriptResponseDto(true, recommendationId = previous.recommendationId, recommendation = previous, steps = emptyList()))
        val model = DoctorViewModel(repository).also(models::add)
        val candidate = previous.copy(recommendationId = "rec_other", scheduleId = "S002")
        client.scriptResponse = { ScriptResponseDto(false, recommendationId = candidate.recommendationId, steps = emptyList()) }
        var navigated = false
        model.selectRecommendationAndNavigate(candidate) { navigated = true }
        runCurrent()
        assertFalse(navigated)
        assertEquals(previous, repository.getSelectedRecommendation())
        assertNull(model.selectingRecommendationId.value)
        client.scriptResponse = { ScriptResponseDto(true, recommendationId = candidate.recommendationId, steps = emptyList()) }
        model.selectRecommendationAndNavigate(candidate) { navigated = true }
        runCurrent()
        assertTrue(navigated)
        assertEquals(candidate, repository.getSelectedRecommendation())
    }

    @Test
    fun revalidationExceptionKeepsSelectionAndNeverNavigates() = runTest(dispatcher) {
        repository.setActiveCaseId(caseId)
        val previous = recommendation()
        repository.selectRecommendation(previous)
        val model = DoctorViewModel(repository).also(models::add)
        client.scriptResponse = { error("SQL unavailable") }
        model.selectRecommendationAndNavigate(previous.copy(recommendationId = "rec_other")) { fail("failed request navigated") }
        runCurrent()
        assertEquals(previous, repository.getSelectedRecommendation())
        assertNull(model.selectingRecommendationId.value)
    }

    @Test
    fun mismatchedRevalidatedRecommendationIdIsRejectedBeforeSelection() = runTest(dispatcher) {
        repository.setActiveCaseId(caseId)
        val model = DoctorViewModel(repository).also(models::add)
        client.scriptResponse = { ScriptResponseDto(true, recommendationId = "rec_other", recommendation = recommendation(), steps = emptyList()) }
        model.selectRecommendationAndNavigate(recommendation()) { fail("mismatched ID navigated") }
        runCurrent()
        assertNull(repository.getSelectedRecommendation())
    }

    @Test
    fun revalidationOfDifferentScheduleCannotOverwriteSelectedIdentity() = runTest(dispatcher) {
        repository.setActiveCaseId(caseId)
        val previous = recommendation()
        repository.selectRecommendation(previous)
        val model = DoctorViewModel(repository).also(models::add)
        client.scriptResponse = {
            ScriptResponseDto(true, recommendationId = previous.recommendationId,
                recommendation = previous.copy(scheduleId = "S_other"), steps = emptyList())
        }
        model.selectRecommendationAndNavigate(previous) { fail("different schedule navigated") }
        runCurrent()
        assertEquals(previous, repository.getSelectedRecommendation())
    }

    @Test
    fun changingActiveCaseWhileRevalidatingCannotSaveIntoAnotherHistory() = runTest(dispatcher) {
        repository.setActiveCaseId(caseId)
        val model = DoctorViewModel(repository).also(models::add)
        val response = CompletableDeferred<ScriptResponseDto>()
        client.scriptResponse = { response.await() }
        model.selectRecommendationAndNavigate(recommendation()) { fail("stale case navigated") }
        runCurrent()
        repository.clearRecommendationFlow()
        repository.setActiveCaseId("${caseId}_new")
        response.complete(ScriptResponseDto(true, recommendationId = recommendation().recommendationId,
            recommendation = recommendation(), steps = emptyList()))
        runCurrent()
        assertNull(repository.getSelectedRecommendation())
        assertEquals("${caseId}_new", repository.getActiveCaseId())
    }

    @Test
    fun successfulRecommendSavesSnapshotThroughRepositoryApiForConfirmedCase() = runTest(dispatcher) {
        val model = liveChat()
        client.chatResponse = { result(stage = "recommending", confirmed = true) }
        model.chooseRecommendation { }
        runCurrent()
        repository.setActiveVisitType("initial")
        client.recommendResponse = {
            RecommendationResultDto(caseId, RecommendationColumnsDto(listOf(recommendation())), totalCount = 1)
        }
        repository.recommend(caseId, "initial")
        assertEquals(HistoryStatus.UNCOMPLETED, repository.getHistoryById(caseId)?.status)
        assertEquals(recommendation().recommendationId, repository.getHistoryById(caseId)?.recommendations?.single()?.recommendationId)
    }

    private suspend fun liveChat(confirmed: Boolean = false): ChatViewModel {
        repository.saveToHistory(history())
        repository.consumeTriageResult(result(
            stage = if (confirmed) "recommending" else "waiting_confirmation", confirmed = confirmed
        ))
        client.resumeResponse = { CaseResumeDto(caseId, "initial",
            result(stage = if (confirmed) "recommending" else "waiting_confirmation", confirmed = confirmed).conversationState,
            true, "negative", false, null, result().departmentResult, null, emptyList(), true) }
        return ChatViewModel(repository, ttsSessionFactory = { ttsSession })
            .also { models.add(it); it.openHistory(caseId); dispatcher.scheduler.runCurrent() }
    }

    private fun history() = History(caseId, "2026/10/09", "AI 問診", "summary", HistoryStatus.UNCOMPLETED)

    private fun result(stage: String = "waiting_confirmation", confirmed: Boolean = false) = TriageResultDto(
        caseId, null, ConversationStateDto(stage = stage, confirmed = confirmed, departmentStatus = "resolved"),
        UrgencyResultDto(needMoreInfo = false),
        DepartmentResultDto(deptId = 7, parentDept = "內科系", childDept = "一般內科"),
        needMoreInfo = false, reply = "後端回覆"
    )

    private fun recommendation() = RecommendationItemDto(
        recommendationId = "rec_first", parentDept = "內科系", childDept = "一般內科", doctor = "測試醫師",
        date = "2099-10-13", session = "下午", score = 50.0, doctorId = "D001", scheduleId = "S001", deptId = 7
    )

    private class FakeClient : MedicalApiClient() {
        val chatRequests = mutableListOf<ChatRequest>()
        val scriptRequests = mutableListOf<ScriptRequest>()
        val resumeRequests = mutableListOf<String>()
        var resumeResponse: suspend (String) -> CaseResumeDto = { throw MedicalApiException("expired", statusCode = 404) }
        override suspend fun resumeCase(caseId: String): CaseResumeDto {
            resumeRequests.add(caseId)
            return resumeResponse(caseId)
        }
        var chatResponse: suspend (ChatRequest) -> TriageResultDto = { error("unexpected chat call") }
        var recommendResponse: suspend (RecommendRequest) -> RecommendationResultDto = { error("unexpected recommend call") }
        var scriptResponse: suspend (ScriptRequest) -> ScriptResponseDto = { error("unexpected script call") }
        override suspend fun chat(request: ChatRequest): TriageResultDto {
            chatRequests.add(request)
            return chatResponse(request)
        }
        override suspend fun recommend(request: RecommendRequest) = recommendResponse(request)
        override suspend fun generateScript(request: ScriptRequest): ScriptResponseDto {
            scriptRequests.add(request)
            return scriptResponse(request)
        }
    }
}
