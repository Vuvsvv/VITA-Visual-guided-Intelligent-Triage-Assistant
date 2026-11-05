package com.example.medicalaiguidance

import androidx.lifecycle.viewModelScope
import com.example.medicalaiguidance.model.VisitPlan
import com.example.medicalaiguidance.network.*
import com.example.medicalaiguidance.repository.MedicalRepository
import com.example.medicalaiguidance.repository.TtsSession
import com.example.medicalaiguidance.viewmodel.ChatViewModel
import com.example.medicalaiguidance.viewmodel.requiresUrgentStop
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.test.*
import org.junit.Assert.*
import org.junit.Test

@OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
class UrgentChatStopTest {
    private val warning = "您回報的狀況可能需要立即處理，因此不會繼續掛號導引。請立即前往急診；情況危急請撥打 119。"
    private fun response() = TriageResultDto(
        "urgent_test", null, ConversationStateDto(clarificationStatus = "unresolved"),
        UrgencyResultDto(), reply = warning
    )

    @Test fun recognizesSafetyPositiveReplyWithoutWarningFlag() {
        assertTrue(response().requiresUrgentStop())
        assertTrue(response().copy(reply = null, conversationState = ConversationStateDto(clarificationStatus = "urgent")).requiresUrgentStop())
        assertFalse(response().copy(reply = "是否有突然胸痛？", conversationState = ConversationStateDto(clarificationStatus = "safety_check")).requiresUrgentStop())
    }

    @Test fun closingDialogDoesNotEnableTextVoiceOrRecommendation() = runTest {
        Dispatchers.setMain(StandardTestDispatcher(testScheduler))
        var calls = 0
        val client = object : MedicalApiClient() {
            override suspend fun chat(request: ChatRequest): TriageResultDto {
                calls++
                return response()
            }
        }
        val repository = MedicalRepository(client)
        repository.clearRecommendationFlow()
        repository.clearChatMessages()
        val session = TtsSession(backgroundScope) { _, _, _ -> error("No TTS expected") }
        val model = ChatViewModel(repository, ttsSessionFactory = { session })
        try {
            model.startNewConversation(VisitPlan.INITIAL)
            advanceUntilIdle()
            assertTrue(model.isUrgentStopped.value)
            assertTrue(model.showUrgentDialog.value)
            model.dismissUrgentDialog()
            assertFalse(model.showUrgentDialog.value)
            model.onInputTextChanged("我要掛號")
            model.acceptVoiceTranscript("繼續")
            model.sendMessage({ fail("must not navigate") })
            model.chooseRecommendation { fail("must not recommend") }
            advanceUntilIdle()
            assertEquals("", model.inputText.value)
            assertEquals(1, calls)
            assertFalse(model.startVoiceRecording())
            model.restartFromHistory()
            assertFalse(model.isUrgentStopped.value)
        } finally {
            model.viewModelScope.cancel()
            session.close()
            repository.deleteHistory("urgent_test")
            repository.clearRecommendationFlow()
            repository.clearChatMessages()
            Dispatchers.resetMain()
        }
    }
}
