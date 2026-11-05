package com.example.medicalaiguidance

import androidx.lifecycle.viewModelScope
import com.example.medicalaiguidance.model.ChatMessage
import com.example.medicalaiguidance.model.History
import com.example.medicalaiguidance.model.HistoryStatus
import com.example.medicalaiguidance.model.MessageSender
import com.example.medicalaiguidance.model.VisitPlan
import com.example.medicalaiguidance.model.hasUnresolvedSafety
import com.example.medicalaiguidance.model.hasUrgentSymptoms
import com.example.medicalaiguidance.network.*
import com.example.medicalaiguidance.repository.MedicalRepository
import com.example.medicalaiguidance.repository.TtsSession
import com.example.medicalaiguidance.viewmodel.ChatStopReason
import com.example.medicalaiguidance.viewmodel.ChatViewModel
import com.example.medicalaiguidance.viewmodel.requiresSafetyUnresolvedStop
import com.example.medicalaiguidance.viewmodel.requiresUrgentStop
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel
import kotlinx.coroutines.test.*
import org.junit.Assert.*
import org.junit.Test

@OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
class SafetyUnresolvedStopTest {
    private val reply = "因為還沒能確認您是否有急迫症狀，系統不能繼續一般掛號導引。若身體不適，請洽醫療人員協助評估；若情況緊急，請立即尋求緊急醫療協助。"
    private val legacyReply = "目前無法可靠確認急迫症狀，不能繼續一般掛號導引。請由醫療人員協助評估；若情況緊急，請立即尋求緊急醫療協助。"
    private fun response() = TriageResultDto(
        "safety_unresolved_test", null,
        ConversationStateDto(clarificationStatus = "unresolved", replySource = "SAFETY_UNRESOLVED_STOP"),
        UrgencyResultDto(), reply = reply
    )

    @Test fun onlyTheSafetyMarkerStopsAnUnresolvedCase() {
        assertTrue(response().requiresSafetyUnresolvedStop())
        assertFalse(response().requiresUrgentStop())
        val departmentUnresolved = response().copy(
            reply = "目前無法可靠釐清症狀。",
            conversationState = ConversationStateDto(clarificationStatus = "unresolved")
        )
        assertFalse(departmentUnresolved.requiresSafetyUnresolvedStop())
        assertFalse(departmentUnresolved.requiresUrgentStop())
    }

    @Test fun historyLabelSeparatesUnresolvedFromUrgent() {
        val history = History(
            "h", "2026/11/03", "初診", "", HistoryStatus.UNCOMPLETED,
            chatMessages = listOf(ChatMessage(content = reply, sender = MessageSender.AI))
        )
        assertTrue(history.hasUnresolvedSafety())
        assertFalse(history.hasUrgentSymptoms())
        val legacy = history.copy(chatMessages = listOf(ChatMessage(content = legacyReply, sender = MessageSender.AI)))
        assertTrue(legacy.hasUnresolvedSafety())
    }

    @Test fun unresolvedStopShowsDialogAndRemovesInput() = runTest {
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
            assertEquals(ChatStopReason.SAFETY_UNRESOLVED, model.stopReason.value)
            assertEquals(reply, model.urgentWarning.value)
            model.dismissUrgentDialog()
            model.onInputTextChanged("我要掛號")
            model.sendMessage({ fail("must not navigate") })
            advanceUntilIdle()
            assertEquals("", model.inputText.value)
            assertEquals(1, calls)
            model.restartFromHistory()
            assertFalse(model.isUrgentStopped.value)
        } finally {
            model.viewModelScope.cancel()
            session.close()
            repository.deleteHistory("safety_unresolved_test")
            repository.clearRecommendationFlow()
            repository.clearChatMessages()
            Dispatchers.resetMain()
        }
    }
}
