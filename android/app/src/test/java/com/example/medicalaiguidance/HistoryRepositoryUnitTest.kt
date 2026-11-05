package com.example.medicalaiguidance

import com.example.medicalaiguidance.model.ChatMessage
import com.example.medicalaiguidance.model.History
import com.example.medicalaiguidance.model.HistoryRecommendation
import com.example.medicalaiguidance.model.HistoryStatus
import com.example.medicalaiguidance.model.MessageSender
import com.example.medicalaiguidance.network.RecommendationItemDto
import com.example.medicalaiguidance.network.ScriptResponseDto
import com.example.medicalaiguidance.repository.TtsSession
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.*
import kotlinx.coroutines.test.*
import com.example.medicalaiguidance.repository.MedicalRepository
import com.example.medicalaiguidance.repository.parseHistoryJson
import com.example.medicalaiguidance.viewmodel.ChatViewModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.assertThrows
import org.junit.Test
import org.json.JSONException

class HistoryRepositoryUnitTest {
    @Test
    fun historyStateFlowUpdatesImmediatelyAfterSaveAndDelete() {
        val repository = MedicalRepository()
        val historyId = "case_reactive_history_${System.nanoTime()}"
        val historyFlow = repository.getAllHistory()
        val syntheticHistory = History(
            id = historyId,
            date = "2026/08/15",
            typeTitle = "一般內科",
            summaryText = "synthetic reactive history",
            status = HistoryStatus.COMPLETED
        )

        repository.saveToHistory(syntheticHistory)
        assertTrue(historyFlow.value.any { it.id == historyId })

        repository.deleteHistory(historyId)
        assertTrue(historyFlow.value.none { it.id == historyId })
    }

    @Test
    fun malformedHistoryJsonRaisesExplicitParseError() {
        assertThrows(JSONException::class.java) {
            parseHistoryJson("{synthetic malformed history}")
        }
    }

    @Test
    fun revalidatedDoctorAndTimeAreSavedWithoutCompletingHistory() {
        val repository = MedicalRepository()
        val historyId = "case_selected_snapshot_${System.nanoTime()}"
        repository.saveToHistory(
            History(
                id = historyId,
                date = "2026/08/15",
                typeTitle = "一般內科",
                summaryText = "synthetic completed history",
                status = HistoryStatus.UNCOMPLETED
            )
        )
        repository.setActiveCaseId(historyId)
        val item = RecommendationItemDto(
                recommendationId = "synthetic_selected_doctor",
                parentDept = "內科系",
                childDept = "一般內科",
                doctor = "測試醫師",
                date = "2026-08-20",
                session = "上午",
                sessionTime = "09:00-09:30",
                room = "320診",
                score = 1.0
            )
        repository.consumeScriptResponse(ScriptResponseDto(true, recommendationId = item.recommendationId, recommendation = item))

        val history = repository.getHistoryById(historyId)
        val selected = history?.recommendations?.single()
        assertEquals("synthetic_selected_doctor", history?.selectedRecommendationId)
        assertEquals("測試醫師", selected?.doctor)
        assertEquals("09:00-09:30", selected?.sessionTime)
        assertNull(history?.completedAt)
        assertEquals(HistoryStatus.UNCOMPLETED, history?.status)
        repository.deleteHistory(historyId)
    }

    @Test
    fun completedHistoryJsonRestoresRecommendationSnapshot() {
        val parsed = parseHistoryJson(
            """
            [{
              "id":"case_synthetic_snapshot",
              "date":"2026/08/15",
              "typeTitle":"一般內科",
              "summaryText":"synthetic",
              "status":"COMPLETED",
              "completedAt":"2026/08/15 10:30",
              "selectedRecommendationId":"rec_1",
              "chatMessages":[],
              "recommendations":[{
                "recommendationId":"rec_1",
                "parentDepartment":"內科系",
                "department":"一般內科",
                "doctor":"測試醫師",
                "date":"2026-08-20",
                "session":"上午",
                "sessionTime":"09:00-09:30",
                "room":"320診"
              }]
            }]
            """.trimIndent()
        ).single()

        assertEquals("2026/08/15 10:30", parsed.completedAt)
        assertEquals("rec_1", parsed.selectedRecommendationId)
        assertEquals("測試醫師", parsed.recommendations.single().doctor)
    }

    @Test
    fun historyDisplayTitleIsNotUsedAsRecommendationIdentity() {
        val parsed = parseHistoryJson(
            """
            [{
              "id":"case_mismatched_snapshot",
              "date":"2026/08/27",
              "typeTitle":"一般內科",
              "summaryText":"synthetic",
              "status":"COMPLETED",
              "selectedRecommendationId":"rec_throat",
              "chatMessages":[],
              "recommendations":[{
                "recommendationId":"rec_throat",
                "parentDepartment":"五官科",
                "department":"喉科",
                "doctor":"錯誤醫師",
                "date":"2026-09-01",
                "session":"上午"
              }]
            }]
            """.trimIndent()
        ).single()

        assertEquals("rec_throat", parsed.selectedRecommendationId)
        assertEquals("喉科", parsed.recommendations.single().department)
    }

    @Test
    fun completedHistoryCannotBeReopenedOrLoseItsSnapshot() {
        val repository = MedicalRepository()
        val historyId = "case_revision_clears_snapshot_${System.nanoTime()}"
        repository.saveToHistory(
            History(
                id = historyId,
                date = "2026/08/27",
                typeTitle = "一般內科",
                summaryText = "synthetic completed history",
                status = HistoryStatus.COMPLETED,
                recommendations = listOf(
                    HistoryRecommendation(
                        recommendationId = "rec_internal",
                        parentDepartment = "內科系",
                        department = "一般內科",
                        doctor = "測試醫師",
                        date = "2026-09-01",
                        session = "上午"
                    )
                ),
                selectedRecommendationId = "rec_internal"
            )
        )

        repository.saveCurrentChatToHistory(
            historyId = historyId,
            summaryText = "重新編輯中",
            completed = false,
            allowReopen = true
        )

        val revised = repository.getHistoryById(historyId)
        assertEquals(HistoryStatus.COMPLETED, revised?.status)
        assertEquals("rec_internal", revised?.selectedRecommendationId)
        assertEquals(1, revised?.recommendations?.size)
        repository.deleteHistory(historyId)
    }

    @Test
    fun openingHistoryRestoresItsDepartmentInsteadOfLeakingPreviousCase() {
        val repository = MedicalRepository()
        val historyId = "case_history_isolation_${System.nanoTime()}"
        repository.clearRecommendationFlow()
        repository.saveToHistory(
            History(
                id = historyId,
                date = "2026/08/15",
                typeTitle = "睡眠醫學中心",
                summaryText = "synthetic history",
                status = HistoryStatus.COMPLETED,
                chatMessages = listOf(
                    ChatMessage(content = "synthetic message", sender = MessageSender.USER)
                )
            )
        )
        repository.selectRecommendation(
            RecommendationItemDto(
                recommendationId = "synthetic_recommendation",
                parentDept = "內科系",
                childDept = "一般內科",
                doctor = "測試醫師",
                date = "2026-08-16",
                session = "上午",
                score = 1.0
            )
        )

        repository.loadHistoryIntoCurrentChat(historyId)
        repository.saveCurrentChatToHistory(
            historyId = historyId,
            summaryText = "updated synthetic history",
            completed = true
        )

        assertEquals("睡眠醫學中心", repository.getHistoryById(historyId)?.typeTitle)
        assertNull(repository.getActiveCaseId())
        repository.deleteHistory(historyId)
    }

    @Test
    fun missingHistoryClearsPreviousConversationContext() {
        val repository = MedicalRepository()
        repository.clearChatMessages()
        repository.addMessage(ChatMessage(content = "synthetic stale message", sender = MessageSender.AI))
        repository.setActiveCaseId("case_stale")

        val result = repository.loadHistoryIntoCurrentChat("missing_${System.nanoTime()}")

        assertNull(result)
        assertNull(repository.getActiveCaseId())
        assertTrue(repository.getChatMessages().isEmpty())
    }

    @Test
    @OptIn(ExperimentalCoroutinesApi::class)
    fun completedHistoryIsReadOnlyAndCannotRestoreDoctorActions() = runTest {
        Dispatchers.setMain(StandardTestDispatcher(testScheduler))
        val repository = MedicalRepository()
        val historyId = "case_completed_read_only_${System.nanoTime()}"
        repository.saveToHistory(
            History(
                id = historyId,
                date = "2026/08/15",
                typeTitle = "一般內科",
                summaryText = "synthetic completed history",
                status = HistoryStatus.COMPLETED,
                chatMessages = listOf(
                    ChatMessage(content = "synthetic AI reply", sender = MessageSender.AI)
                )
            )
        )
        val session = TtsSession(CoroutineScope(SupervisorJob() + StandardTestDispatcher(testScheduler))) { _, _, _ -> error("TTS") }
        val viewModel = ChatViewModel(repository, ttsSessionFactory = { session })
        var doctorNavigationRequested = false

        viewModel.openHistory(historyId)
        viewModel.onInputTextChanged("synthetic edit attempt")
        viewModel.chooseRecommendation { doctorNavigationRequested = true }

        assertTrue(viewModel.isHistoryReadOnly.value)
        assertFalse(viewModel.showDecisionButtons.value)
        assertFalse(viewModel.showDoctorButton.value)
        assertFalse(doctorNavigationRequested)
        assertTrue(viewModel.inputText.value.isEmpty())
        repository.deleteHistory(historyId)
        viewModel.viewModelScope.cancel()
        session.close()
        Dispatchers.resetMain()
    }
}
