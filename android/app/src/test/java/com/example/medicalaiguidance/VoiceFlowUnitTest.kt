package com.example.medicalaiguidance

import androidx.lifecycle.viewModelScope
import com.example.medicalaiguidance.model.ChatMessage
import com.example.medicalaiguidance.model.MessageSender
import com.example.medicalaiguidance.network.TtsRequest
import com.example.medicalaiguidance.network.VoiceTtsResponseDto
import com.example.medicalaiguidance.network.toJson
import com.example.medicalaiguidance.repository.MedicalRepository
import com.example.medicalaiguidance.repository.TtsSession
import com.example.medicalaiguidance.viewmodel.ChatViewModel
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async
import kotlinx.coroutines.cancel
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.advanceTimeBy
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

@OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
class VoiceFlowUnitTest {
    @Test
    fun backgroundLimitIsOneAndOnlyLatestUnsentQuestionIsGenerated() = runTest {
        val calls = mutableListOf<String>()
        val gate = CompletableDeferred<Unit>()
        val session = TtsSession(CoroutineScope(coroutineContext + SupervisorJob())) { _, text, _ ->
            calls.add(text)
            if (text == "Q1") gate.await()
            VoiceTtsResponseDto(audioBase64 = text)
        }
        try {
            session.prefetch("Q1", "taiwanese")
            session.prefetch("Q2", "taiwanese")
            session.prefetch("Q3", "taiwanese")
            assertEquals(listOf("Q1"), calls)
            gate.complete(Unit)
            runCurrent()
            assertEquals(listOf("Q1", "Q3"), calls)
            assertEquals("Q1", session.get("Q1", "taiwanese").audioBase64)
            assertEquals(2, calls.size)
        } finally { session.close() }
    }

    @Test
    fun pausePreservesReadyAndDispatchedAudioAndResumesOnlyLatest() = runTest {
        val calls = mutableListOf<String>()
        val gate = CompletableDeferred<Unit>()
        val session = TtsSession(CoroutineScope(coroutineContext + SupervisorJob())) { _, text, _ ->
            calls.add(text)
            if (text == "running") gate.await()
            VoiceTtsResponseDto(audioBase64 = text)
        }
        try {
            session.prefetch("ready", "taiwanese")
            session.prefetch("running", "taiwanese")
            session.prefetch("old queued", "taiwanese")
            session.setPrefetchPaused(true)
            session.prefetch("current", "taiwanese")
            gate.complete(Unit)
            runCurrent()
            assertEquals(listOf("ready", "running"), calls)
            assertEquals("ready", session.get("ready", "taiwanese").audioBase64)
            assertEquals("running", session.get("running", "taiwanese").audioBase64)
            session.setPrefetchPaused(false)
            assertEquals(listOf("ready", "running", "current"), calls)
        } finally { session.close() }
    }

    @Test
    fun manualPlaybackPromotesUnsentPrefetchWithoutDuplicateHttp() = runTest {
        val calls = mutableListOf<String>()
        val gate = CompletableDeferred<Unit>()
        val session = TtsSession(CoroutineScope(coroutineContext + SupervisorJob())) { _, text, _ ->
            calls.add(text)
            if (text == "slow background") gate.await()
            VoiceTtsResponseDto(audioBase64 = text)
        }
        try {
            session.prefetch("slow background", "taiwanese")
            session.prefetch("play now", "taiwanese")
            assertEquals(1, calls.size)
            assertEquals("play now", session.get("play now", "taiwanese").audioBase64)
            gate.complete(Unit)
            runCurrent()
            session.prefetch("play now", "taiwanese")
            assertEquals(listOf("slow background", "play now"), calls)
        } finally { session.close() }
    }

    @Test
    fun viewModelVoiceInputPausesPrefetchAndDraftCompletionResumesIt() = runTest {
        Dispatchers.setMain(StandardTestDispatcher(testScheduler))
        val calls = mutableListOf<String>()
        val session = TtsSession(CoroutineScope(coroutineContext + SupervisorJob())) { _, text, _ ->
            calls.add(text)
            VoiceTtsResponseDto(audioBase64 = text)
        }
        val model = ChatViewModel(ttsSessionFactory = { session })
        try {
            model.startNewConversation()
            assertTrue(model.beginSystemVoiceInput())
            model.prepareSpeech(ChatMessage(content = "current", sender = MessageSender.AI), "taiwanese")
            runCurrent()
            assertTrue(calls.isEmpty())
            model.finishSystemVoiceInput("使用者回答")
            assertEquals(listOf("current"), calls)
            assertEquals("使用者回答", model.inputText.value)
            assertTrue(model.messages.value.isEmpty())
            assertFalse(model.isAiThinking.value)
            model.onInputTextChanged("使用者修改")
            assertEquals("使用者修改", model.inputText.value)
        } finally {
            session.close()
            model.viewModelScope.cancel()
            Dispatchers.resetMain()
        }
    }

    @Test
    fun systemAsrCancellationAndDuplicateCallbackPreserveDraft() {
        val model = ChatViewModel()
        model.onInputTextChanged("原有草稿")
        assertTrue(model.beginSystemVoiceInput())
        assertFalse(model.beginSystemVoiceInput())
        model.finishSystemVoiceInput(null)
        assertEquals("原有草稿", model.inputText.value)
        assertFalse(model.isListening.value)
        assertTrue(model.beginSystemVoiceInput())
        model.finishSystemVoiceInput("辨識文字")
        model.onInputTextChanged("使用者修正")
        model.finishSystemVoiceInput("辨識文字")
        assertEquals("使用者修正", model.inputText.value)
        assertFalse(model.isAiThinking.value)
    }

    @Test
    fun transcriptOnlyUpdatesEditableDraftAndBlankResultPreservesText() {
        val repository = MedicalRepository()
        val model = ChatViewModel(repository)
        val before = model.messages.value
        model.onInputTextChanged("原先輸入")
        model.acceptVoiceTranscript("  ")
        assertEquals("原先輸入", model.inputText.value)
        model.acceptVoiceTranscript("  頭痛兩天  ")
        model.acceptVoiceTranscript("  頭痛兩天  ")
        assertEquals("頭痛兩天", model.inputText.value)
        assertEquals(before, model.messages.value)
        assertFalse(model.isAiThinking.value)
        model.onInputTextChanged("頭痛三天")
        assertEquals("頭痛三天", model.inputText.value)
    }

    @Test
    fun rapidSendSetsGuardBeforeCoroutineDispatch() = runTest {
        Dispatchers.setMain(StandardTestDispatcher(testScheduler))
        val model = ChatViewModel()
        try {
            model.onInputTextChanged("手動輸入")
            model.sendMessage({})
            assertTrue(model.isAiThinking.value)
            model.sendMessage({})
            assertEquals(1, model.viewModelScope.coroutineContext[Job]!!.children.count())
            // Cancel queued work: this guard test must never contact a real backend.
        } finally {
            model.viewModelScope.cancel()
            Dispatchers.resetMain()
        }
    }

    @Test
    fun prefetchAndRepeatedPlaybackShareOneInFlightRequestAndCache() = runTest {
        var calls = 0
        val gate = CompletableDeferred<VoiceTtsResponseDto>()
        val session = TtsSession(CoroutineScope(coroutineContext + SupervisorJob())) { _, _, _ ->
            calls++
            gate.await()
        }
        try {
            val prefetch = async { session.get("Q1", "chinese") }
            runCurrent()
            val playback = async { session.get("Q1", "chinese", retryFailure = true) }
            val repeat = async { session.get(" Q1 ", "chinese", retryFailure = true) }
            runCurrent()
            assertEquals(1, calls)
            prefetch.cancel() // Leaving a screen must not cancel another waiter.
            gate.complete(VoiceTtsResponseDto(audioBase64 = "YXVkaW8="))
            assertEquals(playback.await(), repeat.await())
            session.get("Q1", "chinese", retryFailure = true)
            assertEquals(1, calls)
        } finally { session.close() }
    }

    @Test
    fun differentQuestionsLanguagesAndSessionsNeverShareAudio() = runTest {
        var calls = 0
        fun newSession() = TtsSession(CoroutineScope(coroutineContext + SupervisorJob())) { id, text, lang ->
            calls++
            VoiceTtsResponseDto(audioBase64 = "$id:$lang:$text")
        }
        val first = newSession()
        val second = newSession()
        try {
            val q1 = first.get("Q1", "chinese")
            assertNotEquals(q1, first.get("Q2", "chinese"))
            assertNotEquals(q1, first.get("Q1", "taiwanese"))
            assertNotEquals(q1, second.get("Q1", "chinese"))
            assertEquals(q1, first.get("Q1", "chinese"))
            assertEquals(4, calls)
        } finally { first.close(); second.close() }
    }

    @Test
    fun prefetchFailureIsQuietAndExplicitPlaybackCanRetry() = runTest {
        var calls = 0
        val session = TtsSession(CoroutineScope(coroutineContext + SupervisorJob())) { _, _, _ ->
            calls++
            if (calls == 1) error("gateway unavailable")
            VoiceTtsResponseDto(audioBase64 = "YXVkaW8=")
        }
        try {
            assertTrue(session.get("Q1", "taiwanese").ttsFailed)
            assertTrue(session.get("Q1", "taiwanese").ttsFailed)
            assertEquals(1, calls)
            assertFalse(session.get("Q1", "taiwanese", retryFailure = true).ttsFailed)
            assertEquals(2, calls)
        } finally { session.close() }
    }

    @Test
    fun cleanupCancelsPendingAudioAndPreventsReuse() = runTest {
        val gate = CompletableDeferred<VoiceTtsResponseDto>()
        val session = TtsSession(CoroutineScope(coroutineContext + SupervisorJob())) { _, _, _ -> gate.await() }
        val wait = async { session.get("Q1", "chinese") }
        runCurrent()
        session.close()
        gate.complete(VoiceTtsResponseDto(audioBase64 = "old"))
        runCurrent()
        assertTrue(wait.isCancelled)
        assertTrue(session.closed)
        try {
            session.get("Q1", "chinese")
            fail("closed session accepted playback")
        } catch (_: IllegalStateException) { }
    }

    @Test
    fun localAudioExpiresAfterIdleTtl() = runTest {
        var calls = 0
        val session = TtsSession(CoroutineScope(coroutineContext + SupervisorJob()), ttlMillis = 100) { _, _, _ ->
            calls++
            VoiceTtsResponseDto(audioBase64 = "audio$calls")
        }
        try {
            val first = session.get("Q1", "chinese")
            runCurrent()
            advanceTimeBy(101)
            runCurrent()
            assertNotEquals(first, session.get("Q1", "chinese"))
        } finally { session.close() }
    }

    @Test
    fun ttsRequestIncludesSessionWithoutChangingExistingFields() {
        val body = JSONObject(TtsRequest("Q1", "taiwanese", sessionId = "session").toJson())
        assertEquals("session", body.getString("session_id"))
        assertEquals("Q1", body.getString("text"))
        assertEquals("taiwanese", body.getString("lang"))
        assertEquals(1.0, body.getDouble("speed"), 0.0)
        assertFalse(JSONObject(TtsRequest("legacy").toJson()).has("session_id"))
    }
}
