package com.example.medicalaiguidance.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.example.medicalaiguidance.model.ChatMessage
import com.example.medicalaiguidance.model.History
import com.example.medicalaiguidance.model.HistoryStatus
import com.example.medicalaiguidance.model.MessageSender
import com.example.medicalaiguidance.model.VisitPlan
import com.example.medicalaiguidance.network.BatchAnswerDto
import com.example.medicalaiguidance.network.QuestionItemDto
import com.example.medicalaiguidance.network.TriageResultDto
import com.example.medicalaiguidance.network.VoiceChatResponseDto
import com.example.medicalaiguidance.repository.MedicalRepository
import com.example.medicalaiguidance.repository.TtsSession
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import com.example.medicalaiguidance.util.AudioRecorder
import com.example.medicalaiguidance.util.AudioPlayer
import com.example.medicalaiguidance.util.FixedTriageAudioResolver
import com.example.medicalaiguidance.util.SystemTextSpeaker
import java.io.File
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

internal data class ChatLatencyTrace(
    val requestStartElapsedMs: Long,
    val responseReceivedElapsedMs: Long,
    val replyMessageId: String
)

internal fun currentQuestionAnswer(question: QuestionItemDto?, answer: String): BatchAnswerDto? =
    question?.let { current ->
        answer.trim().takeIf { it.isNotBlank() }?.let { BatchAnswerDto(current.key, it) }
    }

internal fun TriageResultDto.isReadyForRecommendation(): Boolean {
    val suggestedDepartment = departmentResult ?: triageCase?.departmentResult
    val hasNoPendingQuestion = !needMoreInfo &&
        nextQuestion.isNullOrBlank() &&
        triage.nextQuestion.isNullOrBlank() &&
        questionBatch.isEmpty()
    val isReadyStage = conversationState.stage == "waiting_confirmation" ||
        conversationState.stage == "recommending"
    return hasNoPendingQuestion &&
        suggestedDepartment?.childDept?.isNotBlank() == true &&
        isReadyStage
}

internal fun VoiceChatResponseDto.isReadyForRecommendation(): Boolean {
    val isReadyStage = stage == "waiting_confirmation" || stage == "recommending"
    return !needMoreInfo &&
        departmentResult?.childDept?.isNotBlank() == true &&
        isReadyStage
}

class ChatViewModel(
    private val repository: MedicalRepository = MedicalRepository(),
    private val ttsSessionFactory: () -> TtsSession = repository::beginTtsSession
) : ViewModel() {
    private val _inputText = MutableStateFlow("")
    val inputText: StateFlow<String> = _inputText.asStateFlow()

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _pendingChatLatency = MutableStateFlow<ChatLatencyTrace?>(null)
    internal val pendingChatLatency: StateFlow<ChatLatencyTrace?> = _pendingChatLatency.asStateFlow()

    private val _isAiThinking = MutableStateFlow(false)
    val isAiThinking: StateFlow<Boolean> = _isAiThinking.asStateFlow()

    private val _showDoctorButton = MutableStateFlow(false)
    val showDoctorButton: StateFlow<Boolean> = _showDoctorButton.asStateFlow()

    private val _showDecisionButtons = MutableStateFlow(false)
    val showDecisionButtons: StateFlow<Boolean> = _showDecisionButtons.asStateFlow()

    private val _selectedVisitType = MutableStateFlow<VisitPlan?>(null)
    val selectedVisitType: StateFlow<VisitPlan?> = _selectedVisitType.asStateFlow()

    private val _currentQuestionBatch = MutableStateFlow<List<QuestionItemDto>>(emptyList())
    val currentQuestionBatch: StateFlow<List<QuestionItemDto>> = _currentQuestionBatch.asStateFlow()

    private val _currentQuestionIndex = MutableStateFlow(0)
    val currentQuestionIndex: StateFlow<Int> = _currentQuestionIndex.asStateFlow()

    private val _currentBatchQuestion = MutableStateFlow<QuestionItemDto?>(null)
    val currentBatchQuestion: StateFlow<QuestionItemDto?> = _currentBatchQuestion.asStateFlow()

    private val _batchAnswers = MutableStateFlow<Map<String, String>>(emptyMap())
    val batchAnswers: StateFlow<Map<String, String>> = _batchAnswers.asStateFlow()

    private val _batchValidationError = MutableStateFlow<String?>(null)
    val batchValidationError: StateFlow<String?> = _batchValidationError.asStateFlow()

    private val _chatError = MutableStateFlow<String?>(null)
    val chatError: StateFlow<String?> = _chatError.asStateFlow()

    private val _urgentWarning = MutableStateFlow<String?>(null)
    val urgentWarning: StateFlow<String?> = _urgentWarning.asStateFlow()

    private val _isHistoryReadOnly = MutableStateFlow(false)
    val isHistoryReadOnly: StateFlow<Boolean> = _isHistoryReadOnly.asStateFlow()

    private val _openedHistory = MutableStateFlow<History?>(null)
    val openedHistory: StateFlow<History?> = _openedHistory.asStateFlow()

    private val _isConfirmingRecommendation = MutableStateFlow(false)
    val isConfirmingRecommendation: StateFlow<Boolean> = _isConfirmingRecommendation.asStateFlow()

    private val _isListening = MutableStateFlow(false)
    val isListening: StateFlow<Boolean> = _isListening.asStateFlow()

    private val _isVoiceTranscribing = MutableStateFlow(false)
    val isVoiceTranscribing: StateFlow<Boolean> = _isVoiceTranscribing.asStateFlow()

    private val _speakingMessageId = MutableStateFlow<String?>(null)
    val speakingMessageId: StateFlow<String?> = _speakingMessageId.asStateFlow()

    private val _voiceStatusMessage = MutableStateFlow<String?>(null)
    val voiceStatusMessage: StateFlow<String?> = _voiceStatusMessage.asStateFlow()

    private val _voiceStatusMessageId = MutableStateFlow<String?>(null)
    val voiceStatusMessageId: StateFlow<String?> = _voiceStatusMessageId.asStateFlow()

    private val audioRecorder = AudioRecorder()
    private var ttsSession: TtsSession? = null
    private var playbackJob: Job? = null
    private var playbackGeneration = 0L
    private var playingAudio: AudioPlayer? = null
    private var playingSpeaker: SystemTextSpeaker? = null
    private var transcriptionJob: Job? = null
    private var transcriptionGeneration = 0L

    private var caseId: String? = null
    private var activeHistoryId: String = "hist_${System.currentTimeMillis()}"
    private var openedHistoryId: String? = null
    private var conversationInitialized: Boolean = false

    init {
        publishMessages()
    }

    fun onInputTextChanged(text: String) {
        if (_isHistoryReadOnly.value) return
        _inputText.value = text
    }

    fun startNewConversation(visitPlan: VisitPlan = VisitPlan.UNKNOWN) {
        if (conversationInitialized) return
        conversationInitialized = true
        openedHistoryId = null
        caseId = null
        activeHistoryId = "hist_${System.currentTimeMillis()}"
        _inputText.value = ""
        _showDecisionButtons.value = false
        _showDoctorButton.value = false
        _isHistoryReadOnly.value = false
        _openedHistory.value = null
        _isConfirmingRecommendation.value = false
        _selectedVisitType.value = visitPlan.takeUnless { it == VisitPlan.UNKNOWN }
        _currentQuestionBatch.value = emptyList()
        _currentQuestionIndex.value = 0
        _currentBatchQuestion.value = null
        _batchAnswers.value = emptyMap()
        _batchValidationError.value = null
        _chatError.value = null
        _urgentWarning.value = null
        repository.clearRecommendationFlow()
        _selectedVisitType.value?.let { repository.setActiveVisitType(it.apiValue) }
        repository.clearChatMessages()
        ttsSession = ttsSessionFactory()
        stopVoicePlayback()
        _messages.value = emptyList()
        _pendingChatLatency.value = null
        if (visitPlan == VisitPlan.INITIAL || visitPlan == VisitPlan.FOLLOW_UP) {
            startTriage(visitPlan)
        }
    }

    fun openHistory(historyId: String) {
        if (openedHistoryId == historyId) return
        conversationInitialized = true
        cancelVoiceRecording()
        ttsSession = ttsSessionFactory()
        _inputText.value = ""
        val history = repository.loadHistoryIntoCurrentChat(historyId)
        _openedHistory.value = history
        openedHistoryId = history?.id
        activeHistoryId = history?.id ?: "hist_${System.currentTimeMillis()}"
        caseId = history?.id?.takeIf { it.startsWith("case_") }
        _selectedVisitType.value = VisitPlan.fromApiValue(history?.visitType).takeUnless { it == VisitPlan.UNKNOWN }
        _currentQuestionBatch.value = emptyList()
        _currentQuestionIndex.value = 0
        _currentBatchQuestion.value = null
        _batchAnswers.value = emptyMap()
        _batchValidationError.value = null
        _chatError.value = null
        _urgentWarning.value = null
        _isHistoryReadOnly.value = history?.status == HistoryStatus.COMPLETED
        _showDecisionButtons.value = false
        _showDoctorButton.value = false
        _isConfirmingRecommendation.value = false
        stopVoicePlayback()
        publishMessages()
        _pendingChatLatency.value = null
    }

    private fun startTriage(visitPlan: VisitPlan) {
        if (_isAiThinking.value) return
        _isAiThinking.value = true
        viewModelScope.launch {
            _chatError.value = null
            try {
                val result = repository.startBatchTriage(visitPlan.apiValue)
                caseId = result.caseId
                consumeTriageResponse(result)
            } catch (error: Exception) {
                val message = error.message ?: "無法開始問診，請稍後再試。"
                _chatError.value = message
                repository.addMessage(ChatMessage(content = message, sender = MessageSender.AI))
            } finally {
                publishMessages()
                _isAiThinking.value = false
            }
        }
    }

    fun sendMessage(
        onAnalysisComplete: () -> Unit,
        requestStartElapsedMs: Long? = null
    ) {
        val text = _inputText.value.trim()
        if (_isHistoryReadOnly.value || text.isBlank() || _isAiThinking.value ||
            _isListening.value || _isVoiceTranscribing.value) return
        if (_currentBatchQuestion.value != null) {
            submitCurrentBatchAnswer(text, onAnalysisComplete)
            return
        }

        _isAiThinking.value = true
        viewModelScope.launch {
            var latencyTrace: ChatLatencyTrace? = null
            repository.addMessage(ChatMessage(content = text, sender = MessageSender.USER))
            publishMessages()
            _inputText.value = ""
            _showDecisionButtons.value = false
            stopVoicePlayback()

            try {
                val result = repository.chat(
                    caseId = caseId,
                    message = text,
                    visitType = _selectedVisitType.value?.apiValue
                )
                val responseReceivedElapsedMs = SystemClock.elapsedRealtime()
                caseId = result.caseId
                repository.setActiveCaseId(caseId)

                updateBatchAndSafetyState(result)
                val reply = _currentBatchQuestion.value?.question
                    ?: result.reply
                    ?: result.nextQuestion
                    ?: result.triage.warningMessage
                    ?: "I received your information. Please continue describing your symptoms or preferred visit time."

                val replyMessage = ChatMessage(content = reply, sender = MessageSender.AI)
                repository.addMessage(replyMessage)
                if (requestStartElapsedMs != null) {
                    latencyTrace = ChatLatencyTrace(
                        requestStartElapsedMs = requestStartElapsedMs,
                        responseReceivedElapsedMs = responseReceivedElapsedMs,
                        replyMessageId = replyMessage.id
                    )
                }

                val stage = result.conversationState.stage
                val readyForRecommendation = result.isReadyForRecommendation()

                _showDecisionButtons.value = readyForRecommendation
                _showDoctorButton.value = readyForRecommendation
                saveHistory(completed = readyForRecommendation, summary = reply)

                if (stage == "recommending") {
                    onAnalysisComplete()
                }
            } catch (error: Exception) {
                val message = error.message ?: "Unable to connect to the backend server. Please try again later."
                _chatError.value = message
                repository.addMessage(ChatMessage(content = message, sender = MessageSender.AI))
                saveHistory(completed = false, summary = message)
            } finally {
                publishMessages()
                latencyTrace?.let { _pendingChatLatency.value = it }
                _isAiThinking.value = false
            }
        }
    }

    private fun submitCurrentBatchAnswer(
        answer: String,
        onAnalysisComplete: () -> Unit
    ) {
        val currentQuestion = _currentBatchQuestion.value ?: return
        if (_isHistoryReadOnly.value || _isAiThinking.value) return
        val keyedAnswer = currentQuestionAnswer(currentQuestion, answer) ?: return

        _batchAnswers.value = mapOf(keyedAnswer.key to keyedAnswer.answer)
        _batchValidationError.value = null
        _inputText.value = ""
        stopVoicePlayback()
        repository.addMessage(ChatMessage(content = answer.trim(), sender = MessageSender.USER))

        publishMessages()
        submitCurrentQuestionAnswer(currentQuestion, keyedAnswer, onAnalysisComplete)
    }

    private fun submitCurrentQuestionAnswer(
        currentQuestion: QuestionItemDto,
        keyedAnswer: BatchAnswerDto,
        onAnalysisComplete: () -> Unit
    ) {
        val activeCase = caseId ?: repository.getActiveCaseId()
        val visitPlan = _selectedVisitType.value?.takeIf {
            it == VisitPlan.INITIAL || it == VisitPlan.FOLLOW_UP
        }
        if (activeCase.isNullOrBlank() || visitPlan == null) {
            _batchValidationError.value = "問診案件狀態已失效，請返回首頁重新開始。"
            return
        }
        _isAiThinking.value = true
        viewModelScope.launch {
            _batchValidationError.value = null
            _chatError.value = null
            try {
                val result = repository.submitBatchAnswers(
                    caseId = activeCase,
                    visitType = visitPlan.apiValue,
                    answers = listOf(keyedAnswer)
                )
                caseId = result.caseId
                consumeTriageResponse(result, onAnalysisComplete)
            } catch (error: Exception) {
                val message = error.message ?: "送出回答失敗，請稍後再試。"
                _chatError.value = message
                repository.addMessage(ChatMessage(content = message, sender = MessageSender.AI))
                repository.addMessage(ChatMessage(content = currentQuestion.question, sender = MessageSender.AI))
                saveHistory(completed = false, summary = message)
            } finally {
                publishMessages()
                _isAiThinking.value = false
            }
        }
    }

    internal fun markChatLatencyRendered(replyMessageId: String) {
        if (_pendingChatLatency.value?.replyMessageId == replyMessageId) {
            _pendingChatLatency.value = null
        }
    }

    fun beginSystemVoiceInput(): Boolean {
        if (_isHistoryReadOnly.value || _isAiThinking.value || _isListening.value ||
            _isVoiceTranscribing.value) return false
        stopVoicePlayback()
        _isListening.value = true
        updatePrefetchPause()
        return true
    }

    fun finishSystemVoiceInput(text: String?) {
        if (!_isListening.value) return
        _isListening.value = false
        text?.let(::acceptVoiceTranscript)
        updatePrefetchPause()
    }

    /** ASR only edits the draft. Only the Send button/IME invokes sendMessage. */
    fun acceptVoiceTranscript(text: String) {
        if (_isHistoryReadOnly.value) return
        val transcript = text.trim()
        if (transcript.isNotBlank()) _inputText.value = transcript
    }

    // ── 語音錄音（國台語通用）──
    fun startVoiceRecording(): Boolean {
        if (
            _isHistoryReadOnly.value ||
            _isAiThinking.value ||
            _isListening.value ||
            _isVoiceTranscribing.value
        ) return false

        stopVoicePlayback()
        ttsSession?.setPrefetchPaused(true)
        val started = audioRecorder.start()
        _isListening.value = started
        updatePrefetchPause()
        if (!started) {
            _voiceStatusMessageId.value = null
            _voiceStatusMessage.value = "無法啟動麥克風，請確認錄音權限與裝置狀態。"
        } else {
            _voiceStatusMessage.value = null
        }
        return started
    }

    fun cancelVoiceRecording() {
        transcriptionGeneration++
        transcriptionJob?.cancel()
        transcriptionJob = null
        audioRecorder.cancel()
        _isListening.value = false
        _isVoiceTranscribing.value = false
        // Leaving the screen must not resume a deferred background request.
        ttsSession?.setPrefetchPaused(true)
    }

    fun stopTaiwaneseRecordingAndTranscribe() {
        if (_isHistoryReadOnly.value || !_isListening.value || _isVoiceTranscribing.value) return
        _isListening.value = false
        val wavBytes = audioRecorder.stop()
        if (wavBytes.isEmpty()) {
            _voiceStatusMessage.value = "未錄到語音，請再試一次。"
            updatePrefetchPause()
            return
        }
        _isVoiceTranscribing.value = true
        _voiceStatusMessageId.value = null
        _voiceStatusMessage.value = "台語語音辨識中"
        val generation = ++transcriptionGeneration
        transcriptionJob = viewModelScope.launch {
            try {
                val result = repository.transcribeVoice(wavBytes, "taiwanese")
                if (generation != transcriptionGeneration) return@launch
                if (!result.asrFailed && result.text.isNotBlank()) {
                    acceptVoiceTranscript(result.text)
                    _voiceStatusMessage.value = null
                } else {
                    _voiceStatusMessage.value = result.error ?: "台語辨識失敗，請改用文字輸入。"
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                Log.w(TAG, "/voice/asr failed lang=taiwanese", error)
                if (generation == transcriptionGeneration) {
                    _voiceStatusMessage.value = "台語辨識失敗，請改用文字輸入。"
                }
            } finally {
                if (generation == transcriptionGeneration) {
                    _isVoiceTranscribing.value = false
                    updatePrefetchPause()
                }
            }
        }
    }

    private fun updatePrefetchPause() {
        ttsSession?.setPrefetchPaused(_isListening.value || _isVoiceTranscribing.value)
    }

    fun prepareSpeech(message: ChatMessage, lang: String) {
        if (message.sender != MessageSender.AI || message.content.isBlank()) return
        if (FixedTriageAudioResolver.resolve(message.content, lang) != null) return
        val session = ttsSession?.takeUnless { it.closed } ?: return
        // Both the UI effect and microphone callbacks can reach this method.
        // Session admission is synchronous, before any HTTP coroutine starts.
        session.prefetch(message.content, lang, paused = _isListening.value || _isVoiceTranscribing.value)
    }

    fun stopVoicePlayback() {
        playbackGeneration++
        playbackJob?.cancel()
        playbackJob = null
        playingAudio?.release()
        playingAudio = null
        playingSpeaker?.stop()
        playingSpeaker = null
        clearVoicePlaybackState()
    }

    fun speakMessage(
        message: ChatMessage,
        audioPlayer: AudioPlayer,
        systemTextSpeaker: SystemTextSpeaker,
        context: Context,
        cacheDir: File,
        lang: String
    ) {
        if (message.sender != MessageSender.AI || message.content.isBlank() ||
            _isListening.value || _isVoiceTranscribing.value) return
        val localResourceId = FixedTriageAudioResolver.resolve(message.content, lang)
        val session = if (localResourceId == null) {
            ttsSession?.takeUnless { it.closed } ?: return
        } else {
            null
        }
        stopVoicePlayback()
        playingAudio = audioPlayer
        playingSpeaker = systemTextSpeaker
        val generation = playbackGeneration
        fun isCurrent() = generation == playbackGeneration &&
            (session == null || (!session.closed && ttsSession === session))
        _speakingMessageId.value = message.id
        _voiceStatusMessageId.value = message.id
        _voiceStatusMessage.value = "語音準備中"

        fun systemFallback() {
            if (!isCurrent()) return
            _voiceStatusMessage.value = "後端語音暫不可用，改用系統朗讀"
            systemTextSpeaker.speak(message.content, "chinese",
                onStart = { if (isCurrent()) _voiceStatusMessage.value = null },
                onDone = { if (isCurrent()) clearVoicePlaybackState(message.id) },
                onError = {
                    if (isCurrent()) {
                        _speakingMessageId.value = null
                        _voiceStatusMessage.value = "語音暫不可用"
                    }
                })
        }

        if (localResourceId != null) {
            _voiceStatusMessage.value = null
            audioPlayer.playRawResource(
                context = context,
                rawResourceId = localResourceId,
                onDone = { if (isCurrent()) clearVoicePlaybackState(message.id) },
                onError = { systemFallback() }
            )
            return
        }

        val dynamicSession = requireNotNull(session)
        playbackJob = viewModelScope.launch {
            try {
                var result = dynamicSession.get(message.content, lang, retryFailure = true)
                if (!isCurrent()) return@launch
                if ((result.ttsFailed || result.audioBase64.isBlank()) && lang == "taiwanese") {
                    _voiceStatusMessage.value = "台語語音暫不可用，改用國語播放"
                    result = dynamicSession.get(message.content, "chinese", retryFailure = true)
                }
                if (!isCurrent()) return@launch
                if (!result.ttsFailed && result.audioBase64.isNotBlank()) {
                    _voiceStatusMessage.value = null
                    audioPlayer.playBase64(result.audioBase64, cacheDir, result.audioFormat,
                        onDone = { if (isCurrent()) clearVoicePlaybackState(message.id) },
                        onError = { systemFallback() })
                } else {
                    systemFallback()
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                Log.w(TAG, "TTS playback unavailable", error)
                systemFallback()
            }
        }
    }

    override fun onCleared() {
        cancelVoiceRecording()
        stopVoicePlayback()
        super.onCleared()
    }

    fun chooseRecommendation(onConfirmed: () -> Unit) {
        if (_isHistoryReadOnly.value || _isConfirmingRecommendation.value) return

        val activeCase = caseId ?: repository.getActiveCaseId()
        if (activeCase.isNullOrBlank()) {
            repository.addMessage(
                ChatMessage(
                    content = "目前沒有可確認的問診案件，請先完成症狀問答。",
                    sender = MessageSender.AI
                )
            )
            publishMessages()
            return
        }

        viewModelScope.launch {
            _isConfirmingRecommendation.value = true
            try {
                val result = repository.confirmTriage(activeCase)
                caseId = result.caseId
                repository.setActiveCaseId(result.caseId)
                onConfirmed()
            } catch (error: Exception) {
                val message = error.message ?: "無法確認問診結果，請稍後再試。"
                repository.addMessage(ChatMessage(content = message, sender = MessageSender.AI))
                publishMessages()
            } finally {
                _isConfirmingRecommendation.value = false
            }
        }
    }

    fun continueEditing() {
        if (_isHistoryReadOnly.value || _isConfirmingRecommendation.value) return
        _showDecisionButtons.value = false
        _showDoctorButton.value = false
        val activeCase = caseId ?: repository.getActiveCaseId()
        if (activeCase.isNullOrBlank()) {
            val prompt = "好的，請直接補充想修改的症狀、科別、嚴重程度或希望看診時間，我會延續這次問診重新整理。"
            repository.addMessage(ChatMessage(content = prompt, sender = MessageSender.AI))
            publishMessages()
            saveHistory(completed = false, summary = prompt)
            return
        }

        _isAiThinking.value = true
        viewModelScope.launch {
            try {
                val result = repository.requestRevision(activeCase)
                caseId = result.caseId
                repository.setActiveCaseId(result.caseId)
                val prompt = result.reply
                    ?: result.nextQuestion
                    ?: "好的，我會延續這次問診紀錄重新整理。請補充想修改的症狀、科別、嚴重程度或看診日期/時段。"
                repository.addMessage(ChatMessage(content = prompt, sender = MessageSender.AI))
                saveHistory(completed = false, summary = prompt)
            } catch (error: Exception) {
                val prompt = "已切回可修改狀態。請直接補充想修改的症狀、科別、嚴重程度或看診時間。"
                repository.addMessage(ChatMessage(content = prompt, sender = MessageSender.AI))
                saveHistory(completed = false, summary = prompt)
            } finally {
                publishMessages()
                _isAiThinking.value = false
            }
        }
    }

    private fun consumeTriageResponse(
        result: TriageResultDto,
        onAnalysisComplete: () -> Unit = {}
    ) {
        repository.setActiveCaseId(result.caseId)
        updateBatchAndSafetyState(result)
        val reply = _currentBatchQuestion.value?.question
            ?: result.reply
            ?: result.nextQuestion
            ?: result.triage.warningMessage
            ?: if (result.questionBatch.isNotEmpty()) "請一次回答以下問題。" else "請繼續描述症狀或可看診時間。"
        repository.addMessage(ChatMessage(content = reply, sender = MessageSender.AI))

        val stage = result.conversationState.stage
        val readyForRecommendation = result.isReadyForRecommendation()
        _showDecisionButtons.value = readyForRecommendation
        _showDoctorButton.value = readyForRecommendation
        saveHistory(completed = readyForRecommendation, summary = reply)
        if (stage == "recommending") onAnalysisComplete()
    }

    private fun updateBatchAndSafetyState(result: TriageResultDto) {
        _chatError.value = null
        _currentQuestionBatch.value = result.questionBatch
        _currentQuestionIndex.value = 0
        _currentBatchQuestion.value = result.questionBatch.firstOrNull()
        _batchAnswers.value = emptyMap()
        _batchValidationError.value = null
        _urgentWarning.value = result.triage.warningMessage
            ?.takeIf { result.triage.warningRequired }
        val responseVisitPlan = VisitPlan.fromApiValue(result.triageCase?.visitType)
        if (_selectedVisitType.value == null && responseVisitPlan != VisitPlan.UNKNOWN) {
            _selectedVisitType.value = responseVisitPlan
        }
    }

    private fun saveHistory(completed: Boolean, summary: String) {
        repository.saveCurrentChatToHistory(
            historyId = caseId ?: activeHistoryId,
            summaryText = summary,
            completed = completed
        )
    }

    private fun publishMessages() {
        val updated = repository.getChatMessages().toList()
        if (updated.lastOrNull()?.id != _messages.value.lastOrNull()?.id) stopVoicePlayback()
        _messages.value = updated
    }

    private fun clearVoicePlaybackState(messageId: String? = null) {
        if (messageId != null && _speakingMessageId.value != messageId) return
        _speakingMessageId.value = null
        _voiceStatusMessageId.value = null
        _voiceStatusMessage.value = null
    }

    private companion object {
        const val TAG = "ChatViewModel"
    }
}

internal fun missingRequiredQuestionKeys(
    questions: List<QuestionItemDto>,
    answers: Map<String, String>
): List<String> = questions
    .filter { it.required && answers[it.key].orEmpty().isBlank() }
    .map { it.key }
