package com.example.medicalaiguidance.screen

import android.Manifest
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.os.SystemClock
import android.speech.RecognizerIntent
import android.util.Log
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import com.example.medicalaiguidance.R
import com.example.medicalaiguidance.model.History
import com.example.medicalaiguidance.model.HistoryRecommendation
import com.example.medicalaiguidance.model.MessageSender
import com.example.medicalaiguidance.model.VisitPlan
import com.example.medicalaiguidance.navigation.Route
import com.example.medicalaiguidance.util.AudioPlayer
import com.example.medicalaiguidance.util.SystemTextSpeaker
import com.example.medicalaiguidance.viewmodel.ChatViewModel
import java.util.Locale
import kotlinx.coroutines.delay
import androidx.compose.foundation.border
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.unit.LayoutDirection
import androidx.core.content.ContextCompat
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    navController: NavHostController,
    historyId: String? = null,
    startNew: Boolean = false,
    visitPlan: VisitPlan = VisitPlan.UNKNOWN,
    viewModel: ChatViewModel = viewModel()
) {
    val bgGradient = Brush.verticalGradient(
        colors = listOf(Color(0xFFF5F9F9), Color(0xFFE8F2F1), Color(0xFFF8FAFA))
    )
    val primaryDark = Color(0xFF2F6F73)
    val hintGray = Color(0xFF8FA3A6)
    val micBgColor = Color(0xFFE5F2F2)

    val messages by viewModel.messages.collectAsState()
    val pendingChatLatency by viewModel.pendingChatLatency.collectAsState()
    val inputText by viewModel.inputText.collectAsState()
    val isAiThinking by viewModel.isAiThinking.collectAsState()
    val isListening by viewModel.isListening.collectAsState()
    val isVoiceTranscribing by viewModel.isVoiceTranscribing.collectAsState()
    val showDecisionButtons by viewModel.showDecisionButtons.collectAsState()
    val currentBatchQuestion by viewModel.currentBatchQuestion.collectAsState()
    val urgentWarning by viewModel.urgentWarning.collectAsState()
    val isHistoryReadOnly by viewModel.isHistoryReadOnly.collectAsState()
    val openedHistory by viewModel.openedHistory.collectAsState()
    val isConfirmingRecommendation by viewModel.isConfirmingRecommendation.collectAsState()
    val speakingMessageId by viewModel.speakingMessageId.collectAsState()
    val voiceStatusMessage by viewModel.voiceStatusMessage.collectAsState()
    val selectedVisitType by viewModel.selectedVisitType.collectAsState()
    var selectedLanguage by remember { mutableStateOf("國語") }
    var showHistoryRecommendations by remember(openedHistory?.id) { mutableStateOf(false) }
    var isInputFocused by remember { mutableStateOf(false) }
    val canSendMessage = !isHistoryReadOnly && inputText.isNotBlank() &&
        !isAiThinking && !isListening && !isVoiceTranscribing
    val navigateToDoctorSelection = {
        val effectiveVisitPlan = selectedVisitType ?: visitPlan
        if (effectiveVisitPlan != VisitPlan.UNKNOWN) {
            navController.navigate(Route.selectDoctor(effectiveVisitPlan))
        }
    }

    val listState = rememberLazyListState()
    val context = LocalContext.current
    val audioPlayer = remember { AudioPlayer() }
    val systemTextSpeaker = remember { SystemTextSpeaker(context) }
    val keyboardController = LocalSoftwareKeyboardController.current
    val focusManager = LocalFocusManager.current
    val speechLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val transcript = result.data
                ?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)
                ?.firstOrNull()
                .orEmpty()
            viewModel.finishSystemVoiceInput(transcript)
        } else {
            viewModel.finishSystemVoiceInput(null)
        }
    }
    val launchVoiceInput = {
        if (viewModel.beginSystemVoiceInput()) {
            keyboardController?.hide()
            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.TAIWAN.toLanguageTag())
                putExtra(
                    RecognizerIntent.EXTRA_PROMPT,
                    currentBatchQuestion?.question ?: "請說出你的症狀或想掛號的需求"
                )
            }
            try {
                speechLauncher.launch(intent)
            } catch (_: ActivityNotFoundException) {
                viewModel.finishSystemVoiceInput(null)
                Toast.makeText(context, "這台手機沒有可用的語音輸入服務", Toast.LENGTH_LONG).show()
            }
        }
    }
    val microphonePermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            if (!viewModel.startVoiceRecording()) {
                Toast.makeText(
                    context,
                    "無法啟動麥克風，請確認裝置狀態。",
                    Toast.LENGTH_LONG
                ).show()
            }
        } else {
            Toast.makeText(
                context,
                "需要麥克風權限才能使用台語語音輸入，請在系統設定中開啟權限。",
                Toast.LENGTH_LONG
            ).show()
        }
    }
    val isTaiwaneseMode = selectedLanguage == "台語"
    val currentVoiceLang = if (isTaiwaneseMode) "taiwanese" else "chinese"
    val handleMicClick = {
        if (isTaiwaneseMode) {
            keyboardController?.hide()
            if (!isAiThinking && !isVoiceTranscribing) {
                if (isListening) {
                    viewModel.stopTaiwaneseRecordingAndTranscribe()
                } else {
                    if (
                        ContextCompat.checkSelfPermission(
                            context,
                            Manifest.permission.RECORD_AUDIO
                        ) == PackageManager.PERMISSION_GRANTED
                    ) {
                        if (!viewModel.startVoiceRecording()) {
                            Toast.makeText(
                                context,
                                "無法啟動麥克風，請確認錄音權限與裝置狀態。",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    } else {
                        microphonePermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    }
                }
            }
        } else {
            launchVoiceInput()
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            viewModel.cancelVoiceRecording()
            viewModel.stopVoicePlayback()
            audioPlayer.release()
            systemTextSpeaker.shutdown()
        }
    }

    LaunchedEffect(historyId, startNew, visitPlan) {
        when {
            historyId != null -> viewModel.openHistory(historyId)
            startNew -> viewModel.startNewConversation(visitPlan)
        }
    }

    // Text is already in Compose state; preparation never gates rendering or plays audio.
    LaunchedEffect(messages.lastOrNull()?.id, currentVoiceLang) {
        viewModel.stopVoicePlayback()
        messages.lastOrNull()?.takeIf { it.sender == MessageSender.AI }?.let {
            viewModel.prepareSpeech(it, currentVoiceLang)
        }
    }

    LaunchedEffect(
        messages.lastOrNull()?.id,
        isAiThinking,
        showDecisionButtons,
        inputText,
        isInputFocused
    ) {
        if (messages.isNotEmpty()) {
            val conversationStartIndex = messages.indexOfLast { it.sender == MessageSender.USER }
                .takeIf { it >= 0 }
                ?: messages.lastIndex
            delay(80)
            listState.animateScrollToItem(conversationStartIndex)
            if (isInputFocused || isAiThinking || showDecisionButtons) {
                delay(220)
                listState.animateScrollToItem(conversationStartIndex)
            }
        }
    }

    LaunchedEffect(pendingChatLatency, messages.lastOrNull()?.id) {
        val trace = pendingChatLatency ?: return@LaunchedEffect
        if (messages.none { it.id == trace.replyMessageId }) return@LaunchedEffect

        withFrameNanos { }
        val replyRenderedElapsedMs = SystemClock.elapsedRealtime()
        val networkBackendMs = trace.responseReceivedElapsedMs - trace.requestStartElapsedMs
        val uiUpdateMs = replyRenderedElapsedMs - trace.responseReceivedElapsedMs
        val totalMs = replyRenderedElapsedMs - trace.requestStartElapsedMs
        Log.d(
            "ChatLatency",
            "request_start=${trace.requestStartElapsedMs} " +
                "response_received=${trace.responseReceivedElapsedMs} " +
                "reply_rendered=$replyRenderedElapsedMs " +
                "network_backend_ms=$networkBackendMs " +
                "ui_update_ms=$uiUpdateMs " +
                "total_ms=$totalMs"
        )
        viewModel.markChatLatencyRendered(trace.replyMessageId)
    }

    val performSendMessage = {
        if (canSendMessage) {
            viewModel.sendMessage(
                onAnalysisComplete = {
                    navigateToDoctorSelection()
                },
                requestStartElapsedMs = SystemClock.elapsedRealtime()
            )
        }
    }
    // 畫一個正下方的倒三角形
    val TriangleShape = object : Shape {
        override fun createOutline(
            size: androidx.compose.ui.geometry.Size,
            layoutDirection: LayoutDirection,
            density: androidx.compose.ui.unit.Density
        ): androidx.compose.ui.graphics.Outline {
            val path = Path().apply {
                moveTo(0f, 0f)                         // 左上點
                lineTo(size.width, 0f)                 // 右上點
                lineTo(size.width / 2f, size.height)   // 下方尖角點（置中）
                close()
            }
            return androidx.compose.ui.graphics.Outline.Generic(path)
        }
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(bgGradient)
            .imePadding()
    ) {
        // ---- Message list (bottom layer) ----
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.TopCenter
        ) {
            if (messages.isEmpty()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 150.dp), // 稍微留點空間給下方的羊
                    contentAlignment = Alignment.Center
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {

                        // 1. 氣泡框主體 + 倒三角組合
                        Column(
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            // 圓角文字泡泡 (70% 透明度白底)
                            Box(
                                modifier = Modifier
                                    .background(
                                        color = Color.White.copy(alpha = 0.5f),
                                        shape = RoundedCornerShape(30.dp) // 還原圖片中非常圓潤的膠囊圓角
                                    )
                                    .padding(horizontal = 36.dp, vertical = 24.dp), // 內距拉開讓氣泡飽滿
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text = "不舒服嗎？請告訴我",
                                    color = primaryDark,
                                    fontSize = 26.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }

                            // 正下方的倒三角 (同樣繼承 70% 透明度白底)
                            Box(
                                modifier = Modifier
                                    .width(24.dp)   // 三角形的寬度
                                    .height(12.dp)  // 三角形的高度
                                    .background(
                                        color = Color.White.copy(alpha = 0.5f),
                                        shape = TriangleShape
                                    )
                            )
                        }

                        // 間距推開
                        Spacer(modifier = Modifier.height(24.dp))

                        // 2. 待在對話框正下方
                        Image(
                            painter = painterResource(id = R.drawable.sheep_4),
                            contentDescription = null,
                            modifier = Modifier.size(130.dp),
                            contentScale = ContentScale.Fit
                        )
                    }
                }
            }

            LazyColumn(
                state = listState,
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(
                    start = 18.dp,
                    end = 18.dp,
                    top = 16.dp,
                    bottom = if (isHistoryReadOnly) 32.dp else 152.dp
                ),
                verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.Bottom)
            ) {
                items(messages, key = { it.id }) { msg ->
                    val isUser = msg.sender == MessageSender.USER
                    val isLastMessage = msg.id == messages.lastOrNull()?.id

                    Column(modifier = Modifier.fillMaxWidth()) {
                        RealBubbleItem(
                            messageContent = msg.content,
                            isUser = isUser,
                            primaryDark = primaryDark,
                            isSpeaking = speakingMessageId == msg.id,
                            isPlaybackLocked = speakingMessageId != null,
                            onSpeakClicked = {
                                viewModel.speakMessage(
                                    message = msg,
                                    audioPlayer = audioPlayer,
                                    systemTextSpeaker = systemTextSpeaker,
                                    context = context,
                                    cacheDir = context.cacheDir,
                                    lang = currentVoiceLang
                                )
                            }
                        )

                        if (showDecisionButtons && !isHistoryReadOnly && !isUser && isLastMessage) {
                            AnimatedVisibility(
                                visible = true,
                                enter = fadeIn() + slideInVertically(initialOffsetY = { it / 2 })
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(top = 20.dp), // 按鈕與對話框間距
                                    horizontalArrangement = Arrangement.spacedBy(14.dp)
                                ) {
                                    ChatActionButton(
                                        text = "我想修改",
                                        containerColor = Color(0xFFD5E5E5),
                                        contentColor = primaryDark,
                                        modifier = Modifier.weight(1f),
                                        enabled = !isConfirmingRecommendation,
                                        onClick = { viewModel.continueEditing() }
                                    )
                                    ChatActionButton(
                                        text = "看推薦醫師",
                                        containerColor = primaryDark,
                                        contentColor = Color.White,
                                        modifier = Modifier.weight(1f),
                                        isLoading = isConfirmingRecommendation,
                                        onClick = {
                                            keyboardController?.hide()
                                            focusManager.clearFocus(force = true)
                                            viewModel.chooseRecommendation {
                                                navigateToDoctorSelection()
                                            }
                                        }
                                    )
                                }
                            }
                        }
                    }
                }
                urgentWarning?.let { warning ->
                    item(key = "urgent_warning") {
                        UrgentWarningCard(warning = warning)
                    }
                }
                val historySnapshot = openedHistory
                if (isHistoryReadOnly && historySnapshot != null) {
                    item(key = "completed_history_${historySnapshot.id}") {
                        CompletedHistorySnapshot(
                            history = historySnapshot,
                            showRecommendations = showHistoryRecommendations,
                            onToggleRecommendations = {
                                showHistoryRecommendations = !showHistoryRecommendations
                            },
                            primaryDark = primaryDark
                        )
                    }
                }
                if (isAiThinking) {
                    item {
                        AiThinkingBubble(primaryDark = primaryDark)
                    }
                }

                item(key = "chat_bottom_anchor") {
                    Spacer(modifier = Modifier.height(1.dp))
                }
            }
        }

        // ----頂部遮罩層 漸進式變淡 Header (floating overlay, fades to transparent) ----
        Row(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(
                            Color(0xFFF5F9F9).copy(alpha = 0.95f),
                            Color(0xFFEBF3F2).copy(alpha = 0.75f),
                            Color(0xFFEBF3F2).copy(alpha = 0.35f),
                            Color(0xFFF8FAFA).copy(alpha = 0f)
                        )
                    )
                )
                .statusBarsPadding()
                .padding(horizontal = 22.dp, vertical = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(48.dp)
                    .shadow(elevation = 4.dp, shape = RoundedCornerShape(16.dp))
                    .background(Color.White, shape = RoundedCornerShape(16.dp))
                    .clickable { navController.popBackStack() },
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                    contentDescription = "返回",
                    tint = primaryDark,
                    modifier = Modifier.size(22.dp)
                )
            }

            Spacer(modifier = Modifier.weight(1f))

            if (isHistoryReadOnly) {
                Text(
                    text = "歷史紀錄（僅供查看）",
                    color = primaryDark,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold
                )
            } else {
                LanguageToggle(
                    selectedLanguage = selectedLanguage,
                    onLanguageSelected = { if (!isListening && !isVoiceTranscribing) selectedLanguage = it },
                    primaryDark = primaryDark,
                    selectedColor = micBgColor
                )
            }
        }

        // ---- 使用者輸入框 Input bar (floating overlay at bottom) ----
        if (!isHistoryReadOnly) Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .navigationBarsPadding()
                .padding(start = 20.dp, end = 20.dp, bottom = 16.dp)
                .heightIn(min = 68.dp, max = 150.dp)
                .shadow(elevation = 1.dp, shape = RoundedCornerShape(34.dp))
                .border(
                    width = 0.5.dp,
                    color = primaryDark.copy(alpha = 0.08f),
                    shape = RoundedCornerShape(30.dp)
                )
                .background(Color.White, shape = RoundedCornerShape(34.dp))
                .padding(start = 8.dp, end = 20.dp, top = 4.dp, bottom = 4.dp),
            verticalArrangement = Arrangement.Center
        ) {
            voiceStatusMessage?.let { status ->
                Text(status, color = primaryDark, fontSize = 13.sp, maxLines = 2,
                    modifier = Modifier.padding(start = 16.dp, top = 6.dp))
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(52.dp)
                        .background(
                            color = if (isListening) Color(0xFFE74C3C) else micBgColor,
                            shape = CircleShape
                        )
                        .alpha(if (isVoiceTranscribing) 0.45f else 1f)
                        .clickable(enabled = !isVoiceTranscribing) { handleMicClick() },
                    contentAlignment = Alignment.Center
                ) {
                    if (isListening) {
                        Icon(
                            imageVector = Icons.Default.Stop,
                            contentDescription = "停止錄音",
                            tint = Color.White,
                            modifier = Modifier.size(26.dp)
                        )
                    } else {
                        Icon(
                            painter = painterResource(id = R.drawable.ic_mic),
                            contentDescription = "語音輸入",
                            tint = primaryDark,
                            modifier = Modifier.size(26.dp)
                        )
                    }
                }

                Spacer(modifier = Modifier.width(14.dp))

                Box(
                    modifier = Modifier
                        .weight(1f)
                        .padding(vertical = 6.dp)
                ) {
                    if (isVoiceTranscribing) {
                        VoiceTranscribingIndicator(primaryDark = primaryDark)
                    } else if (isListening) {
                        VoiceWaveform(primaryDark = primaryDark)
                    } else if (inputText.isEmpty()) {
                        Text(
                            text = "點我詢問",
                            color = hintGray,
                            fontSize = 18.sp
                        )
                    }
                    if (!isListening && !isVoiceTranscribing) {
                        BasicTextField(
                            value = inputText,
                            onValueChange = { viewModel.onInputTextChanged(it) },
                            maxLines = 3,
                            textStyle = TextStyle(color = primaryDark, fontSize = 18.sp),
                            cursorBrush = SolidColor(primaryDark),
                            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
                            keyboardActions = KeyboardActions(onSearch = { performSendMessage() }),
                            modifier = Modifier
                                .fillMaxWidth()
                                .onFocusChanged { isInputFocused = it.isFocused }
                        )
                    }
                }

                Spacer(modifier = Modifier.width(12.dp))

                Icon(
                    painter = painterResource(id = R.drawable.ic_send),
                    contentDescription = "送出",
                    tint = if (canSendMessage) primaryDark else Color.LightGray,
                    modifier = Modifier
                        .size(34.dp)
                        .clickable(enabled = canSendMessage) { performSendMessage() }
                )
            }
        }
    }
}

@Composable
private fun CompletedHistorySnapshot(
    history: History,
    showRecommendations: Boolean,
    onToggleRecommendations: () -> Unit,
    primaryDark: Color
) {
    val selectedRecommendation = history.recommendations.firstOrNull {
        it.recommendationId == history.selectedRecommendationId
    }
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 8.dp, bottom = 12.dp)
            .background(Color.White, RoundedCornerShape(22.dp))
            .border(1.dp, primaryDark.copy(alpha = 0.12f), RoundedCornerShape(22.dp))
            .padding(18.dp)
    ) {
        Column(modifier = Modifier.fillMaxWidth()) {
            Text(
                text = "問診完成摘要",
                color = primaryDark,
                fontSize = 19.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(14.dp))
            HistorySnapshotRow("最終就醫科別", history.typeTitle)
            HistorySnapshotRow(
                "就診類型",
                when (history.visitType) {
                    "initial" -> "初診"
                    "followup" -> "複診"
                    "return_visit" -> "回診"
                    else -> "未記錄"
                }
            )
            HistorySnapshotRow("問診完成時間", history.completedAt ?: history.date)
            HistorySnapshotRow(
                "當時選擇醫師",
                selectedRecommendation?.doctor ?: "未選擇或舊紀錄未保存"
            )
            HistorySnapshotRow(
                "當時選擇時段",
                selectedRecommendation?.displayHistoryTime() ?: "未選擇或舊紀錄未保存"
            )

            Spacer(modifier = Modifier.height(8.dp))
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color(0xFFEAF3F2), RoundedCornerShape(14.dp))
                    .clickable(
                        enabled = history.recommendations.isNotEmpty(),
                        onClick = onToggleRecommendations
                    )
                    .padding(horizontal = 14.dp, vertical = 13.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = if (history.recommendations.isEmpty()) {
                            "此舊紀錄未保存醫師推薦"
                        } else {
                            "查看醫師推薦（${history.recommendations.size}）"
                        },
                        color = primaryDark,
                        fontWeight = FontWeight.Bold,
                        fontSize = 15.sp
                    )
                    if (history.recommendations.isNotEmpty()) {
                        Text(
                            text = if (showRecommendations) "收起" else "展開",
                            color = primaryDark,
                            fontSize = 14.sp
                        )
                    }
                }
            }

            if (showRecommendations) {
                Spacer(modifier = Modifier.height(10.dp))
                Text(
                    text = "以下為當時結果，僅供查看，無法重新選擇。",
                    color = Color(0xFF708080),
                    fontSize = 13.sp
                )
                Spacer(modifier = Modifier.height(8.dp))
                history.recommendations.forEach { recommendation ->
                    ReadOnlyRecommendationItem(
                        recommendation = recommendation,
                        isSelected = recommendation.recommendationId == history.selectedRecommendationId,
                        primaryDark = primaryDark
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }
    }
}

@Composable
private fun HistorySnapshotRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(
            text = label,
            color = Color(0xFF708080),
            fontSize = 14.sp,
            modifier = Modifier.width(104.dp)
        )
        Text(
            text = value,
            color = Color(0xFF263A3B),
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.weight(1f)
        )
    }
}

@Composable
private fun ReadOnlyRecommendationItem(
    recommendation: HistoryRecommendation,
    isSelected: Boolean,
    primaryDark: Color
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(
                if (isSelected) Color(0xFFDDEEEB) else Color(0xFFF5F8F8),
                RoundedCornerShape(14.dp)
            )
            .border(
                1.dp,
                if (isSelected) primaryDark.copy(alpha = 0.35f) else Color.Transparent,
                RoundedCornerShape(14.dp)
            )
            .padding(13.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = recommendation.doctor,
                color = Color(0xFF263A3B),
                fontWeight = FontWeight.Bold,
                fontSize = 16.sp
            )
            if (isSelected) {
                Text("當時選擇", color = primaryDark, fontSize = 13.sp, fontWeight = FontWeight.Bold)
            }
        }
        Spacer(modifier = Modifier.height(4.dp))
        Text(recommendation.department, color = Color(0xFF607575), fontSize = 14.sp)
        Text(recommendation.displayHistoryTime(), color = Color(0xFF607575), fontSize = 14.sp)
    }
}

private fun HistoryRecommendation.displayHistoryTime(): String =
    listOfNotNull(
        date.takeIf { it.isNotBlank() },
        (sessionTime ?: session).takeIf { it.isNotBlank() },
        room?.takeIf { it.isNotBlank() }
    ).joinToString(" ").ifBlank { "未保存時段" }

@Composable
private fun UrgentWarningCard(warning: String) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color(0xFFFFF3E0), RoundedCornerShape(18.dp))
            .border(1.dp, Color(0xFFE6A23C), RoundedCornerShape(18.dp))
            .padding(16.dp)
    ) {
        Text("急迫症狀提醒", color = Color(0xFFB45F06), fontSize = 18.sp, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(6.dp))
        Text(warning, color = Color(0xFF704214), fontSize = 15.sp, lineHeight = 22.sp)
    }
}

@Composable
fun AiThinkingBubble(primaryDark: Color) {
    val transition = rememberInfiniteTransition(label = "aiThinkingDots")
    val dotOffsets = List(3) { index ->
        transition.animateFloat(
            initialValue = 0f,
            targetValue = -6f,
            animationSpec = infiniteRepeatable(
                animation = tween(durationMillis = 420, delayMillis = index * 130),
                repeatMode = RepeatMode.Reverse
            ),
            label = "aiThinkingDot$index"
        )
    }

    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        modifier = Modifier
            .shadow(
                elevation = 1.dp,
                shape = RoundedCornerShape(
                    topStart = 24.dp,
                    topEnd = 24.dp,
                    bottomStart = 4.dp,
                    bottomEnd = 24.dp
                )
            )
            .background(
                Color.White,
                shape = RoundedCornerShape(
                    topStart = 24.dp,
                    topEnd = 24.dp,
                    bottomStart = 4.dp,
                    bottomEnd = 24.dp
                )
            )
            .padding(horizontal = 18.dp, vertical = 14.dp)
    ) {

        dotOffsets.forEach { offset ->
            Text(
                text = "•",
                color = primaryDark,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.offset(y = offset.value.dp)
            )
        }
        /*Text(
            text = "正在分析中請稍後",
            color = primaryDark,
            fontSize = 16.sp,
            fontWeight = FontWeight.Medium
        )*/
    }
}

@Composable
fun VoiceWaveform(
    primaryDark: Color,
    modifier: Modifier = Modifier.fillMaxWidth()
) {
    val transition = rememberInfiniteTransition(label = "voiceWaveform")
    val barHeights = List(7) { index ->
        transition.animateFloat(
            initialValue = 10f,
            targetValue = listOf(28f, 18f, 34f, 22f, 38f, 20f, 30f)[index],
            animationSpec = infiniteRepeatable(
                animation = tween(durationMillis = 460 + index * 35, delayMillis = index * 70),
                repeatMode = RepeatMode.Reverse
            ),
            label = "voiceWaveBar$index"
        )
    }

    Row(
        modifier = modifier
            .height(44.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically
    ) {
        barHeights.forEach { height ->
            Box(
                modifier = Modifier
                    .padding(horizontal = 3.dp)
                    .width(5.dp)
                    .height(height.value.dp)
                    .background(primaryDark.copy(alpha = 0.82f), RoundedCornerShape(5.dp))
            )
        }
    }
}

@Composable
fun VoiceTranscribingIndicator(primaryDark: Color) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .height(44.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically
    ) {
        VoiceWaveform(
            primaryDark = primaryDark,
            modifier = Modifier.width(88.dp)
        )
        Spacer(modifier = Modifier.width(12.dp))
        Text(
            text = "正在轉錄",
            color = primaryDark,
            fontSize = 18.sp,
            fontWeight = FontWeight.SemiBold
        )
    }
}

@Composable
fun RealBubbleItem(
    messageContent: String,
    isUser: Boolean,
    primaryDark: Color,
    isSpeaking: Boolean = false,
    isPlaybackLocked: Boolean = false,
    onSpeakClicked: () -> Unit = {}
) {
    val warningOrange = Color(0xFFE6A23C)
    val inactiveIcon = Color(0xFF8FA3A6)
    val bubbleShape = if (isUser) {
        RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp, bottomStart = 24.dp, bottomEnd = 4.dp)
    } else {
        RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp, bottomStart = 4.dp, bottomEnd = 24.dp)
    }

    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = if (isUser) Alignment.End else Alignment.Start
    ) {
        if (isUser) {
            Box(
                modifier = Modifier
                    .widthIn(max = 280.dp)
                    .shadow(elevation = 1.dp, shape = bubbleShape)
                    .background(primaryDark, shape = bubbleShape)
                    .padding(horizontal = 16.dp, vertical = 12.dp)
            ) {
                Text(
                    text = messageContent,
                    color = Color.White,
                    fontSize = 16.sp,
                    lineHeight = 24.sp
                )
            }
            return@Column
        }

        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier
                .padding(start = 4.dp, bottom = 6.dp)
                .background(Color(0xFFFDF6EC), shape = RoundedCornerShape(6.dp))
                .padding(horizontal = 8.dp, vertical = 4.dp)
        ) {
            Box(
                modifier = Modifier
                    .size(6.dp)
                    .background(warningOrange, shape = CircleShape)
            )
            Spacer(modifier = Modifier.width(6.dp))
            Text(
                text = "本資訊僅供參考！若有緊急症狀請立即就醫",
                color = warningOrange,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold
            )
        }

        Box(
            modifier = Modifier
                .widthIn(max = 280.dp)
                .shadow(elevation = 1.dp, shape = bubbleShape)
                .background(Color.White, shape = bubbleShape)
                .clip(bubbleShape)
        ) {
            Column(modifier = Modifier.fillMaxWidth()) {
                Text(  // AI 回復文字
                    text = messageContent.withBoldDepartment(),
                    color = primaryDark, //Color(0xFF556666)
                    fontSize = 16.sp,
                    lineHeight = 24.sp,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp)
                )

                Box(
                    modifier = Modifier
                        .padding(horizontal = 16.dp)
                        .fillMaxWidth()
                        .height(1.dp)
                        .background(Color(0xFFEEEEEE))
                )

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Row(
                        modifier = Modifier
                            .alpha(if (isPlaybackLocked && !isSpeaking) 0.45f else 1f)
                            .clickable(enabled = !isPlaybackLocked) { onSpeakClicked() },
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.VolumeUp,
                            contentDescription = "播放語音",
                            tint = if (isSpeaking) primaryDark else inactiveIcon,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "播放",
                            color = if (isSpeaking) primaryDark else inactiveIcon,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Normal
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun LanguageToggle(
    selectedLanguage: String,
    onLanguageSelected: (String) -> Unit,
    primaryDark: Color,
    selectedColor: Color
) {
    val itemWidth = 64.dp
    val indicatorOffset by animateDpAsState(
        targetValue = if (selectedLanguage == "台語") itemWidth else 0.dp,
        animationSpec = tween(durationMillis = 260),
        label = "languageToggleOffset"
    )

    Box(
        modifier = Modifier
            .height(58.dp)
            .width(itemWidth * 2 + 12.dp)
            .shadow(elevation = 4.dp, shape = RoundedCornerShape(30.dp))
            .background(Color.White, shape = RoundedCornerShape(30.dp))
            .padding(6.dp)
    ) {
        Box(
            modifier = Modifier
                .offset(x = indicatorOffset)
                .height(46.dp)
                .width(itemWidth)
                .background(selectedColor, RoundedCornerShape(24.dp))
        )

        Row(verticalAlignment = Alignment.CenterVertically) {
            LanguageToggleItem(
                text = "國語",
                primaryDark = primaryDark,
                onClick = { onLanguageSelected("國語") }
            )
            LanguageToggleItem(
                text = "台語",
                primaryDark = primaryDark,
                onClick = { onLanguageSelected("台語") }
            )
        }
    }
}

@Composable
fun LanguageToggleItem(
    text: String,
    primaryDark: Color,
    onClick: () -> Unit
) {
    Box(
        modifier = Modifier
            .height(46.dp)
            .width(64.dp)
            .clickable { onClick() },
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = text,
            color = primaryDark,
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

private fun String.withBoldDepartment() = buildAnnotatedString {
    val target = "檢傷結果"
    val startIndex = indexOf(target)
    if (startIndex == -1) {
        append(this@withBoldDepartment)
        return@buildAnnotatedString
    }

    append(substring(0, startIndex))
    withStyle(SpanStyle(fontWeight = FontWeight.ExtraBold)) {
        append(target)
    }
    append(substring(startIndex + target.length))
}

@Composable
fun ChatActionButton(
    text: String,
    containerColor: Color,
    contentColor: Color,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    isLoading: Boolean = false,
    onClick: () -> Unit
) {
    Box(
        modifier = modifier
            .height(54.dp)
            .shadow(elevation = 1.dp, shape = RoundedCornerShape(18.dp))
            .background(containerColor, shape = RoundedCornerShape(18.dp))
            .clickable(enabled = enabled && !isLoading) { onClick() },
        contentAlignment = Alignment.Center
    ) {
        if (isLoading) {
            CircularProgressIndicator(
                color = contentColor,
                modifier = Modifier.size(22.dp),
                strokeWidth = 2.5.dp
            )
        } else {
            Text(
                text = text,
                color = contentColor,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}
