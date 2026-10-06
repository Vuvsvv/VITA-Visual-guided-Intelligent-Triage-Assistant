package com.example.medicalaiguidance.screen

import android.content.Context
import android.content.Intent
import android.media.MediaPlayer
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.widget.Toast
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.OpenInNew
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Popup
import androidx.compose.ui.window.PopupProperties
import androidx.navigation.NavHostController
import com.example.medicalaiguidance.R
import kotlinx.coroutines.launch
import java.util.Locale

private data class GuideStep(
    val stepTag: String,
    val title: String,
    val description: String,
    val accent: Color,
    val screenshotResList: List<Int>,
    val singleHighlight: String,
    val isPermissionStep: Boolean = false,
    val taigiAudioResourceName: String,
    val taiwaneseDescription: String? = null,
    val taiwaneseHighlight: String? = null
)

private enum class VoiceLanguage(val label: String) {
    MANDARIN("國語朗讀"),
    TAIWANESE("台語朗讀")
}

private enum class NarrationSource {
    MANDARIN_TTS,
    TAIGI_AUDIO
}

@Composable
fun GettingStartedScreen(navController: NavHostController) {
    val primary = Color(0xFF036A6D)
    val context = LocalContext.current
    val mainHandler = remember { Handler(Looper.getMainLooper()) }
    val coroutineScope = rememberCoroutineScope()
    var selectedLanguage by remember { mutableStateOf(VoiceLanguage.MANDARIN) }
    var isPlaying by remember { mutableStateOf(false) }
    var ttsReady by remember { mutableStateOf(false) }
    var tts by remember { mutableStateOf<TextToSpeech?>(null) }
    var taigiPlayer by remember { mutableStateOf<MediaPlayer?>(null) }
    var narrationSource by remember { mutableStateOf<NarrationSource?>(null) }

    DisposableEffect(context) {
        var engine: TextToSpeech? = null
        engine = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = engine?.setLanguage(Locale.TAIWAN)
                if (result == TextToSpeech.LANG_MISSING_DATA ||
                    result == TextToSpeech.LANG_NOT_SUPPORTED
                ) {
                    engine?.language = Locale.CHINESE
                }
                mainHandler.post { ttsReady = true }
            } else {
                mainHandler.post { ttsReady = false }
            }
        }
        engine?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                mainHandler.post {
                    if (narrationSource == NarrationSource.MANDARIN_TTS) isPlaying = true
                }
            }

            override fun onDone(utteranceId: String?) {
                mainHandler.post {
                    if (narrationSource == NarrationSource.MANDARIN_TTS) {
                        narrationSource = null
                        isPlaying = false
                    }
                }
            }

            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) {
                mainHandler.post {
                    if (narrationSource == NarrationSource.MANDARIN_TTS) {
                        narrationSource = null
                        isPlaying = false
                    }
                }
            }
        })
        tts = engine

        onDispose {
            engine?.stop()
            engine?.shutdown()
            taigiPlayer?.release()
            mainHandler.removeCallbacksAndMessages(null)
            tts = null
        }
    }

    val steps = remember {
        listOf(
            GuideStep(
                stepTag = "STEP 01",
                title = "訴說症狀\n我們幫您分析",
                description = "國語、台語都能講，像聊天一樣輕鬆釐清不適。",
                accent = Color(0xFF3B7B78),
                screenshotResList = listOf(R.drawable.preview_step_chat),
                singleHighlight = "支援台語／國語雙語音問診",
                taigiAudioResourceName = "getting_started_symptom_intro_taigi",
                taiwaneseDescription = "講國語、講台語攏會通，親像開講同款輕鬆釐清無爽快。",
                taiwaneseHighlight = "有支援台語佮國語雙語音問診"
            ),
            GuideStep(
                stepTag = "STEP 02",
                title = "科別與醫師\n幫您找出適合選擇",
                description = "不知道看哪一科？自動為您比對專長與合適門診。",
                accent = Color(0xFF356E70),
                screenshotResList = listOf(R.drawable.preview_step_doctor),
                singleHighlight = "依症狀自動排序推薦專科醫師",
                taigiAudioResourceName = "getting_started_recommendation_intro_taigi",
                taiwaneseDescription = "毋知愛看佗一科？系統會自動共你比對專長佮適合的門診。",
                taiwaneseHighlight = "照症頭自動推薦專科醫師"
            ),
            GuideStep(
                stepTag = "STEP 03",
                title = "啟用導引\n享有紅框提示",
                description = "開啟無障礙服務權限，系統才能為您標示位置。",
                accent = Color(0xFF1E5B5D),
                screenshotResList = listOf(
                    R.drawable.preview_step_permission_1,
                    R.drawable.preview_step_permission_2,
                    R.drawable.preview_step_permission_3
                ),
                singleHighlight = "前往設定開啟導引權限",
                isPermissionStep = true,
                taigiAudioResourceName = "getting_started_accessibility_intro_taigi",
                taiwaneseDescription = "請開啟無障礙服務權限，系統才會幫你報路標位置。",
                taiwaneseHighlight = "去設定開啟權限"
            ),
            GuideStep(
                stepTag = "STEP 04",
                title = "掛號動線\n一步步指引您",
                description = "畫面提供清晰紅框提示，照著引導點擊不按錯。",
                accent = Color(0xFF2A5F62),
                screenshotResList = listOf(R.drawable.preview_step_summary),
                singleHighlight = "榮總 App 視覺紅框操作導引",
                taigiAudioResourceName = "getting_started_booking_guide_intro_taigi",
                taiwaneseDescription = "螢幕會有清楚的紅框提示，綴住點就袂按毋著。",
                taiwaneseHighlight = "榮總 App 視覺紅框報路導引"
            ),
            GuideStep(
                stepTag = "STEP 05",
                title = "問診歷程\n隨時回溯查閱",
                description = "完整保留評估摘要與掛號建議，回診就醫好安心。",
                accent = Color(0xFF245053),
                screenshotResList = listOf(R.drawable.preview_step_history),
                singleHighlight = "完整保留摘要，隨時查閱",
                taigiAudioResourceName = "getting_started_history_intro_taigi",
                taiwaneseDescription = "完整保留問診摘要佮掛號建議，回診看病真安心。",
                taiwaneseHighlight = "完整保留摘要，隨時會使查看"
            )
        )
    }

    val pagerState = rememberPagerState(initialPage = 0) { steps.size }
    val currentStep = steps[pagerState.currentPage]

    LaunchedEffect(pagerState.currentPage) {
        tts?.stop()
        taigiPlayer?.release()
        taigiPlayer = null
        narrationSource = null
        isPlaying = false
    }

    fun stopSpeech() {
        tts?.stop()
        taigiPlayer?.release()
        taigiPlayer = null
        narrationSource = null
        isPlaying = false
    }

    fun startSpeech(step: GuideStep, language: VoiceLanguage) {
        stopSpeech()
        if (language == VoiceLanguage.TAIWANESE) {
            val audioResourceId = context.resources.getIdentifier(
                step.taigiAudioResourceName,
                "raw",
                context.packageName
            )
            if (audioResourceId == 0) {
                Toast.makeText(context, "本頁台語語音檔尚未提供", Toast.LENGTH_SHORT).show()
                return
            }
            val player = MediaPlayer.create(context, audioResourceId)
            if (player == null) {
                Toast.makeText(context, "本頁台語語音無法播放", Toast.LENGTH_SHORT).show()
                return
            }
            taigiPlayer = player
            narrationSource = NarrationSource.TAIGI_AUDIO
            player.setOnCompletionListener {
                if (taigiPlayer === player) {
                    player.release()
                    taigiPlayer = null
                    narrationSource = null
                    isPlaying = false
                }
            }
            player.setOnErrorListener { _, _, _ ->
                if (taigiPlayer === player) {
                    player.release()
                    taigiPlayer = null
                    narrationSource = null
                    isPlaying = false
                }
                true
            }
            player.start()
            isPlaying = true
            return
        }

        val engine = tts
        if (!ttsReady || engine == null) {
            Toast.makeText(context, "語音引擎載入中，請稍候", Toast.LENGTH_SHORT).show()
            return
        }

        val languageResult = engine.setLanguage(Locale.TAIWAN)
        if (
            languageResult == TextToSpeech.LANG_MISSING_DATA ||
            languageResult == TextToSpeech.LANG_NOT_SUPPORTED
        ) {
            val fallbackResult = engine.setLanguage(Locale.CHINESE)
            if (
                fallbackResult == TextToSpeech.LANG_MISSING_DATA ||
                fallbackResult == TextToSpeech.LANG_NOT_SUPPORTED
            ) {
                Toast.makeText(context, "裝置未提供國語語音", Toast.LENGTH_SHORT).show()
                return
            }
        }
        val textToRead = listOf(
            step.title.replace("\n", "，"),
            step.description,
            step.singleHighlight
        ).joinToString("。") { it.trim().trimEnd('。') } + "。"
        val utteranceId = "GUIDE_${step.stepTag}_${System.currentTimeMillis()}"
        narrationSource = NarrationSource.MANDARIN_TTS
        val result = engine.speak(textToRead, TextToSpeech.QUEUE_FLUSH, null, utteranceId)
        if (result == TextToSpeech.SUCCESS) {
            isPlaying = true
        } else {
            narrationSource = null
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            drawRect(
                brush = Brush.linearGradient(
                    colors = listOf(Color(0xFFE8F2F0), Color(0xFFFBFCFC), Color(0xFFE5F0EE)),
                    start = Offset.Zero,
                    end = Offset(size.width, size.height)
                )
            )
            drawCircle(
                color = Color(0xFFC2D9D5).copy(alpha = .45f),
                radius = size.width * .62f,
                center = Offset(size.width * .92f, size.height * .82f)
            )
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(horizontal = 20.dp, vertical = 10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .size(42.dp)
                        .shadow(3.dp, RoundedCornerShape(14.dp))
                        .background(Color.White, RoundedCornerShape(14.dp))
                        .clickable {
                            stopSpeech()
                            navController.popBackStack()
                        },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "返回首頁",
                        tint = primary,
                        modifier = Modifier.size(22.dp)
                    )
                }

                ExpandingPillIndicator(
                    pageCount = steps.size,
                    currentPage = pagerState.currentPage,
                    activeColor = primary,
                    inactiveColor = Color(0xFFD2E1DE)
                )

                VoiceDropdownButton(
                    isPlaying = isPlaying,
                    selectedLanguage = selectedLanguage,
                    accent = currentStep.accent,
                    onStopPlay = ::stopSpeech,
                    onLanguageSelect = { language ->
                        selectedLanguage = language
                        startSpeech(currentStep, language)
                    }
                )
            }

            Spacer(Modifier.height(14.dp))

            HorizontalPager(
                state = pagerState,
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) { pageIndex ->
                val step = steps[pageIndex]
                Column(
                    modifier = Modifier.fillMaxSize(),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = step.title,
                        color = Color(0xFF162B2B),
                        fontSize = 24.sp,
                        lineHeight = 31.sp,
                        fontWeight = FontWeight.ExtraBold,
                        textAlign = TextAlign.Center
                    )
                    Spacer(Modifier.height(6.dp))
                    Text(
                        text = step.description,
                        color = Color(0xFF4D6B6B),
                        fontSize = 14.sp,
                        lineHeight = 20.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(horizontal = 8.dp)
                    )
                    Spacer(Modifier.height(8.dp))

                    GuideHighlight(
                        text = step.singleHighlight,
                        accent = step.accent,
                        clickable = step.isPermissionStep,
                        onClick = {
                            stopSpeech()
                            openAccessibilitySettings(context)
                        }
                    )
                    Spacer(Modifier.height(10.dp))

                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .fillMaxWidth(),
                        contentAlignment = Alignment.Center
                    ) {
                        MinimalDeviceShell(screenshotResList = step.screenshotResList)
                    }
                }
            }

            Spacer(Modifier.height(10.dp))
            val isLastPage = pagerState.currentPage == steps.lastIndex
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(50.dp)
                    .shadow(
                        elevation = 6.dp,
                        shape = RoundedCornerShape(25.dp),
                        ambientColor = primary.copy(alpha = .18f),
                        spotColor = primary.copy(alpha = .28f)
                    )
                    .background(
                        Brush.horizontalGradient(listOf(Color(0xFF4C8787), primary)),
                        RoundedCornerShape(25.dp)
                    )
                    .clickable {
                        stopSpeech()
                        if (isLastPage) {
                            navController.popBackStack()
                        } else {
                            coroutineScope.launch {
                                pagerState.animateScrollToPage(pagerState.currentPage + 1)
                            }
                        }
                    },
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = if (isLastPage) "開始使用" else "下一步",
                    color = Color.White,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
private fun GuideHighlight(
    text: String,
    accent: Color,
    clickable: Boolean,
    onClick: () -> Unit
) {
    val modifier = Modifier
        .background(accent.copy(alpha = if (clickable) .12f else .08f), RoundedCornerShape(18.dp))
        .then(if (clickable) Modifier.clickable(onClick = onClick) else Modifier)
        .padding(horizontal = 12.dp, vertical = 5.dp)

    Row(modifier = modifier, verticalAlignment = Alignment.CenterVertically) {
        Icon(
            imageVector = if (clickable) Icons.AutoMirrored.Filled.OpenInNew else Icons.Default.CheckCircle,
            contentDescription = null,
            tint = accent,
            modifier = Modifier.size(14.dp)
        )
        Spacer(Modifier.size(5.dp))
        Text(
            text = text,
            color = accent,
            fontSize = 12.5.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
private fun VoiceDropdownButton(
    isPlaying: Boolean,
    selectedLanguage: VoiceLanguage,
    accent: Color,
    onStopPlay: () -> Unit,
    onLanguageSelect: (VoiceLanguage) -> Unit
) {
    var menuOpen by remember { mutableStateOf(false) }
    val transition = rememberInfiniteTransition(label = "audio playing")
    val waveScale by transition.animateFloat(
        initialValue = .95f,
        targetValue = 1.08f,
        animationSpec = infiniteRepeatable(tween(600), RepeatMode.Reverse),
        label = "audio scale"
    )

    Box {
        Box(
            modifier = Modifier
                .size(42.dp)
                .shadow(3.dp, RoundedCornerShape(14.dp))
                .background(
                    if (isPlaying) accent.copy(alpha = .15f) else Color.White,
                    RoundedCornerShape(14.dp)
                )
                .border(
                    width = 1.2.dp,
                    color = if (isPlaying) accent else Color(0xFFD6E4E2),
                    shape = RoundedCornerShape(14.dp)
                )
                .clickable {
                    if (isPlaying) onStopPlay()
                    menuOpen = !menuOpen
                }
                .graphicsLayer {
                    if (isPlaying) {
                        scaleX = waveScale
                        scaleY = waveScale
                    }
                },
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = if (isPlaying) Icons.Default.Stop else Icons.AutoMirrored.Filled.VolumeUp,
                contentDescription = if (isPlaying) "停止朗讀" else "語音朗讀選項",
                tint = accent,
                modifier = Modifier.size(22.dp)
            )
        }

        if (menuOpen) {
            Popup(
                alignment = Alignment.TopEnd,
                offset = IntOffset(x = 0, y = 120),
                onDismissRequest = { menuOpen = false },
                properties = PopupProperties(focusable = true)
            ) {
                Column(
                    modifier = Modifier
                        .width(140.dp)
                        .shadow(8.dp, RoundedCornerShape(14.dp))
                        .background(Color.White, RoundedCornerShape(14.dp))
                        .border(1.dp, Color(0xFFD6E4E2), RoundedCornerShape(14.dp))
                        .padding(vertical = 4.dp)
                ) {
                    VoiceLanguage.entries.forEach { language ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    menuOpen = false
                                    onLanguageSelect(language)
                                }
                                .padding(horizontal = 14.dp, vertical = 10.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = language.label,
                                fontSize = 14.sp,
                                fontWeight = if (selectedLanguage == language) FontWeight.Bold else FontWeight.Medium,
                                color = if (selectedLanguage == language) accent else Color(0xFF2C3E3E)
                            )
                            if (selectedLanguage == language) {
                                Icon(
                                    imageVector = Icons.Default.Check,
                                    contentDescription = null,
                                    tint = accent,
                                    modifier = Modifier.size(16.dp)
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ExpandingPillIndicator(
    pageCount: Int,
    currentPage: Int,
    activeColor: Color,
    inactiveColor: Color
) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(5.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        repeat(pageCount) { index ->
            val selected = index == currentPage
            val width by animateDpAsState(
                targetValue = if (selected) 22.dp else 6.dp,
                animationSpec = tween(280),
                label = "indicator width"
            )
            val color by animateColorAsState(
                targetValue = if (selected) activeColor else inactiveColor,
                animationSpec = tween(280),
                label = "indicator color"
            )
            Box(
                modifier = Modifier
                    .size(width = width, height = 6.dp)
                    .clip(CircleShape)
                    .background(color)
            )
        }
    }
}

@Composable
private fun MinimalDeviceShell(screenshotResList: List<Int>) {
    val screenshotPagerState = rememberPagerState(initialPage = 0) { screenshotResList.size }
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Box(
            modifier = Modifier
                .fillMaxHeight(.96f)
                .aspectRatio(9f / 19.5f)
                .shadow(
                    elevation = 6.dp,
                    shape = RoundedCornerShape(20.dp),
                    spotColor = Color(0x1A000000),
                    ambientColor = Color(0x0A000000)
                )
                .background(Color.White, RoundedCornerShape(20.dp))
                .border(1.2.dp, Color(0xFFDCE8E6), RoundedCornerShape(20.dp))
                .clip(RoundedCornerShape(20.dp))
        ) {
            HorizontalPager(
                state = screenshotPagerState,
                modifier = Modifier.fillMaxSize()
            ) { screenshotIndex ->
                Image(
                    painter = painterResource(screenshotResList[screenshotIndex]),
                    contentDescription = "步驟畫面截圖 第 ${screenshotIndex + 1} 張",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize()
                )
            }
            Box(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 6.dp)
                    .size(7.5.dp)
                    .clip(CircleShape)
                    .background(Color(0xFF262626).copy(alpha = .85f))
            )
            if (screenshotResList.size > 1) {
                Row(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(bottom = 10.dp)
                        .background(Color.Black.copy(alpha = .35f), RoundedCornerShape(10.dp))
                        .padding(horizontal = 6.dp, vertical = 3.dp),
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    repeat(screenshotResList.size) { index ->
                        val selected = index == screenshotPagerState.currentPage
                        Box(
                            modifier = Modifier
                                .size(if (selected) 5.5.dp else 4.dp)
                                .clip(CircleShape)
                                .background(if (selected) Color.White else Color.White.copy(alpha = .5f))
                        )
                    }
                }
            }
        }
    }
}

private fun openAccessibilitySettings(context: Context) {
    try {
        context.startActivity(
            Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
        )
    } catch (_: Exception) {
        context.startActivity(
            Intent(Settings.ACTION_SETTINGS).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
        )
    }
}
