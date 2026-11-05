package com.example.medicalaiguidance.screen

import com.example.medicalaiguidance.network.MedicalApiClient
import com.example.medicalaiguidance.network.TtsRequest
import com.example.medicalaiguidance.util.AudioPlayer
import com.example.medicalaiguidance.network.VoiceTtsResponseDto
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import android.content.Context
import android.graphics.BitmapFactory
import androidx.compose.foundation.BorderStroke
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
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.StarHalf
import androidx.compose.material.icons.filled.CalendarToday
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.MedicalServices
import androidx.compose.material.icons.automirrored.filled.VolumeUp
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Schedule
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.School
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.filled.StarOutline
import androidx.compose.material.icons.filled.ThumbUp
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.navigation.NavController
import com.example.medicalaiguidance.model.Doctor
import com.example.medicalaiguidance.model.DoctorProfile
import com.example.medicalaiguidance.model.VisitPlan
import com.example.medicalaiguidance.navigation.Route
import com.example.medicalaiguidance.network.FallbackDepartmentDto
import com.example.medicalaiguidance.network.RecommendationItemDto
import com.example.medicalaiguidance.viewmodel.DoctorUiState
import com.example.medicalaiguidance.viewmodel.DoctorViewModel
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale
import kotlin.math.roundToInt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/**
 * Stateful 元件：負責與 ViewModel 和 NavController 溝通，處理資料與導覽邏輯。
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DoctorSelectionScreen(
    navController: NavController,
    visitPlan: VisitPlan,
    viewModel: DoctorViewModel = viewModel()
) {
    val uiState by viewModel.uiState.collectAsState()
    val showSheet by viewModel.showBottomSheet.collectAsState()
    val selectedRecommendation by viewModel.selectedRecommendation.collectAsState()
    val selectingRecommendationId by viewModel.selectingRecommendationId.collectAsState()
    val sheetState = rememberModalBottomSheetState()
    val context = LocalContext.current
    val speechScope = rememberCoroutineScope()
    val speech = remember(context, speechScope) { DoctorCardSpeech.create(context, speechScope) }
    val speechLanguage by com.example.medicalaiguidance.repository.VoiceLanguagePreference
        .observe(context).collectAsState()
    LaunchedEffect(speechLanguage) { speech.stop() }
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(speech, lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_STOP) speech.stop()
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            speech.close()
        }
    }

    val doctorProfiles by produceState<List<DoctorProfile>>(emptyList(), context) {
        value = withContext(Dispatchers.IO) {
            loadDoctorProfilesFromAssets(context)
        }
    }

    LaunchedEffect(visitPlan) {
        viewModel.fetchRecommendations(visitPlan.apiValue)
    }

    // BottomSheet 邏輯保留在此，因為它需要存取 ViewModel 的行為
    if (showSheet && selectedRecommendation != null) {
        val item = selectedRecommendation!!
        val profile = findDoctorProfile(doctorProfiles, item)
        ModalBottomSheet(
            onDismissRequest = { viewModel.dismissBottomSheet() },
            sheetState = sheetState,
            containerColor = Color.White,
            shape = RoundedCornerShape(topStart = 36.dp, topEnd = 36.dp),
            sheetMaxWidth = Dp.Unspecified,
            dragHandle = null
        ) {
            DoctorSpecialtyContent(
                doctor = item.toDoctor(profile),
                profile = profile,
                recommendation = item,
                onClose = { viewModel.dismissBottomSheet() }
            )
        }
    }

    // 將資料傳遞給無狀態的 UI 元件
    DoctorSelectionScreenContent(
        uiState = uiState,
        doctorProfiles = doctorProfiles,
        selectingRecommendationId = selectingRecommendationId,
        onBack = { navController.popBackStack() },
        onRetry = { viewModel.fetchRecommendations(visitPlan.apiValue) },
        onCardClick = { recommendation ->
            speech.stop()
            viewModel.selectRecommendationAndNavigate(recommendation) {
                navController.navigate(Route.CONFIRM_NEED)
            }
        },
        onSpecialtyClick = { recommendation ->
            speech.stop()
            viewModel.onRecommendationDetailClick(recommendation)
        },
        speechLanguage = speechLanguage,
        speakingRecommendationId = speech.activeId,
        speechError = speech.error,
        onSpeak = { item ->
            speech.toggle(item.recommendationId, item.spokenIntroduction(), speechLanguage)
        }
    )
}

/**
 * Stateless 元件：純 UI 畫面，不依賴 ViewModel，方便進行 @Preview 預覽。
 */
@Composable
fun DoctorSelectionScreenContent(
    uiState: DoctorUiState,
    doctorProfiles: List<DoctorProfile>,
    selectingRecommendationId: String?,
    onBack: () -> Unit,
    onRetry: () -> Unit,
    onCardClick: (RecommendationItemDto) -> Unit,
    onSpecialtyClick: (RecommendationItemDto) -> Unit,
    speechLanguage: String = "國語",
    speakingRecommendationId: String? = null,
    speechError: String? = null,
    onSpeak: (RecommendationItemDto) -> Unit = {}
) {
    val primaryDark = Color(0xFF376F72)
    val lightBg = Color(0xFFF2FAF8)

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(lightBg)
    ) {
        HeaderBar(
            textDarkColor = primaryDark,
            onBack = onBack,
            speechLanguage = speechLanguage
        )
        speechError?.let {
            Text(it, color = Color(0xFFB00020), fontSize = 13.sp,
                modifier = Modifier.padding(horizontal = 24.dp, vertical = 4.dp))
        }

        when (val state = uiState) {
            DoctorUiState.Loading -> {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    CircularProgressIndicator(color = primaryDark)
                }
            }

            is DoctorUiState.Error -> {
                ErrorState(message = state.message, primaryDark = primaryDark, onRetry = onRetry)
            }

            is DoctorUiState.NoSlots -> {
                NoSlotsState(message = state.message, primaryDark = primaryDark, onRetry = onRetry)
            }

            is DoctorUiState.Success -> {
                val initialMode = if (state.specialtyFirst.isNotEmpty()) {
                    DoctorRecommendationMode.SPECIALTY_FIRST
                } else {
                    DoctorRecommendationMode.TIME_FIRST
                }
                var mode by remember(state.caseId, state.specialtyFirst.size, state.timeFirst.size) {
                    mutableStateOf(initialMode)
                }

                LaunchedEffect(state.caseId, state.specialtyFirst.size, state.timeFirst.size) {
                    val selectedList = if (mode == DoctorRecommendationMode.SPECIALTY_FIRST) {
                        state.specialtyFirst
                    } else {
                        state.timeFirst
                    }
                    if (selectedList.isEmpty()) {
                        mode = initialMode
                    }
                }

                val baseRecommendations = if (mode == DoctorRecommendationMode.SPECIALTY_FIRST) {
                    state.specialtyFirst
                } else {
                    state.timeFirst
                }
                val hasAnyRecommendations = state.specialtyFirst.isNotEmpty() || state.timeFirst.isNotEmpty()

                // 判斷兩種排序結果是否完全相同
                val isIdenticalSorting = remember(state.specialtyFirst, state.timeFirst) {
                    if (state.specialtyFirst.isEmpty() || state.timeFirst.isEmpty()) {
                        false
                    } else if (state.specialtyFirst.size != state.timeFirst.size) {
                        false
                    } else {
                        // 提取兩邊的 ID 並依序比對，若完全一樣則回傳 true
                        val specialtyIds = state.specialtyFirst.map { it.recommendationId }
                        val timeIds = state.timeFirst.map { it.recommendationId }
                        specialtyIds == timeIds
                    }
                }

                if (!hasAnyRecommendations) {
                    EmptyRecommendationState(state.fallbackDepartments)
                } else {
                    // ======= 氣泡過濾邏輯區塊 =======
                    // 1. 取得所有 AI 推薦的日期，過濾空值、去重複、並依照日期排序
                    val sortedAvailableDates = remember(baseRecommendations) {
                        baseRecommendations
                            .mapNotNull { it.date.takeIf { d -> d.isNotBlank() } }
                            .distinct()
                            .sorted()
                    }

                    // 2. 記錄目前點選了哪一天的氣泡
                    var selectedFilterDate by remember { mutableStateOf<String?>(null) }

                    // 切換看診時間/專長模式時，清除選擇
                    LaunchedEffect(mode) {
                        selectedFilterDate = null
                    }

                    // 3. 過濾出要顯示的卡片
                    val displayedRecommendations = remember(baseRecommendations, selectedFilterDate) {
                        displayedDoctorRecommendations(baseRecommendations, selectedFilterDate)
                    }

                    LazyColumn(
                        modifier = Modifier
                            .fillMaxSize()
                            .navigationBarsPadding(),
                        contentPadding = PaddingValues(start = 24.dp, top = 12.dp, end = 24.dp, bottom = 112.dp),
                        verticalArrangement = Arrangement.spacedBy(20.dp)
                    ) {
                        item {
                            Column(verticalArrangement = Arrangement.spacedBy(15.dp)) {
                                FilterSummaryCard(
                                    primaryDark = primaryDark,
                                    mode = mode,
                                    specialtyEnabled = state.specialtyFirst.isNotEmpty(),
                                    timeEnabled = state.timeFirst.isNotEmpty(),
                                    isIdenticalSorting = isIdenticalSorting,
                                    onModeSelected = { mode = it }
                                )
                                Text(
                                    text = "班表更新可能有誤差，實際名額以醫院當下系統為準。",
                                    color = Color(0xFF738286),
                                    fontSize = 13.sp,
                                    lineHeight = 18.sp,
                                    fontWeight = FontWeight.Medium,
                                    textAlign = TextAlign.Center,
                                    modifier = Modifier.fillMaxWidth().padding(horizontal = 4.dp)
                                )

                                // 顯示新的氣泡篩選器 (兩種排序模式都顯示)
                                if (sortedAvailableDates.isNotEmpty()) {
                                    DateBubbleFilter(
                                        availableDates = sortedAvailableDates,
                                        selectedDate = selectedFilterDate,
                                        onDateSelected = { dateString ->
                                            selectedFilterDate = dateString
                                        },
                                        modifier = Modifier.padding(top = 4.dp)
                                    )
                                }
                            }
                        }

                        // 顯示過濾後的卡片
                        items(displayedRecommendations, key = { it.recommendationId }) { recommendation ->
                            val profile = findDoctorProfile(doctorProfiles, recommendation)
                            RecommendationDoctorCard(
                                recommendation = recommendation,
                                profile = profile,
                                primaryDark = primaryDark,
                                showSpecialtyButton = profile != null ||
                                        recommendation.specialtyTags.isNotEmpty() ||
                                        recommendation.hasRecommendationDetailInfo(),
                                isSelecting = selectingRecommendationId == recommendation.recommendationId,
                                isSpeaking = speakingRecommendationId == recommendation.recommendationId,
                                onSpeak = { onSpeak(recommendation) },
                                onCardClick = { onCardClick(recommendation) },
                                onSpecialtyClick = { onSpecialtyClick(recommendation) }
                            )
                        }
                    }
                }
            }
        }
    }
}

// ================= 輔助元件與函式 =================

private enum class DoctorRecommendationMode {
    SPECIALTY_FIRST,
    TIME_FIRST
}

@Composable
private fun HeaderBar(
    textDarkColor: Color,
    onBack: () -> Unit,
    speechLanguage: String
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp, vertical = 20.dp)
            .statusBarsPadding()
            .height(48.dp),
        contentAlignment = Alignment.Center
    ) {
        Box(
            modifier = Modifier
                .align(Alignment.CenterStart)
                .size(48.dp)
                .shadow(elevation = 4.dp, shape = RoundedCornerShape(16.dp))
                .background(Color.White, shape = RoundedCornerShape(16.dp))
                .clickable(onClick = onBack),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                contentDescription = "返回",
                tint = textDarkColor,
                modifier = Modifier.size(22.dp)
            )
        }

        Text(
            text = "選擇醫師",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            color = textDarkColor,
            modifier = Modifier.align(Alignment.Center)
        )
        Box(modifier = Modifier.align(Alignment.CenterEnd)) {
            Text("語音：$speechLanguage", fontSize = 12.sp, color = textDarkColor)
        }
    }
}

@Composable
private fun FilterSummaryCard(
    primaryDark: Color,
    mode: DoctorRecommendationMode,
    specialtyEnabled: Boolean = true,
    timeEnabled: Boolean = true,
    isIdenticalSorting: Boolean = false,
    onModeSelected: (DoctorRecommendationMode) -> Unit = {}
) {
    val subtitle = if (isIdenticalSorting) {
        "目前兩種排序結果相同"
    } else if (mode == DoctorRecommendationMode.SPECIALTY_FIRST) {
        "優先比對症狀與醫師專長"
    } else {
        "優先符合可看診時間"
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .wrapContentHeight(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFFDCEFEA)),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            Icon(
                imageVector = if (mode == DoctorRecommendationMode.SPECIALTY_FIRST) Icons.Filled.MedicalServices else Icons.Default.Schedule,
                contentDescription = null,
                tint = Color(0xFFABCBC4).copy(alpha = 0.5f),
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .offset(x = 12.dp, y = 12.dp)
                    .size(100.dp)
            )
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(14.dp)
            ) {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    color = Color.White.copy(alpha = 0.7f)
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(4.dp),
                        horizontalArrangement = Arrangement.spacedBy(4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        FilterToggleButton(
                            text = "醫師專長",
                            selected = mode == DoctorRecommendationMode.SPECIALTY_FIRST,
                            enabled = specialtyEnabled,
                            primaryDark = primaryDark,
                            modifier = Modifier.weight(1f),
                            onClick = { onModeSelected(DoctorRecommendationMode.SPECIALTY_FIRST) }
                        )
                        FilterToggleButton(
                            text = "看診時間",
                            selected = mode == DoctorRecommendationMode.TIME_FIRST,
                            enabled = timeEnabled,
                            primaryDark = primaryDark,
                            modifier = Modifier.weight(1f),
                            onClick = { onModeSelected(DoctorRecommendationMode.TIME_FIRST) }
                        )
                    }
                }
                Spacer(modifier = Modifier.height(10.dp))
                Column(
                    modifier = Modifier.padding(start = 10.dp)
                ) {
                    Text(
                        text = "已為您排序",
                        fontSize = 15.sp,
                        color = Color(0xFF4C5A5A),
                        fontWeight = FontWeight.Medium
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = subtitle,
                        fontSize = 20.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = Color(0xFF1F3030)
                    )
                }
            }
        }
    }
}

@Composable
private fun FilterToggleButton(
    text: String,
    selected: Boolean,
    enabled: Boolean,
    primaryDark: Color,
    modifier: Modifier = Modifier,
    onClick: () -> Unit
) {
    val textColor = when {
        selected -> Color.White
        enabled -> Color(0xFF4D585A)
        else -> Color(0xFF9DA8AA)
    }
    val backgroundColor = if (selected) primaryDark else Color.Transparent

    Surface(
        modifier = modifier
            .height(38.dp)
            .clickable(enabled = enabled && !selected, onClick = onClick),
        shape = RoundedCornerShape(19.dp),
        color = backgroundColor
    ) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(
                text = text,
                color = textColor,
                fontSize = 16.sp,
                fontWeight = FontWeight.Bold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}

/**
 * 氣泡樣式的日期篩選器 (LazyRow 水平滑動)
 */
@Composable
fun DateBubbleFilter(
    availableDates: List<String>,
    selectedDate: String?,
    onDateSelected: (String?) -> Unit,
    modifier: Modifier = Modifier
) {
    val today = remember { Calendar.getInstance(Locale.TAIWAN) }

    LazyRow(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        contentPadding = PaddingValues(horizontal = 4.dp, vertical = 4.dp)
    ) {
        items(availableDates) { dateStr ->
            val isSelected = (dateStr == selectedDate)
            val labelText = formatBubbleDate(dateStr, today)

            Surface(
                shape = RoundedCornerShape(50),
                color = if (isSelected) Color(0xFF376F72) else Color(0xFFE8F4F1),
                border = if (!isSelected) BorderStroke(1.dp, Color(0xFFBCE3DC)) else null,
                modifier = Modifier
                    .clip(RoundedCornerShape(50))
                    .clickable {
                        onDateSelected(if (isSelected) null else dateStr)
                    }
            ) {
                Text(
                    text = labelText,
                    color = if (isSelected) Color.White else Color(0xFF376F72),
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                    fontWeight = FontWeight.Bold,
                    fontSize = 14.sp
                )
            }
        }
    }
}

private fun formatBubbleDate(dateString: String, today: Calendar): String {
    val cal = parseVisitDate(dateString) ?: return dateString
    val relative = relativeDayText(cal, today)

    val month = cal.get(Calendar.MONTH) + 1
    val day = cal.get(Calendar.DAY_OF_MONTH)
    val week = weekdayText(cal)

    val prefix = if (relative.isNotBlank()) "$relative " else ""

    return String.format(Locale.TAIWAN, "%s%d/%02d (%s)", prefix, month, day, week)
}

@Composable
private fun RecommendationDoctorCard(
    recommendation: RecommendationItemDto,
    profile: DoctorProfile?,
    primaryDark: Color,
    showSpecialtyButton: Boolean,
    isSelecting: Boolean,
    isSpeaking: Boolean,
    onSpeak: () -> Unit,
    onCardClick: () -> Unit,
    onSpecialtyClick: () -> Unit
) {
    val visitTime = formatVisitTime(recommendation)

    Card(
        modifier = Modifier.fillMaxWidth().wrapContentHeight(),
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 1.5.dp)
    ) {
        Column(modifier = Modifier.padding(18.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Surface(
                    modifier = Modifier.size(76.dp),
                    shape = RoundedCornerShape(16.dp),
                    color = Color(0xFFE1E8E8)
                ) {
                    DoctorPhoto(
                        photoUrl = profile?.photoUrl,
                        modifier = Modifier
                            .fillMaxSize()
                            .clip(RoundedCornerShape(16.dp))
                    )
                }

                Spacer(modifier = Modifier.width(16.dp))

                Column(modifier = Modifier.weight(1f)) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = recommendation.doctor.ifBlank { "未命名醫師" },
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold,
                            color = Color(0xFF1F3030),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        if (recommendation.hasRecommendationBadge()) {
                            Surface(shape = RoundedCornerShape(50), color = Color(0xFFFDF6EC)) {
                                Text(
                                    text = "推薦選擇",
                                    fontSize = 13.sp,
                                    fontWeight = FontWeight.Bold,
                                    color = Color(0xFFE6A23C),
                                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp)
                                )
                            }
                        }
                        Spacer(modifier = Modifier.width(6.dp))
                        Box(
                            modifier = Modifier
                                .size(32.dp)
                                .clip(CircleShape)
                                .background(Color(0xFFE8F4F1))
                                .clickable(role = Role.Button, onClick = onSpeak),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = if (isSpeaking) Icons.Default.Stop else Icons.AutoMirrored.Filled.VolumeUp,
                                contentDescription = if (isSpeaking) "停止播放${recommendation.doctor}醫師資訊" else "播放${recommendation.doctor}醫師資訊",
                                tint = primaryDark,
                                modifier = Modifier.size(16.dp)
                            )
                        }
                    }

                    Spacer(modifier = Modifier.height(4.dp))
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = recommendation.childDept.ifBlank { recommendation.parentDept }.ifBlank { "科別待確認" },
                            fontSize = 14.sp,
                            color = Color(0xFF607575),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f)
                        )

                    }
                    if (visitTime.text.isNotBlank()) {
                        Spacer(modifier = Modifier.height(6.dp))
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.CalendarToday,
                                contentDescription = null,
                                tint = primaryDark,
                                modifier = Modifier.size(15.dp)
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                text = visitTime.text,
                                color = primaryDark,
                                fontWeight = FontWeight.SemiBold,
                                fontSize = 14.sp,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(18.dp))

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                if (showSpecialtyButton) {
                    Surface(
                        modifier = Modifier
                            .weight(1f)
                            .height(44.dp)
                            .clickable(onClick = onSpecialtyClick),
                        shape = RoundedCornerShape(12.dp),
                        color = Color.White,
                        border = BorderStroke(1.2.dp, primaryDark)
                    ) {
                        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                            Text("詳情資訊", color = primaryDark, fontSize = 15.sp, fontWeight = FontWeight.Bold)
                        }
                    }
                }
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .height(44.dp)
                        .background(primaryDark, RoundedCornerShape(12.dp))
                        .clickable(enabled = !isSelecting, onClick = onCardClick),
                    contentAlignment = Alignment.Center
                ) {
                    if (isSelecting) {
                        CircularProgressIndicator(color = Color.White, modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                    } else {
                        Text("選擇", color = Color.White, fontSize = 15.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}

/** Without a date filter, show each doctor once (their best-ranked slot); a date filter shows all of that day's slots. */
internal fun displayedDoctorRecommendations(
    recommendations: List<RecommendationItemDto>,
    selectedDate: String?
): List<RecommendationItemDto> =
    if (selectedDate == null) {
        recommendations.distinctBy { it.doctorId?.takeIf(String::isNotBlank) ?: it.doctor }
    } else {
        recommendations.filter { it.date == selectedDate }
    }

private const val COMPACT_REASON_MAX_LENGTH = 55
private const val DETAIL_REASON_MAX_LENGTH = 68

internal fun RecommendationItemDto.compactCardReason(): String? = specialtyTags
    .filter { it.isNotBlank() }
    .takeIf { it.isNotEmpty() }
    ?.joinToString("、")
    ?.let { "SQL 專長資料：$it" }
    ?.take(COMPACT_REASON_MAX_LENGTH)

@Composable
private fun DoctorSpecialtyContent(
    doctor: Doctor,
    profile: DoctorProfile?,
    recommendation: RecommendationItemDto?,
    onClose: () -> Unit
) {
    val accentOrange = Color(0xFFE58A2A)
    val textDark = Color(0xFF151515)
    val configuration = LocalConfiguration.current
    val maxSheetHeight = (configuration.screenHeightDp * 0.8f).dp
    var showExperience by remember { mutableStateOf(false) }
    val titleText = profile?.titles?.takeIf { it.isNotEmpty() }?.joinToString("、") ?: doctor.title
    val educationItems = profile?.education?.takeIf { it.isNotEmpty() } ?: listOf("尚無學歷資料")
    val specialtyItems = (profile?.specialtyTags?.takeIf { tags -> tags.any { it.isNotBlank() } }
        ?: doctor.specialties).mapIndexed { index, specialty ->
        specialty.trimStart().removePrefix("${index + 1}.").trim()
    }.filter { it.isNotBlank() }
    val experienceItems = buildList {
        profile?.currentPositions?.takeIf { it.isNotEmpty() }?.let { positions ->
            add("現職")
            addAll(positions)
        }
        profile?.experience?.takeIf { it.isNotEmpty() }?.let { experience ->
            add("經歷")
            addAll(experience)
        }
    }
    val specialtyBreakdown = recommendation?.let { buildRecommendationBreakdown(it, BreakdownType.SPECIALTY) }
    val timeBreakdown = recommendation?.let { buildRecommendationBreakdown(it, BreakdownType.TIME) }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(max = maxSheetHeight)
            .background(Color.White)
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(20.dp),
            contentAlignment = Alignment.TopCenter
        ) {
            Box(
                modifier = Modifier
                    .padding(top = 10.dp)
                    .size(width = 40.dp, height = 4.dp)
                    .background(Color(0xFFD8D8D8), RoundedCornerShape(50))
            )
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(start = 34.dp, end = 24.dp, top = 20.dp, bottom = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(text = doctor.name, fontSize = 24.sp, fontWeight = FontWeight.Bold, color = textDark)
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text = titleText,
                fontSize = 15.sp,
                fontWeight = FontWeight.Medium,
                color = Color(0xFF666666),
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f)
            )
            Spacer(modifier = Modifier.width(12.dp))
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .background(Color(0xFFE8E8E8), CircleShape)
                    .clickable(onClick = onClose),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.Close,
                    contentDescription = "關閉",
                    tint = Color(0xFF222222),
                    modifier = Modifier.size(18.dp)
                )
            }
        }

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 34.dp)
                .padding(top = 16.dp, bottom = 42.dp)
        ) {
            Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.Top) {
                Box(
                    modifier = Modifier
                        .size(width = 88.dp, height = 98.dp)
                        .background(Color(0xFFD8D8D8), RoundedCornerShape(12.dp))
                        .clip(RoundedCornerShape(12.dp)),
                    contentAlignment = Alignment.Center
                ) {
                    DoctorPhoto(
                        photoUrl = profile?.photoUrl ?: doctor.imageUrl,
                        modifier = Modifier.fillMaxSize()
                    )
                }

                Spacer(modifier = Modifier.width(20.dp))

                Column(modifier = Modifier.weight(1f)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = Icons.Default.School,
                            contentDescription = null,
                            tint = Color.Black,
                            modifier = Modifier.size(20.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(text = "學歷", fontSize = 16.sp, fontWeight = FontWeight.Bold, color = textDark)
                    }

                    Spacer(modifier = Modifier.height(8.dp))
                    educationItems.forEach { item ->
                        Text(
                            text = item,
                            fontSize = 14.sp,
                            lineHeight = 22.sp,
                            color = textDark,
                            modifier = Modifier.padding(bottom = 4.dp)
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(28.dp))
            if (specialtyBreakdown != null || timeBreakdown != null) {
                SectionPill(
                    icon = {
                        Icon(
                            imageVector = Icons.Default.ThumbUp,
                            contentDescription = null,
                            tint = accentOrange,
                            modifier = Modifier.size(22.dp)
                        )
                    },
                    title = "推薦理由",
                    accentColor = accentOrange
                )
                Spacer(modifier = Modifier.height(14.dp))
                RecommendationReasonBlock(
                    specialtyBreakdown = specialtyBreakdown,
                    timeBreakdown = timeBreakdown,
                    textDark = textDark,
                    accentColor = accentOrange
                )
                Spacer(modifier = Modifier.height(32.dp))
            }

            SectionPill(
                icon = {
                    Icon(
                        imageVector = Icons.Default.MedicalServices,
                        contentDescription = null,
                        tint = accentOrange,
                        modifier = Modifier.size(22.dp)
                    )
                },
                title = "專長",
                accentColor = accentOrange
            )

            Spacer(modifier = Modifier.height(14.dp))
            ProfileBulletList(
                items = specialtyItems.ifEmpty { listOf("尚無醫師專長") },
                textDark = textDark
            )

            Spacer(modifier = Modifier.height(32.dp))
            SectionPill(
                icon = {
                    Icon(
                        imageVector = Icons.Default.Person,
                        contentDescription = null,
                        tint = accentOrange,
                        modifier = Modifier.size(22.dp)
                    )
                },
                title = "經歷",
                accentColor = accentOrange,
                trailing = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = if (showExperience) "收合" else "展開",
                            color = accentOrange,
                            fontSize = 15.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Icon(
                            imageVector = if (showExperience) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                            contentDescription = null,
                            tint = accentOrange,
                            modifier = Modifier.size(24.dp)
                        )
                    }
                },
                onClick = { showExperience = !showExperience }
            )

            if (showExperience) {
                Spacer(modifier = Modifier.height(14.dp))
                Column(
                    modifier = Modifier.padding(start = 28.dp, end = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    if (experienceItems.isEmpty()) {
                        Text(
                            text = "尚無經歷資料",
                            color = textDark,
                            fontSize = 15.sp,
                            lineHeight = 22.sp
                        )
                    } else {
                        experienceItems.forEach { item ->
                            val isTitle = item == "現職" || item == "經歷"
                            Text(
                                text = item,
                                color = textDark,
                                fontSize = if (isTitle) 16.sp else 14.sp,
                                lineHeight = 22.sp,
                                fontWeight = if (isTitle) FontWeight.Bold else FontWeight.Medium
                            )
                        }
                    }
                }
            }
        }
    }
}

private enum class BreakdownType {
    SPECIALTY,
    TIME
}

private data class RecommendationBreakdown(
    val title: String,
    val description: String
)

@Composable
private fun RecommendationReasonBlock(
    specialtyBreakdown: RecommendationBreakdown?,
    timeBreakdown: RecommendationBreakdown?,
    textDark: Color,
    accentColor: Color
) {
    Column(
        modifier = Modifier.padding(start = 28.dp, end = 8.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        specialtyBreakdown?.let {
            RecommendationBreakdownRow(
                breakdown = it,
                icon = Icons.Default.Psychology,
                textDark = textDark,
                accentColor = accentColor
            )
        }
        timeBreakdown?.let {
            RecommendationBreakdownRow(
                breakdown = it,
                icon = Icons.Default.Schedule,
                textDark = textDark,
                accentColor = accentColor
            )
        }
    }
}

@Composable
private fun RecommendationBreakdownRow(
    breakdown: RecommendationBreakdown,
    icon: ImageVector,
    textDark: Color,
    accentColor: Color
) {
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = accentColor,
                modifier = Modifier.size(18.dp)
            )
            Text(
                text = "${breakdown.title}：",
                color = accentColor,
                fontSize = 15.sp,
                fontWeight = FontWeight.Bold
            )


        }
        Text(
            text = breakdown.description,
            color = textDark,
            fontSize = 14.sp,
            lineHeight = 22.sp,
            fontWeight = FontWeight.Medium
        )
    }
}


private fun buildRecommendationBreakdown(
    recommendation: RecommendationItemDto,
    type: BreakdownType
): RecommendationBreakdown? {
    return when (type) {
        BreakdownType.SPECIALTY -> recommendation.specialtyDescription().takeIf { it.isNotBlank() }?.let {
            RecommendationBreakdown(title = SPECIALTY_MATCH_TITLE, description = it)
        }
        BreakdownType.TIME -> recommendation.timeDescription().takeIf { it.isNotBlank() }?.let {
            RecommendationBreakdown(title = TIME_MATCH_TITLE, description = it)
        }
    }
}


internal const val SPECIALTY_MATCH_TITLE = "科別建議"
internal const val TIME_MATCH_TITLE = "看診時間"

internal fun RecommendationItemDto.specialtyDescription(): String {
    return departmentReason.trim().ifBlank {
        childDept.ifBlank { parentDept }.trim().takeIf { it.isNotBlank() }
            ?.let { "目前建議您看${it}門診。" }.orEmpty()
    }
}

internal fun RecommendationItemDto.timeDescription(): String {
    return timeReason.trim().ifBlank {
        if (date.isBlank() || session.isBlank()) "" else "此門診的看診時間為${naturalVisitTime()}。"
    }
}


private fun RecommendationItemDto.naturalVisitTime(): String {
    val parsedDate = parseVisitDate(date)
    val sessionLabel = inferSessionLabel(session, slot, sessionTime, parseTimeRange(sessionTime)?.substringBefore("-"))
    if (parsedDate == null) {
        return listOf(date, sessionLabel).filter { it.isNotBlank() }.joinToString(" ")
            .ifBlank { "目前可預約時段" }
    }
    val month = parsedDate.get(Calendar.MONTH) + 1
    val day = parsedDate.get(Calendar.DAY_OF_MONTH)
    return "$month 月 $day 日週${weekdayText(parsedDate)}$sessionLabel"
}


private fun RecommendationItemDto.departmentBasisReason(): String? =
    reasons.firstOrNull { it.startsWith("科別依據：") }?.takeIf { it.isNotBlank() }

private fun RecommendationItemDto.friendlyDepartmentSummary(): String? {
    return childDept.ifBlank { parentDept }.takeIf { it.isNotBlank() }?.let { "已確認科別：$it。" }
}


@Composable
private fun ProfileBulletList(
    items: List<String>,
    textDark: Color,
    contentPadding: PaddingValues = PaddingValues(start = 28.dp, end = 8.dp),
    fontSize: androidx.compose.ui.unit.TextUnit = 16.sp,
    lineHeight: androidx.compose.ui.unit.TextUnit = 28.sp,
    itemSpacing: androidx.compose.ui.unit.Dp = 14.dp,
    fontWeight: FontWeight = FontWeight.SemiBold
) {
    Column(
        modifier = Modifier.padding(contentPadding),
        verticalArrangement = Arrangement.spacedBy(itemSpacing)
    ) {
        items.filter { it.isNotBlank() }.forEach { item ->
            Row(modifier = Modifier.fillMaxWidth(), verticalAlignment = Alignment.Top) {
                Text(
                    text = "•",
                    color = textDark,
                    fontSize = fontSize,
                    lineHeight = lineHeight,
                    fontWeight = fontWeight,
                    modifier = Modifier.width(20.dp)
                )
                Text(
                    text = item,
                    color = textDark,
                    fontSize = fontSize,
                    lineHeight = lineHeight,
                    fontWeight = fontWeight,
                    modifier = Modifier.weight(1f)
                )
            }
        }
    }
}

@Composable
private fun DoctorPhoto(
    photoUrl: String?,
    modifier: Modifier = Modifier
) {
    val imageBitmap by produceState<androidx.compose.ui.graphics.ImageBitmap?>(null, photoUrl) {
        value = null
        if (photoUrl.isNullOrBlank()) return@produceState
        value = withContext(Dispatchers.IO) {
            runCatching {
                URL(photoUrl).openStream().use { stream ->
                    BitmapFactory.decodeStream(stream)?.asImageBitmap()
                }
            }.getOrNull()
        }
    }

    if (imageBitmap != null) {
        Image(
            bitmap = imageBitmap!!,
            contentDescription = null,
            modifier = modifier,
            contentScale = ContentScale.Crop
        )
    } else {
        Box(
            modifier = modifier.background(Color(0xFFD8D8D8)),
            contentAlignment = Alignment.Center
        ) {
            Icon(
                imageVector = Icons.Default.Person,
                contentDescription = null,
                tint = Color(0xFF9A9A9A),
                modifier = Modifier.size(42.dp)
            )
        }
    }
}

@Composable
private fun SectionPill(
    icon: @Composable () -> Unit,
    title: String,
    accentColor: Color,
    trailing: @Composable (() -> Unit)? = null,
    onClick: (() -> Unit)? = null
) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .height(48.dp)
            .shadow(
                elevation = 1.5.dp,
                shape = RoundedCornerShape(24.dp),
                ambientColor = Color.Black.copy(alpha = 0.04f),
                spotColor = Color.Black.copy(alpha = 0.06f)
            )
            .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier),
        shape = RoundedCornerShape(24.dp),
        color = Color(0xFFFFF8F1)
    ) {
        Row(
            modifier = Modifier
                .fillMaxSize()
                .padding(start = 18.dp, end = 18.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            icon()
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text = title,
                color = accentColor,
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.weight(1f))
            trailing?.invoke()
        }
    }
}

@Composable
private fun ErrorState(
    message: String,
    primaryDark: Color,
    onRetry: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(28.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(text = "載入推薦失敗", color = Color(0xFFB00020), fontWeight = FontWeight.Bold, fontSize = 20.sp)
        Spacer(modifier = Modifier.height(10.dp))
        Text(text = message, color = Color(0xFF3C4A4A), textAlign = TextAlign.Center)
        Spacer(modifier = Modifier.height(18.dp))
        Box(
            modifier = Modifier
                .height(46.dp)
                .background(primaryDark, RoundedCornerShape(10.dp))
                .clickable(onClick = onRetry)
                .padding(horizontal = 22.dp),
            contentAlignment = Alignment.Center
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Refresh, null, tint = Color.White, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text("重新整理", color = Color.White, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun EmptyRecommendationState(fallbackDepartments: List<FallbackDepartmentDto>) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .padding(24.dp),
        shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White)
    ) {
        Column(modifier = Modifier.padding(18.dp)) {
            Text("目前沒有可推薦醫師", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = Color(0xFF1F3030))
            Spacer(modifier = Modifier.height(8.dp))
            Text("無可用醫師或時段，請回到問診調整偏好或稍後再試。", color = Color(0xFF607575))
            fallbackDepartments.forEach {
                Spacer(modifier = Modifier.height(8.dp))
                Text("${it.childDept}: ${it.reason}", color = Color(0xFF3C4A4A), fontSize = 14.sp)
            }
        }
    }
}

@Composable
private fun NoSlotsState(
    message: String,
    primaryDark: Color,
    onRetry: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize().padding(28.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text("查無相符班表", color = primaryDark, fontWeight = FontWeight.Bold, fontSize = 21.sp)
        Spacer(modifier = Modifier.height(10.dp))
        Text(message, color = Color(0xFF3C4A4A), textAlign = TextAlign.Center)
        Spacer(modifier = Modifier.height(18.dp))
        Box(
            modifier = Modifier.height(46.dp).background(primaryDark, RoundedCornerShape(10.dp))
                .clickable(onClick = onRetry).padding(horizontal = 22.dp),
            contentAlignment = Alignment.Center
        ) {
            Text("重新查詢", color = Color.White, fontWeight = FontWeight.Bold)
        }
    }
}

private fun RecommendationItemDto.toDoctor(profile: DoctorProfile?): Doctor =
    Doctor(
        id = doctorId ?: recommendationId,
        name = doctor.ifBlank { profile?.name ?: "未命名醫師" },
        departmentId = childDept.ifBlank { parentDept },
        title = profile?.titles?.takeIf { it.isNotEmpty() }?.joinToString("、")
            ?: room?.takeIf { it.isNotBlank() }
            ?: childDept.ifBlank { parentDept },
        specialties = profile?.specialtyTags
            ?.map { it.trim() }
            ?.filter { it.isNotBlank() }
            ?.takeIf { it.isNotEmpty() }
            ?: specialtyTags.map { it.trim() }.filter { it.isNotBlank() },
        imageUrl = profile?.photoUrl,
        availableSlots = listOf(date to (slot.ifBlank { sessionTime ?: session }))
    )

private data class VisitTimeDisplay(val text: String)

internal fun RecommendationItemDto.hasRecommendationBadge(): Boolean = isBestMatch || rank == 1

/** Shared natural-language script for Mandarin and Taiwanese speech synthesis. */
internal fun RecommendationItemDto.spokenIntroduction(): String {
    val name = normalizeDoctorName(doctor).ifBlank { "此位" }
    val introduction = if (hasRecommendationBadge()) "推薦您${name}醫師。" else "${name}醫師。"
    val department = childDept.ifBlank { parentDept }.trim()
    val dateText = parseVisitDate(date)?.let {
        "${it.get(Calendar.MONTH) + 1}月${it.get(Calendar.DAY_OF_MONTH)}日星期${weekdayText(it)}"
    }.orEmpty()
    val range = sequenceOf(sessionTime, slot, session)
        .mapNotNull { parseTimeRange(it) }
        .firstOrNull()
    val times = range?.split("-").orEmpty().mapNotNull { raw ->
        val hour = raw.substringBefore(':').toIntOrNull()
        val minute = raw.substringAfter(':').toIntOrNull()
        if (hour != null && minute != null && hour in 0..23 && minute in 0..59) hour to minute else null
    }
    val timeText = if (times.isEmpty()) {
        inferSessionLabel(session, slot, sessionTime, null)
    } else {
        times.mapIndexed { index, (hour, minute) ->
            val period = spokenPeriod(hour)
            val prefix = if (index == 0 || period != spokenPeriod(times[index - 1].first)) period else ""
            val clockHour = (hour % 12).let { if (it == 0) 12 else it }
            val minuteText = when (minute) {
                0 -> ""
                30 -> "半"
                in 1..9 -> "零${spokenNumber(minute)}分"
                else -> "${spokenNumber(minute)}分"
            }
            "$prefix${spokenNumber(clockHour)}點$minuteText"
        }.joinToString("到")
    }
    return buildString {
        append(introduction)
        if (department.isNotBlank()) append("科別為${department}。")
        if (dateText.isNotBlank() || timeText.isNotBlank()) append("時間為$dateText$timeText。")
    }
}

private fun spokenPeriod(hour: Int): String = when (hour) {
    in 0..5 -> "凌晨"
    in 6..11 -> "上午"
    in 12..17 -> "下午"
    else -> "晚上"
}

private fun spokenNumber(number: Int): String {
    val digits = listOf("零", "一", "二", "三", "四", "五", "六", "七", "八", "九")
    return when {
        number < 10 -> digits[number]
        number < 20 -> "十" + if (number % 10 == 0) "" else digits[number % 10]
        else -> digits[number / 10] + "十" + if (number % 10 == 0) "" else digits[number % 10]
    }
}

private fun formatVisitTime(
    item: RecommendationItemDto,
    today: Calendar = Calendar.getInstance(Locale.TAIWAN)
): VisitTimeDisplay {
    val date = parseVisitDate(item.date)
    val timeRange = parseTimeRange(item.sessionTime)
        ?: parseTimeRange(item.slot)
        ?: parseTimeRange(item.session)
    val timeStart = timeRange?.substringBefore("-")
    val sessionLabel = inferSessionLabel(item.session, item.slot, item.sessionTime, timeStart)

    if (date == null) {
        val fallback = listOf(item.date, sessionLabel, timeRange ?: item.displayTime())
            .filter { it.isNotBlank() }
            .joinToString(" ")
        return VisitTimeDisplay(text = fallback)
    }

    val dayLabel = relativeDayText(date, today)
    val dateText = "${date.get(Calendar.MONTH) + 1}/${date.get(Calendar.DAY_OF_MONTH)} (${weekdayText(date)})"
    val displayText = listOfNotNull(dayLabel, dateText, sessionLabel, timeRange)
        .filter { it.isNotBlank() }
        .joinToString(" ")

    return VisitTimeDisplay(text = displayText)
}

private fun parseVisitDate(raw: String): Calendar? {
    val text = raw.trim()
    if (text.isBlank()) return null
    val datePart = Regex("""\d{4}[-/]\d{1,2}[-/]\d{1,2}""")
        .find(text)
        ?.value
        ?: return null
    val parser = SimpleDateFormat("yyyy-MM-dd", Locale.TAIWAN).apply {
        isLenient = false
    }
    return runCatching {
        val parsed = parser.parse(datePart.replace('/', '-')) ?: return null
        Calendar.getInstance(Locale.TAIWAN).apply {
            time = parsed
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }
    }.getOrNull()
}

private fun parseTimeRange(raw: String?): String? {
    val text = raw?.trim().orEmpty()
    if (text.isBlank()) return null
    val times = Regex("""\d{1,2}:\d{2}""")
        .findAll(text)
        .map { it.value.padStart(5, '0') }
        .toList()
    return when {
        times.size >= 2 -> "${times[0]}-${times[1]}"
        times.size == 1 -> times[0]
        else -> null
    }
}

private fun inferSessionLabel(
    session: String,
    slot: String,
    sessionTime: String?,
    startTime: String?
): String {
    val raw = listOf(session, slot, sessionTime.orEmpty()).joinToString(" ")
    return when {
        raw.contains("上午") || raw.contains("早") -> "上午"
        raw.contains("下午") || raw.contains("午") -> "下午"
        raw.contains("夜") || raw.contains("晚") -> "晚上"
        startTime != null -> {
            val hour = startTime.substringBefore(":").toIntOrNull() ?: return ""
            when (hour) {
                in 0..11 -> "上午"
                in 12..17 -> "下午"
                else -> "晚上"
            }
        }
        else -> ""
    }
}

private fun relativeDayText(date: Calendar, today: Calendar): String =
    when (daysBetween(today, date)) {
        0 -> "今"
        1 -> "明"
        2 -> "後天"
        else -> ""
    }

private fun daysBetween(today: Calendar, target: Calendar): Int {
    val start = today.startOfDay()
    val end = target.startOfDay()
    val millisPerDay = 24L * 60L * 60L * 1000L
    return ((end.timeInMillis - start.timeInMillis) / millisPerDay).toInt()
}

private fun Calendar.startOfDay(): Calendar =
    (clone() as Calendar).apply {
        set(Calendar.HOUR_OF_DAY, 0)
        set(Calendar.MINUTE, 0)
        set(Calendar.SECOND, 0)
        set(Calendar.MILLISECOND, 0)
    }

private fun weekdayText(date: Calendar): String =
    when (date.get(Calendar.DAY_OF_WEEK)) {
        Calendar.MONDAY -> "一"
        Calendar.TUESDAY -> "二"
        Calendar.WEDNESDAY -> "三"
        Calendar.THURSDAY -> "四"
        Calendar.FRIDAY -> "五"
        Calendar.SATURDAY -> "六"
        else -> "日"
    }

private fun RecommendationItemDto.displayTime(): String =
    sessionTime?.takeIf { it.isNotBlank() } ?: slot.ifBlank { session }

private fun RecommendationItemDto.hasRecommendationDetailInfo(): Boolean =
    listOf(matchReason.orEmpty(), specialtyTags.joinToString(), reasons.joinToString())
        .any { it.isNotBlank() }

private fun String.isDisplayableAiReason(): Boolean {
    val text = trim()
    return text.isNotBlank() &&
            !text.startsWith("掛號別：") &&
            !text.startsWith("班表：") &&
            !text.startsWith("診間：")
}

private fun loadDoctorProfilesFromAssets(context: Context): List<DoctorProfile> {
    return runCatching {
        val json = context.assets.open("teacher_profiles.json")
            .bufferedReader(Charsets.UTF_8)
            .use { it.readText() }
        val array = JSONArray(json)
        (0 until array.length()).mapNotNull { index ->
            array.optJSONObject(index)?.toDoctorProfile()
        }
    }.getOrDefault(emptyList())
}

private fun JSONObject.toDoctorProfile(): DoctorProfile {
    return DoctorProfile(
        name = optString("name", ""),
        photoUrl = optNullableString("photo_url"),
        education = optStringOrList("education"),
        currentPositions = optStringOrList("current_positions"),
        experience = optStringOrList("experience"),
        specialtyTags = optStringOrList("specialty_tags"),
        titles = optStringOrList("titles"),
        tid = if (isNull("tid")) null else optInt("tid")
    )
}

private fun JSONObject.optStringOrList(name: String): List<String> {
    if (isNull(name)) return emptyList()
    optJSONArray(name)?.let { array ->
        return (0 until array.length())
            .mapNotNull { index -> array.optString(index).trim().takeIf { it.isNotBlank() } }
    }
    return optString(name, "")
        .split("；", ";", "\n")
        .flatMap { it.split(Regex("\\s+-\\s+|^-\\s+")) }
        .map { it.trim().trimStart('-', ' ') }
        .filter { it.isNotBlank() }
}

private fun JSONObject.optNullableString(name: String): String? =
    if (isNull(name)) null else optString(name).takeIf { it.isNotBlank() }

private fun findDoctorProfile(
    profiles: List<DoctorProfile>,
    item: RecommendationItemDto
): DoctorProfile? {
    val doctorName = normalizeDoctorName(item.doctor)
    if (doctorName.isBlank()) return null
    return profiles.firstOrNull { normalizeDoctorName(it.name) == doctorName }
        ?: profiles.firstOrNull {
            val profileName = normalizeDoctorName(it.name)
            profileName.isNotBlank() && (profileName.contains(doctorName) || doctorName.contains(profileName))
        }
}

private fun normalizeDoctorName(name: String): String =
    name
        .replace("醫師", "")
        .replace("醫生", "")
        .replace(" ", "")
        .trim()


// ================= 預覽 (Preview) 函式 =================

@Preview(showBackground = true, device = "id:pixel_5", name = "成功狀態預覽 (醫師專長)")
@Composable
fun DoctorSelectionScreenPreview() {
    val dummyRecommendation = RecommendationItemDto(
        recommendationId = "rec_001",
        doctorId = "doc_001",
        doctor = "王大明 醫師",
        parentDept = "內科",
        childDept = "心臟內科",
        date = "2026-06-01",
        session = "上午",
        slot = "09:00",
        sessionTime = "09:00-12:00",
        score = 95.0,
        specialtyScore = 95.0,
        timeScore = 90.0,
        isBestMatch = true,
        rank = 1,
        reasons = listOf("專長依據：擅長心血管疾病"),
        specialtyTags = listOf("高血壓", "心律不整"),
        matchReason = "醫師專長符合您的症狀",
        room = "201診"
    )

    val dummySuccessState = DoctorUiState.Success(
        caseId = "test_case_id",
        specialtyFirst = listOf(dummyRecommendation),
        timeFirst = listOf(dummyRecommendation),
        fallbackDepartments = emptyList()
    )

    DoctorSelectionScreenContent(
        uiState = dummySuccessState,
        doctorProfiles = emptyList(),
        selectingRecommendationId = null,
        onBack = {},
        onRetry = {},
        onCardClick = {},
        onSpecialtyClick = {}
    )
}

@Preview(showBackground = true, device = "id:pixel_5", name = "看診時間優先狀態預覽")
@Composable
fun DoctorSelectionTimeFirstPreview() {
    val dummyTimeRecommendation = RecommendationItemDto(
        recommendationId = "rec_002",
        doctorId = "doc_002",
        doctor = "林美玲 醫師",
        parentDept = "內科",
        childDept = "心臟內科",
        date = "2026-08-28",
        session = "下午",
        slot = "14:00",
        sessionTime = "14:00-17:00",
        score = 90.0,
        specialtyScore = 85.0,
        timeScore = 98.0,
        isBestMatch = true,
        rank = 1,
        reasons = listOf("時間依據：為您安排最近的門診時間"),
        specialtyTags = listOf("一般內科", "心血管保健"),
        matchReason = "時間最符合您的需求",
        room = "105診"
    )

    val dummyTimeState = DoctorUiState.Success(
        caseId = "test_case_id_time",
        specialtyFirst = emptyList(),
        timeFirst = listOf(dummyTimeRecommendation),
        fallbackDepartments = emptyList()
    )

    DoctorSelectionScreenContent(
        uiState = dummyTimeState,
        doctorProfiles = emptyList(),
        selectingRecommendationId = null,
        onBack = {},
        onRetry = {},
        onCardClick = {},
        onSpecialtyClick = {}
    )
}

@Preview(showBackground = true, device = "id:pixel_5", name = "真實假資料測試 (看診時間優先)")
@Composable
fun DoctorSelectionRealDataPreview() {
    val mockTimeFirstList = listOf(
        RecommendationItemDto(
            recommendationId = "rec_t_case_mock_20260828_001_001",
            doctorId = "2608",
            doctor = "侯重光",
            parentDept = "內科系",
            childDept = "一般內科",
            date = "2026-09-03",
            session = "上午診",
            slot = "2608",
            sessionTime = "09:00-12:00",
            room = "2608",
            specialtyTags = listOf("一般內科", "健康管理", "代謝疾病"),
            specialtyScore = 78.0,
            timeScore = 100.0,
            matchReason = "醫師可提供一般內科、健康管理及代謝疾病的初步評估。",
            score = 93.4,
            reasons = listOf(
                "科別依據：最終建議科別為一般內科。",
                "時間依據：2026-09-03 上午診完全符合可看診時段。",
                "狀態依據：Schedule.status=open，判斷為可掛號。",
                "專長依據：醫師專長包含一般內科、健康管理及代謝疾病。",
                "排序依據：時間優先欄使用 time_score*70 + specialty_score*30。"
            ),
            rank = 1,
            isBestMatch = false
        ),
        RecommendationItemDto(
            recommendationId = "rec_t_case_mock_20260828_001_002",
            doctorId = "2607",
            doctor = "葉筱芸",
            parentDept = "內科系",
            childDept = "一般內科",
            date = "2026-09-02",
            session = "下午診",
            slot = "2607",
            sessionTime = "13:30-17:00",
            room = "2607",
            specialtyTags = listOf("一般內科", "感染症", "發燒評估"),
            specialtyScore = 86.0,
            timeScore = 90.0,
            matchReason = "醫師專長可協助一般內科症狀及感染相關問題的初步評估。",
            score = 88.8,
            reasons = listOf(
                "科別依據：最終建議科別為一般內科。",
                "時間依據：2026-09-02 下午診符合可看診時段。",
                "狀態依據：Schedule.status=open，判斷為可掛號。",
                "專長依據：醫師專長包含一般內科、感染症及發燒評估。"
            ),
            rank = 2,
            isBestMatch = false
        ),
        RecommendationItemDto(
            recommendationId = "rec_t_case_mock_20260828_001_003",
            doctorId = "2606",
            doctor = "林炯熙",
            parentDept = "內科系",
            childDept = "一般內科",
            date = "2026-09-01",
            session = "上午診",
            slot = "2606",
            sessionTime = "08:30-12:00",
            room = "2606",
            specialtyTags = listOf("一般內科", "慢性病整合照護", "高血壓"),
            specialtyScore = 92.0,
            timeScore = 80.0,
            matchReason = "醫師專長與一般內科常見症狀及慢性病整合評估相符。",
            score = 83.6,
            reasons = listOf(
                "科別依據：最終建議科別為一般內科。",
                "時間依據：2026-09-01 上午診符合可看診時段。",
                "狀態依據：Schedule.status=open，判斷為可掛號。",
                "專長依據：醫師專長包含一般內科及慢性病整合照護。"
            ),
            rank = 3,
            isBestMatch = false
        )
    )

    val dummyState = DoctorUiState.Success(
        caseId = "case_mock_20260828_001",
        specialtyFirst = mockTimeFirstList, // 兩邊給一樣的資料，可以順便測試「排序結果相同」的提示
        timeFirst = mockTimeFirstList,
        fallbackDepartments = emptyList()
    )

    DoctorSelectionScreenContent(
        uiState = dummyState,
        doctorProfiles = emptyList(),
        selectingRecommendationId = null,
        onBack = {},
        onRetry = {},
        onCardClick = {},
        onSpecialtyClick = {}
    )
}

/** One playback owner for the entire doctor list, using live backend TTS. */
internal class DoctorCardSpeech(
    private val scope: CoroutineScope,
    private val requestSpeech: suspend (TtsRequest) -> VoiceTtsResponseDto,
    private val playSpeech: (VoiceTtsResponseDto, () -> Unit, () -> Unit) -> Unit,
    private val stopSpeech: () -> Unit
) {
    var activeId by mutableStateOf<String?>(null)
        private set
    var error by mutableStateOf<String?>(null)
        private set
    private var generation = 0
    private var job: Job? = null

    fun stop() {
        generation++
        job?.cancel()
        job = null
        activeId = null
        error = null
        stopSpeech()
    }

    fun toggle(id: String, text: String, language: String) {
        val wasActive = activeId == id
        stop()
        if (wasActive) return
        val lang = when (language) {
            "國語" -> "chinese"
            "台語" -> "taiwanese"
            else -> { error = "不支援此語音語言。"; return }
        }
        if (text.isBlank()) {
            error = "目前沒有可播放的醫師資訊。"
            return
        }
        activeId = id
        val token = generation
        val done: () -> Unit = { if (token == generation) activeId = null }
        val failed: () -> Unit = {
            if (token == generation) {
                activeId = null
                error = "語音播放失敗，請確認網路連線後再試。"
            }
        }
        job = scope.launch {
            try {
                // The backend supports uncached requests without a session_id.
                // Only the natural introduction is sent, never the full card or scores.
                val result = requestSpeech(TtsRequest(text = text, lang = lang))
                if (token != generation) return@launch
                if (result.ttsFailed || result.audioBase64.isBlank()) failed()
                else playSpeech(result, done, failed)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (_: Exception) {
                failed()
            }
        }
    }

    fun close() = stop()

    companion object {
        fun create(context: Context, scope: CoroutineScope): DoctorCardSpeech {
            val player = AudioPlayer()
            val api = MedicalApiClient()
            return DoctorCardSpeech(
                scope = scope,
                requestSpeech = api::tts,
                playSpeech = { result, done, failed ->
                    player.playBase64(result.audioBase64, context.cacheDir, result.audioFormat,
                        onDone = done, onError = failed)
                },
                stopSpeech = player::release
            )
        }
    }
}
