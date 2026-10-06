package com.example.medicalaiguidance.screen

import android.app.DatePickerDialog
import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.CalendarToday
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FilterChip
import androidx.compose.material3.FilterChipDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import com.example.medicalaiguidance.navigation.Route
import com.example.medicalaiguidance.network.RecommendationItemDto
import com.example.medicalaiguidance.network.ReferenceDepartmentDto
import com.example.medicalaiguidance.viewmodel.QuickSearchReferenceLoadState
import com.example.medicalaiguidance.viewmodel.QuickSearchUiState
import com.example.medicalaiguidance.viewmodel.QuickSearchViewModel
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale

// === Design System Tokens ===
private val GreenPrimary = Color(0xFF2F6F73)
private val GreenContainer = Color(0xFFD1EAE8)
private val GreenOnContainer = Color(0xFF052B2C)
private val BgColor = Color(0xFFF2FAF8)
private val BorderColor = Color(0xFFB5CBCB)
private val TextSecondary = Color(0xFF5A7070)
private val ErrorColor = Color(0xFFBA1A1A)

private val DisableBgGray = Color(0xFFE0E0E0)
private val DisableTextGray = Color(0xFF9E9E9E)
private val DepartmentDropdownMaxHeight = 420.dp

private data class SessionOption(val label: String, val value: String)
private data class SelectionOption(val value: String, val label: String, val supportingText: String = "")

private val QuickSearchPeriods = listOf(
    SessionOption("上午診", "morning"),
    SessionOption("下午診", "afternoon"),
    SessionOption("夜診", "evening")
)

@Composable
fun QuickSearchScreen(
    navController: NavHostController,
    viewModel: QuickSearchViewModel = viewModel()
) {
    val form by viewModel.form.collectAsState()
    val uiState by viewModel.uiState.collectAsState()
    val validationError by viewModel.validationError.collectAsState()
    val selectingScheduleId by viewModel.selectingScheduleId.collectAsState()
    val departments by viewModel.departments.collectAsState()
    val departmentLoadState by viewModel.departmentLoadState.collectAsState()
    val loading = uiState == QuickSearchUiState.Loading

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(BgColor)
            .statusBarsPadding(),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 24.dp, vertical = 20.dp),
                contentAlignment = Alignment.Center
            ) {
                Box(
                    modifier = Modifier
                        .align(Alignment.CenterStart)
                        .size(48.dp)
                        .shadow(elevation = 4.dp, shape = RoundedCornerShape(16.dp))
                        .background(Color.White, RoundedCornerShape(16.dp))
                        .clickable { navController.popBackStack() },
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "返回",
                        tint = GreenPrimary,
                        modifier = Modifier.size(22.dp)
                    )
                }
                Text(
                    text = "查詢門診班表",
                    color = GreenPrimary,
                    fontSize = 24.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        item {
            ReturnVisitFormCard(
                department = form.department,
                departments = departments,
                departmentLoadState = departmentLoadState,
                availableDate = form.availableDate,
                selectedPeriod = form.period,
                validationError = validationError,
                loading = loading,
                onDepartmentSelected = viewModel::selectDepartment,
                onDepartmentSelectionCleared = viewModel::clearDepartmentSelection,
                onRetryDepartments = viewModel::loadDepartments,
                onDateSelected = viewModel::updateAvailableDate,
                onPeriodSelected = viewModel::selectPeriod,
                onSubmit = viewModel::submit
            )
        }

        when (val state = uiState) {
            QuickSearchUiState.Idle, QuickSearchUiState.Loading -> Unit
            is QuickSearchUiState.Error -> item {
                ReturnVisitMessage(state.message, ErrorColor)
            }
            is QuickSearchUiState.Success -> {
                if (state.results.isEmpty()) {
                    item { ReturnVisitMessage("目前沒有符合條件的可預約班表。", GreenPrimary) }
                } else {
                    item {
                        Text(
                            text = "符合條件的可預約班表",
                            color = GreenPrimary,
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.padding(horizontal = 24.dp, vertical = 4.dp)
                        )
                    }
                    items(state.results, key = { it.recommendationId }) { schedule ->
                        ReturnVisitRecommendationCard(
                            item = schedule,
                            isSelecting = selectingScheduleId == schedule.recommendationId,
                            onClick = {
                                viewModel.selectScheduleAndNavigate(schedule) {
                                    navController.navigate(Route.CONFIRM_NEED)
                                }
                            }
                        )
                    }
                }
            }
        }

        item { Spacer(modifier = Modifier.height(24.dp)) }
    }
}

@Composable
private fun ReturnVisitFormCard(
    department: String,
    departments: List<ReferenceDepartmentDto>,
    departmentLoadState: QuickSearchReferenceLoadState,
    availableDate: String,
    selectedPeriod: String,
    validationError: String?,
    loading: Boolean,
    onDepartmentSelected: (String) -> Unit,
    onDepartmentSelectionCleared: () -> Unit,
    onRetryDepartments: () -> Unit,
    onDateSelected: (String) -> Unit,
    onPeriodSelected: (String) -> Unit,
    onSubmit: () -> Unit
) {
    val context = LocalContext.current

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp)
            .background(Color.White, RoundedCornerShape(20.dp))
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                text = "設定查詢條件",
                color = GreenPrimary,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold
            )
            Text(
                text = "請選擇科別與方便看診的時間，系統將列出符合條件的實際班表。",
                color = TextSecondary,
                fontSize = 14.sp,
                lineHeight = 20.sp
            )
        }

        // 1. 就診科別
        SearchableSelectionField(
            selectedValue = department,
            resetKey = "department",
            label = "就診科別（必填）",
            placeholder = "搜尋並選擇科別",
            options = sortQuickSearchDepartments(departments).map {
                SelectionOption(value = it.deptId, label = it.childDept, supportingText = it.parentDept)
            },
            enabled = !loading && departmentLoadState == QuickSearchReferenceLoadState.Ready,
            loadState = departmentLoadState,
            onSelected = onDepartmentSelected,
            onSelectionCleared = onDepartmentSelectionCleared,
            onRetry = onRetryDepartments
        )

        // 2. 可看診日期
        DatePickerField(
            value = availableDate,
            onDateSelected = onDateSelected,
            enabled = !loading,
            onOpenPicker = { openReturnVisitDatePicker(context, availableDate, onDateSelected) }
        )

        // 3. 可看診時段
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(
                text = "可看診時段",
                color = GreenPrimary,
                fontSize = 15.sp,
                fontWeight = FontWeight.SemiBold
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                QuickSearchPeriods.forEach { session ->
                    val selected = session.value == selectedPeriod
                    FilterChip(
                        selected = selected,
                        onClick = { onPeriodSelected(session.value) },
                        enabled = !loading,
                        label = { Text(session.label, fontSize = 14.sp) },
                        shape = RoundedCornerShape(10.dp),
                        colors = FilterChipDefaults.filterChipColors(
                            containerColor = Color.Transparent,
                            labelColor = TextSecondary,
                            selectedContainerColor = GreenContainer,
                            selectedLabelColor = GreenOnContainer
                        ),
                        border = FilterChipDefaults.filterChipBorder(
                            enabled = true,
                            selected = selected,
                            borderColor = BorderColor,
                            selectedBorderColor = GreenPrimary
                        )
                    )
                }
            }
        }

        validationError?.let {
            Text(text = it, color = ErrorColor, fontSize = 13.sp)
        }

        Spacer(modifier = Modifier.height(8.dp))

        // 4. 查詢按鈕 (UX優化：套用灰色 Disable 狀態)
        Button(
            onClick = onSubmit,
            enabled = !loading && department.isNotBlank(),
            shape = RoundedCornerShape(14.dp),
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = GreenPrimary,
                contentColor = Color.White,
                // 套用標準的 Disabled 灰階色彩
                disabledContainerColor = DisableBgGray,
                disabledContentColor = DisableTextGray
            )
        ) {
            if (loading) {
                CircularProgressIndicator(color = Color.White, modifier = Modifier.size(22.dp), strokeWidth = 2.5.dp)
            } else {
                Text("尋找門診", fontSize = 17.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun SearchableSelectionField(
    selectedValue: String,
    resetKey: String,
    label: String,
    placeholder: String,
    options: List<SelectionOption>,
    enabled: Boolean,
    loadState: QuickSearchReferenceLoadState,
    onSelected: (String) -> Unit,
    onSelectionCleared: () -> Unit,
    onRetry: () -> Unit
) {
    var query by rememberSaveable(resetKey) { mutableStateOf(selectedValue) }
    var expanded by remember(resetKey) { mutableStateOf(false) }
    var textFieldWidth by remember { mutableIntStateOf(0) }
    val interactionSource = remember { MutableInteractionSource() }
    val isPressed by interactionSource.collectIsPressedAsState()

    LaunchedEffect(selectedValue, loadState) {
        when {
            selectedValue.isNotBlank() -> query = selectedValue
            loadState is QuickSearchReferenceLoadState.Error -> query = ""
        }
    }

    LaunchedEffect(isPressed, enabled) {
        if (isPressed && enabled) expanded = true
    }

    val filteredOptions = if (shouldShowAllSelectionOptions(query, selectedValue)) {
        options
    } else {
        options.filter {
            it.label.contains(query, ignoreCase = true) ||
                    it.supportingText.contains(query, ignoreCase = true)
        }
    }

    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Box(modifier = Modifier.fillMaxWidth()) {
            OutlinedTextField(
                value = query,
                onValueChange = { value ->
                    query = value
                    expanded = true
                    if (value != selectedValue) onSelectionCleared()
                },
                label = { Text(label) },
                placeholder = { Text(placeholder) },
                enabled = enabled,
                singleLine = true,
                interactionSource = interactionSource,
                trailingIcon = {
                    if (loadState == QuickSearchReferenceLoadState.Loading) {
                        CircularProgressIndicator(color = GreenPrimary, modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                    } else {
                        Icon(
                            imageVector = Icons.Default.ArrowDropDown,
                            contentDescription = "展開$label",
                            tint = GreenPrimary,
                            modifier = Modifier.clickable(enabled = enabled) { expanded = true }
                        )
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .onGloballyPositioned { coordinates -> textFieldWidth = coordinates.size.width }
                    .onFocusChanged { if (it.isFocused && enabled) expanded = true },
                shape = RoundedCornerShape(12.dp),
                colors = returnVisitTextFieldColors()
            )
            val density = androidx.compose.ui.platform.LocalDensity.current
            DropdownMenu(
                expanded = expanded && enabled,
                onDismissRequest = { expanded = false },
                modifier = Modifier
                    .width(with(density) { textFieldWidth.toDp() })
                    .heightIn(max = DepartmentDropdownMaxHeight)
                    .background(Color.White)
            ) {
                if (filteredOptions.isEmpty()) {
                    DropdownMenuItem(text = { Text("找不到符合的選項", color = TextSecondary) }, onClick = {}, enabled = false)
                } else {
                    filteredOptions.forEach { option ->
                        DropdownMenuItem(
                            contentPadding = androidx.compose.foundation.layout.PaddingValues(vertical = 8.dp, horizontal = 16.dp),
                            text = {
                                Column(
                                    modifier = Modifier.padding(vertical = 4.dp),
                                    verticalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Text(text = option.label, color = GreenPrimary, fontSize = 16.sp, fontWeight = FontWeight.Medium)
                                    if (option.supportingText.isNotBlank()) {
                                        Text(text = option.supportingText, color = TextSecondary, fontSize = 13.sp)
                                    }
                                }
                            },
                            onClick = {
                                query = option.label
                                expanded = false
                                onSelected(option.value)
                            }
                        )
                    }
                }
            }
        }

        when (loadState) {
            QuickSearchReferenceLoadState.Idle -> Unit
            QuickSearchReferenceLoadState.Loading -> Text("科別清單載入中…", color = TextSecondary, fontSize = 12.sp)
            QuickSearchReferenceLoadState.Ready -> Unit
            is QuickSearchReferenceLoadState.Error -> Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(text = loadState.message, color = ErrorColor, fontSize = 12.sp, modifier = Modifier.weight(1f))
                TextButton(onClick = onRetry) { Text("重新載入", color = GreenPrimary) }
            }
        }
    }
}

internal fun shouldShowAllSelectionOptions(query: String, selectedValue: String): Boolean =
    query.isBlank() || (selectedValue.isNotBlank() && query == selectedValue)

internal fun sortQuickSearchDepartments(
    departments: List<ReferenceDepartmentDto>
): List<ReferenceDepartmentDto> = departments.sortedBy {
    quickSearchDepartmentGroupRank(it.parentDept)
}

internal fun quickSearchDepartmentGroupRank(parentDepartment: String): Int {
    val normalized = parentDepartment.trim()
    return when {
        normalized == "內科系" || normalized == "一般內科" -> 0
        normalized == "外科系" -> 1
        normalized.startsWith("婦幼") -> 2
        normalized == "五官科" -> 3
        normalized == "其他科" -> 4
        normalized == "大我門診" -> 5
        normalized == "整合門診" -> 6
        normalized.equals("AI輔助門診", ignoreCase = true) -> 7
        else -> Int.MAX_VALUE
    }
}

// === 以下為共用模組 ===

@Composable
private fun DatePickerField(
    value: String,
    onDateSelected: (String) -> Unit,
    enabled: Boolean,
    onOpenPicker: () -> Unit
) {
    Box(modifier = Modifier.fillMaxWidth()) {
        OutlinedTextField(
            value = value,
            onValueChange = onDateSelected,
            label = { Text("可看診日期") },
            placeholder = { Text("請選擇日期") },
            readOnly = true,
            enabled = enabled,
            singleLine = true,
            trailingIcon = {
                Icon(
                    imageVector = Icons.Default.CalendarToday,
                    contentDescription = "選擇日期",
                    tint = GreenPrimary,
                    modifier = Modifier.clickable(enabled = enabled) { onOpenPicker() }
                )
            },
            modifier = Modifier
                .fillMaxWidth()
                .clickable(enabled = enabled) { onOpenPicker() },
            shape = RoundedCornerShape(12.dp),
            colors = returnVisitTextFieldColors()
        )
        Box(
            modifier = Modifier
                .matchParentSize()
                .clickable(enabled = enabled) { onOpenPicker() }
        )
    }
}

@Composable
private fun returnVisitTextFieldColors() = OutlinedTextFieldDefaults.colors(
    focusedBorderColor = GreenPrimary,
    unfocusedBorderColor = BorderColor,
    focusedLabelColor = GreenPrimary,
    cursorColor = GreenPrimary
)

@Composable
private fun ReturnVisitMessage(message: String, color: Color) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp)
            .background(Color.White, RoundedCornerShape(16.dp))
            .padding(18.dp)
    ) {
        Text(text = message, color = color, fontSize = 15.sp, lineHeight = 22.sp)
    }
}

@Composable
private fun ReturnVisitRecommendationCard(
    item: RecommendationItemDto,
    isSelecting: Boolean,
    onClick: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 24.dp)
            .background(Color.White, RoundedCornerShape(16.dp))
            .clickable(enabled = !isSelecting, onClick = onClick)
            .padding(18.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Text(
            text = item.doctor.ifBlank { "醫師待確認" },
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold,
            color = GreenPrimary
        )
        Text(
            text = item.childDept.ifBlank { item.parentDept },
            color = TextSecondary,
            fontSize = 14.sp
        )
        Text(
            text = listOf(item.date, item.sessionTime ?: item.session, item.room ?: item.slot)
                .filter { it.isNotBlank() }
                .joinToString("  ·  "),
            color = Color(0xFF263A3B),
            fontSize = 14.sp,
            fontWeight = FontWeight.Medium
        )
        item.reasons.firstOrNull()?.let {
            Text(text = it, color = TextSecondary, fontSize = 13.sp)
        }
        if (isSelecting) {
            CircularProgressIndicator(
                color = GreenPrimary,
                modifier = Modifier.size(20.dp),
                strokeWidth = 2.dp
            )
        } else {
            Text(
                text = "選擇此班表並進入掛號導引",
                color = GreenPrimary,
                fontSize = 13.sp,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}

private fun openReturnVisitDatePicker(
    context: Context,
    currentValue: String,
    onDateSelected: (String) -> Unit
) {
    val today = Calendar.getInstance(Locale.TAIWAN).apply {
        set(Calendar.HOUR_OF_DAY, 0)
        set(Calendar.MINUTE, 0)
        set(Calendar.SECOND, 0)
        set(Calendar.MILLISECOND, 0)
    }
    val calendar = parseDateToCalendar(currentValue)?.takeIf { !it.before(today) } ?: today
    DatePickerDialog(
        context,
        { _, year, month, dayOfMonth ->
            val selected = Calendar.getInstance(Locale.TAIWAN).apply {
                set(Calendar.YEAR, year)
                set(Calendar.MONTH, month)
                set(Calendar.DAY_OF_MONTH, dayOfMonth)
            }
            onDateSelected(SimpleDateFormat("yyyy-MM-dd", Locale.TAIWAN).format(selected.time))
        },
        calendar.get(Calendar.YEAR),
        calendar.get(Calendar.MONTH),
        calendar.get(Calendar.DAY_OF_MONTH)
    ).apply {
        datePicker.minDate = today.timeInMillis
    }.show()
}

private fun parseDateToCalendar(value: String): Calendar? =
    runCatching {
        val date = SimpleDateFormat("yyyy-MM-dd", Locale.TAIWAN).apply { isLenient = false }.parse(value.replace('/', '-')) ?: return@runCatching null
        Calendar.getInstance(Locale.TAIWAN).apply { time = date }
    }.getOrNull()
