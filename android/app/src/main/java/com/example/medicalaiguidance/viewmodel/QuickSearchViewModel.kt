package com.example.medicalaiguidance.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.medicalaiguidance.network.QuickSearchRequest
import com.example.medicalaiguidance.network.RecommendationItemDto
import com.example.medicalaiguidance.network.ReferenceDepartmentDto
import com.example.medicalaiguidance.repository.MedicalRepository
import java.text.SimpleDateFormat
import java.util.Locale
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class QuickSearchForm(
    val department: String = "",
    val departmentId: String = "",
    val parentDepartment: String = "",
    val availableDate: String = "",
    val period: String = ""
)

sealed interface QuickSearchUiState {
    data object Idle : QuickSearchUiState
    data object Loading : QuickSearchUiState
    data class Success(
        val caseId: String,
        val results: List<RecommendationItemDto>
    ) : QuickSearchUiState
    data class Error(val message: String) : QuickSearchUiState
}

sealed interface QuickSearchReferenceLoadState {
    data object Idle : QuickSearchReferenceLoadState
    data object Loading : QuickSearchReferenceLoadState
    data object Ready : QuickSearchReferenceLoadState
    data class Error(val message: String) : QuickSearchReferenceLoadState
}

class QuickSearchViewModel(
    private val repository: MedicalRepository = MedicalRepository()
) : ViewModel() {
    private val _form = MutableStateFlow(QuickSearchForm())
    val form: StateFlow<QuickSearchForm> = _form.asStateFlow()

    private val _uiState = MutableStateFlow<QuickSearchUiState>(QuickSearchUiState.Idle)
    val uiState: StateFlow<QuickSearchUiState> = _uiState.asStateFlow()

    private val _validationError = MutableStateFlow<String?>(null)
    val validationError: StateFlow<String?> = _validationError.asStateFlow()

    private val _selectingScheduleId = MutableStateFlow<String?>(null)
    val selectingScheduleId: StateFlow<String?> = _selectingScheduleId.asStateFlow()

    private val _departments = MutableStateFlow<List<ReferenceDepartmentDto>>(emptyList())
    val departments: StateFlow<List<ReferenceDepartmentDto>> = _departments.asStateFlow()

    private val _departmentLoadState = MutableStateFlow<QuickSearchReferenceLoadState>(
        QuickSearchReferenceLoadState.Idle
    )
    val departmentLoadState: StateFlow<QuickSearchReferenceLoadState> = _departmentLoadState.asStateFlow()

    init {
        loadDepartments()
    }

    fun loadDepartments() {
        if (_departmentLoadState.value == QuickSearchReferenceLoadState.Loading) return
        viewModelScope.launch {
            _departmentLoadState.value = QuickSearchReferenceLoadState.Loading
            _departments.value = emptyList()
            try {
                val loaded = repository.referenceDepartments().distinctBy { it.deptId }
                _departments.value = loaded
                _departmentLoadState.value = if (loaded.isEmpty()) {
                    QuickSearchReferenceLoadState.Error("目前沒有可選擇的正式科別，請重新載入。")
                } else {
                    QuickSearchReferenceLoadState.Ready
                }
            } catch (_: Exception) {
                _departments.value = emptyList()
                clearDepartmentSelection()
                _departmentLoadState.value = QuickSearchReferenceLoadState.Error(
                    "無法載入正式科別，請重新載入。"
                )
            }
        }
    }

    fun selectDepartment(deptId: String) {
        val selected = _departments.value.firstOrNull { it.deptId == deptId }
        if (selected == null) {
            clearDepartmentSelection()
            _validationError.value = "請從正式科別清單選擇就診科別。"
            return
        }
        _form.value = _form.value.copy(
            department = selected.childDept,
            departmentId = selected.deptId,
            parentDepartment = selected.parentDept
        )
        clearResultAndError()
    }

    fun clearDepartmentSelection() {
        _form.value = _form.value.copy(
            department = "",
            departmentId = "",
            parentDepartment = ""
        )
        clearResultAndError()
    }

    fun updateAvailableDate(value: String) = updateForm { copy(availableDate = value) }

    fun selectPeriod(value: String) = updateForm {
        copy(period = value.takeIf { it in QUICK_SEARCH_PERIODS }.orEmpty())
    }

    fun submit() {
        if (_uiState.value == QuickSearchUiState.Loading) return
        val current = _form.value
        val normalizedDate = normalizeQuickSearchDate(current.availableDate)
        val departmentId = current.departmentId.toIntOrNull()
        val departmentIsValid = _departments.value.any {
            it.deptId == current.departmentId && it.childDept == current.department
        } && departmentId != null && departmentId > 0
        val error = quickSearchValidationError(current, normalizedDate, departmentIsValid)
        if (error != null) {
            _validationError.value = error
            return
        }

        viewModelScope.launch {
            _validationError.value = null
            _uiState.value = QuickSearchUiState.Loading
            try {
                repository.clearRecommendationFlow()
                repository.setActiveVisitType("quick_search")
                val result = repository.quickSearch(
                    QuickSearchRequest(
                        deptId = requireNotNull(departmentId),
                        date = requireNotNull(normalizedDate),
                        period = current.period
                    )
                )
                _uiState.value = QuickSearchUiState.Success(
                    caseId = result.caseId,
                    results = result.results
                )
            } catch (_: Exception) {
                _uiState.value = QuickSearchUiState.Error(
                    "無法查詢正式門診班表，請稍後再試。"
                )
            }
        }
    }

    fun selectScheduleAndNavigate(
        schedule: RecommendationItemDto,
        onNavigate: () -> Unit
    ) {
        if (_selectingScheduleId.value != null) return
        val state = _uiState.value as? QuickSearchUiState.Success ?: return
        val selectionError = quickSearchSelectionError(schedule)
        if (selectionError != null) {
            _uiState.value = QuickSearchUiState.Error(selectionError)
            return
        }

        viewModelScope.launch {
            _selectingScheduleId.value = schedule.recommendationId
            try {
                repository.setActiveCaseId(state.caseId)
                repository.selectRecommendation(schedule)
                val script = repository.generateScript(
                    caseId = state.caseId,
                    recommendationId = schedule.recommendationId,
                    recommendation = schedule
                )
                if (!script.isSuccess || script.steps.isEmpty()) {
                    _uiState.value = QuickSearchUiState.Error(
                        script.message ?: "後端未回傳可用的掛號導引步驟。"
                    )
                    return@launch
                }
                onNavigate()
            } catch (error: Exception) {
                _uiState.value = QuickSearchUiState.Error(
                    error.message ?: "無法產生掛號導引腳本，請稍後再試。"
                )
            } finally {
                _selectingScheduleId.value = null
            }
        }
    }

    private fun updateForm(transform: QuickSearchForm.() -> QuickSearchForm) {
        _form.value = _form.value.transform()
        clearResultAndError()
    }

    private fun clearResultAndError() {
        _validationError.value = null
        if (_uiState.value is QuickSearchUiState.Error || _uiState.value is QuickSearchUiState.Success) {
            _uiState.value = QuickSearchUiState.Idle
        }
    }
}

internal val QUICK_SEARCH_PERIODS = setOf("morning", "afternoon", "evening")

internal fun quickSearchSelectionError(schedule: RecommendationItemDto): String? = when {
    schedule.recommendationId.isBlank() -> "此班表缺少選擇識別資料，請重新查詢。"
    schedule.scheduleId.isNullOrBlank() -> "此班表缺少 schedule_id，請重新查詢。"
    schedule.doctorId.isNullOrBlank() -> "此班表缺少 doctor_id，請重新查詢。"
    schedule.deptId == null || schedule.deptId <= 0 -> "此班表缺少 dept_id，請重新查詢。"
    schedule.date.isBlank() -> "此班表缺少看診日期，請重新查詢。"
    schedule.session.isBlank() -> "此班表缺少看診時段，請重新查詢。"
    else -> null
}

internal fun quickSearchValidationError(
    form: QuickSearchForm,
    normalizedDate: String? = normalizeQuickSearchDate(form.availableDate),
    departmentIsValid: Boolean = form.department.isNotBlank() && form.departmentId.isNotBlank()
): String? = when {
    !departmentIsValid -> "請從正式科別清單選擇就診科別。"
    normalizedDate == null -> "請選擇有效的可看診日期。"
    form.period !in QUICK_SEARCH_PERIODS -> "請選擇上午診、下午診或夜診。"
    else -> null
}

internal fun normalizeQuickSearchDate(value: String): String? = runCatching {
    val parser = SimpleDateFormat("yyyy-MM-dd", Locale.TAIWAN).apply { isLenient = false }
    val parsed = parser.parse(value.replace('/', '-')) ?: return@runCatching null
    parser.format(parsed)
}.getOrNull()
