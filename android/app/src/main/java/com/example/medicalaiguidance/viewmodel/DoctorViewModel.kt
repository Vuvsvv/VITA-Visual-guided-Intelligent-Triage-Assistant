package com.example.medicalaiguidance.viewmodel

import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.medicalaiguidance.network.FallbackDepartmentDto
import com.example.medicalaiguidance.network.MedicalApiException
import com.example.medicalaiguidance.network.RecommendationItemDto
import com.example.medicalaiguidance.repository.MedicalRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed interface DoctorUiState {
    object Loading : DoctorUiState

    data class Success(
        val caseId: String,
        val specialtyFirst: List<RecommendationItemDto>,
        val timeFirst: List<RecommendationItemDto>,
        val fallbackDepartments: List<FallbackDepartmentDto> = emptyList(),
        val totalCount: Int = 0
    ) : DoctorUiState

    data class NoSlots(val message: String) : DoctorUiState
    data class Error(val message: String) : DoctorUiState
}

class DoctorViewModel(
    private val repository: MedicalRepository = MedicalRepository()
) : ViewModel() {
    companion object {
        private const val TAG = "DoctorViewModel"
    }

    private val _uiState = MutableStateFlow<DoctorUiState>(DoctorUiState.Loading)
    val uiState: StateFlow<DoctorUiState> = _uiState.asStateFlow()

    private val _selectedRecommendation = MutableStateFlow<RecommendationItemDto?>(null)
    val selectedRecommendation: StateFlow<RecommendationItemDto?> = _selectedRecommendation.asStateFlow()

    private val _showBottomSheet = MutableStateFlow(false)
    val showBottomSheet: StateFlow<Boolean> = _showBottomSheet.asStateFlow()

    private val _selectingRecommendationId = MutableStateFlow<String?>(null)
    val selectingRecommendationId: StateFlow<String?> = _selectingRecommendationId.asStateFlow()

    fun fetchRecommendations(visitType: String) {
        val caseId = repository.getActiveCaseId()
        if (caseId.isNullOrBlank()) {
            Log.w(TAG, "Skip /recommend: active case_id is missing")
            _uiState.value = DoctorUiState.Error("尚未取得已確認的問診案件，請先回到問診流程完成確認。")
            return
        }

        viewModelScope.launch {
            _uiState.value = DoctorUiState.Loading
            try {
                Log.d(TAG, "Calling /recommend case_id=$caseId visit_type=$visitType")
                val result = repository.recommend(caseId, visitType)
                val specialtyFirst = result.recommendations.specialtyFirst
                val timeFirst = result.recommendations.timeFirst
                val allItems = specialtyFirst + timeFirst
                val first = allItems.firstOrNull()
                Log.d(
                    TAG,
                    "/recommend returned case_id=${result.caseId} total_count=${result.totalCount} " +
                        "items=${allItems.size} first_doctor=${first?.doctor.orEmpty()} " +
                        "first_schedule_id=${first?.scheduleId.orEmpty()} first_date=${first?.date.orEmpty()} " +
                        "first_room=${first?.room.orEmpty()}"
                )
                logFirstRecommendation("specialtyFirst", specialtyFirst)
                logFirstRecommendation("timeFirst", timeFirst)
                Log.d(
                    TAG,
                    "/recommend priority_lists_same_order=${sameOrder(specialtyFirst, timeFirst)} " +
                        "specialty_count=${specialtyFirst.size} time_count=${timeFirst.size}"
                )
                _uiState.value = DoctorUiState.Success(
                    caseId = result.caseId,
                    specialtyFirst = specialtyFirst,
                    timeFirst = timeFirst,
                    fallbackDepartments = result.fallbackDepartments,
                    totalCount = result.totalCount
                )
            } catch (error: Exception) {
                Log.e(TAG, "/recommend failed case_id=$caseId", error)
                val noSlotsMessage = intentionalNoSlotsMessage(error)
                _uiState.value = if (noSlotsMessage != null) {
                    DoctorUiState.NoSlots(noSlotsMessage)
                } else {
                    DoctorUiState.Error(error.message ?: "無法載入後端推薦結果，請稍後再試。")
                }
            }
        }
    }

    private fun logFirstRecommendation(label: String, items: List<RecommendationItemDto>) {
        val first = items.firstOrNull()
        Log.d(
            TAG,
            "$label first doctor=${first?.doctor.orEmpty()} schedule_id=${first?.scheduleId.orEmpty()} " +
                "date=${first?.date.orEmpty()} time_score=${first?.timeScore} total_score=${first?.score}"
        )
    }

    private fun sameOrder(
        specialtyFirst: List<RecommendationItemDto>,
        timeFirst: List<RecommendationItemDto>
    ): Boolean {
        val specialtyIds = specialtyFirst.map { it.scheduleId ?: it.recommendationId }
        val timeIds = timeFirst.map { it.scheduleId ?: it.recommendationId }
        return specialtyIds == timeIds
    }

    fun onRecommendationDetailClick(item: RecommendationItemDto) {
        _selectedRecommendation.value = item
        _showBottomSheet.value = true
    }

    fun dismissBottomSheet() {
        _showBottomSheet.value = false
        _selectedRecommendation.value = null
    }

    fun selectRecommendationAndNavigate(item: RecommendationItemDto, onNavigate: () -> Unit) {
        val caseId = repository.getActiveCaseId()
        if (caseId.isNullOrBlank()) {
            _uiState.value = DoctorUiState.Error("尚未取得 case_id，無法產生掛號導引腳本。")
            return
        }

        viewModelScope.launch {
            _selectingRecommendationId.value = item.recommendationId
            try {
                repository.selectRecommendation(item)
                val script = repository.generateScript(caseId, item.recommendationId)
                if (!script.isSuccess || script.steps.isEmpty()) {
                    _uiState.value = DoctorUiState.Error(script.message ?: "後端未回傳可用的掛號導引步驟。")
                    return@launch
                }
                onNavigate()
            } catch (error: Exception) {
                _uiState.value = DoctorUiState.Error(error.message ?: "無法產生掛號導引腳本，請稍後再試。")
            } finally {
                _selectingRecommendationId.value = null
            }
        }
    }
}

internal fun intentionalNoSlotsMessage(error: Exception): String? {
    val apiError = error as? MedicalApiException ?: return null
    if (apiError.statusCode != 503) return null
    return apiError.detail
        ?.trim()
        ?.takeIf { it.startsWith("目前找不到") && it.endsWith("班表。") }
}
