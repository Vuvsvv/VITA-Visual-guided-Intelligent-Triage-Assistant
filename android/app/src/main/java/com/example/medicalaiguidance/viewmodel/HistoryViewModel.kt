package com.example.medicalaiguidance.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.medicalaiguidance.model.History
import com.example.medicalaiguidance.model.HistoryStatus
import com.example.medicalaiguidance.repository.MedicalRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn

sealed interface HistoryUiState {
    object Loading : HistoryUiState
    data class Success(val filteredHistory: List<History>) : HistoryUiState
    data class Error(val message: String) : HistoryUiState
}

class HistoryViewModel(
    private val repository: MedicalRepository = MedicalRepository()
) : ViewModel() {
    private val _selectedTab = MutableStateFlow(0)
    val selectedTab: StateFlow<Int> = _selectedTab.asStateFlow()

    val uiState: StateFlow<HistoryUiState> = combine(
        repository.getAllHistory(),
        _selectedTab,
        repository.getHistoryLoadError()
    ) { history, selectedTab, loadError ->
        if (loadError != null) {
            HistoryUiState.Error(loadError)
        } else {
            val filtered = when (selectedTab) {
                1 -> history.filter { it.status == HistoryStatus.COMPLETED }
                2 -> history.filter { it.status == HistoryStatus.UNCOMPLETED }
                else -> history
            }
            HistoryUiState.Success(filtered)
        }
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5_000),
        initialValue = HistoryUiState.Loading
    )

    fun onTabSelected(index: Int) {
        _selectedTab.value = index.coerceIn(0, 2)
    }

    fun deleteHistory(id: String) {
        repository.deleteHistory(id)
    }
}
