package com.example.medicalaiguidance.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.medicalaiguidance.model.History
import com.example.medicalaiguidance.repository.MedicalRepository
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn

class HomeViewModel(
    private val repository: MedicalRepository = MedicalRepository()
) : ViewModel() {
    val recentHistory: StateFlow<List<History>> = repository.getAllHistory()
        .map { history -> history.take(2) }
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5_000),
            initialValue = emptyList()
        )
}
