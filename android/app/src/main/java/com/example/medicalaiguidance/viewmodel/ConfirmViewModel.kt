package com.example.medicalaiguidance.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.launch
import com.example.medicalaiguidance.model.Appointment
import com.example.medicalaiguidance.model.VisitPlan
import com.example.medicalaiguidance.network.ScriptResponseDto
import com.example.medicalaiguidance.repository.MedicalRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class ConfirmViewModel(
    private val repository: MedicalRepository = MedicalRepository()
) : ViewModel() {
    private var launchingHospital = false

    fun launchHospitalAfterVoiceCleanup(launch: () -> Unit) {
        if (launchingHospital) return
        launchingHospital = true
        viewModelScope.launch {
            try {
                repository.cleanupTtsBeforeHospitalLaunch()
                launch()
            } finally {
                launchingHospital = false
            }
        }
    }
    private val _appointmentInfo = MutableStateFlow<Appointment?>(null)
    val appointmentInfo: StateFlow<Appointment?> = _appointmentInfo.asStateFlow()

    private val _visitPlan = MutableStateFlow(VisitPlan.UNKNOWN)
    val visitPlan: StateFlow<VisitPlan> = _visitPlan.asStateFlow()

    private val _scriptInfo = MutableStateFlow<ScriptResponseDto?>(null)
    val scriptInfo: StateFlow<ScriptResponseDto?> = _scriptInfo.asStateFlow()

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    init {
        loadConfirmedAppointment()
    }

    fun loadConfirmedAppointment() {
        runCatching {
            repository.getConfirmedAppointment()
        }.onSuccess { appointment ->
            _appointmentInfo.value = appointment
            _visitPlan.value = VisitPlan.fromApiValue(repository.getActiveVisitType())
            _scriptInfo.value = repository.getCurrentGuidanceScript()
            _errorMessage.value = null
        }.onFailure { error ->
            _appointmentInfo.value = null
            _visitPlan.value = VisitPlan.UNKNOWN
            _scriptInfo.value = null
            _errorMessage.value = error.message ?: "尚未取得後端推薦結果，請先完成推薦流程。"
        }
    }
}
