package com.example.medicalaiguidance.model


data class Appointment(
    val id: String,
    val date: String,         // 例如：2026年5月19日
    val dayOfWeek: String,    // 例如：(二)
    val timeSlot: String,     // 上午 / 下午 / 晚上
    val department: Department,
    val doctor: Doctor
)