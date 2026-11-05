package com.example.medicalaiguidance.model

data class Doctor(
    val id: String,
    val name: String,
    val departmentId: String,
    val title: String,
    val specialties: List<String> = emptyList(),
    val imageUrl: String? = null,
    // 新增門診班表欄位：Pair(星期幾, 時段)
    val availableSlots: List<Pair<String, String>> = emptyList()
)