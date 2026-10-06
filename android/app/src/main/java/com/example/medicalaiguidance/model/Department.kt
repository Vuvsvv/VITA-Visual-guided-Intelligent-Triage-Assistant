package com.example.medicalaiguidance.model

data class Department(
    val id: String,
    val name: String,         // 大類，例如：外科系
    val clinicName: String    // 細目，例如：骨科部、關節重建
)