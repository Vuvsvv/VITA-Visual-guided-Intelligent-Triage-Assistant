package com.example.medicalaiguidance.model

data class DoctorProfile(
    val name: String,
    val photoUrl: String?,
    val education: List<String>,
    val currentPositions: List<String>,
    val experience: List<String>,
    val specialtyTags: List<String>,
    val titles: List<String>,
    val tid: Int?
)
