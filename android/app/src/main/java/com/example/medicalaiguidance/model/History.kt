package com.example.medicalaiguidance.model

enum class HistoryStatus {
    ALL,        // 全部
    COMPLETED,  // 已完成
    UNCOMPLETED // 未完成（評估中）
}

data class History(
    val id: String,
    val date: String,
    val typeTitle: String,
    val summaryText: String,
    val status: HistoryStatus,
    val chatMessages: List<ChatMessage> = emptyList(),
    val completedAt: String? = null,
    val recommendations: List<HistoryRecommendation> = emptyList(),
    val selectedRecommendationId: String? = null,
    val visitType: String? = null
)

data class HistoryRecommendation(
    val recommendationId: String,
    val parentDepartment: String,
    val department: String,
    val doctor: String,
    val date: String,
    val session: String,
    val sessionTime: String? = null,
    val room: String? = null
)
