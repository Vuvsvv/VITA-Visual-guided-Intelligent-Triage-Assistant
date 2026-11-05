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
    val room: String? = null,
    val category: String? = null
)

internal fun HistoryRecommendation.recommendationCategory(): String =
    category?.takeIf { it in setOf("specialty", "time") }
        ?: when {
            recommendationId.startsWith("rec_s_") -> "specialty"
            recommendationId.startsWith("rec_t_") -> "time"
            else -> "legacy"
        }

internal fun String.visibleAssistantText(): String =
    replace("已確認分診結果，可呼叫 /recommend 取得推薦掛號方案。", "")
        .replace("請確認後取得推薦掛號方案。", "")
        .trim()

internal fun String.historyDisplayDate(): String {
    val match = Regex("(\\d{4})[-/](\\d{1,2})[-/](\\d{1,2})").find(this) ?: return this
    return "${match.groupValues[1]}/${match.groupValues[2].padStart(2, '0')}/${match.groupValues[3].padStart(2, '0')}"
}

internal fun HistoryRecommendation.historyDisplaySession(): String {
    val raw = session.trim().lowercase()
    return when {
        raw.contains("上午") || raw.contains("早") || raw == "morning" -> "上午診"
        raw.contains("下午") || raw.contains("午") || raw == "afternoon" -> "下午診"
        raw.contains("晚") || raw.contains("夜") || raw == "evening" -> "夜診"
        else -> {
            val hour = Regex("(\\d{1,2}):\\d{2}").find(sessionTime.orEmpty().ifBlank { raw })
                ?.groupValues?.get(1)?.toIntOrNull()
            when (hour) {
                in 0..11 -> "上午診"
                in 12..17 -> "下午診"
                in 18..23 -> "夜診"
                else -> "未保存時段"
            }
        }
    }
}

internal fun resolveHistorySummary(summary: String, messages: List<ChatMessage>): String =
    summary.visibleAssistantText().takeIf { it.isNotBlank() }
        ?: messages.asReversed().asSequence()
            .filter { it.sender == MessageSender.AI }
            .map { it.content.visibleAssistantText() }
            .firstOrNull { it.isNotBlank() }
        ?: "問診紀錄"

internal fun History.hasUrgentSymptoms(): Boolean =
    (listOf(summaryText) + chatMessages.filter { it.sender == MessageSender.AI }.map { it.content })
        .any { it.contains("119") && (it.contains("急診") || it.contains("情況危急")) }

/** Words in the backend's SAFETY_UNRESOLVED_REPLY; keep in sync with conversation_service.py. */
internal const val SAFETY_UNRESOLVED_MARKER = "還沒能確認您是否有急迫症狀"
/** The earlier SAFETY_UNRESOLVED_REPLY wording, still in saved histories. */
private const val LEGACY_SAFETY_UNRESOLVED_MARKER = "無法可靠確認急迫症狀"

internal fun String.isSafetyUnresolvedReply(): Boolean =
    contains(SAFETY_UNRESOLVED_MARKER) || contains(LEGACY_SAFETY_UNRESOLVED_MARKER)

internal fun History.hasUnresolvedSafety(): Boolean =
    (listOf(summaryText) + chatMessages.filter { it.sender == MessageSender.AI }.map { it.content })
        .any { it.isSafetyUnresolvedReply() }
