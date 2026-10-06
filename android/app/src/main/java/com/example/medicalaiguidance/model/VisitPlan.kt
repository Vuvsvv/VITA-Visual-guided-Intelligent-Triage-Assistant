package com.example.medicalaiguidance.model

enum class VisitPlan(
    val routeValue: String,
    val apiValue: String,
    val displayName: String,
) {
    INITIAL(routeValue = "initial", apiValue = "initial", displayName = "初診"),
    FOLLOW_UP(routeValue = "followup", apiValue = "followup", displayName = "複診"),
    QUICK_SEARCH(routeValue = "quick_search", apiValue = "quick_search", displayName = "快速查詢"),
    RETURN_VISIT(routeValue = "return_visit", apiValue = "return_visit", displayName = "回診"),
    UNKNOWN(routeValue = "unknown", apiValue = "unknown", displayName = "未記錄");

    companion object {
        fun fromRoute(value: String?): VisitPlan = entries.firstOrNull {
            it.routeValue == value
        } ?: UNKNOWN

        fun fromApiValue(value: String?): VisitPlan = entries.firstOrNull {
            it.apiValue == value
        } ?: UNKNOWN
    }
}
