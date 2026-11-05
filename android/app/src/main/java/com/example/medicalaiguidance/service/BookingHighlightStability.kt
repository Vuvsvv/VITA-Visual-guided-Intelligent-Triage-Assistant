package com.example.medicalaiguidance.service

internal data class HighlightBounds(val left: Int, val top: Int, val right: Int, val bottom: Int)

internal object BookingHighlightStability {
    fun requiresRecheck(keyword: String, departmentStep: Boolean): Boolean = departmentStep ||
        keyword in setOf("選擇看診時間", "選擇看診時間 / 醫師", "填寫個人資料", "確認送出")
    fun canDisplay(previous: HighlightBounds, current: HighlightBounds, screenWidth: Int, tolerance: Int): Boolean {
        if (screenWidth <= 0 || current.right <= current.left || current.bottom <= current.top) return false
        if (current.left < -tolerance || current.right > screenWidth + tolerance) return false
        return kotlin.math.abs(previous.left - current.left) <= tolerance &&
            kotlin.math.abs(previous.top - current.top) <= tolerance &&
            kotlin.math.abs(previous.right - current.right) <= tolerance &&
            kotlin.math.abs(previous.bottom - current.bottom) <= tolerance
    }
}
