package com.example.medicalaiguidance.service

internal object BookingResultPolicy {
    fun isSuccess(texts: List<String>): Boolean {
        val labels = texts.map { it.replace(Regex("\\s+"), "").trimEnd('。', '！', '!') }
        if (labels.any { it.contains("掛號失敗") || it.contains("預約失敗") }) return false

        val hasSuccessMessage = labels.any {
            it in setOf("掛號成功", "預約掛號成功", "已成功掛號", "預約成功")
        }
        val hasReservationNumber = labels.withIndex().any { (index, text) ->
            Regex("^預約號碼[:：]?\\d+$").matches(text) ||
                (text.trimEnd(':', '：') == "預約號碼" &&
                    labels.getOrNull(index + 1)?.matches(Regex("\\d+")) == true)
        }
        return hasSuccessMessage && hasReservationNumber
    }
}
