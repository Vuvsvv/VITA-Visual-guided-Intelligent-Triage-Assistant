package com.example.medicalaiguidance.service

/** Cancellation results require explicit hospital evidence after final confirmation. */
internal object CancellationFlowPolicy {
    enum class Result { SUCCESS, FAILURE, UNKNOWN }

    private fun normalize(value: String): String =
        value.replace(Regex("\\s+"), "").trimEnd('。', '！', '!')

    fun isDataEntry(texts: List<String>, hasEditableField: Boolean): Boolean =
        hasEditableField || texts.any {
            val text = normalize(it)
            text.contains("請輸入身分證") ||
                text.contains("請輸入身份證") ||
                text.contains("請輸入驗證碼")
        }

    fun isRecordList(texts: List<String>): Boolean {
        val labels = texts.map(::normalize).filter(String::isNotBlank)
        return labels.withIndex().any { (index, text) ->
            Regex("預約號碼[:：]?\\d+").containsMatchIn(text) ||
                (text.trimEnd(':', '：') == "預約號碼" &&
                    labels.getOrNull(index + 1)?.matches(Regex("\\d+")) == true)
        }
    }

    fun isExplicitlyEmpty(texts: List<String>): Boolean {
        val labels = texts.map(::normalize)
        return "取消" !in labels && labels.any {
            it in setOf(
                "目前沒有可取消的預約",
                "目前沒有可取消掛號",
                "查無掛號紀錄",
                "查無預約紀錄"
            )
        }
    }

    fun isAbandonmentAction(text: String): Boolean = normalize(text) in
        setOf("放棄動作", "保留預約", "返回", "取消操作")

    fun result(texts: List<String>, confirmationSubmitted: Boolean): Result {
        if (!confirmationSubmitted) return Result.UNKNOWN
        val labels = texts.map(::normalize)
        if (labels.any { it in setOf("取消掛號失敗", "取消失敗", "取消預約失敗") }) {
            return Result.FAILURE
        }
        val cancelledExecution = Regex("^執行情形[:：]已取消$")
        if (labels.any(cancelledExecution::matches) ||
            cancelledExecution.matches(labels.joinToString(""))
        ) {
            return Result.SUCCESS
        }
        if (labels.any {
                it in setOf("取消掛號成功", "取消成功", "取消預約成功", "已成功取消掛號")
            }
        ) {
            return Result.SUCCESS
        }
        return Result.UNKNOWN
    }
}
