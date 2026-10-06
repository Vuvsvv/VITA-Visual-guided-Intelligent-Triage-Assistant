package com.example.medicalaiguidance.service

import android.content.res.Resources
import android.graphics.Rect

/** 取消掛號紅框導引的專屬步驟與按鈕定位規則。 */
internal object CancellationGuidance {
    val script = listOf("掛號查詢", "請輸入身分證號", "取消")

    fun isDataEntryStep(keyword: String): Boolean = keyword == "請輸入身分證號"

    fun isConfirmationAction(text: CharSequence?): Boolean =
        text?.toString()?.replace(Regex("\\s+"), "") == "確認取消掛號"

    fun isAppointmentList(nodes: List<NodeData>): Boolean = CancellationFlowPolicy.isRecordList(
        nodes.filter { it.isVisibleToUser && it.isOnScreen() }
            .flatMap { listOf(it.text, it.contentDescription.orEmpty()) }
    )

    fun findTarget(nodes: List<NodeData>, keyword: String): NodeData? {
        val visibleNodes = nodes.filter { it.isVisibleToUser && it.isOnScreen() }
        return when (keyword) {
            "掛號查詢" -> visibleNodes.firstOrNull { hasMatchText(it, keyword) }
            "請輸入身分證號" -> visibleNodes.firstOrNull {
                hasMatchText(it, keyword) || hasMatchText(it, "身分證號")
            }
            "取消" -> findCancellationAction(nodes)
                .takeIf { it.status != CancellationMatchStatus.NONE_CANCELLABLE }
                ?.demonstrationSourceIndex
                ?.let(nodes::getOrNull)
            else -> null
        }
    }

    fun findCancellationAction(nodes: List<NodeData>): CancellationMatchResult {
        val visibleNodes = nodes.withIndex().filter {
            it.value.isVisibleToUser && it.value.isOnScreen()
        }
        val snapshots = visibleNodes.map { (index, node) ->
            CancellationNodeSnapshot(
                sourceIndex = index,
                text = node.text,
                contentDescription = node.contentDescription,
                left = node.rect.left,
                top = node.rect.top,
                right = node.rect.right,
                bottom = node.rect.bottom,
                isEnabled = node.isEnabled,
                isClickable = node.isClickable,
                clickableAncestorSourceIndex = node.clickableAncestorIndex
            )
        }
        return CancellationAppointmentMatcher.match(snapshots)
    }

    private fun hasMatchText(node: NodeData, keyword: String): Boolean {
        val textNoSpace = node.text.replace(" ", "")
        val descNoSpace = (node.contentDescription ?: "").replace(" ", "")
        if (textNoSpace.isBlank() && descNoSpace.isBlank()) return false
        return textNoSpace.contains(keyword) || descNoSpace.contains(keyword)
    }

}

internal fun NodeData.isOnScreen(): Boolean {
    val screenWidth = Resources.getSystem().displayMetrics.widthPixels
    val screenHeight = Resources.getSystem().displayMetrics.heightPixels
    return rect.width() > 0 && rect.height() > 0 &&
            rect.right > 0 && rect.left < screenWidth &&
            rect.bottom > 0 && rect.top < screenHeight
}
