package com.example.medicalaiguidance.service

internal data class CancellationNodeSnapshot(
    val sourceIndex: Int,
    val text: String,
    val contentDescription: String? = null,
    val left: Int = 0,
    val top: Int = 0,
    val right: Int = 0,
    val bottom: Int = 0,
    val isEnabled: Boolean = true,
    val isClickable: Boolean = false,
    val clickableAncestorSourceIndex: Int? = null
)

internal enum class CancellationMatchStatus {
    SINGLE_CANCELLABLE,
    MULTIPLE_CANCELLABLE,
    NONE_CANCELLABLE
}

internal data class CancellationMatchResult(
    val status: CancellationMatchStatus,
    val sourceIndices: List<Int> = emptyList()
) {
    val demonstrationSourceIndex: Int? get() = sourceIndices.firstOrNull()
}

/** Pure matching logic; Android nodes are reduced to snapshots before entering here. */
internal object CancellationAppointmentMatcher {
    private const val CANCEL = "取消"
    private val excludedLabels = setOf("已取消", "取消掛號成功")

    fun match(nodes: List<CancellationNodeSnapshot>): CancellationMatchResult {
        val actionableIndices = nodes
            .filter { node -> node.isEnabled && node.hasExactCancelLabel() }
            .mapNotNull { label -> label.clickTarget(nodes)?.sourceIndex }
            .distinct()

        val status = when (actionableIndices.size) {
            0 -> CancellationMatchStatus.NONE_CANCELLABLE
            1 -> CancellationMatchStatus.SINGLE_CANCELLABLE
            else -> CancellationMatchStatus.MULTIPLE_CANCELLABLE
        }
        return CancellationMatchResult(status, actionableIndices)
    }

    private fun CancellationNodeSnapshot.clickTarget(
        nodes: List<CancellationNodeSnapshot>
    ): CancellationNodeSnapshot? {
        if (hasExcludedLabel()) return null
        if (isClickable && isEnabled) return this

        // Some WebView layouts expose the exact label as a child and the click action on
        // its parent. Traversal records the actual nearest clickable ancestor.
        val clickableAncestor = clickableAncestorSourceIndex
            ?.let { ancestorIndex -> nodes.firstOrNull { it.sourceIndex == ancestorIndex } }
            ?.takeIf { ancestor ->
                ancestor.isEnabled && ancestor.isClickable && !ancestor.hasExcludedLabel()
            }
        return clickableAncestor ?: this
    }

    private fun CancellationNodeSnapshot.hasExactCancelLabel(): Boolean =
        normalizedValues().any { it == CANCEL }

    private fun CancellationNodeSnapshot.hasExcludedLabel(): Boolean =
        normalizedValues().any(excludedLabels::contains)

    private fun CancellationNodeSnapshot.normalizedValues(): List<String> =
        listOf(text.normalized(), contentDescription.orEmpty().normalized())

    private fun String.normalized(): String =
        replace(Regex("\\s+"), "").trim()
}
