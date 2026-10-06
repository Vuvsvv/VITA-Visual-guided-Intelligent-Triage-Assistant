package com.example.medicalaiguidance

import com.example.medicalaiguidance.service.CancellationAppointmentMatcher
import com.example.medicalaiguidance.service.CancellationMatchStatus
import com.example.medicalaiguidance.service.CancellationNodeSnapshot
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CancellationAppointmentMatcherUnitTest {
    @Test
    fun exactCancelAndAlreadyCancelledMatchesOnlyExactCancel() {
        val result = CancellationAppointmentMatcher.match(
            listOf(
                node(1, "取消", top = 10, clickable = true),
                node(2, "已取消", top = 50, clickable = true)
            )
        )

        assertEquals(CancellationMatchStatus.SINGLE_CANCELLABLE, result.status)
        assertEquals(listOf(1), result.sourceIndices)
    }

    @Test
    fun onlyAlreadyCancelledItemsMeansNoCancellableAppointment() {
        val result = CancellationAppointmentMatcher.match(
            listOf(
                node(1, "已取消", top = 10, clickable = true),
                node(2, "已取消", top = 50, clickable = true)
            )
        )

        assertEquals(CancellationMatchStatus.NONE_CANCELLABLE, result.status)
        assertTrue(result.sourceIndices.isEmpty())
    }

    @Test
    fun twoExactCancelButtonsAreReportedAsMultiple() {
        val result = CancellationAppointmentMatcher.match(
            listOf(
                node(1, "取消", top = 10, clickable = true),
                node(2, "取消", top = 50, clickable = true)
            )
        )

        assertEquals(CancellationMatchStatus.MULTIPLE_CANCELLABLE, result.status)
        assertEquals(listOf(1, 2), result.sourceIndices)
        assertEquals(1, result.demonstrationSourceIndex)
    }

    @Test
    fun multipleCancellableItemsNeverIncludeAlreadyCancelledNode() {
        val result = CancellationAppointmentMatcher.match(
            listOf(
                node(1, "取消", top = 10, clickable = true),
                node(2, "已取消", top = 50, clickable = true),
                node(3, "取消掛號成功", top = 90, clickable = true),
                node(4, "取消", top = 130, clickable = true)
            )
        )

        assertEquals(CancellationMatchStatus.MULTIPLE_CANCELLABLE, result.status)
        assertEquals(listOf(1, 4), result.sourceIndices)
    }

    @Test
    fun rescanningAfterScrollCanFindExactCancel() {
        val beforeScroll = CancellationAppointmentMatcher.match(
            listOf(node(1, "已取消", top = 10, clickable = true))
        )
        val afterScroll = CancellationAppointmentMatcher.match(
            listOf(node(5, "取消", top = 10, clickable = true))
        )

        assertEquals(CancellationMatchStatus.NONE_CANCELLABLE, beforeScroll.status)
        assertEquals(CancellationMatchStatus.SINGLE_CANCELLABLE, afterScroll.status)
        assertEquals(listOf(5), afterScroll.sourceIndices)
    }

    @Test
    fun exactLabelMayUseNearestEnabledClickableParent() {
        val result = CancellationAppointmentMatcher.match(
            listOf(
                node(1, "", top = 0, bottom = 60, clickable = true),
                node(
                    2,
                    "取消",
                    top = 15,
                    bottom = 45,
                    clickable = false,
                    clickableAncestorIndex = 1
                )
            )
        )

        assertEquals(CancellationMatchStatus.SINGLE_CANCELLABLE, result.status)
        assertEquals(listOf(1), result.sourceIndices)
    }

    @Test
    fun disabledExactCancelIsNotActionable() {
        val result = CancellationAppointmentMatcher.match(
            listOf(node(1, "取消", top = 10, clickable = true, enabled = false))
        )

        assertEquals(CancellationMatchStatus.NONE_CANCELLABLE, result.status)
    }

    @Test
    fun exactEnabledLabelWithoutClickableAncestorUsesItsOwnBounds() {
        val result = CancellationAppointmentMatcher.match(
            listOf(node(9, "取消", top = 10, clickable = false))
        )

        assertEquals(CancellationMatchStatus.SINGLE_CANCELLABLE, result.status)
        assertEquals(listOf(9), result.sourceIndices)
    }

    @Test
    fun exactContentDescriptionIsAccepted() {
        val result = CancellationAppointmentMatcher.match(
            listOf(
                CancellationNodeSnapshot(
                    sourceIndex = 8,
                    text = "",
                    contentDescription = " 取 消 ",
                    left = 10,
                    top = 10,
                    right = 210,
                    bottom = 40,
                    isEnabled = true,
                    isClickable = true
                )
            )
        )

        assertEquals(CancellationMatchStatus.SINGLE_CANCELLABLE, result.status)
        assertEquals(listOf(8), result.sourceIndices)
    }

    @Test
    fun textContainingCancelIsNotAnExactCancelButton() {
        val result = CancellationAppointmentMatcher.match(
            listOf(node(1, "取消預約", top = 10, clickable = true))
        )

        assertEquals(CancellationMatchStatus.NONE_CANCELLABLE, result.status)
    }

    private fun node(
        index: Int,
        text: String,
        top: Int,
        bottom: Int = top + 30,
        clickable: Boolean,
        enabled: Boolean = true,
        clickableAncestorIndex: Int? = null
    ) = CancellationNodeSnapshot(
        sourceIndex = index,
        text = text,
        left = 10,
        top = top,
        right = 210,
        bottom = bottom,
        isEnabled = enabled,
        isClickable = clickable,
        clickableAncestorSourceIndex = clickableAncestorIndex
    )
}
