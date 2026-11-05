package com.example.medicalaiguidance

import com.example.medicalaiguidance.screen.formatConfirmVisitTime
import org.junit.Assert.assertEquals
import org.junit.Test

class ConfirmVisitTimeFormatUnitTest {
    @Test
    fun visitTimeOnlyWrapsBetweenDateWeekdayAndSession() {
        val text = formatConfirmVisitTime(date = "2026-10-06", dayOfWeek = "2026-10-06", timeSlot = "08:30-12:00")

        assertEquals("2026/10/06 (二) 上午診", text.replace("⁠", ""))
        assertEquals(
            listOf("2026/10/06", "(二)", "上午診"),
            text.split(" ").map { it.replace("⁠", "") }
        )
        // Every character inside a segment is joined, so "上午診" cannot be split across lines.
        assertEquals("上⁠午⁠診", text.split(" ").last())
    }
}
