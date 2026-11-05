package com.example.medicalaiguidance

import com.example.medicalaiguidance.service.BookingResultPolicy
import com.example.medicalaiguidance.service.CancellationFlowPolicy
import com.example.medicalaiguidance.service.CancellationFlowPolicy.Result
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CancellationFlowPolicyUnitTest {
    @Test
    fun hospitalExecutionToastCompletesOnlyAfterConfirmation() {
        assertEquals(Result.SUCCESS, CancellationFlowPolicy.result(listOf("執行情形：已取消"), true))
        assertEquals(Result.SUCCESS, CancellationFlowPolicy.result(listOf("執行 情形 : 已取消"), true))
        assertEquals(Result.SUCCESS, CancellationFlowPolicy.result(listOf("執行情形：", "已取消"), true))
        assertEquals(Result.UNKNOWN, CancellationFlowPolicy.result(listOf("執行情形：已取消"), false))
        assertEquals(Result.UNKNOWN, CancellationFlowPolicy.result(listOf("已取消"), true))
        assertEquals(Result.UNKNOWN, CancellationFlowPolicy.result(listOf("執行情形：未取消"), true))
    }

    @Test
    fun formCancelButtonIsNotARecordList() {
        assertFalse(CancellationFlowPolicy.isRecordList(listOf("身分證號", "取消", "查詢")))
        assertFalse(CancellationFlowPolicy.isRecordList(listOf("預約號碼", "請輸入資料")))
        assertTrue(CancellationFlowPolicy.isDataEntry(listOf("已填入測試資料", "取消"), true))
        assertTrue(CancellationFlowPolicy.isDataEntry(listOf("請輸入身分證號"), false))
    }

    @Test
    fun loadedRecordsNeedAnActualReferenceNumber() {
        assertTrue(CancellationFlowPolicy.isRecordList(listOf("預約號碼：006", "取消")))
        assertTrue(CancellationFlowPolicy.isRecordList(listOf("預約號碼", "006", "已取消")))
        assertFalse(CancellationFlowPolicy.isDataEntry(listOf("預約號碼：006", "取消"), false))
    }

    @Test
    fun emptyViewportDoesNotProveNoBookings() {
        assertFalse(CancellationFlowPolicy.isExplicitlyEmpty(listOf("已取消")))
        assertFalse(CancellationFlowPolicy.isExplicitlyEmpty(emptyList()))
        assertTrue(CancellationFlowPolicy.isExplicitlyEmpty(listOf("查無掛號紀錄")))
        assertFalse(CancellationFlowPolicy.isExplicitlyEmpty(listOf("查無掛號紀錄", "取消")))
    }

    @Test
    fun abandonmentIsSeparateFromSubmission() {
        assertTrue(CancellationFlowPolicy.isAbandonmentAction("放棄動作"))
        assertTrue(CancellationFlowPolicy.isAbandonmentAction("返回"))
        assertFalse(CancellationFlowPolicy.isAbandonmentAction("確認取消掛號"))
        assertFalse(CancellationFlowPolicy.isAbandonmentAction("取消"))
    }

    @Test
    fun openingOrDismissingDialogCannotReportSuccess() {
        assertEquals(Result.UNKNOWN, CancellationFlowPolicy.result(listOf("取消掛號成功"), false))
        assertEquals(Result.UNKNOWN, CancellationFlowPolicy.result(listOf("確認取消掛號", "已取消"), true))
    }

    @Test
    fun exactResultsAfterConfirmationAndFailureWins() {
        assertEquals(Result.SUCCESS, CancellationFlowPolicy.result(listOf("取消掛號成功。"), true))
        assertEquals(
            Result.FAILURE,
            CancellationFlowPolicy.result(listOf("取消掛號失敗", "取消掛號成功"), true)
        )
        assertEquals(
            Result.UNKNOWN,
            CancellationFlowPolicy.result(listOf("若取消掛號成功可重新預約"), true)
        )
    }

    @Test
    fun bookingTitleAndNumberAreNotSuccess() {
        assertFalse(
            BookingResultPolicy.isSuccess(listOf("臺北榮總預約掛號", "預約號碼：006", "加入行事曆"))
        )
        assertFalse(BookingResultPolicy.isSuccess(listOf("掛號成功")))
    }

    @Test
    fun bookingNeedsExplicitSuccessAndNumber() {
        assertTrue(BookingResultPolicy.isSuccess(listOf("掛號成功", "預約號碼：006")))
        assertTrue(BookingResultPolicy.isSuccess(listOf("掛號成功。", "預約號碼", "006")))
        assertFalse(BookingResultPolicy.isSuccess(listOf("掛號成功", "預約號碼", "取消")))
        assertFalse(BookingResultPolicy.isSuccess(listOf("掛號失敗", "掛號成功", "預約號碼006")))
    }
}
