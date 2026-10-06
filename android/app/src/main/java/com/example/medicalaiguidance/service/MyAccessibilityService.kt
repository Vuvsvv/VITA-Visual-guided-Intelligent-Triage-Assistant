package com.example.medicalaiguidance.service

import android.accessibilityservice.AccessibilityService
import android.graphics.Rect
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.util.Log
import com.example.medicalaiguidance.model.VisitPlan

enum class AppointmentType {
    INITIAL,
    RETURN_VISIT
}

internal fun VisitPlan.toAccessibilityAppointmentTypeOrNull(): AppointmentType? =
    when (this) {
        VisitPlan.INITIAL -> AppointmentType.INITIAL
        VisitPlan.FOLLOW_UP,
        VisitPlan.QUICK_SEARCH,
        VisitPlan.RETURN_VISIT -> AppointmentType.RETURN_VISIT
        VisitPlan.UNKNOWN -> null
    }

class MyAccessibilityService : AccessibilityService() {
    companion object {
        private const val VGH_PACKAGE_NAME = "tw.com.bicom.VGHTPE"
        private const val TARGET_APP_EXIT_GRACE_MS = 4_000L
        private const val RESULT_POLL_INTERVAL_MS = 350L
        private const val RESULT_WAIT_TIMEOUT_MS = 15_000L
        private const val COMPLETION_MESSAGE_DURATION_MS = 8_000L
        private const val MULTIPLE_CANCELLABLE_MESSAGE =
            "目前有多筆可取消預約，請點選您欲取消預約旁的『取消』按鈕。"
        private const val NO_CANCELLABLE_MESSAGE = "目前沒有可取消的預約。"
        private const val SCROLL_HINT_GRACE_MS = 800L
        private const val STABLE_HIGHLIGHT_DELAY_MS = 180L
        private const val STABLE_RECT_TOLERANCE_DP = 10
        private var pendingDepartment: String? = null
        private var pendingClinic: String? = null
        private var pendingDoctor: String? = null
        private var pendingDate: String? = null
        private var pendingTimeSlot: String? = null
        private var pendingAppointmentType = AppointmentType.INITIAL
        private var pendingGuidanceType: GuidanceType? = null
        private var shouldUpdateScript = false
        private var forceReset = false

        fun updateTarget(
            department: String,
            clinic: String,
            doctor: String,
            date: String,
            timeSlot: String = "",
            appointmentType: AppointmentType
        ) {
            pendingDepartment = department
            pendingClinic = clinic
            pendingDoctor = doctor
            pendingDate = date
            pendingTimeSlot = timeSlot
            pendingAppointmentType = appointmentType
            pendingGuidanceType = GuidanceType.BOOKING
            shouldUpdateScript = true
        }

        fun startCancellationGuidance() {
            pendingDepartment = null
            pendingClinic = null
            pendingDoctor = null
            pendingDate = null
            pendingTimeSlot = null
            pendingAppointmentType = AppointmentType.INITIAL
            pendingGuidanceType = GuidanceType.CANCELLATION
            shouldUpdateScript = true
        }

        private fun clearPendingGuidanceRequest() {
            pendingDepartment = null
            pendingClinic = null
            pendingDoctor = null
            pendingDate = null
            pendingTimeSlot = null
            pendingAppointmentType = AppointmentType.INITIAL
            pendingGuidanceType = null
            shouldUpdateScript = false
            forceReset = false
        }
    }

    private var overlay: OverlayManager? = null
    private val guidanceSession = GuidanceSessionState()
    private var currentStepIndex: Int
        get() = guidanceSession.currentStepIndex
        set(value) {
            guidanceSession.currentStepIndex = value
        }
    private val script: List<String>
        get() = guidanceSession.steps
    private var isDialogOpen = false
    private var lastHighlightAt = 0L
    private var lastScrollHintAt = 0L
    private var waitingForPersonalDataExit = false
    private var stepEnteredAt = 0L
    private var trackedStepIndex = 0
    private val mainHandler = Handler(Looper.getMainLooper())
    private var pendingStableHighlight: Runnable? = null
    private var pendingStableKeyword: String? = null
    private var pendingStableRect: Rect? = null
    private var displayedStableRect: Rect? = null
    private var displayedStableKeyword: String? = null

    private var pendingScrollHintRecheck: Runnable? = null
    private var pendingExitAbort: Runnable? = null
    private var completionMessageUntil = 0L
    private var awaitingBookingResult = false
    private var cancellationListReached = false
    private var cancellationConfirmationVisible = false
    private var awaitingCancellationResult = false
    private var cancellationResultTimeout: Runnable? = null
    private var lastCancellationMessage: String? = null
    private val calendar = java.util.Calendar.getInstance()
    private val rocYear = calendar.get(java.util.Calendar.YEAR) - 1911
    private val month = calendar.get(java.util.Calendar.MONTH) + 1
    private val day = calendar.get(java.util.Calendar.DAY_OF_MONTH)

    override fun onServiceConnected() {
        super.onServiceConnected()
        overlay = OverlayManager(this)
    }

    private fun generateDynamicScript(
        department: String,
        clinic: String,
        doctor: String,
        date: String
    ): List<String> {
        val appointmentDay = date.toAppointmentDayText()
        val baseScript = mutableListOf(
            "行動掛號", "繼續掛號", "依門診科別",
            department, clinic, "選擇看診時間", appointmentDay, doctor,
            "填寫個人資料", "請輸入身分證號"
        )

        if (pendingAppointmentType == AppointmentType.INITIAL) {
            baseScript.addAll(
                listOf("請輸入病患姓名", "民國${rocYear}年", "${month}月", "${day}日")
            )
        }

        baseScript.add("確認送出")
        baseScript.add("SUCCESS_FINISH")
        return baseScript
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event == null) return

        if (forceReset) {
            finishGuidanceSession("手動重設")
        }

        if (shouldUpdateScript) {
            val requestedType = pendingGuidanceType
            val requestedScript = when (requestedType) {
                GuidanceType.CANCELLATION -> CancellationGuidance.script
                GuidanceType.BOOKING -> generateDynamicScript(
                    pendingDepartment ?: "", pendingClinic ?: "",
                    pendingDoctor ?: "", pendingDate ?: ""
                )
                null -> emptyList()
            }
            if (requestedType != null && requestedScript.isNotEmpty()) {
                guidanceSession.start(requestedType, requestedScript)
                prepareInteractionState("建立新導引 session", hideOverlay = true)
            }
            shouldUpdateScript = false
        }

        if (!guidanceSession.canHandleTargetEvent()) {
            if (completionMessageUntil > android.os.SystemClock.uptimeMillis() &&
                rootInActiveWindow?.packageName?.toString() == VGH_PACKAGE_NAME
            ) {
                return
            }
            completionMessageUntil = 0L
            cancelPendingGuidanceCallbacks()
            overlay?.hide()
            return
        }

        val activePackageName = rootInActiveWindow?.packageName?.toString().orEmpty()
        val eventPackageName = event.packageName?.toString().orEmpty()
        if (activePackageName == VGH_PACKAGE_NAME) {
            if (guidanceSession.isAwaitingTargetReturn) {
                cancelPendingExitAbort()
                guidanceSession.markTargetReturned()
                Log.d("vgh_id_detect", "重新進入榮總App，維持目前 active session 步驟")
            }
            if (eventPackageName.isNotBlank() && eventPackageName != VGH_PACKAGE_NAME) {
                Log.d("vgh_id_detect", "忽略非榮總事件，前景仍是榮總 eventPackage=$eventPackageName")
            }
        } else if (activePackageName.isNotBlank()) {
            markLeftVghApp(activePackageName)
            Log.d("vgh_id_detect", "目前前景不是榮總App，隱藏紅框 package=$activePackageName")
            return
        } else if (eventPackageName.isNotBlank() && eventPackageName != VGH_PACKAGE_NAME &&
            (event.eventType == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED ||
                event.eventType == AccessibilityEvent.TYPE_WINDOWS_CHANGED)
        ) {
            markLeftVghApp(eventPackageName)
            Log.d("vgh_id_detect", "離開榮總App，隱藏紅框 package=$eventPackageName")
            return
        }

        if (event.eventType == AccessibilityEvent.TYPE_NOTIFICATION_STATE_CHANGED &&
            eventPackageName == VGH_PACKAGE_NAME &&
            guidanceSession.guidanceType == GuidanceType.CANCELLATION &&
            awaitingCancellationResult
        ) {
            val notificationTexts = event.text.map(CharSequence::toString) +
                listOfNotNull(event.contentDescription?.toString())
            when (CancellationFlowPolicy.result(notificationTexts, confirmationSubmitted = true)) {
                CancellationFlowPolicy.Result.SUCCESS ->
                    completeGuidanceWithMessage("醫院畫面顯示取消成功")
                CancellationFlowPolicy.Result.FAILURE ->
                    stopCancellationGuidance("取消掛號失敗，原預約可能仍有效，請查看醫院提示。")
                CancellationFlowPolicy.Result.UNKNOWN -> Unit
            }
            return
        }

        if (event.eventType == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED) {
            val className = event.className?.toString() ?: ""
            isDialogOpen = className.contains("Dialog") || className.contains("PopupWindow") || className.contains("Menu")
        }

        if (event.eventType == AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED ||
            event.eventType == AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED ||
            event.eventType == AccessibilityEvent.TYPE_VIEW_SCROLLED ||
            event.eventType == AccessibilityEvent.TYPE_VIEW_CLICKED ||
            event.eventType == AccessibilityEvent.TYPE_VIEW_SELECTED) {
            if (guidanceSession.guidanceType == GuidanceType.CANCELLATION &&
                event.eventType == AccessibilityEvent.TYPE_VIEW_CLICKED
            ) {
                val clickedTexts = event.text.map(CharSequence::toString) + listOfNotNull(
                    event.source?.text?.toString(),
                    event.contentDescription?.toString()
                )
                if (clickedTexts.any(CancellationFlowPolicy::isAbandonmentAction)) {
                    resetCancellationInteraction()
                    cancellationListReached = true
                    cancellationMessage("請核對掛號日期、科別與醫師，再點選欲取消紀錄旁的『取消』按鈕。")
                    return
                }
                if (isCancellationConfirmationClicked(event) && cancellationListReached) {
                    beginCancellationResultWait()
                }
            }
            if (guidanceSession.guidanceType == GuidanceType.BOOKING &&
                event.eventType == AccessibilityEvent.TYPE_VIEW_CLICKED &&
                (event.text.map { it.toString().trim() } +
                    listOfNotNull(event.source?.text?.toString()?.trim())).contains("確認送出")
            ) {
                beginBookingResultWait()
            }
            val root = rootInActiveWindow ?: return
            if (guidanceSession.canHandleTargetEvent()) handleStep(root)
        }
    }

    override fun onInterrupt() {
        finishGuidanceSession("AccessibilityService interrupted")
    }
    override fun onDestroy() {
        finishGuidanceSession("AccessibilityService destroyed")
        super.onDestroy()
    }

    private fun markLeftVghApp(packageName: String) {
        lastCancellationMessage = null
        overlay?.hide()
        if (guidanceSession.markTargetLeft()) {
            scheduleExitAbort()
            Log.d(
                "vgh_id_detect",
                "已離開榮總App，${TARGET_APP_EXIT_GRACE_MS}ms 後仍未返回將中止導引 package=$packageName"
            )
        }
    }

    private fun prepareInteractionState(reason: String, hideOverlay: Boolean) {
        awaitingBookingResult = false
        completionMessageUntil = 0L
        resetCancellationInteraction()
        cancelPendingGuidanceCallbacks()
        trackedStepIndex = currentStepIndex
        stepEnteredAt = System.currentTimeMillis()
        isDialogOpen = false
        waitingForPersonalDataExit = false
        lastHighlightAt = 0L
        lastScrollHintAt = 0L
        if (hideOverlay) overlay?.hide()
        Log.d("vgh_id_detect", "重設紅框流程 reason=$reason")
    }

    private fun finishGuidanceSession(reason: String) {
        awaitingBookingResult = false
        completionMessageUntil = 0L
        resetCancellationInteraction()
        cancelPendingGuidanceCallbacks()
        guidanceSession.finish()
        clearPendingGuidanceRequest()
        trackedStepIndex = GuidanceSessionState.NO_ACTIVE_STEP
        stepEnteredAt = 0L
        isDialogOpen = false
        waitingForPersonalDataExit = false
        lastHighlightAt = 0L
        lastScrollHintAt = 0L
        overlay?.hide()
        Log.d("vgh_id_detect", "導引 session 已結束 reason=$reason")
    }

    private fun abortGuidanceSession(reason: String) {
        finishGuidanceSession("中止：$reason")
    }

    private fun scheduleExitAbort() {
        val sessionGeneration = guidanceSession.generation
        cancelPendingExitAbort()
        val task = Runnable {
            pendingExitAbort = null
            if (guidanceSession.shouldAbortAfterExitGrace(sessionGeneration)) {
                abortGuidanceSession("離開榮總App超過 grace period")
            }
        }
        pendingExitAbort = task
        mainHandler.postDelayed(task, TARGET_APP_EXIT_GRACE_MS)
    }

    private fun cancelPendingExitAbort() {
        pendingExitAbort?.let(mainHandler::removeCallbacks)
        pendingExitAbort = null
    }

    private fun cancelPendingGuidanceCallbacks() {
        mainHandler.removeCallbacksAndMessages(null)
        pendingStableHighlight = null
        pendingStableKeyword = null
        pendingStableRect = null
        displayedStableRect = null
        displayedStableKeyword = null
        pendingScrollHintRecheck = null
        pendingExitAbort = null
        cancellationResultTimeout = null
    }

    private fun moveToStep(index: Int, reason: String) {
        val boundedIndex = index.coerceIn(0, script.lastIndex.coerceAtLeast(0))
        if (currentStepIndex == boundedIndex) return
        cancelPendingStableHighlight()
        currentStepIndex = boundedIndex
        trackedStepIndex = boundedIndex
        stepEnteredAt = System.currentTimeMillis()
        Log.d("vgh_id_detect", "切換紅框步驟 index=$boundedIndex keyword=${script.getOrNull(boundedIndex)} reason=$reason")
    }
    private fun isDoctorSessionListVisible(nodes: List<NodeData>): Boolean {
        return nodes
            .filter { it.isOnScreen() }
            .any { it.text.toVisitSessionOrNull() != null }
    }
    private fun advanceStep(reason: String) {
        moveToStep(currentStepIndex + 1, reason)
    }

    private fun handleStep(node: AccessibilityNodeInfo) {
        if (trackedStepIndex != currentStepIndex) {
            trackedStepIndex = currentStepIndex
            stepEnteredAt = System.currentTimeMillis()
        }

        val allNodes = findAllTextNodes(node)
        if (allNodes.isEmpty()) {
            hideOverlayIfStable()
            return
        }

        val finishIndex = script.indexOf("SUCCESS_FINISH")
        val idIndex = script.indexOf("請輸入身分證號")
        if (finishIndex != -1 && idIndex != -1 && currentStepIndex >= idIndex) {
            if (isRegistrationSuccessScreen(allNodes)) {
                if (currentStepIndex < finishIndex) {
                    completeGuidanceWithMessage("醫院畫面顯示掛號成功")
                    return
                }
            }
        }

        if (currentStepIndex >= script.size) return

        val currentKeyword = script[currentStepIndex]

        if (currentKeyword == "SUCCESS_FINISH") {
            completeGuidanceWithMessage("尚無法確認掛號結果，請先查詢預約紀錄。")
            return
        }
        if (awaitingBookingResult) {
            cancelPendingStableHighlight()
            overlay?.hide()
            return
        }
        if (guidanceSession.guidanceType == GuidanceType.CANCELLATION &&
            handleCancellation(allNodes)
        ) return

        val yearCount = allNodes.count { it.text.contains("民國") }
        val monthCount = allNodes.count { it.text.contains("月") && it.text.length < 5 }
        val dayCount = allNodes.count { it.text.contains("日") && it.text.length < 5 }
        val isCalendarDayStep = currentKeyword.isCalendarDayNumber()
        val isDateStep = currentKeyword.contains("年") || currentKeyword.contains("月") || currentKeyword.contains("日")

        if (isDateStep && (isDialogOpen || yearCount > 3 || monthCount > 3 || dayCount > 3)) {
            hideOverlayIfStable()
            return
        }

        if (isCalendarDayStep && currentStepIndex + 1 < script.size) {
            val nextKeyword = script[currentStepIndex + 1]
            val nextIsDoctor = nextKeyword.isPendingDoctorKeyword()
            val nextNode = if (nextIsDoctor) {
                findDoctorInSession(allNodes, nextKeyword)
            } else {
                findBestMatch(allNodes, nextKeyword)
            }
            if (nextNode != null && isTargetAppointmentDateSelected(allNodes, currentKeyword)) {
                advanceStep("日期已選且下一步出現")
            }
        }

        val isWaitStep = currentKeyword.contains("請輸入") || isDateStep || isCalendarDayStep

        if (!isWaitStep && currentStepIndex + 1 < script.size) {
            val nextKeyword = script[currentStepIndex + 1]
            if (nextKeyword.isCalendarDayNumber() && isClinicCalendarVisible(allNodes)) {
                overlay?.hide()
                advanceStep("進入行事曆頁，先隱藏舊紅框")
            } else {
                val nextNode = findBestMatch(allNodes, nextKeyword)
                if (nextNode != null) {
                    advanceStep("下一步目標已出現")
                }
            }
        }

        if (isDepartmentStep(script[currentStepIndex]) && isSubClinicPickerVisible(allNodes)) {
            advanceStep("子科別選單已開啟")
        }

        if (waitingForPersonalDataExit) {
            val submitNode = findBestMatch(allNodes, "確認送出")
            if (submitNode != null) {
                waitingForPersonalDataExit = false
                Log.d("vgh_id_detect", "確認送出已出現在畫面上，解除個資等待狀態")
            } else if (isPersonalDataFormVisible(allNodes)) {
                overlay?.hide()
                Log.d("vgh_id_detect", "等待使用者離開個資表單")
                return
            } else {
                waitingForPersonalDataExit = false
                Log.d("vgh_id_detect", "已離開個資表單，繼續確認送出步驟")
            }
        }

        if (shouldEnterPersonalDataWaiting(allNodes)) {
            val submitIndex = script.indexOf("確認送出")
            if (submitIndex >= 0) moveToStep(submitIndex, "進入個資表單後等待確認送出")
            waitingForPersonalDataExit = true
            overlay?.hide()
            Log.d("vgh_id_detect", "已進入個資表單，等待使用者填完並離開")
            return
        }

        val finalKeyword = script[currentStepIndex]
        if (isPrivatePersonalDataStep(finalKeyword)) {
            val submitNode = findBestMatch(allNodes, "確認送出")
            if (submitNode != null) {
                moveToStep(script.indexOf("確認送出"), "個資段落跳到確認送出")
                highlight(submitNode.rect)
            } else {
                overlay?.hide()
                Log.d("vgh_id_detect", "個資輸入段落隱藏紅框 keyword=$finalKeyword")
            }
            return
        }

        if (finalKeyword.isCalendarDayNumber()) {
            val nextKeyword = script.getOrNull(currentStepIndex + 1)
            val nextIsDoctor = nextKeyword.isPendingDoctorKeyword()
            val nextNode = nextKeyword?.let {
                if (nextIsDoctor) findDoctorInSession(allNodes, it) else findBestMatch(allNodes, it)
            }
            if (isTargetAppointmentDateSelected(allNodes, finalKeyword)) {
                if (nextKeyword != null) advanceStep("日期已選，進入下一步")
                val unavailableStatus = nextKeyword?.let { findDoctorUnavailableStatus(allNodes, it) }
                if (nextIsDoctor && nextKeyword != null && unavailableStatus != null) {
                    showUnavailableSessionHint(nextKeyword, unavailableStatus)
                } else if (nextNode != null) {
                    highlight(nextNode.rect)
                } else if (nextIsDoctor && nextKeyword != null && isDoctorSessionListVisible(allNodes)) {
                    // 補上條件 2：確定已經在診次列表頁
                    showScrollHint("溫馨提醒：請往下滑找到「$nextKeyword」")
                } else {
                    overlay?.hide()
                }
                return
            }

            if (shouldPromptNextCalendarMonth(allNodes)) {
                showScrollHintWhenStepStable("請點右上箭頭或往左滑，切換到下一個月")
                return
            }

            val calendarDayNode = findCalendarDayTarget(allNodes, finalKeyword)
            if (calendarDayNode != null) {
                highlight(calendarDayNode.rect)
            } else {
                overlay?.hide()
                Log.d("vgh_id_detect", "行事曆日期未找到，等待使用者選日期 keyword=$finalKeyword")
            }
            return
        }

        if (isDepartmentStep(finalKeyword)) {
            val departmentTarget = findDepartmentInReservationSection(
                nodes = allNodes,
                departmentName = finalKeyword,
                targetSection = pendingAppointmentType.toReservationSection()
            )
            if (departmentTarget != null) {
                highlight(departmentTarget.rect)
            } else {
                showScrollHintWhenStepStable(
                    "溫馨提醒：請在${pendingAppointmentType.displayName()}找到「$finalKeyword」"
                )
            }
            return
        }

        if (isClinicStep(finalKeyword)) {
            val clinicTarget = findExactVisibleTextNode(allNodes, finalKeyword)
            if (clinicTarget != null) {
                highlight(clinicTarget.rect)
            } else {
                showScrollHint("溫馨提醒：請往下滑找到「$finalKeyword」")
            }
            return
        }

        if (isDoctorStep(finalKeyword)) {
            val dateKeyword = script.getOrNull(currentStepIndex - 1)
            val dateStillCorrect = dateKeyword != null &&
                    dateKeyword.isCalendarDayNumber() &&
                    isTargetAppointmentDateSelected(allNodes, dateKeyword)
            val onSessionList = isDoctorSessionListVisible(allNodes)

            val doctorTarget = findDoctorInSession(allNodes, finalKeyword)
            val unavailableStatus = findDoctorUnavailableStatus(allNodes, finalKeyword)

            if (unavailableStatus != null) {
                showUnavailableSessionHint(finalKeyword, unavailableStatus)
            } else if (doctorTarget != null) {
                highlight(doctorTarget.rect)
            } else if (dateStillCorrect && onSessionList) {
                // 兩個條件都滿足：日期沒跑掉，畫面也確定停在診次列表，只是醫師在螢幕下方
                showScrollHint("溫馨提醒：請往下滑找到「$finalKeyword」")
            } else {
                // 日期可能被改掉，或畫面根本不是診次列表（例如還在切月份、彈窗、載入中）
                hideOverlayIfStable()
            }
            return
        }

        val currentTarget = findBestMatch(allNodes, finalKeyword)

        if (currentTarget != null) {
            highlight(currentTarget.rect)
        } else {
            if (finalKeyword == "確認送出") {
                overlay?.hide()
                Log.d("vgh_id_detect", "等待確認送出出現")
                return
            }

            if (isClinicStep(finalKeyword)) {
                showScrollHint("找不到「$finalKeyword」，請往下滑")
                hideOverlayIfStable()
                return
            }

            if (isDoctorStep(finalKeyword)) {
                showScrollHint("找不到「$finalKeyword」，請往下滑")
                hideOverlayIfStable()
                return
            }

            var foundFuture = false
            for (i in 1..2) {
                if (currentStepIndex + i < script.size) {
                    val futureKeyword = script[currentStepIndex + i]
                    val futureNode = findBestMatch(allNodes, futureKeyword)
                    if (futureNode != null) {
                        moveToStep(currentStepIndex + i, "找到後續步驟")
                        highlight(futureNode.rect)
                        foundFuture = true
                        break
                    }
                }
            }
            if (foundFuture) return

            if (currentStepIndex >= script.indexOf("請輸入身分證號")) {
                val prevKeyword = script[currentStepIndex - 1]
                val prevNode = findBestMatch(allNodes, prevKeyword)
                if (prevNode != null) {
                    moveToStep(currentStepIndex - 1, "回到前一步")
                    highlight(prevNode.rect)
                    return
                }
            }
            if (!isDialogOpen) hideOverlayIfStable()
        }
    }

    private fun stopCancellationGuidance(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        abortGuidanceSession(message)
    }

    private fun completeGuidanceWithMessage(message: String) {
        finishGuidanceSession(message)
        val expiresAt = android.os.SystemClock.uptimeMillis() + COMPLETION_MESSAGE_DURATION_MS
        completionMessageUntil = expiresAt
        overlay?.showMessage(message)
        mainHandler.postDelayed({
            if (completionMessageUntil == expiresAt) {
                completionMessageUntil = 0L
                overlay?.hide()
            }
        }, COMPLETION_MESSAGE_DURATION_MS)
    }

    private fun beginBookingResultWait() {
        if (awaitingBookingResult) return
        awaitingBookingResult = true
        val generation = guidanceSession.generation
        mainHandler.postDelayed({
            if (guidanceSession.isCurrentGeneration(generation) && awaitingBookingResult) {
                completeGuidanceWithMessage("尚無法確認掛號結果，請先查詢預約紀錄，勿重複送出。")
            }
        }, RESULT_WAIT_TIMEOUT_MS)
    }

    private fun resetCancellationInteraction() {
        lastCancellationMessage = null
        cancellationResultTimeout?.let(mainHandler::removeCallbacks)
        cancellationResultTimeout = null
        cancellationListReached = false
        cancellationConfirmationVisible = false
        awaitingCancellationResult = false
    }

    private fun cancellationMessage(message: String) {
        cancelPendingStableHighlight()
        if (lastCancellationMessage == message) return
        lastCancellationMessage = message
        overlay?.showMessage(message)
    }

    private fun beginCancellationResultWait() {
        if (awaitingCancellationResult) return
        awaitingCancellationResult = true
        val generation = guidanceSession.generation
        val deadline = android.os.SystemClock.uptimeMillis() + RESULT_WAIT_TIMEOUT_MS
        val task = object : Runnable {
            override fun run() {
                if (!guidanceSession.isCurrentGeneration(generation) || !awaitingCancellationResult) return
                val root = rootInActiveWindow
                if (root?.packageName?.toString() == VGH_PACKAGE_NAME) {
                    handleCancellation(findAllTextNodes(root))
                }
                if (!guidanceSession.isCurrentGeneration(generation) || !awaitingCancellationResult) return
                if (android.os.SystemClock.uptimeMillis() >= deadline) {
                    stopCancellationGuidance("尚無法確認取消結果，請先查詢預約紀錄，勿重複送出。")
                } else {
                    mainHandler.postDelayed(this, RESULT_POLL_INTERVAL_MS)
                }
            }
        }
        cancellationResultTimeout = task
        mainHandler.postDelayed(task, RESULT_POLL_INTERVAL_MS)
    }

    private fun handleCancellation(nodes: List<NodeData>): Boolean {
        val visible = nodes.filter { it.isVisibleToUser && it.isOnScreen() }
        val texts = visible.flatMap { listOf(it.text, it.contentDescription.orEmpty()) }

        if (!awaitingCancellationResult &&
            CancellationFlowPolicy.isDataEntry(texts, visible.any(NodeData::isEditable))
        ) {
            cancellationListReached = false
            cancellationConfirmationVisible = false
            lastCancellationMessage = null
            cancelPendingStableHighlight()
            overlay?.hide()
            val dataEntryIndex = script.indexOf("請輸入身分證號")
            if (dataEntryIndex >= 0) {
                moveToStep(dataEntryIndex, "填寫查詢資料，暫停清單提示")
            }
            return true
        }

        when (CancellationFlowPolicy.result(texts, awaitingCancellationResult)) {
            CancellationFlowPolicy.Result.SUCCESS -> {
                completeGuidanceWithMessage("醫院畫面顯示取消成功")
                return true
            }
            CancellationFlowPolicy.Result.FAILURE -> {
                stopCancellationGuidance("取消掛號失敗，原預約可能仍有效，請查看醫院提示。")
                return true
            }
            CancellationFlowPolicy.Result.UNKNOWN -> Unit
        }

        val confirmationVisible = texts.any(CancellationGuidance::isConfirmationAction)
        if (cancellationListReached && confirmationVisible && !awaitingCancellationResult) {
            cancellationConfirmationVisible = true
            val message = "請核對掛號日期、科別與醫師，再確認是否取消掛號。"
            val confirmTarget = visible
                .filter {
                    it.isEnabled &&
                        (CancellationGuidance.isConfirmationAction(it.text) ||
                            CancellationGuidance.isConfirmationAction(it.contentDescription))
                }
                .minWithOrNull(
                    compareBy<NodeData> { if (it.isClickable) 0 else 1 }
                        .thenBy { it.rect.width().toLong() * it.rect.height() }
                )
            if (confirmTarget != null) {
                lastCancellationMessage = null
                highlight(confirmTarget.rect, message)
            } else {
                cancellationMessage(message)
            }
            return true
        }

        val explicitlyEmpty = CancellationFlowPolicy.isExplicitlyEmpty(texts)
        val listVisible = CancellationGuidance.isAppointmentList(nodes) || explicitlyEmpty
        if (cancellationConfirmationVisible && !confirmationVisible && !awaitingCancellationResult) {
            cancellationConfirmationVisible = false
            if (!listVisible) {
                cancellationMessage("尚未確認取消結果，請查看醫院畫面或返回掛號查詢。")
                return true
            }
        }
        if (awaitingCancellationResult) {
            cancellationMessage("正在等待取消結果，請稍候")
            return true
        }
        if (!listVisible) {
            if (cancellationListReached ||
                script.getOrNull(currentStepIndex)?.let(CancellationGuidance::isDataEntryStep) == true
            ) {
                lastCancellationMessage = null
                cancelPendingStableHighlight()
                overlay?.hide()
                return true
            }
            return false
        }

        cancellationListReached = true
        cancellationConfirmationVisible = false
        val cancellationIndex = script.indexOf("取消")
        if (cancellationIndex >= 0) moveToStep(cancellationIndex, "取得掛號紀錄，等待使用者選擇")
        if (explicitlyEmpty) {
            stopCancellationGuidance(NO_CANCELLABLE_MESSAGE)
            return true
        }

        val match = CancellationGuidance.findCancellationAction(nodes)
        when (match.status) {
            CancellationMatchStatus.SINGLE_CANCELLABLE -> {
                match.demonstrationSourceIndex
                    ?.let(nodes::getOrNull)
                    ?.let {
                        highlight(
                            it.rect,
                            "請核對掛號日期、科別與醫師，再點選欲取消紀錄旁的『取消』按鈕。"
                        )
                    }
                    ?: stopCancellationGuidance(NO_CANCELLABLE_MESSAGE)
            }
            CancellationMatchStatus.MULTIPLE_CANCELLABLE -> {
                match.demonstrationSourceIndex
                    ?.let(nodes::getOrNull)
                    ?.let { highlight(it.rect, MULTIPLE_CANCELLABLE_MESSAGE) }
                    ?: cancellationMessage(MULTIPLE_CANCELLABLE_MESSAGE)
            }
            CancellationMatchStatus.NONE_CANCELLABLE -> {
                cancellationMessage("目前畫面尚未顯示可取消按鈕，請往下滑查看其他預約。")
            }
        }
        return true
    }

    /** The confirmation click starts result observation; it is not proof of success. */
    private fun isCancellationConfirmationClicked(event: AccessibilityEvent): Boolean {
        if (guidanceSession.guidanceType != GuidanceType.CANCELLATION || event.eventType != AccessibilityEvent.TYPE_VIEW_CLICKED) return false

        return event.text.any(CancellationGuidance::isConfirmationAction) ||
                CancellationGuidance.isConfirmationAction(event.source?.text) ||
                CancellationGuidance.isConfirmationAction(event.contentDescription)
    }

    private fun findBestMatch(nodes: List<NodeData>, keyword: String): NodeData? {
        if (keyword.isBlank()) return null
        val visibleNodes = nodes.filter { it.isOnScreen() }

        if (guidanceSession.guidanceType == GuidanceType.CANCELLATION) {
            CancellationGuidance.findTarget(nodes, keyword)?.let { return it }
        }

        if (keyword.isCalendarDayNumber()) {
            val calendarNumberNodes = visibleNodes
                .filter { node -> node.text.trim().toIntOrNull()?.let { it in 1..31 } == true }
            val calendarRows = calendarNumberNodes
                .groupBy { it.rect.centerY() / 40 }
                .values
                .filter { row -> row.size >= 3 }
                .flatten()
            val calendarTop = calendarRows.minOfOrNull { it.rect.top } ?: 180
            val calendarBottom = calendarRows.maxOfOrNull { it.rect.bottom } ?: 720

            val exactCalendarDayNodes = visibleNodes
                .filter { it.text.trim() == keyword }
                .filter { it.rect.centerY() in calendarTop..calendarBottom }

            val bestCalendarDay = exactCalendarDayNodes.minWithOrNull(
                compareBy<NodeData> { kotlin.math.abs(it.rect.centerY() - 520) }
                    .thenBy { it.rect.width() * it.rect.height() }
            )
            return bestCalendarDay
        }

        if (keyword == "行動掛號") {
            return visibleNodes.find { it.text.contains(keyword) }
        }

        val matches = visibleNodes.filter { it.text.contains(keyword) }
        if (matches.isEmpty()) return null

        if (keyword == "依門診科別") {
            val screenWidth = screenSize().first
            val exactTab = matches
                .filter { it.text.trim() == keyword }
                .filter { it.rect.centerX() < screenWidth / 2 }
                .minWithOrNull(
                    compareBy<NodeData> { it.text.length }
                        .thenBy { it.rect.width() * it.rect.height() }
                )
            if (exactTab != null) return exactTab

            val combinedTabs = matches
                .filter { it.rect.centerX() < screenWidth / 2 || it.rect.width() > screenWidth / 2 }
                .minByOrNull { it.rect.width() * it.rect.height() }
            return combinedTabs?.let { node ->
                val leftTabRect = Rect(
                    node.rect.left,
                    node.rect.top,
                    node.rect.left + node.rect.width() / 2,
                    node.rect.bottom
                )
                NodeData(keyword, leftTabRect)
            }
        }

        val isAppointmentTimeHeading = keyword == "選擇看診時間" || keyword == "選擇看診時間 / 醫師"
        val filteredMatches = if (isAppointmentTimeHeading || keyword.length >= 6) {
            matches.filter { it.text.length <= keyword.length + 6 }
        } else {
            matches
        }

        if (filteredMatches.isEmpty()) return null

        val bestMatch = filteredMatches.minWithOrNull(
            compareBy<NodeData> { if (it.text.trim() == keyword) 0 else 1 }
                .thenBy { it.text.length }
                .thenBy { it.rect.width() * it.rect.height() }
        )

        return if (isAppointmentTimeHeading) {
            bestMatch?.withCompactTopBounds()
        } else {
            bestMatch
        }
    }

    private fun NodeData.withCompactTopBounds(): NodeData {
        if (rect.height() <= 300) return this

        val compactHeight = 96
        return copy(
            rect = Rect(
                rect.left,
                rect.top,
                rect.right,
                rect.top + compactHeight
            )
        )
    }

    private fun isClinicCalendarVisible(nodes: List<NodeData>): Boolean {
        val hasMonthTitle = nodes.any { it.text.contains("月") && it.text.contains("202") }
        val weekdayCount = nodes.count {
            it.text.trim() in setOf("週日", "週一", "週二", "週三", "週四", "週五", "週六")
        }
        val hasClinicSession = nodes.any {
            it.text.contains("上午診") || it.text.contains("下午診") || it.text.contains("晚上診")
        }
        return hasMonthTitle && weekdayCount >= 5 && hasClinicSession
    }

    private fun findCalendarDayTarget(nodes: List<NodeData>, dayText: String): NodeData? {
        val exactNode = findBestMatch(nodes, dayText)
        if (exactNode != null) return exactNode

        val inferredRect = inferCalendarDayRect(nodes, dayText.toIntOrNull() ?: return null)
        if (inferredRect != null) {
            return NodeData(dayText, inferredRect)
        }
        return null
    }

    private fun shouldPromptNextCalendarMonth(nodes: List<NodeData>): Boolean {
        val targetYearMonth = pendingDate.toYearMonthOrNull() ?: return false
        val visibleYearMonth = visibleCalendarYearMonth(nodes) ?: return false
        return visibleYearMonth.first < targetYearMonth.first ||
                (visibleYearMonth.first == targetYearMonth.first && visibleYearMonth.second < targetYearMonth.second)
    }

    private fun visibleCalendarYearMonth(nodes: List<NodeData>): Pair<Int, Int>? {
        return nodes
            .filter { it.isOnScreen() }
            .mapNotNull { node ->
                parseCalendarYearMonthTitle(node.text)?.let { node.rect.top to it }
            }
            .minByOrNull { it.first }
            ?.second
    }

    private fun parseCalendarYearMonthTitle(text: String): Pair<Int, Int>? {
        val normalized = text.replace(" ", "")
        val yearFirst = Regex("""(20\d{2}).*?(\d{1,2})\s*月""")
        val monthFirst = Regex("""(\d{1,2})\s*月.*?(20\d{2})""")

        yearFirst.find(normalized)?.let { match ->
            val year = match.groupValues.getOrNull(1)?.toIntOrNull() ?: return null
            val month = match.groupValues.getOrNull(2)?.toIntOrNull() ?: return null
            if (month in 1..12) return year to month
        }

        monthFirst.find(normalized)?.let { match ->
            val month = match.groupValues.getOrNull(1)?.toIntOrNull() ?: return null
            val year = match.groupValues.getOrNull(2)?.toIntOrNull() ?: return null
            if (month in 1..12) return year to month
        }

        return null
    }

    private fun isTargetAppointmentDateSelected(nodes: List<NodeData>, dayText: String): Boolean {
        val targetDay = dayText.toIntOrNull() ?: return false
        val visibleNodes = nodes.filter { it.isOnScreen() }
        val monthTitle = visibleNodes
            .filter { it.text.contains("月") && it.text.contains("202") }
            .minByOrNull { it.rect.top }

        val selectedTopCardDay = monthTitle?.let { title ->
            visibleNodes
                .filter { node -> node.text.trim().toIntOrNull()?.let { it in 1..31 } == true }
                .filter { it.rect.centerY() < title.rect.centerY() }
                .maxByOrNull { it.rect.height() * it.rect.width() }
                ?.text
                ?.trim()
                ?.toIntOrNull()
        }
        if (selectedTopCardDay != null) {
            return selectedTopCardDay == targetDay
        }

        val selectedCalendarDay = visibleNodes
            .filter { it.isSelected }
            .mapNotNull { it.text.trim().toIntOrNull() }
            .firstOrNull { it in 1..31 }
        if (selectedCalendarDay != null) {
            return selectedCalendarDay == targetDay
        }
        return false
    }

    private fun inferCalendarDayRect(nodes: List<NodeData>, targetDay: Int): Rect? {
        val visibleNodes = nodes.filter { it.isOnScreen() }
        val monthTitle = visibleNodes
            .filter { it.text.contains("月") && it.text.contains("202") }
            .minByOrNull { it.rect.top }
            ?: return null

        val weekdayLabels = setOf("週日", "週一", "週二", "週三", "週四", "週五", "週六")
        val weekdayRow = visibleNodes
            .filter { it.text.trim() in weekdayLabels }
            .filter { it.rect.top > monthTitle.rect.bottom }
            .filter { it.rect.top < monthTitle.rect.bottom + 220 }
            .groupBy { it.rect.centerY() / 50 }
            .values
            .maxByOrNull { row -> row.map { it.text.trim() }.distinct().size }
            ?: return null

        val weekdayByText = weekdayRow
            .groupBy { it.text.trim() }
            .mapValues { (_, nodesInColumn) ->
                nodesInColumn.minWithOrNull(
                    compareBy<NodeData> { it.rect.width() * it.rect.height() }
                        .thenBy { it.rect.top }
                )
            }
        val appointmentWeekdayNodes = listOf("週一", "週二", "週三", "週四", "週五", "週六")
            .map { label -> weekdayByText[label] ?: return null }

        val yearMonth = pendingDate.toYearMonthOrNull() ?: return null
        val dayOfWeek = dayOfWeekOfDate(yearMonth.first, yearMonth.second, targetDay)
        if (dayOfWeek == java.util.Calendar.SUNDAY) {
            return null
        }
        val firstDayColumn = firstDayColumnOfMonth(yearMonth.first, yearMonth.second)
        val zeroBasedIndex = firstDayColumn + targetDay - 1
        val row = zeroBasedIndex / 7
        val column = dayOfWeek - java.util.Calendar.MONDAY

        val xCenter = appointmentWeekdayNodes[column].rect.centerX()
        val weekdayBottom = weekdayRow.maxOf { it.rect.bottom }
        val numberNodes = visibleNodes
            .filter { node -> node.text.trim().toIntOrNull()?.let { it in 1..31 } == true }
            .filter { it.rect.top > weekdayBottom }

        val cellHeight = numberNodes
            .map { it.rect.centerY() }
            .distinct()
            .sorted()
            .zipWithNext { a, b -> b - a }
            .filter { it > 20 }
            .minOrNull() ?: 66
        val firstRowY = numberNodes
            .filter { it.text.trim().toIntOrNull() in 1..7 }
            .minOfOrNull { it.rect.centerY() }
            ?: (weekdayBottom + cellHeight)
        val yCenter = firstRowY + row * cellHeight
        val size = 58

        val screenWidth = android.content.res.Resources.getSystem().displayMetrics.widthPixels
        val screenHeight = android.content.res.Resources.getSystem().displayMetrics.heightPixels
        val rect = Rect(
            xCenter - size / 2,
            yCenter - size / 2,
            xCenter + size / 2,
            yCenter + size / 2
        )
        if (rect.left < 0 || rect.right > screenWidth || rect.top <= monthTitle.rect.bottom || rect.bottom > screenHeight) {
            return null
        }
        return rect
    }

    private fun isDepartmentStep(keyword: String): Boolean =
        pendingDepartment?.isNotBlank() == true &&
                keyword == pendingDepartment &&
                currentStepIndex == script.indexOf(pendingDepartment)

    private fun findDepartmentInReservationSection(
        nodes: List<NodeData>,
        departmentName: String,
        targetSection: ReservationSection
    ): NodeData? {
        val visibleNodes = nodes.filter { it.isOnScreen() }
        val departmentCandidates = visibleNodes.filter { node ->
            node.text.normalizedLabel() == departmentName.normalizedLabel()
        }.ifEmpty {
            visibleNodes.filter { node ->
                node.text.contains(departmentName) && node.text.length <= departmentName.length + 4
            }
        }
        if (departmentCandidates.isEmpty()) return null

        val reservationHeaders = visibleNodes
            .mapNotNull { node ->
                node.text.toReservationSectionOrNull()?.let { section -> node to section }
            }
            .sortedBy { it.first.rect.centerY() }

        if (reservationHeaders.isEmpty()) {
            return departmentCandidates.bestDepartmentCandidate()
        }

        val candidatesInSection = departmentCandidates.filterByReservationSection(
            reservationHeaders = reservationHeaders,
            targetSection = targetSection
        )

        return candidatesInSection.bestDepartmentCandidate()
    }

    private fun List<NodeData>.filterByReservationSection(
        reservationHeaders: List<Pair<NodeData, ReservationSection>>,
        targetSection: ReservationSection
    ): List<NodeData> =
        filter { candidate ->
            val nearestHeader = reservationHeaders
                .filter { it.first.rect.centerY() <= candidate.rect.centerY() }
                .maxByOrNull { it.first.rect.centerY() }
            nearestHeader?.second == targetSection
        }.ifEmpty {
            val targetHeader = reservationHeaders.firstOrNull { it.second == targetSection }?.first
            val nextHeader = targetHeader?.let { header ->
                reservationHeaders
                    .map { it.first }
                    .filter { it.rect.centerY() > header.rect.centerY() }
                    .minByOrNull { it.rect.centerY() }
            }
            if (targetHeader == null) {
                emptyList()
            } else {
                filter { candidate ->
                    candidate.rect.centerY() > targetHeader.rect.centerY() &&
                            (nextHeader == null || candidate.rect.centerY() < nextHeader.rect.centerY())
                }
            }
        }

    private fun List<NodeData>.bestDepartmentCandidate(): NodeData? =
        minWithOrNull(
            compareBy<NodeData> { it.rect.width() * it.rect.height() }
                .thenBy { it.rect.top }
        )

    private fun isSubClinicPickerVisible(nodes: List<NodeData>): Boolean {
        val nextKeyword = script.getOrNull(currentStepIndex + 1)
        if (nextKeyword != pendingClinic) return false
        if (nodes.none { it.text.trim() == "CANCEL" }) return false

        val currentDepartment = pendingDepartment ?: return false
        return nodes.any { node ->
            val text = node.text.trim()
            text != currentDepartment &&
                    text.contains("科") &&
                    !text.endsWith("系") &&
                    text != "依門診科別"
        }
    }

    private fun isClinicStep(keyword: String): Boolean =
        pendingClinic?.isNotBlank() == true &&
                keyword == pendingClinic &&
                currentStepIndex == script.indexOf(pendingClinic)

    private fun isDoctorStep(keyword: String): Boolean =
        pendingDoctor?.isNotBlank() == true &&
                keyword == pendingDoctor &&
                currentStepIndex == script.indexOf(pendingDoctor)

    private fun String?.isPendingDoctorKeyword(): Boolean =
        pendingDoctor?.isNotBlank() == true && this == pendingDoctor

    private fun isPrivatePersonalDataStep(keyword: String): Boolean =
        keyword == "請輸入身分證號" ||
                keyword == "請輸入病患姓名" ||
                keyword == "民國${rocYear}年" ||
                keyword == "${month}月" ||
                keyword == "${day}日"

    private fun shouldEnterPersonalDataWaiting(nodes: List<NodeData>): Boolean {
        val personalStartIndex = script.indexOf("填寫個人資料")
        val submitIndex = script.indexOf("確認送出")
        if (personalStartIndex < 0 || submitIndex < 0) return false
        if (currentStepIndex < personalStartIndex || currentStepIndex >= submitIndex) return false
        return isPersonalDataFormVisible(nodes)
    }

    private fun isPersonalDataFormVisible(nodes: List<NodeData>): Boolean {
        val hasIdField = nodes.any { it.text.trim() == "身分證號" || it.text.trim() == "請輸入身分證號" }

        if (pendingAppointmentType == AppointmentType.RETURN_VISIT) {
            return hasIdField
        }

        val hasNameField = nodes.any { it.text.trim() == "姓名" || it.text.trim() == "請輸入病患姓名" }
        val hasBirthdayField = nodes.any { it.text.trim() == "出生年月日" }
        return hasIdField && hasNameField && hasBirthdayField
    }

    private fun findExactVisibleTextNode(nodes: List<NodeData>, keyword: String): NodeData? =
        nodes.asSequence()
            .filter { it.isOnScreen() }
            .filter { it.text.trim() == keyword }
            .minWithOrNull(
                compareBy<NodeData> { it.rect.width() * it.rect.height() }
                    .thenBy { it.rect.top }
            )

    private fun findDoctorInSession(nodes: List<NodeData>, doctorName: String): NodeData? {
        val visibleNodes = nodes.filter { it.isOnScreen() }
        val doctorCandidates = visibleNodes
            .filter { it.text.contains(doctorName) }
            .filter { it.text.length <= doctorName.length + 12 }

        if (doctorCandidates.isEmpty()) return null

        val targetSession = pendingTimeSlot.toVisitSessionOrNull()
        if (targetSession.isNullOrBlank()) {
            return doctorCandidates.bestDoctorCandidate(doctorName)
        }

        val sessionHeaders = visibleNodes
            .mapNotNull { node -> node.text.toVisitSessionOrNull()?.let { session -> node to session } }
            .sortedBy { it.first.rect.centerY() }

        if (sessionHeaders.isEmpty()) {
            return doctorCandidates.bestDoctorCandidate(doctorName)
        }

        val candidatesInSession = doctorCandidates.filter { candidate ->
            val nearestHeader = sessionHeaders
                .filter { it.first.rect.centerY() <= candidate.rect.centerY() }
                .maxByOrNull { it.first.rect.centerY() }
            nearestHeader?.second == targetSession
        }

        return candidatesInSession.bestDoctorCandidate(doctorName)
    }

    private fun findDoctorUnavailableStatus(nodes: List<NodeData>, doctorName: String): UnavailableSlotStatus? {
        val visibleNodes = nodes.filter { it.isOnScreen() }
        val doctorCandidates = visibleNodes
            .filter { it.text.contains(doctorName) }
            .filter { it.text.length <= doctorName.length + 12 }

        if (doctorCandidates.isEmpty()) return null

        val targetSession = pendingTimeSlot.toVisitSessionOrNull()
        val sessionHeaders = visibleNodes
            .mapNotNull { node -> node.text.toVisitSessionOrNull()?.let { session -> node to session } }
            .sortedBy { it.first.rect.centerY() }

        val candidatesInSession = if (targetSession.isNullOrBlank() || sessionHeaders.isEmpty()) {
            doctorCandidates
        } else {
            doctorCandidates.filter { candidate ->
                val nearestHeader = sessionHeaders
                    .filter { it.first.rect.centerY() <= candidate.rect.centerY() }
                    .maxByOrNull { it.first.rect.centerY() }
                nearestHeader?.second == targetSession
            }
        }

        val unavailableStatusNodes = visibleNodes.mapNotNull { node ->
            node.text.toUnavailableSlotStatusOrNull()?.let { status -> node to status }
        }

        var matchedStatus: UnavailableSlotStatus? = null
        candidatesInSession.firstOrNull { doctor ->
            val rowNodes = visibleNodes.filter { node ->
                kotlin.math.abs(node.rect.centerY() - doctor.rect.centerY()) <= 95
            }
            val rowText = rowNodes.joinToString("") { it.text }.normalizedLabel()
            matchedStatus = doctor.text.toUnavailableSlotStatusOrNull()
                ?: rowText.toUnavailableSlotStatusOrNull()
                        ?: unavailableStatusNodes.firstOrNull { (statusNode, _) ->
                    kotlin.math.abs(statusNode.rect.centerY() - doctor.rect.centerY()) <= 95
                }?.second
            matchedStatus != null
        }
        return matchedStatus
    }

    private fun List<NodeData>.bestDoctorCandidate(doctorName: String): NodeData? =
        minWithOrNull(
            compareBy<NodeData> { if (it.text.trim() == doctorName) 0 else 1 }
                .thenBy { it.text.length }
                .thenBy { it.rect.width() * it.rect.height() }
        )

    private fun showUnavailableSessionHint(doctorName: String, status: UnavailableSlotStatus) {
        showScrollHint("此診次${doctorName}已${status.label}\n在畫面自行切換其他日期 / 選擇其他醫師")
    }

    private fun showScrollHint(message: String) {
        val now = System.currentTimeMillis()
        if (now - lastScrollHintAt < 2500) return
        lastScrollHintAt = now
        lastHighlightAt = now
        overlay?.showMessage(message)
    }

    private fun showScrollHintWhenStepStable(message: String) {
        val elapsed = System.currentTimeMillis() - stepEnteredAt
        if (elapsed < SCROLL_HINT_GRACE_MS) {
            scheduleScrollHintRecheck(message, SCROLL_HINT_GRACE_MS - elapsed)
            return
        }
        showScrollHint(message)
    }

    private fun scheduleScrollHintRecheck(message: String, delay: Long) {
        val key = "$currentStepIndex|$message"
        val sessionGeneration = guidanceSession.generation
        pendingScrollHintRecheck?.let(mainHandler::removeCallbacks)
        val task = Runnable {
            pendingScrollHintRecheck = null
            if (!guidanceSession.isCurrentGeneration(sessionGeneration)) return@Runnable
            if ("$currentStepIndex|$message" == key) {
                showScrollHint(message)
            }
        }
        pendingScrollHintRecheck = task
        mainHandler.postDelayed(task, delay.coerceAtLeast(0))
    }

    private fun cancelPendingScrollHintRecheck() {
        pendingScrollHintRecheck?.let(mainHandler::removeCallbacks)
        pendingScrollHintRecheck = null
    }

    private fun findAllTextNodes(root: AccessibilityNodeInfo): List<NodeData> {
        val result = mutableListOf<NodeData>()
        traverse(root, result)
        return result
    }

    private fun traverse(
        node: AccessibilityNodeInfo?,
        list: MutableList<NodeData>,
        nearestClickableAncestorIndex: Int? = null
    ) {
        if (node == null) return

        val rawText = node.text?.toString() ?: ""
        val rawDesc = node.contentDescription?.toString() ?: ""
        val rawHint = if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            node.hintText?.toString() ?: ""
        } else { "" }

        val isFilled = rawText.isNotEmpty() && !rawText.contains("請輸入") && rawText != rawHint && rawText != rawDesc
        val combinedText = if (isFilled) rawText else "$rawText $rawDesc $rawHint".trim()

        // 收集節點條件放寬：有文字，或是可點擊，通通抓進來！
        var currentNodeIndex: Int? = null
        if (combinedText.isNotEmpty() || node.isClickable || node.isEditable) {
            val rect = android.graphics.Rect()
            node.getBoundsInScreen(rect)
            if (rect.width() > 0 && rect.height() > 0) {
                currentNodeIndex = list.size
                list.add(
                    NodeData(
                        text = combinedText,
                        rect = rect,
                        isSelected = node.isSelected,
                        contentDescription = rawDesc.takeIf { it.isNotBlank() },
                        isEnabled = node.isEnabled,
                        isClickable = node.isClickable,
                        clickableAncestorIndex = nearestClickableAncestorIndex,
                        isEditable = node.isEditable,
                        isVisibleToUser = node.isVisibleToUser
                    )
                )
            }
        }
        val clickableAncestorForChildren = if (node.isEnabled && node.isClickable) {
            currentNodeIndex ?: nearestClickableAncestorIndex
        } else {
            nearestClickableAncestorIndex
        }
        for (i in 0 until node.childCount) {
            traverse(node.getChild(i), list, clickableAncestorForChildren)
        }
    }

    private fun highlight(rect: Rect, message: String? = null) {
        val keyword = script.getOrNull(currentStepIndex).orEmpty()
        val key = "$currentStepIndex|$keyword|${message.orEmpty()}"
        val targetRect = normalizeHighlightRect(rect, keyword) ?: run {
            overlay?.hide()
            return
        }
        val tolerance = dp(STABLE_RECT_TOLERANCE_DP)
        if (displayedStableKeyword == key && displayedStableRect?.isCloseTo(targetRect, tolerance) == true) {
            displayedStableRect = targetRect
            showHighlightImmediately(targetRect, message)
            return
        }
        scheduleStableHighlight(key, targetRect, message)
    }

    /**
     * Accessibility events can arrive while the target app is still animating or scrolling.
     * Draw only after the latest candidate rect survives a short quiet period.
     */
    private fun scheduleStableHighlight(key: String, rect: Rect, message: String?) {
        val sessionGeneration = guidanceSession.generation
        val tolerance = dp(STABLE_RECT_TOLERANCE_DP)
        if (pendingStableKeyword == key && pendingStableRect?.isCloseTo(rect, tolerance) == true) return

        val replacingVisibleTarget = displayedStableKeyword != key ||
                displayedStableRect?.isCloseTo(rect, tolerance) != true
        cancelPendingStableHighlight()
        pendingStableKeyword = key
        pendingStableRect = rect
        if (replacingVisibleTarget) overlay?.hide()
        val task = Runnable {
            if (!guidanceSession.isCurrentGeneration(sessionGeneration)) return@Runnable
            val stableKey = pendingStableKeyword
            val stableRect = pendingStableRect
            pendingStableHighlight = null
            pendingStableKeyword = null
            pendingStableRect = null
            val currentKey = "$currentStepIndex|${script.getOrNull(currentStepIndex).orEmpty()}|${message.orEmpty()}"
            if (stableKey != currentKey) return@Runnable
            val finalRect = stableRect ?: return@Runnable

            displayedStableKeyword = stableKey
            displayedStableRect = finalRect
            showHighlightImmediately(finalRect, message)
            Log.d("vgh_id_detect", "stable highlight key=$stableKey rect=${finalRect.toShortString()}")
        }
        pendingStableHighlight = task
        mainHandler.postDelayed(task, STABLE_HIGHLIGHT_DELAY_MS)
    }

    private fun cancelPendingStableHighlight() {
        pendingStableHighlight?.let(mainHandler::removeCallbacks)
        pendingStableHighlight = null
        pendingStableKeyword = null
        pendingStableRect = null
        displayedStableRect = null
        displayedStableKeyword = null
        cancelPendingScrollHintRecheck()
    }

    private fun showHighlightImmediately(rect: Rect, message: String? = null) {
        lastHighlightAt = System.currentTimeMillis()
        overlay?.show(
            rect,
            message,
            messageAtTop = guidanceSession.guidanceType == GuidanceType.CANCELLATION &&
                cancellationConfirmationVisible && !awaitingCancellationResult
        )
    }

    private fun normalizeHighlightRect(rect: Rect, keyword: String): Rect? {
        val (screenWidth, screenHeight) = screenSize()
        if (screenWidth <= 0 || screenHeight <= 0 || rect.width() <= 0 || rect.height() <= 0) return null

        val usesPreviousRange = keyword == "選擇看診時間" ||
                keyword == "選擇看診時間 / 醫師" ||
                keyword == "填寫個人資料"
        val shouldCompact = if (usesPreviousRange) {
            rect.height() > screenHeight * 0.7f
        } else {
            rect.height() > screenHeight * 0.7f || rect.width() > screenWidth * 0.92f
        }
        val compactRect = if (shouldCompact) {
            val halfWidth = (screenWidth * 0.22f).toInt()
            val halfHeight = (screenHeight * 0.05f).toInt()
            Rect(rect.centerX() - halfWidth, rect.centerY() - halfHeight, rect.centerX() + halfWidth, rect.centerY() + halfHeight)
        } else rect
        return compactRect.clampedToScreen(screenWidth, screenHeight)
            .takeIf { it.width() >= dp(8) && it.height() >= dp(8) }
    }

    private fun Rect.isCloseTo(other: Rect, tolerancePx: Int): Boolean =
        kotlin.math.abs(left - other.left) <= tolerancePx &&
                kotlin.math.abs(top - other.top) <= tolerancePx &&
                kotlin.math.abs(right - other.right) <= tolerancePx &&
                kotlin.math.abs(bottom - other.bottom) <= tolerancePx

    private fun Rect.clampedToScreen(screenWidth: Int, screenHeight: Int): Rect {
        val maxRight = screenWidth.coerceAtLeast(1)
        val maxBottom = screenHeight.coerceAtLeast(1)
        val clampedLeft = left.coerceIn(0, maxRight - 1)
        val clampedTop = top.coerceIn(0, maxBottom - 1)
        val clampedRight = right.coerceIn(clampedLeft + 1, maxRight)
        val clampedBottom = bottom.coerceIn(clampedTop + 1, maxBottom)
        return Rect(clampedLeft, clampedTop, clampedRight, clampedBottom)
    }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density).toInt()

    private fun screenSize(): Pair<Int, Int> {
        val windowManager = getSystemService(WINDOW_SERVICE) as android.view.WindowManager
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            windowManager.currentWindowMetrics.bounds.let { it.width() to it.height() }
        } else {
            @Suppress("DEPRECATION")
            android.util.DisplayMetrics().also { windowManager.defaultDisplay.getRealMetrics(it) }
                .let { it.widthPixels to it.heightPixels }
        }
    }

    private fun hideOverlayIfStable() {
        val elapsed = System.currentTimeMillis() - lastHighlightAt
        if (elapsed < 1500) {
            return
        }
        overlay?.hide()
    }

    private fun isRegistrationSuccessScreen(nodes: List<NodeData>): Boolean =
        BookingResultPolicy.isSuccess(
            nodes.filter { it.isVisibleToUser && it.isOnScreen() }
                .flatMap { listOf(it.text, it.contentDescription.orEmpty()) }
                .filter(String::isNotBlank)
        )
}

// 修改 NodeData，加上 isClickable 屬性
data class NodeData(
    val text: String,
    val rect: Rect,
    val isSelected: Boolean = false,
    val contentDescription: String? = null,
    val isEnabled: Boolean = true,
    val isClickable: Boolean = false,
    val clickableAncestorIndex: Int? = null,
    val isEditable: Boolean = false,
    val isVisibleToUser: Boolean = true
)

private enum class ReservationSection {
    INITIAL,
    RETURN_VISIT
}

private fun AppointmentType.toReservationSection(): ReservationSection =
    when (this) {
        AppointmentType.INITIAL -> ReservationSection.INITIAL
        AppointmentType.RETURN_VISIT -> ReservationSection.RETURN_VISIT
    }

private fun AppointmentType.displayName(): String =
    when (this) {
        AppointmentType.INITIAL -> "初診預約"
        AppointmentType.RETURN_VISIT -> "複診掛號"
    }

private enum class UnavailableSlotStatus(val label: String) {
    FULL("額滿"),
    LEAVE("請假"),
    CLOSED("關診"),
    STOPPED("停診")
}

private fun String.toAppointmentDayText(): String {
    val trimmed = trim()
    if (trimmed.isBlank()) return trimmed
    val normalized = trimmed.replace('-', '/')
    return normalized
        .split('/')
        .lastOrNull()
        ?.takeIf { it.all(Char::isDigit) }
        ?.toIntOrNull()
        ?.toString()
        ?: trimmed
}

private fun String.isCalendarDayNumber(): Boolean =
    toIntOrNull()?.let { it in 1..31 } == true

private fun String?.toYearMonthOrNull(): Pair<Int, Int>? {
    val normalized = this?.trim().orEmpty().replace('-', '/')
    val parts = normalized.split('/')
    if (parts.size < 2) return null
    val year = parts.getOrNull(0)?.toIntOrNull() ?: return null
    val month = parts.getOrNull(1)?.toIntOrNull() ?: return null
    if (month !in 1..12) return null
    return year to month
}

private fun firstDayColumnOfMonth(year: Int, month: Int): Int {
    val calendar = java.util.Calendar.getInstance(java.util.Locale.TAIWAN).apply {
        set(java.util.Calendar.YEAR, year)
        set(java.util.Calendar.MONTH, month - 1)
        set(java.util.Calendar.DAY_OF_MONTH, 1)
    }
    return calendar.get(java.util.Calendar.DAY_OF_WEEK) - java.util.Calendar.SUNDAY
}

private fun dayOfWeekOfDate(year: Int, month: Int, day: Int): Int {
    val calendar = java.util.Calendar.getInstance(java.util.Locale.TAIWAN).apply {
        set(java.util.Calendar.YEAR, year)
        set(java.util.Calendar.MONTH, month - 1)
        set(java.util.Calendar.DAY_OF_MONTH, day)
    }
    return calendar.get(java.util.Calendar.DAY_OF_WEEK)
}

private fun String.normalizedLabel(): String =
    filterNot { it.isWhitespace() }

private fun String.isUnavailableSlotText(): Boolean {
    return toUnavailableSlotStatusOrNull() != null
}

private fun String.toUnavailableSlotStatusOrNull(): UnavailableSlotStatus? {
    val value = normalizedLabel()
    return when {
        value.contains("請假") -> UnavailableSlotStatus.LEAVE
        value.contains("關診") -> UnavailableSlotStatus.CLOSED
        value.contains("停診") || value.contains("停止掛號") -> UnavailableSlotStatus.STOPPED
        value.contains("額滿") || value.contains("已額滿") || value.contains("預約額滿") -> UnavailableSlotStatus.FULL
        else -> null
    }
}

private fun String.toReservationSectionOrNull(): ReservationSection? {
    val value = trim()
    return when {
        value.contains("初診預約") -> ReservationSection.INITIAL
        value.contains("複診掛號") || value.contains("復診掛號") -> ReservationSection.RETURN_VISIT
        else -> null
    }
}

private fun String?.toVisitSessionOrNull(): String? {
    val value = this?.trim().orEmpty()
    return when {
        value.contains("上午") -> "上午診"
        value.contains("下午") -> "下午診"
        value.contains("晚上") || value.contains("夜間") || value.contains("夜診") -> "夜間診"
        else -> null
    }
}
