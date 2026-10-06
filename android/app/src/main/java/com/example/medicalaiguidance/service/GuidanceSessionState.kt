package com.example.medicalaiguidance.service

internal enum class GuidanceType {
    BOOKING,
    CANCELLATION
}

/** In-memory lifecycle for one explicitly requested accessibility guidance session. */
internal class GuidanceSessionState {
    var isActive: Boolean = false
        private set
    var guidanceType: GuidanceType? = null
        private set
    var steps: List<String> = emptyList()
        private set
    var currentStepIndex: Int = NO_ACTIVE_STEP
    var generation: Long = 0L
        private set
    var isAwaitingTargetReturn: Boolean = false
        private set

    fun start(type: GuidanceType, newSteps: List<String>) {
        generation += 1
        guidanceType = type
        steps = newSteps.toList()
        currentStepIndex = 0
        isAwaitingTargetReturn = false
        isActive = steps.isNotEmpty()
        if (!isActive) {
            guidanceType = null
            currentStepIndex = NO_ACTIVE_STEP
        }
    }

    fun finish() {
        generation += 1
        isActive = false
        guidanceType = null
        steps = emptyList()
        currentStepIndex = NO_ACTIVE_STEP
        isAwaitingTargetReturn = false
    }

    fun canHandleTargetEvent(): Boolean =
        isActive && guidanceType != null && steps.isNotEmpty() && currentStepIndex in steps.indices

    fun isCurrentGeneration(value: Long): Boolean = canHandleTargetEvent() && generation == value

    fun markTargetLeft(): Boolean {
        if (!canHandleTargetEvent() || isAwaitingTargetReturn) return false
        isAwaitingTargetReturn = true
        return true
    }

    fun markTargetReturned() {
        isAwaitingTargetReturn = false
    }

    fun shouldAbortAfterExitGrace(expectedGeneration: Long): Boolean =
        isCurrentGeneration(expectedGeneration) && isAwaitingTargetReturn

    companion object {
        const val NO_ACTIVE_STEP = -1
    }
}
