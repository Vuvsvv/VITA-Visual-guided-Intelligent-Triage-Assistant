package com.example.medicalaiguidance

import com.example.medicalaiguidance.service.GuidanceSessionState
import com.example.medicalaiguidance.service.GuidanceType
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class GuidanceSessionStateUnitTest {
    private val bookingSteps = listOf("行動掛號", "SUCCESS_FINISH")
    private val cancellationSteps = listOf("掛號查詢", "CANCELLATION_FINISH")

    @Test
    fun bookingSuccessClearsTheWholeSession() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.BOOKING, bookingSteps)

        session.finish()

        assertFinished(session)
    }

    @Test
    fun cancellationSuccessClearsTheWholeSession() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.CANCELLATION, cancellationSteps)

        session.finish()

        assertFinished(session)
    }

    @Test
    fun targetEventAfterBookingSuccessCannotShowStepZero() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.BOOKING, bookingSteps)
        session.finish()

        assertFalse(session.canHandleTargetEvent())
        assertEquals(GuidanceSessionState.NO_ACTIVE_STEP, session.currentStepIndex)
    }

    @Test
    fun targetEventAfterCancellationSuccessCannotShowStepZero() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.CANCELLATION, cancellationSteps)
        session.finish()

        assertFalse(session.canHandleTargetEvent())
        assertEquals(GuidanceSessionState.NO_ACTIVE_STEP, session.currentStepIndex)
    }

    @Test
    fun aNewBookingSessionStillStartsAtStepZero() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.BOOKING, bookingSteps)
        session.finish()

        session.start(GuidanceType.BOOKING, bookingSteps)

        assertTrue(session.canHandleTargetEvent())
        assertEquals(GuidanceType.BOOKING, session.guidanceType)
        assertEquals(0, session.currentStepIndex)
        assertEquals("行動掛號", session.steps.first())
    }

    @Test
    fun aNewCancellationSessionStillStartsAtStepZero() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.CANCELLATION, cancellationSteps)
        session.finish()

        session.start(GuidanceType.CANCELLATION, cancellationSteps)

        assertTrue(session.canHandleTargetEvent())
        assertEquals(GuidanceType.CANCELLATION, session.guidanceType)
        assertEquals(0, session.currentStepIndex)
        assertEquals("掛號查詢", session.steps.first())
    }

    @Test
    fun delayedCallbackFromFinishedSessionIsInvalidated() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.BOOKING, bookingSteps)
        val oldGeneration = session.generation

        session.finish()

        assertFalse(session.isCurrentGeneration(oldGeneration))
    }

    @Test
    fun finishClearsLastGuidanceTypeBeforeAnotherTargetEvent() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.CANCELLATION, cancellationSteps)

        session.finish()

        assertNull(session.guidanceType)
        assertFalse(session.canHandleTargetEvent())
    }

    @Test
    fun returningWithinExitGraceKeepsTheActiveStep() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.BOOKING, bookingSteps)
        session.currentStepIndex = 1
        val generation = session.generation

        assertTrue(session.markTargetLeft())
        session.markTargetReturned()

        assertFalse(session.shouldAbortAfterExitGrace(generation))
        assertTrue(session.canHandleTargetEvent())
        assertEquals(1, session.currentStepIndex)
    }

    @Test
    fun remainingOutsideAfterExitGraceAllowsSessionAbort() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.CANCELLATION, cancellationSteps)
        val generation = session.generation

        assertTrue(session.markTargetLeft())
        assertTrue(session.shouldAbortAfterExitGrace(generation))
        session.finish()

        assertFinished(session)
    }

    @Test
    fun bookingGuidanceIsAbortedAfterRemainingOutsideTheTarget() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.BOOKING, bookingSteps)
        val generation = session.generation

        session.markTargetLeft()
        assertTrue(session.shouldAbortAfterExitGrace(generation))
        session.finish()

        assertFinished(session)
    }

    @Test
    fun openingTargetAfterAbortCannotRestartStepZero() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.BOOKING, bookingSteps)
        session.markTargetLeft()
        session.finish()

        assertFalse(session.canHandleTargetEvent())
        assertEquals(GuidanceSessionState.NO_ACTIVE_STEP, session.currentStepIndex)
    }

    @Test
    fun newBookingCanStartAtStepZeroAfterAbort() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.BOOKING, bookingSteps)
        session.markTargetLeft()
        session.finish()

        session.start(GuidanceType.BOOKING, bookingSteps)

        assertEquals(GuidanceType.BOOKING, session.guidanceType)
        assertEquals(0, session.currentStepIndex)
        assertTrue(session.canHandleTargetEvent())
    }

    @Test
    fun newCancellationCanStartAtStepZeroAfterAbort() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.CANCELLATION, cancellationSteps)
        session.markTargetLeft()
        session.finish()

        session.start(GuidanceType.CANCELLATION, cancellationSteps)

        assertEquals(GuidanceType.CANCELLATION, session.guidanceType)
        assertEquals(0, session.currentStepIndex)
        assertTrue(session.canHandleTargetEvent())
    }

    @Test
    fun repeatedOutsideEventsDoNotRestartTheExitGracePeriod() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.BOOKING, bookingSteps)

        assertTrue(session.markTargetLeft())
        assertFalse(session.markTargetLeft())
    }

    @Test
    fun exitCallbackFromOldSessionCannotAbortANewSession() {
        val session = GuidanceSessionState()
        session.start(GuidanceType.BOOKING, bookingSteps)
        val oldGeneration = session.generation
        session.markTargetLeft()
        session.finish()
        session.start(GuidanceType.CANCELLATION, cancellationSteps)

        assertFalse(session.shouldAbortAfterExitGrace(oldGeneration))
        assertTrue(session.canHandleTargetEvent())
        assertEquals(GuidanceType.CANCELLATION, session.guidanceType)
    }

    private fun assertFinished(session: GuidanceSessionState) {
        assertFalse(session.isActive)
        assertNull(session.guidanceType)
        assertTrue(session.steps.isEmpty())
        assertEquals(GuidanceSessionState.NO_ACTIVE_STEP, session.currentStepIndex)
        assertFalse(session.isAwaitingTargetReturn)
        assertFalse(session.canHandleTargetEvent())
    }
}
