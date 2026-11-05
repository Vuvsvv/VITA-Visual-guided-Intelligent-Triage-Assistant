package com.example.medicalaiguidance

import com.example.medicalaiguidance.model.VisitPlan
import com.example.medicalaiguidance.network.RecommendationItemDto
import com.example.medicalaiguidance.repository.MedicalRepository
import com.example.medicalaiguidance.service.AppointmentType
import com.example.medicalaiguidance.service.toAccessibilityAppointmentTypeOrNull
import com.example.medicalaiguidance.viewmodel.ConfirmViewModel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class AppointmentTypeMappingUnitTest {
    @Test
    fun initialMapsToInitial() {
        assertEquals(
            AppointmentType.INITIAL,
            VisitPlan.INITIAL.toAccessibilityAppointmentTypeOrNull()
        )
    }

    @Test
    fun followupMapsToFollowupReservationSection() {
        assertEquals(
            AppointmentType.FOLLOWUP,
            VisitPlan.FOLLOW_UP.toAccessibilityAppointmentTypeOrNull()
        )
    }

    @Test
    fun quickSearchMapsToFollowupReservationSection() {
        assertEquals(
            AppointmentType.FOLLOWUP,
            VisitPlan.QUICK_SEARCH.toAccessibilityAppointmentTypeOrNull()
        )
    }

    @Test
    fun allFormalVisitPlansKeepTheirExpectedMappings() {
        val mappings = mapOf(
            VisitPlan.INITIAL to AppointmentType.INITIAL,
            VisitPlan.FOLLOW_UP to AppointmentType.FOLLOWUP,
            VisitPlan.QUICK_SEARCH to AppointmentType.FOLLOWUP
        )

        mappings.forEach { (visitPlan, expected) ->
            assertEquals(expected, visitPlan.toAccessibilityAppointmentTypeOrNull())
        }
        assertEquals(1, mappings.values.count { it == AppointmentType.INITIAL })
        assertEquals(2, mappings.values.count { it == AppointmentType.FOLLOWUP })
    }

    @Test
    fun visitPlanContainsOnlyFormalProductFlowsAndUnknown() {
        assertEquals(
            setOf(VisitPlan.INITIAL, VisitPlan.FOLLOW_UP, VisitPlan.QUICK_SEARCH, VisitPlan.UNKNOWN),
            VisitPlan.entries.toSet()
        )
    }

    @Test
    fun unknownDoesNotSilentlyFallBackToInitial() {
        assertNull(VisitPlan.UNKNOWN.toAccessibilityAppointmentTypeOrNull())
    }

    @Test
    fun confirmViewModelReadsEachFormalVisitPlanFromRepositoryState() {
        val repository = MedicalRepository()

        listOf(VisitPlan.INITIAL, VisitPlan.FOLLOW_UP, VisitPlan.QUICK_SEARCH).forEach { visitPlan ->
            repository.clearRecommendationFlow()
            repository.setActiveVisitType(visitPlan.apiValue)
            repository.selectRecommendation(testRecommendation())

            val viewModel = ConfirmViewModel(repository)

            assertEquals(visitPlan, viewModel.visitPlan.value)
        }
    }

    private fun testRecommendation() = RecommendationItemDto(
        recommendationId = "rec_mapping_test",
        parentDept = "內科系",
        childDept = "一般內科",
        doctor = "測試醫師",
        date = "2026-09-10",
        session = "上午",
        slot = "一診",
        score = 1.0
    )
}
