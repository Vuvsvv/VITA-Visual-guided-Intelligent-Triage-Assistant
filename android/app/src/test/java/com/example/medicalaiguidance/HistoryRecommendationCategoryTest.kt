package com.example.medicalaiguidance

import com.example.medicalaiguidance.model.History
import com.example.medicalaiguidance.model.HistoryRecommendation
import com.example.medicalaiguidance.model.HistoryStatus
import com.example.medicalaiguidance.model.recommendationCategory
import com.example.medicalaiguidance.repository.parseHistoryJson
import com.example.medicalaiguidance.repository.toJson
import org.junit.Assert.assertEquals
import org.junit.Test

class HistoryRecommendationCategoryTest {
    private fun item(id: String, category: String? = null) = HistoryRecommendation(
        id, "內科", "一般內科", "醫師", "2026-11-02", "上午", category = category
    )

    @Test fun oldIdsRestoreCategoriesWithoutGuessingUnknownRecords() {
        assertEquals("specialty", item("rec_s_case_001").recommendationCategory())
        assertEquals("time", item("rec_t_case_001").recommendationCategory())
        assertEquals("legacy", item("old-id").recommendationCategory())
        assertEquals("time", item("rec_s_case_001", "time").recommendationCategory())
    }

    @Test fun categoriesAndOrderingSurvivePersistence() {
        val items = listOf(item("second", "specialty"), item("first", "specialty"), item("time", "time"))
        val history = History("case", "2026-11-01", "一般內科", "", HistoryStatus.COMPLETED,
            recommendations = items, selectedRecommendationId = "first")
        val restored = parseHistoryJson("[${history.toJson()}]").single()
        assertEquals(items, restored.recommendations)
        assertEquals("first", restored.selectedRecommendationId)
    }
}
