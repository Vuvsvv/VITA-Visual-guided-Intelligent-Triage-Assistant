package com.example.medicalaiguidance

import com.example.medicalaiguidance.network.RecommendationItemDto
import com.example.medicalaiguidance.network.TtsRequest
import com.example.medicalaiguidance.network.VoiceTtsResponseDto
import com.example.medicalaiguidance.screen.DoctorCardSpeech
import com.example.medicalaiguidance.screen.displayedDoctorRecommendations
import com.example.medicalaiguidance.screen.spokenIntroduction
import kotlinx.coroutines.test.runCurrent
import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Test

@OptIn(kotlinx.coroutines.ExperimentalCoroutinesApi::class)
class FrontendPresentationUnitTest {
    private fun item(id: String, doctorId: String?, date: String) = RecommendationItemDto(
        recommendationId = id, doctorId = doctorId, doctor = "測試醫師",
        parentDept = "外科系", childDept = "一般骨科", date = date,
        session = "上午", sessionTime = "08:30-12:00", score = 90.0
    )

    @Test fun doctorListRetainsBackendOrderAndDateFilterExposesAllSlots() {
        val ranked = listOf(item("r1", "d1", "2026-10-22"),
            item("r2", "d1", "2026-10-23"), item("r3", "d2", "2026-10-23"))
        assertEquals(listOf("r1", "r3"), displayedDoctorRecommendations(ranked, null).map { it.recommendationId })
        assertEquals(listOf("r2", "r3"), displayedDoctorRecommendations(ranked, "2026-10-23").map { it.recommendationId })
        assertTrue(displayedDoctorRecommendations(ranked, "2099-01-01").isEmpty())
        assertEquals(3, ranked.size)
    }

    @Test fun missingDoctorIdFallsBackToDoctorName() {
        val ranked = listOf(item("r1", null, "2026-10-22"), item("r2", "", "2026-10-23"))
        assertEquals(listOf("r1"), displayedDoctorRecommendations(ranked, null).map { it.recommendationId })
    }

    @Test fun speechUsesCurrentRecommendationWithoutRoomOrScore() {
        val recommendation = item("r1", "d1", "2026-10-22").copy(rank = 1, room = "3202")
        val text = recommendation.spokenIntroduction()
        assertEquals("推薦您測試醫師。科別為一般骨科。時間為10月22日星期四上午八點半到下午十二點。", text)
        assertFalse(text.contains("3202"))
        assertFalse(recommendation.copy(rank = 2).spokenIntroduction().startsWith("推薦您"))
    }

    @Test fun speakerUsesBackendTtsAndTogglesOff() = runTest {
        val requests = mutableListOf<TtsRequest>()
        var plays = 0
        val speech = DoctorCardSpeech(this, { requests += it; VoiceTtsResponseDto(audioBase64 = "audio") },
            { _, _, _ -> plays++ }, {})
        speech.toggle("r1", "目前醫師資訊", "台語")
        runCurrent()
        assertEquals(listOf(TtsRequest(text = "目前醫師資訊", lang = "taiwanese")), requests)
        assertEquals(1, plays)
        assertEquals("r1", speech.activeId)
        speech.toggle("r1", "目前醫師資訊", "台語")
        assertNull(speech.activeId)
        assertEquals(1, requests.size)
        speech.close()
    }

    @Test fun failedBackendTtsClearsPlaybackAndShowsError() = runTest {
        val speech = DoctorCardSpeech(this, { throw IllegalStateException("unavailable") },
            { _, _, _ -> fail("Unexpected playback") }, {})
        speech.toggle("r1", "目前醫師資訊", "國語")
        runCurrent()
        assertNull(speech.activeId)
        assertNotNull(speech.error)
        speech.close()
    }
}
