package com.example.medicalaiguidance

import android.os.SystemClock
import android.util.Log
import android.media.MediaPlayer
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.medicalaiguidance.network.AvailabilityDto
import com.example.medicalaiguidance.network.BatchAnswerDto
import com.example.medicalaiguidance.network.FollowupRecommendRequest
import com.example.medicalaiguidance.network.MedicalApiClient
import com.example.medicalaiguidance.network.QuickSearchRequest
import com.example.medicalaiguidance.repository.MedicalRepository
import com.example.medicalaiguidance.util.AudioPlayer
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Test-only device E2E for the 0913 benchmark. It uses the production Android
 * repository/API parser against the configured Backend and never changes app behavior.
 */
@RunWith(AndroidJUnit4::class)
class FinalAndroidE2E0913InstrumentedTest {
    @Test
    fun benchmarkTaiwaneseAsrUploadToTranscript() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val audio = context.assets.open("test_taiwanese_asr.wav").use { it.readBytes() }
        repeat(5) { index ->
            val repository = benchmarkRepository()
            val start = SystemClock.elapsedRealtimeNanos()
            val result = repository.transcribeVoice(audio, "taiwanese")
            val latencyMs = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
            val success = !result.asrFailed && result.text.isNotBlank()
            Log.i(
                TAG,
                "flow=taiwanese_asr_user_perceived|run=${index + 1}|latency_ms=${format(latencyMs)}|success=$success|question_count=0|result_count=${if (result.text.isNotBlank()) 1 else 0}|ai_calls=0"
            )
            assertTrue(success)
        }
    }

    @Test
    fun benchmarkDynamicTtsColdClickToAudioStart() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val text = "目前建議科別為 一般骨科，請確認後取得推薦掛號方案。"
        listOf("chinese", "taiwanese").forEach { language ->
            repeat(5) { index ->
                val repository = benchmarkRepository()
                val session = repository.beginTtsSession()
                val audioPlayer = AudioPlayer()
                var failed = false
                val start = SystemClock.elapsedRealtimeNanos()
                val result = session.get(text, language)
                assertFalse(result.ttsFailed)
                audioPlayer.playBase64(
                    result.audioBase64,
                    context.cacheDir,
                    result.audioFormat,
                    onError = { failed = true }
                )
                val latencyMs = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
                SystemClock.sleep(10)
                val mediaPlayer = audioPlayer.currentMediaPlayerForBenchmark()
                val success = !failed && mediaPlayer.isPlaying
                Log.i(
                    TAG,
                    "flow=dynamic_${language}_tts_user_perceived|run=${index + 1}|latency_ms=${format(latencyMs)}|success=$success|question_count=0|result_count=1|ai_calls=0"
                )
                assertTrue(success)
                audioPlayer.release()
                session.close()
            }
        }
    }

    @Test
    fun benchmarkInitialAndFollowupTriageToRecommendations() = runBlocking {
        listOf("initial", "followup").forEach { visitType ->
            repeat(5) { index ->
                val repository = benchmarkRepository()
                repository.clearRecommendationFlow()
                val start = SystemClock.elapsedRealtimeNanos()
                var result = repository.startBatchTriage(visitType)
                var questionCount = 0
                while (result.needMoreInfo && questionCount < 12) {
                    val key = result.questionBatch.firstOrNull()?.key
                        ?: result.conversationState.lastQuestionKey
                        ?: error("Missing next question key")
                    val answer = answers[key] ?: error("No controlled answer for $key")
                    result = repository.submitBatchAnswers(
                        caseId = result.caseId,
                        visitType = visitType,
                        answers = listOf(BatchAnswerDto(key, answer))
                    )
                    questionCount += 1
                }
                assertTrue("Department result should be available", result.departmentResult != null)
                val confirmed = repository.confirmTriage(result.caseId)
                assertTrue(confirmed.conversationState.confirmed)
                val recommendation = repository.recommend(result.caseId, visitType)
                val count = recommendation.totalCount
                val latencyMs = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
                val success = count > 0 && recommendation.fallbackDepartments.isEmpty()
                Log.i(
                    TAG,
                    "flow=$visitType|run=${index + 1}|latency_ms=${format(latencyMs)}|success=$success|question_count=$questionCount|result_count=$count|ai_calls=not_device_observable"
                )
                assertTrue(success)
                assertTrue(recommendation.recommendations.specialtyFirst.size <= 5)
                assertTrue(recommendation.recommendations.timeFirst.size <= 5)
            }
        }
    }

    @Test
    fun benchmarkQuickSearch() = runBlocking {
        repeat(5) { index ->
            val repository = benchmarkRepository()
            val start = SystemClock.elapsedRealtimeNanos()
            val result = repository.quickSearch(
                QuickSearchRequest(deptId = 1298, date = "2026-09-14", period = "afternoon")
            )
            val latencyMs = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
            val success = result.results.isNotEmpty()
            Log.i(
                TAG,
                "flow=quick_search|run=${index + 1}|latency_ms=${format(latencyMs)}|success=$success|question_count=0|result_count=${result.totalCount}|ai_calls=0"
            )
            assertTrue(success)
        }
    }

    @Test
    fun benchmarkReturnVisit() = runBlocking {
        repeat(5) { index ->
            val repository = benchmarkRepository()
            val start = SystemClock.elapsedRealtimeNanos()
            val result = repository.followupRecommend(
                FollowupRecommendRequest(
                    childDept = "一般骨科",
                    parentDept = "外科系",
                    deptId = "1298",
                    originalDoctor = "邱方遙",
                    originalDoctorId = "7149",
                    followupReason = "依原醫師回診",
                    availability = AvailabilityDto(
                        preferredDates = listOf("2026-09-14"),
                        preferredSessions = listOf("下午")
                    )
                )
            )
            val latencyMs = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
            val success = result.recommendations.isNotEmpty() &&
                result.department?.deptId == 1298 &&
                result.recommendations.all { it.doctorId == "7149" }
            Log.i(
                TAG,
                "flow=return_visit|run=${index + 1}|latency_ms=${format(latencyMs)}|success=$success|question_count=0|result_count=${result.totalCount}|ai_calls=0"
            )
            assertTrue(success)
            assertFalse(result.recommendations.any { it.doctorId != "7149" })
        }
    }

    private fun format(value: Double): String = "%.3f".format(java.util.Locale.US, value)

    private fun benchmarkRepository(): MedicalRepository =
        MedicalRepository(MedicalApiClient("http://10.0.2.2:18080"))

    private fun AudioPlayer.currentMediaPlayerForBenchmark(): MediaPlayer {
        val field = AudioPlayer::class.java.getDeclaredField("player")
        field.isAccessible = true
        return field.get(this) as MediaPlayer
    }

    companion object {
        private const val TAG = "BENCH0913E2E"
        private val answers = mapOf(
            "symptom" to "膝蓋卡卡的",
            "red_flags" to "都沒有",
            "body_part" to "膝蓋",
            "duration" to "一個月",
            "severity" to "中等",
            "preferred_days" to "哪天都可以",
            "preferred_sessions" to "都可以",
            "department_clarification" to "活動時膝蓋會卡住"
        )
    }
}
