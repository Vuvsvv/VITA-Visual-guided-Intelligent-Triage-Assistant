package com.example.medicalaiguidance

import android.media.MediaPlayer
import android.os.SystemClock
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.medicalaiguidance.util.AudioPlayer
import com.example.medicalaiguidance.util.FixedTriageAudioResolver
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/** Test-only 0913 benchmark instrumentation. It does not change app runtime behavior. */
@RunWith(AndroidJUnit4::class)
class FinalBenchmark0913InstrumentedTest {
    @Test
    fun benchmarkFixedPrerecordedAudioStartup() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val cases = listOf(
            Case(
                subgroup = "zh_short",
                text = "請描述目前最主要的不舒服症狀。",
                language = "chinese",
                expectedResource = R.raw.triage_symptom_01_zh
            ),
            Case(
                subgroup = "zh_long_red_flag",
                text = "請問是否有突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛等急迫症狀？",
                language = "chinese",
                expectedResource = R.raw.triage_emergency_symptoms_01_zh
            ),
            Case(
                subgroup = "taigi_short",
                text = "請描述目前最主要的不舒服症狀。",
                language = "taiwanese",
                expectedResource = R.raw.triage_symptom_01_taigi
            ),
            Case(
                subgroup = "taigi_long_red_flag",
                text = "請問是否有突發胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛等急迫症狀？",
                language = "taiwanese",
                expectedResource = R.raw.triage_emergency_symptoms_01_taigi
            )
        )

        cases.forEach { case ->
            repeat(20) { index ->
                val resource = FixedTriageAudioResolver.resolve(case.text, case.language)
                assertEquals(case.expectedResource, resource)
                val player = AudioPlayer()
                var failed = false
                val start = SystemClock.elapsedRealtimeNanos()
                player.playRawResource(context, resource!!, onError = { failed = true })
                val latencyMs = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0
                SystemClock.sleep(10)
                val mediaPlayer = player.currentMediaPlayerForBenchmark()
                val success = !failed && mediaPlayer.isPlaying
                Log.i(
                    TAG,
                    "subgroup=${case.subgroup}|run=${index + 1}|latency_ms=${"%.3f".format(java.util.Locale.US, latencyMs)}|success=$success|local_audio_hit=true|voice_tts_called=false|resource_id=$resource"
                )
                assertFalse(failed)
                assertTrue(mediaPlayer.isPlaying)
                player.release()
            }
        }
    }

    private fun AudioPlayer.currentMediaPlayerForBenchmark(): MediaPlayer {
        val field = AudioPlayer::class.java.getDeclaredField("player")
        field.isAccessible = true
        return field.get(this) as MediaPlayer
    }

    private data class Case(
        val subgroup: String,
        val text: String,
        val language: String,
        val expectedResource: Int
    )

    companion object {
        private const val TAG = "BENCH0913"
    }
}
