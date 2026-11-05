package com.example.medicalaiguidance

import android.media.MediaPlayer
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.example.medicalaiguidance.util.AudioPlayer
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotSame
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class FixedTriageAudioPlaybackInstrumentedTest {
    @Test
    fun zhAndTaigiShortAndLongResourcesStartAndSwitchWithoutOverlap() {
        val context = ApplicationProvider.getApplicationContext<android.content.Context>()
        val audioPlayer = AudioPlayer()
        val resources = listOf(
            R.raw.triage_symptom_01_zh,
            R.raw.triage_emergency_symptoms_01_zh,
            R.raw.triage_symptom_01_taigi,
            R.raw.triage_emergency_symptoms_01_taigi
        )

        try {
            var previous: MediaPlayer? = null
            resources.forEach { resourceId ->
                var failed = false
                audioPlayer.playRawResource(context, resourceId, onError = { failed = true })
                Thread.sleep(150)
                val current = audioPlayer.currentMediaPlayerForTest()
                assertFalse(failed)
                assertTrue(current.isPlaying)
                previous?.let { old ->
                    assertNotSame(old, current)
                    assertTrue(old.isReleasedOrStopped())
                }
                previous = current
            }
        } finally {
            audioPlayer.release()
        }
    }

    private fun AudioPlayer.currentMediaPlayerForTest(): MediaPlayer {
        val field = AudioPlayer::class.java.getDeclaredField("player")
        field.isAccessible = true
        return field.get(this) as MediaPlayer
    }

    private fun MediaPlayer.isReleasedOrStopped(): Boolean =
        try {
            !isPlaying
        } catch (_: IllegalStateException) {
            true
        }
}
