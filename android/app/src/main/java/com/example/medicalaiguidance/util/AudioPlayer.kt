package com.example.medicalaiguidance.util

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.util.Base64
import android.util.Log
import java.io.File
import java.io.FileOutputStream

class AudioPlayer {
    private var player: MediaPlayer? = null
    private var currentFile: File? = null

    fun playBase64(
        base64Audio: String,
        cacheDir: File,
        audioFormat: String = "wav",
        onDone: () -> Unit = {},
        onError: () -> Unit = {}
    ) {
        if (base64Audio.isBlank()) {
            onError()
            return
        }

        try {
            val sanitizedBase64 = base64Audio.substringAfter("base64,", base64Audio).trim()
            val bytes = Base64.decode(sanitizedBase64, Base64.DEFAULT)
            val extension = base64Audio.dataUrlAudioFormat()?.toAudioFileExtension()
                ?: audioFormat.toAudioFileExtension()
            release()
            val audioFile = File.createTempFile("tts_", ".$extension", cacheDir)
            currentFile = audioFile
            FileOutputStream(audioFile).use { it.write(bytes) }

            Log.d(TAG, "Playing TTS audio path=${audioFile.absolutePath} format=$extension bytes=${bytes.size}")
            player = MediaPlayer().apply {
                setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                setDataSource(audioFile.absolutePath)
                setOnCompletionListener {
                    release()
                    onDone()
                }
                setOnErrorListener { _, what, extra ->
                    Log.w(TAG, "MediaPlayer playback failed what=$what extra=$extra format=$extension")
                    release()
                    onError()
                    true
                }
                prepare()
                start()
            }
        } catch (error: Exception) {
            Log.w(TAG, "Unable to play TTS audio format=$audioFormat base64_len=${base64Audio.length}", error)
            release()
            onError()
        }
    }

    fun playRawResource(
        context: Context,
        rawResourceId: Int,
        onDone: () -> Unit = {},
        onError: () -> Unit = {}
    ) {
        try {
            release()
            val localPlayer = MediaPlayer.create(context, rawResourceId)
            if (localPlayer == null) {
                Log.w(TAG, "Unable to create MediaPlayer for raw resource id=$rawResourceId")
                onError()
                return
            }
            player = localPlayer.apply {
                setOnCompletionListener {
                    release()
                    onDone()
                }
                setOnErrorListener { _, what, extra ->
                    Log.w(TAG, "Raw resource playback failed what=$what extra=$extra resource_id=$rawResourceId")
                    release()
                    onError()
                    true
                }
                start()
                Log.d(TAG, "Playing fixed triage raw resource id=$rawResourceId")
            }
        } catch (error: Exception) {
            Log.w(TAG, "Unable to play raw audio resource id=$rawResourceId", error)
            release()
            onError()
        }
    }

    fun release() {
        player?.release()
        player = null
        currentFile?.delete()
        currentFile = null
    }

    private fun String.toAudioFileExtension(): String {
        val normalized = substringBefore(';')
            .substringAfterLast('/')
            .removePrefix(".")
            .lowercase()
            .trim()
        return when (normalized) {
            "", "wave", "x-wav" -> "wav"
            "mpeg", "mpga", "mpg" -> "mp3"
            "mp4", "m4a", "aac", "ogg", "wav", "mp3" -> normalized
            else -> "wav"
        }
    }

    private fun String.dataUrlAudioFormat(): String? {
        if (!startsWith("data:", ignoreCase = true)) return null
        val header = substringBefore(',', missingDelimiterValue = "")
        return header
            .removePrefix("data:")
            .substringBefore(';')
            .takeIf { it.isNotBlank() }
    }

    companion object {
        const val TAG = "AudioPlayer"

        fun cleanupExpiredFiles(cacheDir: File, now: Long = System.currentTimeMillis()) {
            cacheDir.listFiles()?.filter {
                it.isFile && it.name.startsWith("tts_") && now - it.lastModified() >= 30 * 60 * 1000L
            }?.forEach {
                if (!it.delete()) Log.w(TAG, "Unable to remove expired temporary TTS file")
            }
        }
    }
}
