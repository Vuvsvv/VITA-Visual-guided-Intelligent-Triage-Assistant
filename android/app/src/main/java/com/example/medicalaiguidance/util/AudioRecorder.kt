package com.example.medicalaiguidance.util

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.ByteArrayOutputStream

class AudioRecorder {
    private var recorder: AudioRecord? = null
    private var recordingThread: Thread? = null
    @Volatile private var isRecording = false
    private val pcmBuffer = ByteArrayOutputStream()

    private val sampleRate = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT

    @SuppressLint("MissingPermission")
    fun start(): Boolean {
        if (isRecording) return true
        val minBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)
            .coerceAtLeast(sampleRate)

        pcmBuffer.reset()
        val audioRecord = try {
            AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                minBufferSize
            )
        } catch (_: RuntimeException) {
            return false
        }
        if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
            audioRecord.release()
            return false
        }
        try {
            audioRecord.startRecording()
        } catch (_: RuntimeException) {
            audioRecord.release()
            return false
        }
        if (audioRecord.recordingState != AudioRecord.RECORDSTATE_RECORDING) {
            audioRecord.release()
            return false
        }
        recorder = audioRecord
        isRecording = true

        recordingThread = Thread {
            val buffer = ByteArray(minBufferSize)
            while (isRecording) {
                val read = recorder?.read(buffer, 0, buffer.size) ?: 0
                if (read > 0) {
                    pcmBuffer.write(buffer, 0, read)
                }
            }
        }.also { it.start() }
        return true
    }

    fun stop(): ByteArray {
        if (!isRecording) return ByteArray(0)
        isRecording = false
        recordingThread?.join(1000)
        recordingThread = null

        runCatching { recorder?.stop() }
        recorder?.release()
        recorder = null

        val pcm = pcmBuffer.toByteArray()
        return if (pcm.isEmpty()) ByteArray(0) else pcmToWav(pcm)
    }

    fun cancel() {
        isRecording = false
        runCatching { recorder?.stop() }
        recorder?.release()
        recorder = null
        recordingThread = null
        pcmBuffer.reset()
    }

    private fun pcmToWav(pcm: ByteArray): ByteArray {
        val out = ByteArrayOutputStream()
        val byteRate = sampleRate * 2
        val totalDataLen = pcm.size + 36

        fun writeInt(value: Int) {
            out.write(value and 0xff)
            out.write((value shr 8) and 0xff)
            out.write((value shr 16) and 0xff)
            out.write((value shr 24) and 0xff)
        }

        fun writeShort(value: Int) {
            out.write(value and 0xff)
            out.write((value shr 8) and 0xff)
        }

        out.write("RIFF".toByteArray())
        writeInt(totalDataLen)
        out.write("WAVE".toByteArray())
        out.write("fmt ".toByteArray())
        writeInt(16)
        writeShort(1)
        writeShort(1)
        writeInt(sampleRate)
        writeInt(byteRate)
        writeShort(2)
        writeShort(16)
        out.write("data".toByteArray())
        writeInt(pcm.size)
        out.write(pcm)
        return out.toByteArray()
    }
}
