package com.example.medicalaiguidance.util

import android.content.Context
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.util.Locale

class SystemTextSpeaker(context: Context) : TextToSpeech.OnInitListener {
    private val mainHandler = Handler(Looper.getMainLooper())
    private var textToSpeech: TextToSpeech? = TextToSpeech(context.applicationContext, this)
    private var isReady = false
    private var pendingSpeech: PendingSpeech? = null

    override fun onInit(status: Int) {
        isReady = status == TextToSpeech.SUCCESS
        val pending = pendingSpeech
        pendingSpeech = null
        if (isReady && pending != null) {
            speakNow(pending)
        } else if (!isReady) {
            pending?.onError?.invoke()
        }
    }

    fun speak(
        text: String,
        lang: String,
        onStart: () -> Unit = {},
        onDone: () -> Unit = {},
        onError: () -> Unit = {}
    ) {
        if (text.isBlank()) {
            onError()
            return
        }

        val speech = PendingSpeech(text, lang, onStart, onDone, onError)
        if (isReady) {
            speakNow(speech)
        } else {
            pendingSpeech = speech
        }
    }

    fun stop() {
        pendingSpeech = null
        textToSpeech?.stop()
    }

    fun shutdown() {
        pendingSpeech = null
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
        isReady = false
    }

    private fun speakNow(speech: PendingSpeech) {
        val engine = textToSpeech
        if (engine == null) {
            speech.onError()
            return
        }

        val locale = if (speech.lang == "taiwanese") Locale.TAIWAN else Locale.TRADITIONAL_CHINESE
        val languageResult = engine.setLanguage(locale)
        if (
            languageResult == TextToSpeech.LANG_MISSING_DATA ||
            languageResult == TextToSpeech.LANG_NOT_SUPPORTED
        ) {
            val fallbackResult = engine.setLanguage(Locale.CHINESE)
            if (
                fallbackResult == TextToSpeech.LANG_MISSING_DATA ||
                fallbackResult == TextToSpeech.LANG_NOT_SUPPORTED
            ) {
                speech.onError()
                return
            }
        }

        val utteranceId = "system_tts_${System.currentTimeMillis()}"
        engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                mainHandler.post { speech.onStart() }
            }

            override fun onDone(utteranceId: String?) {
                mainHandler.post { speech.onDone() }
            }

            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) {
                mainHandler.post { speech.onError() }
            }

            override fun onError(utteranceId: String?, errorCode: Int) {
                mainHandler.post { speech.onError() }
            }
        })

        engine.stop()
        val result = engine.speak(speech.text, TextToSpeech.QUEUE_FLUSH, Bundle(), utteranceId)
        if (result == TextToSpeech.ERROR) {
            speech.onError()
        }
    }

    private data class PendingSpeech(
        val text: String,
        val lang: String,
        val onStart: () -> Unit,
        val onDone: () -> Unit,
        val onError: () -> Unit
    )
}
