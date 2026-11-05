package com.example.medicalaiguidance.repository

import android.content.Context
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** Shared across screens and restored after the app restarts. */
object VoiceLanguagePreference {
    private val selected = MutableStateFlow("國語")
    private val selectedState = selected.asStateFlow()
    private var initialized = false

    @Synchronized
    fun observe(context: Context): StateFlow<String> {
        if (!initialized) {
            selected.value = context.applicationContext
                .getSharedPreferences("voice_preferences", Context.MODE_PRIVATE)
                .getString("language", "國語")
                .let { if (it == "台語") "台語" else "國語" }
            initialized = true
        }
        return selectedState
    }

    @Synchronized
    fun select(context: Context, language: String) {
        require(language == "國語" || language == "台語")
        observe(context)
        selected.value = language
        context.applicationContext.getSharedPreferences("voice_preferences", Context.MODE_PRIVATE)
            .edit().putString("language", language).apply()
    }
}
