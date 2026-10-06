package com.example.medicalaiguidance.repository

import com.example.medicalaiguidance.network.VoiceTtsResponseDto
import java.util.UUID
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** Main-thread confined; shared by prefetch and explicit playback for one consultation. */
class TtsSession(
    private val scope: CoroutineScope,
    private val ttlMillis: Long = 30 * 60 * 1000L,
    private val load: suspend (String, String, String) -> VoiceTtsResponseDto
) {
    val id: String = UUID.randomUUID().toString()
    var closed = false
        private set
    private val audio = mutableMapOf<Pair<String, String>, VoiceTtsResponseDto>()
    private data class Pending(
        val key: Pair<String, String>,
        var foreground: Boolean,
        val result: CompletableDeferred<VoiceTtsResponseDto> = CompletableDeferred(),
        var started: Boolean = false
    )
    private val pending = linkedMapOf<Pair<String, String>, Pending>()
    private var expiry: Job? = null
    private var prefetchPaused = false
    private var latestPrefetch: Pair<String, String>? = null
    private var activeBackground = 0

    /** Only the currently displayed question is eligible for background work. */
    fun prefetch(text: String, lang: String, paused: Boolean = prefetchPaused) {
        if (closed) return
        prefetchPaused = paused
        latestPrefetch = text.trim() to lang
        discardUnsentPrefetch()
        queueLatestPrefetch()
    }

    fun setPrefetchPaused(paused: Boolean) {
        if (closed) return
        prefetchPaused = paused
        if (paused) discardUnsentPrefetch() else queueLatestPrefetch()
    }

    private fun discardUnsentPrefetch() {
        pending.values.filter { !it.started && !it.foreground }.toList().forEach {
            pending.remove(it.key)
            it.result.cancel()
        }
    }

    private fun queueLatestPrefetch() {
        val key = latestPrefetch ?: return
        if (prefetchPaused || closed || audio.containsKey(key)) return
        touch()
        pending.getOrPut(key) { Pending(key, foreground = false) }
        dispatch()
    }

    private fun touch() {
        expiry?.cancel()
        expiry = scope.launch {
            delay(ttlMillis)
            audio.clear()
        }
    }

    /** Explicit playback can promote an unsent prefetch without a second request. */
    suspend fun get(text: String, lang: String, retryFailure: Boolean = false): VoiceTtsResponseDto {
        check(!closed) { "TTS session has ended" }
        touch()
        val key = text.trim() to lang
        audio[key]?.let { if (!it.ttsFailed || !retryFailure) return it }
        val entry = pending.getOrPut(key) { Pending(key, foreground = true) }
        entry.foreground = true
        dispatch()
        // Cancelling a playback waiter never cancels a dispatched HTTP request.
        return entry.result.await()
    }

    private fun dispatch() {
        if (closed) return
        // User playback bypasses queued prefetch. ASR never acquires this queue.
        pending.values.filter { it.foreground && !it.started }.toList().forEach(::start)
        if (!prefetchPaused && activeBackground == 0 && pending.values.none { it.foreground }) {
            pending.values.firstOrNull { !it.started }?.let(::start)
        }
    }

    private fun start(entry: Pending) {
        if (closed || entry.started || pending[entry.key] !== entry) return
        entry.started = true
        val background = !entry.foreground
        if (background) activeBackground++
        // Enter the loader in this main-thread turn: pause only removes work that
        // has not been dispatched. It never attempts to cancel blocking HTTP.
        scope.launch(start = CoroutineStart.UNDISPATCHED) {
            try {
                val result = try {
                    load(id, entry.key.first, entry.key.second).let {
                        if (it.audioBase64.isBlank()) it.copy(ttsFailed = true) else it
                    }
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: Exception) {
                    VoiceTtsResponseDto(ttsFailed = true, error = error.message)
                }
                if (!closed) audio[entry.key] = result
                entry.result.complete(result)
            } catch (cancelled: CancellationException) {
                entry.result.cancel(cancelled)
                throw cancelled
            } finally {
                pending.remove(entry.key)
                if (background) activeBackground--
                dispatch()
            }
        }
    }

    fun close() {
        closed = true
        audio.clear()
        pending.values.forEach { it.result.cancel() }
        pending.clear()
        scope.cancel()
    }
}
