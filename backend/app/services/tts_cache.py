"""Ephemeral, single-worker session TTS cache; never writes audio to disk/DB.

Like case_store, this requires a single application worker. Each key has one
shared task. HTTP cancellation does not cancel generation for other waiters.
"""
import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.services.voice_client import TtsGatewayResult, VoiceGatewayError

logger = logging.getLogger(__name__)
SESSION_TTL_SECONDS = 30 * 60


@dataclass
class Session:
    touched: float
    audio: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)


class TtsSessionCache:
    def __init__(self, ttl=SESSION_TTL_SECONDS, clock=time.monotonic):
        self.ttl = ttl
        self.clock = clock
        self.sessions: dict[str, Session] = {}
        # Short-lived tombstones prevent a delayed POST recreating cleared audio.
        self.closed: dict[str, float] = {}

    def prune(self):
        now = self.clock()
        for session_id, session in list(self.sessions.items()):
            if now - session.touched >= self.ttl:
                self.cleanup(session_id)
                # Expiry evicts audio; a still-open consultation may generate again.
                self.closed.pop(session_id, None)
        self.closed = {key: end for key, end in self.closed.items() if end > now}

    def cleanup(self, session_id: str):
        session = self.sessions.pop(session_id, None)
        self.closed[session_id] = self.clock() + self.ttl
        if session:
            session.audio.clear()
            # Running blocking gateway calls may finish, but can never repopulate.
            for task in session.pending.values():
                task.cancel()
            session.pending.clear()
        logger.info("TTS session cleanup removed=%s", session is not None)

    async def get(self, session_id: str, text: str, lang: str, speed: float,
                  generate: Callable[[], Awaitable[TtsGatewayResult]], trace=None):
        self.prune()
        if session_id in self.closed:
            raise VoiceGatewayError("TTS session 已結束或逾時，請開始新的問診")
        session = self.sessions.setdefault(session_id, Session(self.clock()))
        session.touched = self.clock()
        key = (hashlib.sha256(text.strip().encode()).hexdigest(), lang, speed)
        if key in session.audio:
            if trace:
                trace.set_cache("cache_hit")
            logger.debug("TTS cache hit")
            return session.audio[key]
        task = session.pending.get(key)
        if task is None:
            if trace:
                trace.set_cache("cache_miss", trace.request_id)
            async def run():
                try:
                    result = await generate()
                    if not result.audio_base64:
                        raise VoiceGatewayError("TTS 未回傳音訊")
                    if self.sessions.get(session_id) is session:
                        session.audio[key] = result
                    return result
                finally:
                    session.pending.pop(key, None)

            logger.debug("TTS cache miss")
            task = asyncio.create_task(run(), name=trace.request_id if trace else None)
            # Retrieve failures even when all HTTP callers have disconnected.
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            session.pending[key] = task
        elif trace:
            trace.set_cache("pending_join", task.get_name())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled() or session_id in self.closed:
                raise VoiceGatewayError("TTS session 已結束或逾時") from None
            raise


tts_cache = TtsSessionCache()
