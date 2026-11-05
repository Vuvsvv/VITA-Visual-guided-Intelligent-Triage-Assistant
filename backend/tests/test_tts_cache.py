import asyncio
import threading
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import httpx
from fastapi import UploadFile

from app.main import app
from app.routes import voice
from app.services.tts_cache import TtsSessionCache
from app.services.voice_client import TtsGatewayResult, VoiceGatewayError


class TtsCacheTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = 0
        self.cache = TtsSessionCache(ttl=10, clock=lambda: self.now)
        self.generate = AsyncMock(return_value=TtsGatewayResult("YXVkaW8=", "wav"))

    async def get(self, session="a", text="question", lang="chinese", speed=1.0):
        return await self.cache.get(session, text, lang, speed, self.generate)

    async def test_miss_then_hit_and_trimmed_text(self):
        first = await self.get()
        self.assertIs(first, await self.get(text=" question "))
        self.assertIs(first, await self.get())
        self.generate.assert_awaited_once()

    async def test_text_language_speed_and_session_are_isolated(self):
        await self.get()
        await self.get(text="next question")
        await self.get(lang="taiwanese")
        await self.get(speed=0.8)
        await self.get(session="b")
        self.assertEqual(5, self.generate.await_count)

    async def test_concurrent_calls_share_one_generation(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def slow():
            started.set()
            await release.wait()
            return TtsGatewayResult("YXVkaW8=")

        self.generate.side_effect = slow
        callers = [asyncio.create_task(self.get()) for _ in range(12)]
        await started.wait()
        release.set()
        results = await asyncio.gather(*callers)
        self.assertTrue(all(result is results[0] for result in results))
        self.generate.assert_awaited_once()

    async def test_disconnected_waiter_does_not_cancel_shared_generation(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def slow():
            started.set()
            await release.wait()
            return TtsGatewayResult("YXVkaW8=")

        self.generate.side_effect = slow
        first = asyncio.create_task(self.get())
        await started.wait()
        second = asyncio.create_task(self.get())
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        release.set()
        await second
        self.generate.assert_awaited_once()

    async def test_cleanup_is_idempotent_and_other_session_survives(self):
        await self.get()
        other = await self.get(session="b")
        self.cache.cleanup("a")
        self.cache.cleanup("a")
        self.assertNotIn("a", self.cache.sessions)
        self.assertIs(other, await self.get(session="b"))
        with self.assertRaises(VoiceGatewayError):
            await self.get()
        self.assertEqual(2, self.generate.await_count)

    async def test_cleanup_during_generation_cannot_repopulate_cache(self):
        started = asyncio.Event()

        async def slow():
            started.set()
            await asyncio.Event().wait()

        self.generate.side_effect = slow
        request = asyncio.create_task(self.get())
        await started.wait()
        self.cache.cleanup("a")
        with self.assertRaises(VoiceGatewayError):
            await request
        self.assertNotIn("a", self.cache.sessions)

    async def test_failure_is_retryable_and_not_cached(self):
        self.generate.side_effect = VoiceGatewayError("unavailable")
        with self.assertRaises(VoiceGatewayError):
            await self.get()
        self.generate.side_effect = None
        await self.get()
        self.assertEqual(2, self.generate.await_count)

    async def test_empty_audio_is_not_cached(self):
        self.generate.return_value = TtsGatewayResult("")
        with self.assertRaises(VoiceGatewayError):
            await self.get()
        self.assertFalse(self.cache.sessions["a"].audio)

    async def test_idle_ttl_evicts_and_allows_new_generation(self):
        await self.get()
        self.now = 9
        await self.get(session="b")
        self.now = 10
        self.cache.prune()
        self.assertNotIn("a", self.cache.sessions)
        self.assertIn("b", self.cache.sessions)
        await self.get()
        self.assertEqual(3, self.generate.await_count)

    async def test_tombstones_also_expire(self):
        self.cache.cleanup("a")
        self.now = 11
        self.cache.prune()
        self.assertFalse(self.cache.closed)


class VoiceTtsApiTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cache = TtsSessionCache()
        self.gateway = Mock()
        self.gateway.synthesize_to_base64.return_value = TtsGatewayResult("YXVkaW8=", "mp3")
        self.patches = [
            patch.object(voice, "tts_cache", self.cache),
            patch.object(voice, "_voice_config_error", return_value=None),
            patch.object(voice, "_get_client", return_value=self.gateway),
            patch.object(voice, "get_settings", return_value=SimpleNamespace(voice_default_lang="chinese")),
        ]
        for item in self.patches:
            item.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        self.payload = {"session_id": str(uuid4()), "text": "請描述症狀", "lang": "chinese", "speed": 0.8}

    async def asyncTearDown(self):
        await self.client.aclose()
        for session_id in list(self.cache.sessions):
            self.cache.cleanup(session_id)
        for item in reversed(self.patches):
            item.stop()

    async def test_api_cache_contract_and_speed(self):
        for _ in range(3):
            response = await self.client.post("/voice/tts", json=self.payload)
            self.assertEqual(200, response.status_code)
            self.assertEqual("mp3", response.json()["audio_format"])
            self.assertFalse(response.json()["tts_failed"])
        self.gateway.synthesize_to_base64.assert_called_once_with("請描述症狀", lang="chinese", speed=0.8)

    async def test_cleanup_removes_only_requested_session_even_when_voice_disabled(self):
        other_id = str(uuid4())
        await self.client.post("/voice/tts", json=self.payload)
        await self.client.post("/voice/tts", json={**self.payload, "session_id": other_id})
        with patch.object(voice, "_voice_config_error", return_value="disabled"):
            response = await self.client.post("/voice/tts/cleanup", json={"session_id": self.payload["session_id"]})
        self.assertTrue(response.json()["cleared"])
        self.assertNotIn(self.payload["session_id"], self.cache.sessions)
        self.assertIn(other_id, self.cache.sessions)
        late = await self.client.post("/voice/tts", json=self.payload)
        self.assertTrue(late.json()["tts_failed"])
        self.assertEqual(2, self.gateway.synthesize_to_base64.call_count)

    async def test_failure_does_not_call_chat_and_text_route_is_available(self):
        self.gateway.synthesize_to_base64.side_effect = VoiceGatewayError("unavailable")
        with patch.object(voice, "chat_handler", new=AsyncMock()) as chat:
            response = await self.client.post("/voice/tts", json=self.payload)
            self.assertTrue(response.json()["tts_failed"])
            chat.assert_not_awaited()
        response = await self.client.post("/chat", json={"visit_type": "initial"})
        self.assertEqual(200, response.status_code)
        self.assertTrue(response.json().get("reply") or response.json().get("next_question"))

    async def test_slow_gateway_does_not_block_text_and_duplicate_http_calls_share_work(self):
        started, release = threading.Event(), threading.Event()

        def slow(*args, **kwargs):
            started.set()
            release.wait(timeout=3)
            return TtsGatewayResult("YXVkaW8=")

        self.gateway.synthesize_to_base64.side_effect = slow
        first = asyncio.create_task(self.client.post("/voice/tts", json=self.payload))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            second = asyncio.create_task(self.client.post("/voice/tts", json=self.payload))
            text = await asyncio.wait_for(self.client.post("/chat", json={"visit_type": "initial"}), timeout=1)
            self.assertEqual(200, text.status_code)
            self.assertFalse(first.done())
        finally:
            release.set()
        await asyncio.gather(first, second)
        self.gateway.synthesize_to_base64.assert_called_once()

    async def test_asr_returns_draft_without_calling_chat_or_tts(self):
        self.gateway.transcribe.return_value = "  頭痛兩天  "
        with patch.object(voice, "chat_handler", new=AsyncMock()) as chat:
            result = await voice.voice_asr(UploadFile(filename="test.wav", file=BytesIO(b"RIFF")), "taiwanese")
            self.assertEqual("  頭痛兩天  ", result["text"])
            chat.assert_not_awaited()
            self.gateway.synthesize_to_base64.assert_not_called()

    async def test_validation_and_legacy_tts_compatibility(self):
        for payload in ({}, {"session_id": "invalid"}):
            response = await self.client.post("/voice/tts/cleanup", json=payload)
            self.assertEqual(400, response.status_code)
        for speed in ("invalid", 0, 5):
            response = await self.client.post("/voice/tts", json={**self.payload, "speed": speed})
            self.assertEqual(400, response.status_code)
        response = await self.client.post("/voice/tts", json={"text": "hello", "lang": "chinese"})
        self.assertFalse(response.json()["tts_failed"])

