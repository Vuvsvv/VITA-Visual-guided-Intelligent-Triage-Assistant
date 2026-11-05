import asyncio
import json
import logging
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

import httpx

from app.main import app
from app.routes import voice
from app.services import voice_perf
from app.services.tts_cache import TtsSessionCache
from app.services.voice_client import TtsGatewayResult, VoiceGatewayClient, VoiceGatewayError
from app.services.voice_gateway_executor import VoiceGatewayExecutor


class GatewayConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    async def test_old_shared_executor_can_queue_asr_behind_tts(self):
        # Reproduce the old to_thread resource sharing, not a claim about the
        # remote gateway or the production pool having exactly two workers.
        pool = ThreadPoolExecutor(2)
        loop = asyncio.get_running_loop()
        loop.set_default_executor(pool)
        started = [threading.Event(), threading.Event()]
        release = threading.Event()

        def tts(index):
            started[index].set()
            release.wait(3)

        pending = [asyncio.create_task(asyncio.to_thread(tts, i)) for i in range(2)]
        try:
            while not all(event.is_set() for event in started):
                await asyncio.sleep(0.001)
            asr_started = threading.Event()
            # to_thread ultimately submits this same default-executor work.
            asr = loop.run_in_executor(None, asr_started.set)
            self.assertFalse(asr_started.is_set())
            self.assertFalse(asr.done())
        finally:
            release.set()
        await asyncio.gather(*pending, asr)
        self.assertTrue(asr_started.is_set())
        pool.shutdown()

    async def test_asr_starts_while_tts_pool_is_full_and_new_tts_yields(self):
        executor = VoiceGatewayExecutor(asr_workers=1, tts_workers=2)
        tts_release, asr_release = threading.Event(), threading.Event()
        tts_started = [threading.Event(), threading.Event()]
        asr_started, late_tts_started = threading.Event(), threading.Event()

        def tts(index):
            tts_started[index].set()
            tts_release.wait(3)

        def asr():
            asr_started.set()
            asr_release.wait(3)

        tasks = [asyncio.create_task(executor.run(voice_perf.VoiceTrace("tts"), tts, i)) for i in range(2)]
        try:
            self.assertTrue(await asyncio.to_thread(tts_started[0].wait, 1))
            self.assertTrue(await asyncio.to_thread(tts_started[1].wait, 1))
            tasks.append(asyncio.create_task(executor.run(voice_perf.VoiceTrace("asr"), asr)))
            self.assertTrue(await asyncio.to_thread(asr_started.wait, 1))
            self.assertEqual({"active_asr": 1, "active_tts": 2}, voice_perf.active_calls.snapshot())
            tts_release.set()
            await asyncio.gather(*tasks[:2])
            tasks.append(asyncio.create_task(executor.run(voice_perf.VoiceTrace("tts"), late_tts_started.set)))
            await asyncio.sleep(0.02)
            self.assertFalse(late_tts_started.is_set())
            asr_release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 2)
            self.assertTrue(late_tts_started.is_set())
            self.assertEqual({"active_asr": 0, "active_tts": 0}, voice_perf.active_calls.snapshot())
            self.assertEqual(0, executor.asr_outstanding)
        finally:
            tts_release.set()
            asr_release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            executor.shutdown()

    async def test_cancelled_http_waiter_does_not_hide_running_gateway_or_leak_counter(self):
        executor = VoiceGatewayExecutor()
        started, release, ended = threading.Event(), threading.Event(), threading.Event()

        def slow():
            started.set()
            try:
                release.wait(3)
                raise VoiceGatewayError("sensitive upstream error")
            finally:
                ended.set()

        request = asyncio.create_task(executor.run(voice_perf.VoiceTrace("asr"), slow))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            self.assertEqual(1, voice_perf.active_calls.snapshot()["active_asr"])
            self.assertEqual(1, executor.asr_outstanding)
        finally:
            release.set()
            await asyncio.to_thread(ended.wait, 1)
            # Barrier through the same executor proves the completion callback ran.
            await executor.run(voice_perf.VoiceTrace("asr"), lambda: None)
            executor.shutdown()
        self.assertEqual({"active_asr": 0, "active_tts": 0}, voice_perf.active_calls.snapshot())
        self.assertEqual(0, executor.asr_outstanding)


class VoicePerfApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_timings_cache_join_and_sensitive_content_are_separate(self):
        cache = TtsSessionCache()
        gateway = Mock()
        started, release = threading.Event(), threading.Event()
        secret_text = "PRIVATE_PATIENT_123"
        secret_audio = "PRIVATE_BASE64_AUDIO"

        def slow(*args, **kwargs):
            started.set()
            release.wait(3)
            return TtsGatewayResult(secret_audio)

        gateway.synthesize_to_base64.side_effect = slow
        gateway.transcribe.return_value = secret_text
        payload = {"session_id": str(uuid4()), "text": secret_text, "lang": "taiwanese"}
        with patch.object(voice, "tts_cache", cache), patch.object(voice, "_get_client", return_value=gateway), \
                patch.object(voice, "_voice_config_error", return_value=None), \
                patch.object(voice, "get_settings", return_value=SimpleNamespace(voice_default_lang="taiwanese")), \
                self.assertLogs("uvicorn.error.voice_perf", level="INFO") as logs:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                first = asyncio.create_task(client.post("/voice/tts", json=payload))
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 1))
                    second = asyncio.create_task(client.post("/voice/tts", json=payload))
                    await asyncio.sleep(0.01)
                    asr = await client.post("/voice/asr", files={"file": ("private.wav", b"private audio")}, data={"lang": "taiwanese"})
                    self.assertEqual(secret_text, asr.json()["text"])
                finally:
                    release.set()
                await asyncio.gather(first, second)
                await client.post("/voice/tts", json=payload)
                gateway.synthesize_to_base64.assert_called_once()

        output = "\n".join(logs.output)
        for sensitive in (secret_text, secret_audio, "private.wav", "private audio"):
            self.assertNotIn(sensitive, output)
        events = [json.loads(line.split("VoicePerf: ")[1]) for line in logs.output]
        ready = [event for event in events if event["event"] == "response_ready"]
        self.assertEqual(4, len(ready))
        tts_ready = [event for event in ready if event["type"] == "tts"]
        self.assertEqual({"cache_hit", "cache_miss", "pending_join"}, {event["cache"] for event in tts_ready})
        miss = next(event for event in tts_ready if event["cache"] == "cache_miss")
        join = next(event for event in tts_ready if event["cache"] == "pending_join")
        self.assertEqual(miss["request_id"], join["producer_request_id"])
        self.assertIsNone(join["gateway_start"])
        self.assertTrue(all(event["cache_key"] for event in tts_ready))
        asr = next(event for event in ready if event["type"] == "asr")
        self.assertEqual(asr["request_id"], asr["session_id"])
        for event in (miss, asr):
            self.assertLessEqual(event["request_received"], event["gateway_start"])
            self.assertLessEqual(event["gateway_start"], event["gateway_end"])
            self.assertLessEqual(event["gateway_end"], event["response_ready"])
            self.assertGreaterEqual(event["total_duration_ms"], event["gateway_duration_ms"])
        self.assertTrue(any(event["type"] == "asr" and event["event"] == "gateway_start"
                            and event["active_tts"] == 1 for event in events))

    async def test_upstream_error_logs_never_echo_body_or_key(self):
        gateway = VoiceGatewayClient("https://example.invalid", "SECRET_API_KEY")
        response = Mock(status_code=500, text="SECRET_PATIENT_AND_AUDIO")
        with self.assertLogs("app.services.voice_client", level="WARNING") as logs:
            with self.assertRaises(VoiceGatewayError):
                gateway._parse(response, "taiwanese_tts")
        self.assertNotIn("SECRET", "\n".join(logs.output))
        trace = voice_perf.VoiceTrace("asr")
        with self.assertLogs("uvicorn.error.voice_perf", level="INFO") as logs:
            with self.assertRaises(VoiceGatewayError):
                await voice_perf.run_gateway(trace, Mock(side_effect=VoiceGatewayError("SECRET")))
        self.assertNotIn("SECRET", "\n".join(logs.output))
        self.assertEqual(0, voice_perf.active_calls.snapshot()["active_asr"])
