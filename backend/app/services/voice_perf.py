"""Content-free voice timings. Gateway timestamps are taken in the actual worker.

Counters describe this backend process, not downstream GPU workers. The tiny
metadata lock is never held during HTTP, an executor wait, or model execution.
"""
import asyncio
import contextvars
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from uuid import uuid4
from app.services.voice_gateway_executor import gateway_executor

# Inherit Uvicorn's configured console handler so INFO events are visible without
# changing the application's global logging configuration.
logger = logging.getLogger("uvicorn.error.voice_perf")
current_trace = contextvars.ContextVar("voice_perf", default=None)


class ActiveCalls:
    def __init__(self):
        self.lock = threading.Lock()
        self.counts = {"asr": 0, "tts": 0}

    def snapshot(self, kind=None, delta=0):
        with self.lock:
            if kind:
                self.counts[kind] += delta
            return {"active_asr": self.counts["asr"], "active_tts": self.counts["tts"]}


active_calls = ActiveCalls()


@dataclass
class VoiceTrace:
    kind: str
    request_id: str = field(default_factory=lambda: uuid4().hex)
    received: float = field(default_factory=time.perf_counter)
    received_wall: float = field(default_factory=time.time)
    session_id: str | None = None
    cache: str | None = None
    cache_key: str | None = None
    producer_request_id: str | None = None
    submitted: float | None = None
    start: float | None = None
    end: float | None = None
    ready: float | None = None
    outcome: str = "ok"
    task: str = field(default_factory=lambda: asyncio.current_task().get_name())

    def set_cache(self, state, producer_request_id=None):
        self.cache = state
        self.producer_request_id = producer_request_id
        self.emit(state)

    def timestamp(self, moment):
        return round(self.received_wall + moment - self.received, 6) if moment is not None else None

    def emit(self, event, counts=None, **extra):
        now = time.perf_counter()
        record = {
            "type": self.kind, "event": event, "request_id": self.request_id,
            "session_id": self.session_id or self.request_id,
            "cache": self.cache, "cache_key": self.cache_key,
            "producer_request_id": self.producer_request_id,
            "request_received": self.received_wall,
            "gateway_submitted": self.timestamp(self.submitted),
            "gateway_start": self.timestamp(self.start),
            "gateway_end": self.timestamp(self.end),
            "response_ready": self.timestamp(self.ready),
            "wait_before_gateway_ms": round((self.start - self.received) * 1000, 3) if self.start else None,
            "executor_and_admission_wait_ms": round((self.start - self.submitted) * 1000, 3)
                if self.start and self.submitted else None,
            "gateway_duration_ms": round((self.end - self.start) * 1000, 3) if self.end and self.start else None,
            "total_duration_ms": round(((self.ready or now) - self.received) * 1000, 3),
            "outcome": self.outcome, "pid": os.getpid(),
            "task": self.task, "thread": threading.current_thread().name,
            **(counts if counts is not None else active_calls.snapshot()), **extra,
        }
        logger.info("VoicePerf: %s", json.dumps(record, ensure_ascii=True))

    def gateway_call(self, call, *args, **kwargs):
        self.start = time.perf_counter()
        self.emit("gateway_start", active_calls.snapshot(self.kind, 1))
        try:
            return call(*args, **kwargs)
        except Exception as error:
            # Exception messages may contain upstream payloads or credentials.
            self.outcome = "gateway_error"
            self.emit("gateway_error", error_type=type(error).__name__)
            raise
        finally:
            self.end = time.perf_counter()
            self.emit("gateway_end", active_calls.snapshot(self.kind, -1))


def trace_for(kind):
    return current_trace.get() or VoiceTrace(kind)


async def run_gateway(trace, call, *args, **kwargs):
    trace.submitted = time.perf_counter()
    trace.emit("gateway_submitted")
    return await gateway_executor.run(trace, call, *args, **kwargs)


def safe_cache_key(session_id, text, lang, speed):
    # Session-salted digest only; no medical text or unsalted short-phrase hash.
    return hashlib.sha256(f"{session_id}|{lang}|{speed}|{text.strip()}".encode()).hexdigest()[:16]


class VoicePerfMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        kind = {"/voice/asr": "asr", "/voice/tts": "tts"}.get(scope.get("path"))
        if scope["type"] != "http" or scope.get("method") != "POST" or not kind:
            return await self.app(scope, receive, send)
        trace = VoiceTrace(kind)
        token = current_trace.set(trace)
        trace.emit("request_received")  # Before upload/body parsing and dependencies.

        async def traced_send(message):
            if message["type"] == "http.response.start":
                trace.ready = time.perf_counter()
                if message["status"] >= 400:
                    trace.outcome = "http_error"
                trace.emit("response_ready", http_status=message["status"])
            await send(message)

        try:
            await self.app(scope, receive, traced_send)
        except BaseException as error:
            trace.outcome = "cancelled" if isinstance(error, asyncio.CancelledError) else "error"
            trace.emit("request_interrupted", error_type=type(error).__name__)
            raise
        finally:
            current_trace.reset(token)
