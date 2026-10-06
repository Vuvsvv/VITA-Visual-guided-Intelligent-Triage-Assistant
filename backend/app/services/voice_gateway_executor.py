"""Small process-local admission policy; no downstream cancellation claims.

ASR never shares TTS executor slots. New TTS gateway calls yield while any ASR
is queued/running here. Already dispatched gateway HTTP calls run to completion.
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, CancelledError


class VoiceGatewayExecutor:
    def __init__(self, asr_workers=2, tts_workers=2):
        self.asr_pool = ThreadPoolExecutor(asr_workers, thread_name_prefix="voice-asr")
        self.tts_pool = ThreadPoolExecutor(tts_workers, thread_name_prefix="voice-tts")
        self.condition = threading.Condition()
        self.asr_outstanding = 0

    async def run(self, trace, call, *args, **kwargs):
        is_asr = trace.kind == "asr"
        cancelled = threading.Event()
        if is_asr:
            with self.condition:
                self.asr_outstanding += 1

        def work():
            if not is_asr:
                with self.condition:
                    waiting_for_asr = bool(self.asr_outstanding)
                if waiting_for_asr:
                    trace.emit("asr_priority_wait")
                with self.condition:
                    while self.asr_outstanding and not cancelled.is_set():
                        self.condition.wait()
                    if cancelled.is_set():
                        raise CancelledError()
            # No lock is held across the HTTP request. ASR never waits for TTS.
            return trace.gateway_call(call, *args, **kwargs)

        def complete(_):
            if is_asr:
                with self.condition:
                    self.asr_outstanding -= 1
                    self.condition.notify_all()

        try:
            future = (self.asr_pool if is_asr else self.tts_pool).submit(work)
        except BaseException:
            complete(None)
            raise
        future.add_done_callback(complete)
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            # This only cancels queued work. A running HTTP call remains running,
            # and counters stay active until its actual completion.
            cancelled.set()
            future.cancel()
            with self.condition:
                self.condition.notify_all()
            raise

    def shutdown(self):
        self.asr_pool.shutdown(wait=False, cancel_futures=True)
        self.tts_pool.shutdown(wait=False, cancel_futures=True)


gateway_executor = VoiceGatewayExecutor()
