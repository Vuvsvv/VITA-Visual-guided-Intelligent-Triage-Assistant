from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, Iterator


_PHASE_NAMES = (
    "case_prepare",
    "deterministic_parse",
    "semantic_refinement",
    "rule_engine",
    "department_detection",
    "case_store",
    "ai_reply_total",
    "ai_reply_provider",
    "ai_reply_validation",
    "response_build",
)


def _chat_perf_logger() -> logging.Logger:
    logger = logging.getLogger("ChatPerf")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


@dataclass
class ChatPerf:
    started_at: float = field(default_factory=time.perf_counter)
    durations_ms: dict[str, float] = field(default_factory=dict)
    ai_calls: list[dict[str, Any]] = field(default_factory=list)
    ai_reply_fallback: bool | None = None
    ai_reply_fallback_reason: str | None = None
    ai_reply_validator_accepted: bool | None = None
    ai_reply_parser_invoked: bool = False
    deterministic_next_question: str | None = None
    final_reply: str | None = None
    case_id: str | None = None
    triage_branch: str | None = None
    use_ttas_triage: str | None = None

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started_at = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            self.durations_ms[name] = self.durations_ms.get(name, 0.0) + elapsed_ms

    def summary(self) -> dict[str, Any]:
        calls_by_phase = {
            phase: [call for call in self.ai_calls if call["phase"] == phase]
            for phase in ("semantic_refinement", "department_detection", "ai_reply")
        }
        ai_reply_provider_ms = sum(
            call["latency_ms"] for call in calls_by_phase["ai_reply"]
        )
        summary: dict[str, Any] = {
            "chat_total": _rounded((time.perf_counter() - self.started_at) * 1000),
        }
        for name in _PHASE_NAMES:
            if name == "ai_reply_provider":
                summary[name] = _rounded(ai_reply_provider_ms)
            else:
                summary[name] = _rounded(self.durations_ms.get(name, 0.0))
        summary.update(
            {
                "semantic_ai_called": bool(calls_by_phase["semantic_refinement"]),
                "department_ai_called": bool(calls_by_phase["department_detection"]),
                "ai_reply_ai_called": bool(calls_by_phase["ai_reply"]),
                "ai_call_count": len(self.ai_calls),
                "ai_calls": self.ai_calls,
                "gemini_call_count": len(_gemini_calls(self.ai_calls)),
                "gemini_calls": _gemini_calls(self.ai_calls),
                "ai_reply_validator_accepted": self.ai_reply_validator_accepted,
                "ai_reply_fallback": self.ai_reply_fallback,
                "ai_reply_fallback_reason": self.ai_reply_fallback_reason,
            }
        )
        return summary

    def debug_trace(self) -> dict[str, Any]:
        ai_reply_calls = [
            call for call in self.ai_calls if call["phase"] == "ai_reply"
        ]
        all_models = list(
            dict.fromkeys(
                str(call["model"])
                for call in self.ai_calls
                if call.get("model")
            )
        )
        all_providers = list(
            dict.fromkeys(
                str(call["provider"])
                for call in self.ai_calls
                if call.get("provider")
            )
        )
        ai_reply_raw_metadata = next(
            (
                call
                for call in reversed(ai_reply_calls)
                if call.get("raw_reply_present")
            ),
            {},
        )
        return {
            "case_id": self.case_id,
            "USE_TTAS_TRIAGE": self.use_ttas_triage,
            "triage_branch": self.triage_branch,
            "ai_provider_called": bool(self.ai_calls),
            "ai_reply_provider_called": bool(ai_reply_calls),
            "actual_provider": all_providers or None,
            "actual_model": all_models or None,
            "ai_raw_reply_present": bool(ai_reply_raw_metadata.get("raw_reply_present")),
            "ai_raw_reply_chars": int(ai_reply_raw_metadata.get("raw_reply_chars") or 0),
            "ai_reply_parser_invoked": self.ai_reply_parser_invoked,
            "ai_reply_validator_accepted": self.ai_reply_validator_accepted,
            "fallback": self.ai_reply_fallback,
            "fallback_reason": self.ai_reply_fallback_reason,
            "deterministic_next_question_present": bool(self.deterministic_next_question),
            "final_reply_present": bool(self.final_reply),
            "provider_calls": self.ai_calls,
        }


_current_perf: ContextVar[ChatPerf | None] = ContextVar("chat_perf", default=None)
_current_ai_phase: ContextVar[str] = ContextVar("chat_perf_ai_phase", default="unclassified")


def begin_chat_perf() -> tuple[ChatPerf, Token[ChatPerf | None]]:
    perf = ChatPerf()
    return perf, _current_perf.set(perf)


def finish_chat_perf(perf: ChatPerf, token: Token[ChatPerf | None]) -> None:
    try:
        _chat_perf_logger().info(
            "ChatPerf: %s",
            json.dumps(perf.summary(), ensure_ascii=False, separators=(",", ":")),
        )
        _chat_trace_logger().info(
            "ChatDebugTrace: %s",
            json.dumps(perf.debug_trace(), ensure_ascii=False, separators=(",", ":")),
        )
    finally:
        _current_perf.reset(token)


@contextmanager
def ai_phase(name: str) -> Iterator[None]:
    token = _current_ai_phase.set(name)
    try:
        yield
    finally:
        _current_ai_phase.reset(token)


@contextmanager
def measure_current(name: str) -> Iterator[None]:
    perf = _current_perf.get()
    if perf is None:
        yield
        return
    with perf.measure(name):
        yield


def record_ai_call(
    *,
    latency_ms: float,
    success: bool,
    provider: str | None = None,
    model: str | None = None,
    raw_reply: str | None = None,
    error_type: str | None = None,
    status: str | int | None = None,
) -> None:
    perf = _current_perf.get()
    if perf is None:
        return
    call: dict[str, Any] = {
        "phase": _current_ai_phase.get(),
        "latency_ms": _rounded(latency_ms),
        "success": success,
    }
    if provider:
        call["provider"] = provider
    if model:
        call["model"] = model
    if raw_reply is not None and call["phase"] == "ai_reply":
        call["raw_reply_present"] = True
        call["raw_reply_chars"] = len(raw_reply)
    if error_type:
        call["error_type"] = error_type
    if status is not None:
        call["status"] = str(status)
    perf.ai_calls.append(call)


# Compatibility alias for legacy tests/imports. New runtime code records generic
# AI calls so Cerebras activity is never mislabeled as a Gemini call.
record_gemini_call = record_ai_call


def record_ai_reply_outcome(
    *,
    fallback: bool,
    reason: str | None,
    validator_accepted: bool | None,
    parser_invoked: bool,
) -> None:
    perf = _current_perf.get()
    if perf is None:
        return
    perf.ai_reply_fallback = fallback
    perf.ai_reply_fallback_reason = reason
    perf.ai_reply_validator_accepted = validator_accepted
    perf.ai_reply_parser_invoked = parser_invoked


def record_chat_route_trace(*, case_id: str, triage_branch: str) -> None:
    perf = _current_perf.get()
    if perf is None:
        return
    perf.case_id = case_id
    perf.triage_branch = triage_branch
    perf.use_ttas_triage = os.getenv("USE_TTAS_TRIAGE")


def record_chat_response_trace(
    *,
    deterministic_next_question: str | None,
    final_reply: str | None,
) -> None:
    perf = _current_perf.get()
    if perf is None:
        return
    perf.deterministic_next_question = deterministic_next_question
    perf.final_reply = final_reply


def safe_exception_status(exc: BaseException) -> str | int | None:
    for attribute in ("status_code", "code", "status"):
        value = getattr(exc, attribute, None)
        if value is not None and not callable(value):
            return value
    return None


def _rounded(value: float) -> float:
    return round(value, 1)


def _gemini_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        call
        for call in calls
        if "google" in str(call.get("provider") or "").lower()
        or "gemini" in str(call.get("provider") or "").lower()
    ]


def _chat_trace_logger() -> logging.Logger:
    trace_logger = logging.getLogger("ChatDebugTrace")
    if not trace_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        trace_logger.addHandler(handler)
    trace_logger.setLevel(logging.INFO)
    trace_logger.propagate = False
    return trace_logger
