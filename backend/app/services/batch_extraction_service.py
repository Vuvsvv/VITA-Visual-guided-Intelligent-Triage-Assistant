from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from unittest import case

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.config import get_settings
from app.schemas import BatchAnswer, EvidenceAssertion, Message, SemanticExtraction, TriageCase
from app.services.ai_service import complete_runtime_json as _complete_runtime_json, runtime_ai_available
from app.services.confidence_scoring import ACCEPT_THRESHOLD, MAX_QUESTION_ATTEMPTS
from app.services.department_preference_service import capture_department_preference
from app.services.field_acceptance import (
    BODY_PART_TERMS,
    SYMPTOM_TERMS,
    ai_normalized_value_rejection_reason,
    has_symptom_semantics,
    plausible_semantic_target,
    requires_semantic_refinement,
)
from app.services.question_specs import question_spec_for_field
from app.services.rule_engine import (
    CHECKLIST_FIELD_ORDER,
    apply_semantic_extractions,
    apply_user_message,
    missing_checklist_fields,
)
from app.services.semantic_normalizer import (
    SESSION_ALIASES,
    normalize_preferred_sessions,
)
logger = logging.getLogger(__name__)

_ALLOWED_FIELDS = frozenset((*CHECKLIST_FIELD_ORDER, "onset", "accompanying_symptoms"))
_ALLOWED_STATUSES = {"available", "unavailable", "unknown", "partial", "ambiguous"}
_META_REPLY_TERMS = (
    "我剛剛已經回答了",
    "我剛剛說過了",
    "我剛才說過了",
    "我已經說過了",
    "我前面說過了",
    "剛剛不是說了嗎",
    "我不是說了嗎",
    "剛剛有講",
    "前面講過了",
    "我剛才回答過",
    "不是才說過",
)


async def complete_prompt(prompt: str) -> str:
    """Compatibility seam for tests; production always uses Cerebras JSON mode."""
    return await _complete_runtime_json(prompt, purpose="semantic_extraction")


class _AiExtractionItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    normalized_value: Any = None
    semantic_status: str = "unknown"
    assertion: EvidenceAssertion | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_text: str = ""
    needs_clarification: bool = False
    follow_up_reason: str | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence_type(cls, value: object) -> object:
        if type(value) not in {int, float}:
            raise ValueError("confidence must be a finite number")
        return value


class _AiExtractionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extractions: list[_AiExtractionItem]


@dataclass
class BatchExtractionOutcome:
    deterministic_fields: list[str] = field(default_factory=list)
    accepted_fields: list[str] = field(default_factory=list)
    ai_fields: list[str] = field(default_factory=list)
    unresolved_fields: list[str] = field(default_factory=list)
    ai_attempted: bool = False
    fallback_reason: str | None = None


async def extract_batch_answers(
    case: TriageCase,
    answers: list[BatchAnswer],
    *,
    semantic_first: bool = False,
) -> BatchExtractionOutcome:
    """Consume a keyed batch, calling at most one AI provider for ambiguity."""
    outcome = BatchExtractionOutcome()
    answered_text: dict[str, str] = {}
    ambiguous_fallbacks: dict[str, SemanticExtraction] = {}
    semantic_failure_fallbacks: dict[str, SemanticExtraction] = {}
    semantic_refinement_sources: dict[str, str] = {}
    settings = get_settings()
    provider = "cerebras"

    for answer in answers:
        text = answer.answer.strip()
        if not text:
            continue
        key = str(answer.key)
        append_answer_history = True
        if key in _ALLOWED_FIELDS and key != "red_flags" and _is_meta_reply(text):
            previous_answer = _previous_keyed_answer(case, key)
            already_satisfied = _field_has_accepted_value(case, key)
            semantic_replay = bool(previous_answer and not already_satisfied)
            case.history_records.append(Message(role="user", content=f"[{key}] {text}"))
            logger.info(
                "[SEMANTIC_META_REPLAY] case_id=%s field=%s previous_answer_found=%s "
                "already_satisfied=%s semantic_replay=%s",
                case.case_id,
                key,
                str(bool(previous_answer)).lower(),
                str(already_satisfied).lower(),
                str(semantic_replay).lower(),
            )
            if already_satisfied:
                if key not in outcome.deterministic_fields:
                    outcome.deterministic_fields.append(key)
                if key not in outcome.accepted_fields:
                    outcome.accepted_fields.append(key)
                continue

            # The displayed clarification already incremented the attempt
            # counter. A meta reply is not a new clinical answer, so keep the
            # field below the fallback threshold whether replay succeeds or
            # not. This preserves one real answer opportunity.
            attempts = case.conversation_state.question_attempts.get(key, 0)
            case.conversation_state.question_attempts[key] = min(
                attempts,
                MAX_QUESTION_ATTEMPTS - 1,
            )
            if key in case.conversation_state.consumed_fields:
                case.conversation_state.consumed_fields.remove(key)
            case.conversation_state.field_statuses[key] = "unknown"
            case.conversation_state.field_confidence[key] = 0.0
            case.conversation_state.clarification_reasons[key] = (
                "meta reply did not count as a new clinical answer"
            )
            if not previous_answer:
                continue

            # Replay the prior clinical text through the same deterministic
            # fast-path and semantic-refinement pipeline as a fresh answer.
            # Only the meta reply itself is appended to history.
            text = previous_answer
            append_answer_history = False
        answered_text[key] = text
        before_missing = set(missing_checklist_fields(case, apply_attempt_fallback=False))
        if append_answer_history:
            case.history_records.append(Message(role="user", content=f"[{key}] {text}"))
        if key == "department_clarification":
            case.patient_input.department_context = text
        preference = capture_department_preference(case, text)
        if preference and preference.resolved and "requested_department" not in outcome.accepted_fields:
            outcome.accepted_fields.append("requested_department")
        if key == "red_flags":
            case.conversation_state.field_statuses[key] = "unknown"
            case.conversation_state.field_confidence[key] = 0.0
            case.conversation_state.clarification_reasons[key] = (
                "free-text safety answers require validated Turn Interpreter evidence"
            )
            continue
        if (
            semantic_first
            and runtime_ai_available(settings)
            and key in _ALLOWED_FIELDS
            and key != "red_flags"
            and not _is_structured_choice(key, text)
        ):
            # The key names a question; the answer is still free text.
            apply_user_message(case, text, semantic_first=True, record_history=False)
            semantic_refinement_sources[key] = text
            continue
        if key == "duration" and _looks_ambiguous(text):
            fallback_case = TriageCase(case_id=case.case_id)
            fallback = _deterministic_extraction_for_answer(
                fallback_case,
                key,
                text,
                allow_ambiguous=True,
            )
            if fallback is not None:
                ambiguous_fallbacks[key] = fallback
        extractions = _deterministic_extractions_for_answer(case, key, text)
        accepted_extractions: list[SemanticExtraction] = []
        for extraction in extractions:
            is_primary = extraction.field == key
            accepted_fast_path, fast_path_reason = _deterministic_fast_path_decision(
                extraction.field,
                text,
                extraction,
                is_primary=is_primary,
            )
            logger.info(
                "[SEMANTIC_FAST_PATH] case_id=%s current_field=%s field=%s "
                "accepted=%s reason=%s",
                case.case_id,
                key,
                extraction.field,
                str(accepted_fast_path).lower(),
                fast_path_reason,
            )
            if accepted_fast_path:
                accepted_extractions.append(extraction)
                continue
            if extraction.field != "red_flags":
                # Do not commit a lossy deterministic candidate. Keeping the
                # field empty lets the semantic result land without tripping
                # duplicate_existing_value.
                semantic_refinement_sources.setdefault(extraction.field, text)
                if extraction.field in {"body_part", "severity"}:
                    semantic_failure_fallbacks.setdefault(extraction.field, extraction)
            if is_primary and key == "severity":
                ambiguous_fallbacks[key] = extraction
        extractions = accepted_extractions
        if extractions:
            apply_semantic_extractions(case, extractions)
        if (
            key in _ALLOWED_FIELDS
            and key not in case.conversation_state.consumed_fields
            and not any(item.field == key for item in extractions)
        ):
            case.conversation_state.field_statuses[key] = "unknown"
            case.conversation_state.field_confidence[key] = 0.0
            case.conversation_state.clarification_reasons[key] = "answer did not pass strict field validation"
        after_missing = set(missing_checklist_fields(case, apply_attempt_fallback=False))
        resolved_now = [field_name for field_name in before_missing if field_name not in after_missing]
        for field_name in resolved_now:
            if field_name not in outcome.deterministic_fields:
                outcome.deterministic_fields.append(field_name)
            if field_name not in outcome.accepted_fields:
                outcome.accepted_fields.append(field_name)

    unresolved = [
        field_name
        for field_name in missing_checklist_fields(case, apply_attempt_fallback=False)
        if field_name in answered_text
    ]
    # Red-flag completion is deterministic-only. AI may never mark the screen
    # checked merely by returning a JSON field.
    # A non-empty keyed answer for an unresolved field must reach the semantic
    # provider once when it is available. Failing to understand the answer
    # deterministically is exactly when semantic refinement is needed.
    current_missing = set(missing_checklist_fields(case, apply_attempt_fallback=False))
    semantic_sources = {
        field_name: source_text
        for field_name, source_text in semantic_refinement_sources.items()
        if (semantic_first or field_name in current_missing) and field_name != "red_flags"
    }
    semantic_sources.update({
        field_name: answered_text[field_name]
        for field_name in unresolved
        if field_name != "red_flags" and field_name not in semantic_sources
    })
    if semantic_sources:
        all_missing = missing_checklist_fields(case, apply_attempt_fallback=False)
    for source_text in tuple(semantic_sources.values()):
        for field_name in all_missing:
            if field_name == "red_flags" or field_name in semantic_sources:
                continue

            # preferred_sessions 作為「順便抽取的額外欄位」時，
            # 必須先有明確掛號／看診時段語境。
            # 例如「下午比較嚴重」「下午開始頭痛」中的下午
            # 是症狀時間，不是掛號偏好。
            #
            # 注意：如果目前使用者本來就在回答 preferred_sessions，
            # 該欄位已經存在 semantic_sources，因此不會進到這個 guard。
            # 所以直接回答「下午」仍然可以正常接受。
            if (
                field_name == "preferred_sessions"
                and normalize_preferred_sessions(source_text, None) is None
            ):
                continue

            if plausible_semantic_target(field_name, source_text):
                semantic_sources[field_name] = source_text
    ai_targets = list(semantic_sources)
    if not ai_targets:
        outcome.unresolved_fields = [
            field_name
            for field_name in missing_checklist_fields(case)
            if field_name in answered_text
        ]
        _log_extraction_outcome(answered_text, outcome, provider, settings)
        return outcome

    if not runtime_ai_available(settings):
        outcome.fallback_reason = f"provider_key_missing:{provider}"
        _apply_ambiguous_fallbacks(
            case,
            {**semantic_failure_fallbacks, **ambiguous_fallbacks},
            outcome,
        )
        outcome.unresolved_fields = [
            field_name
            for field_name in missing_checklist_fields(case)
            if field_name in answered_text
        ]
        logger.info("batch extraction skipped reason=%s", outcome.fallback_reason)
        _log_extraction_outcome(answered_text, outcome, provider, settings)
        return outcome

    outcome.ai_attempted = True
    prompt = _build_batch_prompt(case, semantic_sources, ai_targets)
    for field_name in ai_targets:
        logger.info(
            "[SEMANTIC_AI_REQUEST] case_id=%s field=%s answer_chars=%s",
            case.case_id,
            field_name,
            len(semantic_sources[field_name]),
        )
    try:
        raw = await complete_prompt(prompt)
        extractions = _parse_ai_extractions(
            raw,
            semantic_sources,
            ai_targets,
            case_id=case.case_id,
        )
        apply_semantic_extractions(
            case,
            extractions,
            allow_red_flag_completion=False,
        )
        outcome.ai_fields = [item.field for item in extractions]
        remaining = set(missing_checklist_fields(case, apply_attempt_fallback=False))
        for item in extractions:
            if item.field not in remaining and item.field not in outcome.accepted_fields:
                outcome.accepted_fields.append(item.field)
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        suffix = f":{status}" if status is not None else ""
        outcome.fallback_reason = f"{type(exc).__name__}{suffix}"
        _apply_ambiguous_fallbacks(
            case,
            {**semantic_failure_fallbacks, **ambiguous_fallbacks},
            outcome,
        )
        logger.warning(
            "batch extraction provider fallback provider=%s reason=%s",
            provider,
            outcome.fallback_reason,
        )

    outcome.unresolved_fields = [
        field_name
        for field_name in missing_checklist_fields(case)
        if field_name in answered_text
    ]
    _log_extraction_outcome(answered_text, outcome, provider, settings)
    return outcome


def _is_structured_choice(key: str, text: str) -> bool:
    choices = {
        "preferred_days": {"週一", "週二", "週三", "週四", "週五", "週六", "週日"},
        "preferred_sessions": {"上午", "下午", "夜間"},
        "severity": {"mild", "moderate", "severe"},
    }
    return text in choices.get(key, set())


def _log_extraction_outcome(
    answered_text: dict[str, str],
    outcome: BatchExtractionOutcome,
    provider: str,
    settings: Any,
) -> None:
    model = getattr(settings, "cerebras_model", "gpt-oss-120b")
    logger.info(
        "[EXTRACTION] current_question=%s deterministic_fields=%s accepted_fields=%s "
        "semantic_fallback_called=%s semantic_provider=%s semantic_model=%s",
        ",".join(answered_text.keys()) or "none",
        outcome.deterministic_fields,
        outcome.accepted_fields,
        outcome.ai_attempted,
        provider if outcome.ai_attempted else "none",
        model if outcome.ai_attempted else "none",
    )


def _apply_ambiguous_fallbacks(
    case: TriageCase,
    fallbacks: dict[str, SemanticExtraction],
    outcome: BatchExtractionOutcome,
) -> None:
    """Use prior deterministic behavior only when semantic AI cannot respond."""
    if not fallbacks:
        return
    apply_semantic_extractions(case, list(fallbacks.values()))
    remaining = set(missing_checklist_fields(case, apply_attempt_fallback=False))
    for field_name in fallbacks:
        if field_name in remaining:
            continue
        if field_name not in outcome.deterministic_fields:
            outcome.deterministic_fields.append(field_name)
        if field_name not in outcome.accepted_fields:
            outcome.accepted_fields.append(field_name)


def _deterministic_extraction_for_answer(
    case: TriageCase,
    key: str,
    text: str,
    *,
    allow_ambiguous: bool = False,
) -> SemanticExtraction | None:
    if key == "red_flags":
        return None
    temporary = TriageCase(case_id=case.case_id)
    temporary.conversation_state.last_question_key = key
    apply_user_message(temporary, text)
    ambiguous = _looks_ambiguous(text)
    accept_ambiguous = allow_ambiguous or not ambiguous
    status = temporary.conversation_state.field_statuses.get(key, "available")
    confidence = temporary.conversation_state.field_confidence.get(key, 0.9)
    value: Any = None

    if key == "symptom" and accept_ambiguous and has_symptom_semantics(text):
        value = text
    elif key == "body_part" and accept_ambiguous:
        value = temporary.patient_input.body_part
    elif key == "duration" and accept_ambiguous and temporary.patient_input.duration:
        value = temporary.patient_input.duration
    elif key == "severity" and accept_ambiguous:
        case.patient_input.severity_normalized = temporary.patient_input.severity_normalized
        value = (
            {"severity_level": temporary.patient_input.severity_normalized.severity_level}
            if temporary.patient_input.severity_normalized.severity_level
            else None
        )
        status = temporary.patient_input.severity_normalized.semantic_status
        confidence = temporary.patient_input.severity_normalized.confidence
    elif key == "preferred_days" and accept_ambiguous:
        value = temporary.availability.preferred_days
        status = temporary.availability.semantic_status.get(key, status)
        confidence = temporary.availability.confidence.get(key, confidence)
    elif key == "preferred_sessions" and accept_ambiguous:
        value = temporary.availability.preferred_sessions
        status = temporary.availability.semantic_status.get(key, status)
        confidence = temporary.availability.confidence.get(key, confidence)

    if value is None or (isinstance(value, list) and not value and status != "unavailable"):
        return None
    return SemanticExtraction(
        field=key,
        normalized_value=value,
        semantic_status=status,
        confidence=confidence,
        source_text=text,
        needs_clarification=False,
        extractor="deterministic_batch",
    )


def _deterministic_extractions_for_answer(
    case: TriageCase,
    key: str,
    text: str,
) -> list[SemanticExtraction]:
    """Extract the keyed field plus other explicit deterministic slots."""
    primary = _deterministic_extraction_for_answer(case, key, text)
    results = [primary] if primary is not None else []
    if _looks_ambiguous(text):
        return results
    temporary = TriageCase(case_id=case.case_id)
    temporary.conversation_state.last_question_key = key
    apply_user_message(temporary, text)

    def add(field_name: str, value: Any, status: str = "available", confidence: float = 0.9) -> None:
        if field_name == key or value is None or (value == [] and status != "unavailable"):
            return
        results.append(
            SemanticExtraction(
                field=field_name,
                normalized_value=value,
                semantic_status=status,
                confidence=confidence,
                source_text=text,
                needs_clarification=False,
                extractor="deterministic_multi_slot",
            )
        )

    add("body_part", temporary.patient_input.body_part)
    add("duration", temporary.patient_input.duration)
    if temporary.patient_input.severity_normalized.severity_level:
        add(
            "severity",
            {"severity_level": temporary.patient_input.severity_normalized.severity_level},
            temporary.patient_input.severity_normalized.semantic_status,
            temporary.patient_input.severity_normalized.confidence,
        )
    if temporary.availability.preferred_days:
        add(
            "preferred_days",
            temporary.availability.preferred_days,
            temporary.availability.semantic_status.get("preferred_days", "partial"),
            temporary.availability.confidence.get("preferred_days", 0.9),
        )
    if temporary.availability.preferred_dates:
        add(
            "preferred_dates",
            temporary.availability.preferred_dates,
            temporary.availability.semantic_status.get("preferred_dates", "partial"),
            temporary.availability.confidence.get("preferred_dates", 0.94),
        )
    if temporary.availability.preferred_sessions:
        add(
            "preferred_sessions",
            temporary.availability.preferred_sessions,
            temporary.availability.semantic_status.get("preferred_sessions", "partial"),
            temporary.availability.confidence.get("preferred_sessions", 0.86),
        )
    return results


def _looks_ambiguous(text: str) -> bool:
    return requires_semantic_refinement(text)


def _deterministic_fast_path_decision(
    field_name: str,
    text: str,
    extraction: SemanticExtraction,
    *,
    is_primary: bool,
) -> tuple[bool, str]:
    """Keep only high-confidence, low-interpretation answers on the zero-AI path."""
    if field_name == "red_flags":
        return False, "free_text_safety_requires_turn_interpreter"
    if requires_semantic_refinement(text):
        return False, "semantic_refinement_required"
    if extraction.confidence < ACCEPT_THRESHOLD:
        return False, "confidence_below_threshold"
    if field_name == "preferred_days" and _has_compound_day_semantics(text):
        return False, "compound_day_semantics"
    if field_name == "preferred_sessions" and _has_compound_session_semantics(text):
        return False, "compound_session_semantics"
    if field_name == "severity":
        direct_scale_terms = ("輕微", "普通", "中等", "中度", "嚴重", "很痛", "劇痛")
        if extraction.confidence >= 0.85 or any(term in text for term in direct_scale_terms):
            return True, "clear_canonical_value"
        return False, "indirect_severity_semantics"
    if field_name == "body_part":
        if _body_part_candidate_is_coarse(text, extraction.normalized_value):
            reason = "coarse_primary_extraction" if is_primary else "coarse_secondary_extraction"
            return False, reason
    return True, "clear_canonical_value"


def _is_clear_deterministic_fast_path(
    field_name: str,
    text: str,
    extraction: SemanticExtraction,
) -> bool:
    """Compatibility wrapper used by focused tests and callers."""
    accepted_fast_path, _ = _deterministic_fast_path_decision(
        field_name,
        text,
        extraction,
        is_primary=True,
    )
    return accepted_fast_path


_COMPOUND_SESSION_TERMS = (
    "不要",
    "不想",
    "不方便",
    "不能",
    "沒空",
    "不行",
    "但是",
    "可是",
    "不過",
    "除了",
    "其他",
    "其餘",
    "上班",
    "下班",
    "之後",
    "以前",
    "才方便",
)

_AVAILABILITY_POLARITY_TERMS = (
    "不方便",
    "不能",
    "沒空",
    "不行",
    "除了",
    "其他",
    "其餘",
    "但是",
    "可是",
    "不過",
)


def _has_compound_day_semantics(text: str) -> bool:
    """Detect polarity risk around named days; leave its interpretation to AI."""
    has_named_day = bool(
        re.search(r"(?:週|周|星期|禮拜)[一二三四五六日天]", text)
        or any(term in text for term in ("平日", "工作日", "週末", "假日"))
    )
    return has_named_day and any(term in text for term in _AVAILABILITY_POLARITY_TERMS)


def _has_compound_session_semantics(text: str) -> bool:
    mentioned_sessions = {
        canonical
        for canonical, aliases in SESSION_ALIASES.items()
        if any(alias in text for alias in aliases)
    }
    return len(mentioned_sessions) >= 2 or any(term in text for term in _COMPOUND_SESSION_TERMS)


def _body_part_candidate_is_coarse(text: str, value: Any) -> bool:
    """Detect obvious rule compression without constructing another anatomy dictionary."""
    normalized = str(value or "").strip()
    source = text.strip(" ，。！？!?")
    if not normalized or normalized not in source:
        # Existing canonical conversions such as 腸胃 -> 腹 are intentional.
        return False
    if source == normalized:
        return False

    matched_terms = [term for term in BODY_PART_TERMS if term in source]
    longest_match = max(matched_terms, key=len, default="")
    if len(longest_match) > len(normalized):
        # Known canonical compression (膝蓋 -> 膝, 腹部 -> 腹) is safe.
        return False

    source_sides = {side for side in ("左", "右") if side in source}
    normalized_sides = {side for side in ("左", "右") if side in normalized}
    if len(source_sides) == 1 and not normalized_sides:
        return True
    # A short match followed by another CJK character is likely only the
    # prefix of a more specific anatomical phrase (手 + 腕, 腳 + 踝).
    # Symptom characters such as 痛/酸 are excluded so 頭痛 and 手痛 remain
    # valid clear paths.
    if len(normalized) <= 2:
        start = 0
        while True:
            index = source.find(normalized, start)
            if index < 0:
                break
            next_index = index + len(normalized)
            if next_index < len(source):
                suffix = source[next_index:]
                if _is_cjk_character(suffix[0]) and not _starts_with_symptom_expression(suffix):
                    return True
            start = index + 1
    return False


def _is_cjk_character(character: str) -> bool:
    return bool(character) and "\u4e00" <= character <= "\u9fff"


def _starts_with_symptom_expression(value: str) -> bool:
    text = value.lstrip(" ，。！？!?、")
    prefixes = (
        "會",
        "有點",
        "有一點",
        "很",
        "一直",
        "感到",
        "覺得",
        "不太",
        "就",
        "附近",
        "這邊",
        "那邊",
        "周圍",
        "旁邊",
        "位置",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :]
                changed = True
                break
    return not text or text in {"吧", "啦", "啊", "的"} or any(
        text.startswith(term) for term in SYMPTOM_TERMS
    )


def _is_meta_reply(text: str) -> bool:
    normalized = "".join(character for character in text if character not in " \t\r\n，。！？!?、")
    return any(term in normalized for term in _META_REPLY_TERMS)


def _previous_keyed_answer(case: TriageCase, key: str) -> str | None:
    prefix = f"[{key}] "
    for message in reversed(case.history_records):
        if message.role == "user" and message.content.startswith(prefix):
            candidate = message.content[len(prefix) :].strip()
            if candidate and not _is_meta_reply(candidate):
                return candidate
    return None


def _field_has_accepted_value(case: TriageCase, field_name: str) -> bool:
    patient = case.patient_input
    if field_name in {"symptom", "body_part", "duration", "severity"}:
        return bool(getattr(patient, field_name))
    if field_name == "preferred_days":
        return (
            bool(case.availability.preferred_days)
            or case.conversation_state.field_statuses.get(field_name) == "unavailable"
        )
    if field_name == "preferred_sessions":
        return (
            bool(case.availability.preferred_sessions)
            or case.conversation_state.field_statuses.get(field_name) == "unavailable"
        )
    return False


def _build_batch_prompt(
    case: TriageCase,
    answered_text: dict[str, str],
    targets: list[str],
) -> str:
    target_answers = {key: answered_text[key] for key in targets}
    current_questions = {
        key: {
            "current_question_field": key,
            "current_question": question_spec_for_field(key).canonical_text,
            "user_answer": answered_text[key],
        }
        for key in targets
    }
    history = [item.model_dump() for item in case.history_records[-20:]]
    return f"""你是醫療問診的 structured extraction 元件，只能抽取指定欄位。
不得決定 stage、is_complete、confirmed、red_flags_checked、next_question 或推薦流程。

允許欄位：{json.dumps(targets, ensure_ascii=False)}
本批仍需協助解析的 keyed answers：
{json.dumps(target_answers, ensure_ascii=False, indent=2)}

目前正在回答的問題語境（current_question_field、current_question、user_answer）：
{json.dumps(current_questions, ensure_ascii=False, indent=2)}

目前 patient_input：
{json.dumps(case.patient_input.model_dump(), ensure_ascii=False, indent=2)}

目前 availability：
{json.dumps(case.availability.model_dump(), ensure_ascii=False, indent=2)}

最近 history：
{json.dumps(history, ensure_ascii=False, indent=2)}

只輸出 JSON：
{{
  "extractions": [
    {{
      "field": "body_part",
      "normalized_value": "右肩",
      "semantic_status": "available",
      "assertion": null,
      "confidence": 0.9,
      "source_text": "右肩附近",
      "needs_clarification": false,
      "follow_up_reason": null
    }}
  ]
}}
semantic_status 只能是 available、unavailable、unknown、partial、ambiguous。
confidence 必須介於 0 與 1。不得輸出未列在允許欄位中的 field。
key 只表示目前系統正在問的欄位，不代表原句一定回答了該欄位；答非所問時不得硬填。
同一句若明確包含其他允許欄位，可以一併抽取，但不得猜測未提及資訊。
最外層 key 必須且只能是 extractions，不可改成 semantic_extractions 或其他名稱。
source_text 必須是直接支持該 extraction 的最短連續逐字患者原文，不可改寫，不得夾帶無關子句。
只要原句對任一允許欄位有明確資訊，就必須輸出該 extraction；不要因其他欄位不確定而整體回空。
弱語氣不等於無法回答：「吧、可能、大概、應該、好像、差不多、左右」若仍有清楚核心資訊，必須輸出 available（集合欄位可用 partial）、needs_clarification=false。
available 表示資訊足以寫入；partial 表示可用但只涵蓋集合的一部分；ambiguous 只用於互相衝突且無法安全選擇；unknown 只用於未提供資訊或明確表示不知道。
symptom 可正規化為簡短症狀文字，不必受手寫症狀詞表限制；body_part 可正規化為簡短解剖位置，不必受手寫部位詞表限制。
symptom 與 accompanying_symptoms 只要 normalized_value 含 nonempty medical concept，就必須輸出 assertion=present/absent/uncertain。若同一句有不同 polarity，必須拆成不同 extraction 並各自引用最短 source_text；Backend 不會替你解析否定 scope。
body_part 若是未知於既有 canonical 的部位，normalized_value 應保留 source_text 中可逐字找到的核心部位（例如「鎖骨附近」→「鎖骨」、「手腕那邊」→「手腕」）；只有已知 canonical 可改寫（例如「腸胃」→「腹」）。不得把來源中的部位替換成無關部位。
duration 必須正規化為「數字+天／週／個月／年」，例如 3天、2週、6個月、1年；半年轉為 6個月，一年半轉為 18個月。
severity 的 normalized_value 只能是字串 "mild"、"moderate" 或 "severe"；輕微／還好轉為 mild，普通／中等／中度轉為 moderate，嚴重／很嚴重／痛到無法睡覺轉為 severe。不得輸出「輕微」「中等」「嚴重程度低」等其他字串。
preferred_days 與 preferred_sessions 是封閉集合：
- preferred_days 全集只能是 ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]。
- preferred_sessions 全集只能是 ["上午", "下午", "夜間"]；早上等同上午，晚上等同夜間。
- 在目前問題就是該 availability 欄位時，若原文明確排除部分 option，且剩餘 option 可由全集唯一、安全地推導，normalized_value 必須回傳剩餘可用集合，不可只回被排除的 option，也不可因此回 unknown、ambiguous 或空陣列。
- preferred_sessions 例如「我不喜歡早上跟晚上」→ ["下午"]；「不要下午」→ ["上午", "夜間"]；「下午可以，但是晚上不要」→ ["下午"]。
- preferred_days 只有在原文明確表達 complement（例如「其他都可以／其餘都可以」）時才補全集：例如「我下週三有事其他時間都可以」→ ["週一", "週二", "週四", "週五", "週六", "週日"]；「週三跟週五不行，其他都可以」→ ["週一", "週二", "週四", "週六", "週日"]。單獨說「週三不行」不可擅自假設其他六天都可以。
- availability 回傳全部 option 時 semantic_status=available；只回部分可用 option（包含 complement 後的剩餘集合）時 semantic_status=partial。partial 是可直接寫入的有效答案，needs_clarification=false。
- source_text 仍必須是使用者本次回答中的連續逐字原文，不得把 normalized_value 當成 source_text。"""


def _parse_ai_extractions(
    raw: str,
    answered_text: dict[str, str],
    targets: list[str],
    *,
    case_id: str | None = None,
) -> list[SemanticExtraction]:
    text = str(raw).strip().replace("```json", "").replace("```", "").strip()
    try:
        raw_payload = json.loads(text)
        payload = _AiExtractionPayload.model_validate(raw_payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        envelope_keys = sorted(raw_payload.keys()) if isinstance(locals().get("raw_payload"), dict) else []
        logger.warning(
            "[SEMANTIC_AI_DECISION] case_id=%s field=unknown accepted=false "
            "reject_reason=schema_invalid envelope_keys=%s error_type=%s",
            case_id or "unknown",
            envelope_keys,
            type(exc).__name__,
        )
        raise ValueError("invalid batch extraction JSON/schema") from exc

    target_set = set(targets)
    validated: list[SemanticExtraction] = []
    seen: set[str] = set()
    rejected: list[dict[str, str]] = []
    normalized_fields: list[str] = []
    grounded_fields: list[str] = []
    for item in payload.extractions:
        field_name = item.field.strip()
        reason = _semantic_rejection_reason(item, target_set, seen)
        if reason:
            rejected.append({"field": field_name or "empty", "reason": reason})
            _log_semantic_ai_decision(case_id, field_name or "empty", False, reason)
            continue
        normalized_fields.append(field_name)
        source_text = item.source_text.strip()
        logger.info(
            "[SEMANTIC_AI_PARSED] case_id=%s field=%s status=%s confidence=%.3f source_chars=%s",
            case_id or "unknown",
            field_name,
            item.semantic_status,
            item.confidence,
            len(source_text),
        )
        source_required = item.semantic_status in {"available", "partial", "unavailable"}
        if (source_required or source_text) and not _source_text_grounded(
            field_name,
            source_text,
            answered_text,
        ):
            rejected.append({"field": field_name, "reason": "source_not_grounded"})
            _log_semantic_ai_decision(case_id, field_name, False, "source_not_grounded")
            continue
        normalized_reason = ai_normalized_value_rejection_reason(
            field_name,
            item.normalized_value,
            item.semantic_status,
            source_text,
        )
        if normalized_reason:
            rejected.append({"field": field_name, "reason": normalized_reason})
            _log_semantic_ai_decision(case_id, field_name, False, normalized_reason)
            continue
        grounded_fields.append(field_name)
        seen.add(field_name)
        validated.append(
            SemanticExtraction(
                field=field_name,
                normalized_value=item.normalized_value,
                semantic_status=item.semantic_status,
                assertion=item.assertion,
                confidence=item.confidence,
                source_text=source_text,
                needs_clarification=item.needs_clarification,
                follow_up_reason=item.follow_up_reason,
                extractor="ai_batch",
            )
        )
    if not payload.extractions:
        for field_name in targets:
            _log_semantic_ai_decision(case_id, field_name, False, "no_extraction")
    logger.info(
        "[AI] purpose=semantic_validation raw_fields=%s schema_validation=passed "
        "normalized_fields=%s grounded_fields=%s final_fields=%s rejected=%s",
        [item.field for item in payload.extractions],
        normalized_fields,
        grounded_fields,
        [item.field for item in validated],
        rejected,
    )
    return validated


def _semantic_rejection_reason(
    item: _AiExtractionItem,
    target_set: set[str],
    seen: set[str],
) -> str | None:
    field_name = item.field.strip()
    if field_name not in _ALLOWED_FIELDS:
        return "invalid_field"
    if field_name not in target_set:
        return "invalid_field"
    if field_name == "red_flags":
        return "red_flags_deterministic_only"
    if field_name in seen:
        return "duplicate_field"
    if item.semantic_status not in _ALLOWED_STATUSES:
        return "invalid_status"
    if _has_nonempty_medical_concept(field_name, item.normalized_value) and item.assertion is None:
        return "missing_assertion"
    if (
        item.semantic_status in {"available", "partial", "unavailable"}
        and item.confidence < ACCEPT_THRESHOLD
    ):
        return "confidence_too_low"
    return None


def _has_nonempty_medical_concept(field_name: str, value: Any) -> bool:
    if field_name == "symptom":
        return isinstance(value, str) and bool(value.strip())
    if field_name == "accompanying_symptoms":
        return isinstance(value, list) and any(
            isinstance(item, str) and item.strip() for item in value
        )
    return False


def _source_text_grounded(
    field_name: str,
    source_text: str,
    answered_text: dict[str, str],
) -> bool:
    answer = answered_text.get(field_name, "")
    return bool(source_text) and source_text in answer


def _log_semantic_ai_decision(
    case_id: str | None,
    field_name: str,
    accepted_value: bool,
    reject_reason: str | None,
) -> None:
    logger.info(
        "[SEMANTIC_AI_DECISION] case_id=%s field=%s accepted=%s reject_reason=%s",
        case_id or "unknown",
        field_name,
        str(accepted_value).lower(),
        reject_reason or "null",
    )
