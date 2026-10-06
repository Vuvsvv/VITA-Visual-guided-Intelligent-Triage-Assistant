from __future__ import annotations

import math
import re
import logging
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from app.schemas import Message, SemanticExtraction, TriageCase, UrgencyResult
from app.services.clarification_engine import clarification_prompt
from app.services.confidence_scoring import ACCEPT_THRESHOLD, MAX_QUESTION_ATTEMPTS, accepted
from app.services.field_acceptance import (
    DURATION_ONSET_PATTERN,
    ai_normalized_value_valid,
    has_symptom_semantics,
    looks_like_department_request,
    normalize_body_part,
    normalized_value_valid,
)
from app.services.negation_utils import is_negated_keyword, strip_negated_red_flags
from app.services.question_specs import QUESTION_SPECS, question_spec_for_field, select_question_variant
from app.services.semantic_normalizer import (
    NormalizationResult,
    normalize_message,
)

logger = logging.getLogger(__name__)


CHINESE_DURATION_NUMBERS = {
    "一": 1,
    "二": 2,
    "兩": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


RED_FLAG_QUESTION_KEY = "red_flags"
QUESTION_TEXTS = {spec.state_field: spec.canonical_text for spec in QUESTION_SPECS}
CHECKLIST_FIELD_ORDER = tuple(spec.state_field for spec in QUESTION_SPECS)
_EVIDENCE_ASSERTIONS = {"present", "absent", "uncertain"}
_MEDICAL_ASSERTION_FIELDS = {"symptom", "accompanying_symptoms"}
_SEMANTIC_STATUSES = {"available", "unavailable", "unknown", "partial", "ambiguous", "uncertain"}

PREFERRED_DAYS_KEY = "preferred_days"
PREFERRED_DATES_KEY = "preferred_dates"
PREFERRED_SESSIONS_KEY = "preferred_sessions"
ALL_WEEKDAYS = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]
WEEKDAY_DAYS = ["週一", "週二", "週三", "週四", "週五"]
WEEKEND_DAYS = ["週六", "週日"]
ALL_SESSIONS = ["上午", "下午", "夜間"]
ANY_SESSION_TERMS = [
    "都可以", "我都可以", "隨便", "隨便安排", "都行", "皆可", "任何時段都可以",
    "上午下午晚上都可以", "上午下午夜間都可以", "時間沒差", "什麼時段都行",
]
ANY_DAY_TERMS = [
    "哪天都可以", "每天都可以", "日期都可以", "日期沒差", "日期都沒差",
    "任何一天都行", "任何一天都可以",
]
def apply_user_message(
    case: TriageCase,
    message: str,
    *,
    semantic_first: bool = False,
    record_history: bool = True,
    apply_legacy_safety: bool = False,
) -> TriageCase:
    text = message.strip()
    if not text:
        return case

    revision_mode = case.conversation_state.revision_mode
    if record_history:
        case.history_records.append(Message(role="user", content=text))
    if semantic_first:
        if revision_mode and _looks_like_symptom_revision(text):
            _reset_symptom_dependent_fields(case)
            case.patient_input.symptom = ""
            case.patient_input.body_part = None
            case.department_result = None
            for field in ("symptom", "body_part"):
                if field in case.conversation_state.consumed_fields:
                    case.conversation_state.consumed_fields.remove(field)
        return case
    patient = case.patient_input
    symptom_value = text if has_symptom_semantics(text) else None
    if revision_mode and _looks_like_symptom_revision(text):
        _reset_symptom_dependent_fields(case)
    semantic_result = normalize_message(text, case.conversation_state.last_question_key)
    _apply_semantic_result(case, semantic_result)

    if revision_mode and _looks_like_symptom_revision(text) and symptom_value:
        patient.symptom = symptom_value
        case.department_result = None
    elif not patient.symptom and symptom_value:
        patient.symptom = symptom_value

    body_part = _extract_body_part(text)
    if body_part and (
        revision_mode
        or not patient.body_part
        or _is_more_specific_body_part(body_part, patient.body_part)
    ):
        patient.body_part = body_part
        if revision_mode:
            case.department_result = None

    duration = _extract_duration(text)
    if duration:
        patient.duration = duration
        _consume_field(case, "duration", "available", 0.9)

    severity = _extract_severity(text)
    if severity:
        patient.severity = severity

    onset = _extract_onset(text)
    if onset:
        patient.onset = onset

    for symptom in _extract_accompanying_symptoms(text):
        if symptom not in patient.accompanying_symptoms:
            patient.accompanying_symptoms.append(symptom)

    availability = case.availability

    _refresh_collected_fields(case)
    if revision_mode:
        case.confirmed = False
        case.recommendation_generated = False
        case.script_generated = False
        case.selected_recommendation_id = None
        case.conversation_state.confirmed = False
        case.conversation_state.awaiting_confirmation = False
        case.conversation_state.revision_mode = False
    logger.info(
        "apply_user_message case_id=%s history_len=%s collected_fields=%s red_flag_count=%s red_flags_checked=%s last_question_key=%s consumed_fields=%s question_attempts=%s field_statuses=%s",
        case.case_id,
        len(case.history_records),
        patient.collected_fields,
        len(patient.red_flags),
        patient.red_flags_checked,
        case.conversation_state.last_question_key,
        case.conversation_state.consumed_fields,
        case.conversation_state.question_attempts,
        case.conversation_state.field_statuses,
    )
    return case


def evaluate_urgency(case: TriageCase, *, mark_next_question: bool | None = True) -> UrgencyResult:
    """Compatibility wrapper around the official deterministic TTAS evaluator.

    This function intentionally performs no raw-text medical interpretation and
    assigns no project-defined urgency score. Production chat calls the TTAS
    evaluator directly; older structured callers may keep using this wrapper.
    """
    from app.services.ttas_evaluator import apply_ttas_evaluation

    result = apply_ttas_evaluation(case)
    if result.warning_required or mark_next_question is None:
        return result
    next_question = (
        next_question_for(case)
        if mark_next_question
        else (
            question_text_for_field(case, missing[0])
            if (missing := missing_checklist_fields(case))
            else None
        )
    )
    return result.model_copy(
        update={
            "need_more_info": next_question is not None,
            "next_question": next_question,
            "is_final": next_question is None,
        }
    )


def next_question_for(case: TriageCase) -> Optional[str]:
    question_key = _next_question_key(case)
    if question_key is None:
        case.conversation_state.last_question_key = None
        return None

    question_text = _question_text_for(case, question_key)
    _mark_question_asked(case, question_key)
    return question_text


def _next_question_key(case: TriageCase) -> Optional[str]:
    missing = missing_checklist_fields(case)
    return missing[0] if missing else None


def missing_checklist_fields(
    case: TriageCase,
    *,
    apply_attempt_fallback: bool = True,
) -> list[str]:
    """Return deterministic checklist gaps without letting AI control flow."""
    _sync_consumed_fields(case)
    state = case.conversation_state
    missing: list[str] = []
    for field in CHECKLIST_FIELD_ORDER:
        if _field_satisfied(case, field):
            continue
        if field != RED_FLAG_QUESTION_KEY and _question_attempts(case, field) >= MAX_QUESTION_ATTEMPTS:
            if (
                state.clarification_reasons.get(field) == "AI extraction confidence below threshold"
                and state.field_confidence.get(field, 1.0) < ACCEPT_THRESHOLD
            ):
                missing.append(field)
                continue
            if apply_attempt_fallback:
                _fallback_field(case, field)
                continue
        missing.append(field)
    return missing


def question_text_for_field(case: TriageCase, field: str) -> str:
    if field not in CHECKLIST_FIELD_ORDER:
        raise ValueError(f"Unsupported checklist field: {field}")
    return _question_text_for(case, field)


def mark_questions_asked(case: TriageCase, fields: list[str]) -> None:
    for field in fields:
        if field not in CHECKLIST_FIELD_ORDER:
            raise ValueError(f"Unsupported checklist field: {field}")
        _mark_question_asked(case, field)


def _field_satisfied(case: TriageCase, field: str) -> bool:
    patient = case.patient_input
    availability = case.availability
    state = case.conversation_state
    if field in state.consumed_fields:
        return True

    if field == "symptom":
        return bool(patient.symptom)
    if field == RED_FLAG_QUESTION_KEY:
        return patient.red_flags_checked
    if field == "body_part":
        return bool(patient.body_part)
    if field == "duration":
        return bool(patient.duration)
    if field == "severity":
        return bool(patient.severity)
    if field == PREFERRED_DAYS_KEY:
        return bool(availability.preferred_days) or state.field_statuses.get(field) == "unavailable"
    if field == PREFERRED_SESSIONS_KEY:
        return bool(availability.preferred_sessions) or state.field_statuses.get(field) == "unavailable"
    return True


def _fallback_field(case: TriageCase, field: str) -> None:
    state = case.conversation_state
    state.field_statuses[field] = state.field_statuses.get(field, "unknown")
    state.field_confidence[field] = state.field_confidence.get(field, 0.0)
    state.clarification_reasons[field] = "max question attempts reached; explicitly skipped as unknown"
    _consume_field(case, field, state.field_statuses[field], state.field_confidence[field])


def _question_text_for(case: TriageCase, field: str) -> str:
    confidence = case.conversation_state.field_confidence.get(field, 1.0)
    status = case.conversation_state.field_statuses.get(field)
    if _question_attempts(case, field) > 0 and (confidence < ACCEPT_THRESHOLD or status in {"unknown", "ambiguous", "unavailable"}):
        return clarification_prompt(field, status)
    spec = question_spec_for_field(field)
    return select_question_variant(spec.question_id)


def _question_attempts(case: TriageCase, field: str) -> int:
    return case.conversation_state.question_attempts.get(field, 0)


def _apply_semantic_result(case: TriageCase, result: NormalizationResult) -> None:
    patient = case.patient_input
    if result.severity is not None:
        patient.severity_normalized = result.severity
        if result.severity.severity_level and accepted(result.severity.confidence, result.severity.semantic_status):
            patient.severity = result.severity.severity_level

    for extraction in result.extractions:
        _record_semantic_extraction(case, extraction)
        _apply_semantic_extraction(case, extraction)

    if result.extractions:
        case.semantic_extractions.extend(result.extractions)
        case.semantic_extractions = case.semantic_extractions[-50:]


def apply_semantic_extractions(
    case: TriageCase,
    extractions: list[SemanticExtraction],
    *,
    allow_red_flag_completion: bool = True,
) -> None:
    """Apply schema-validated extraction while preserving deterministic gates."""
    accepted_extractions: list[SemanticExtraction] = []
    for extraction in extractions:
        is_ai_extraction = extraction.extractor.startswith("ai")
        if extraction.field not in {
            *CHECKLIST_FIELD_ORDER, PREFERRED_DATES_KEY, "onset", "accompanying_symptoms",
        }:
            if is_ai_extraction:
                _log_ai_apply_decision(case, extraction, False, "invalid_field")
            continue
        has_medical_concept = _has_nonempty_medical_concept(extraction)
        if is_ai_extraction and (
            extraction.semantic_status not in _SEMANTIC_STATUSES
            or type(extraction.confidence) not in {int, float}
            or not math.isfinite(extraction.confidence)
            or not 0.0 <= extraction.confidence <= 1.0
        ):
            _log_ai_apply_decision(case, extraction, False, "invalid_semantic_metadata")
            continue
        if (
            is_ai_extraction
            and has_medical_concept
            and extraction.assertion not in _EVIDENCE_ASSERTIONS
        ):
            _log_ai_apply_decision(case, extraction, False, "missing_or_invalid_assertion")
            continue
        if extraction.field == RED_FLAG_QUESTION_KEY:
            if is_ai_extraction:
                _log_ai_apply_decision(case, extraction, False, "red_flags_not_semantic_evidence")
            continue
        if (
            is_ai_extraction
            and not case.conversation_state.revision_mode
            and extraction.semantic_status in {"available", "partial"}
            and extraction.field != "accompanying_symptoms"
            and not (
                extraction.field == "symptom"
                and extraction.assertion in {"absent", "uncertain"}
            )
            and _field_has_value(case, extraction.field)
        ):
            _log_ai_apply_decision(case, extraction, False, "duplicate_existing_value")
            continue
        value_valid = (
            ai_normalized_value_valid(
                extraction.field,
                extraction.normalized_value,
                extraction.semantic_status,
                extraction.source_text,
            )
            if is_ai_extraction
            else normalized_value_valid(
                extraction.field,
                extraction.normalized_value,
                extraction.semantic_status,
            )
        )
        if (
            extraction.field not in {RED_FLAG_QUESTION_KEY, PREFERRED_DATES_KEY}
            and extraction.semantic_status != "uncertain"
            and not value_valid
        ):
            logger.info(
                "semantic extraction rejected field=%s reason=strict_field_validation",
                extraction.field,
            )
            if is_ai_extraction:
                _log_ai_apply_decision(case, extraction, False, "invalid_normalized_value")
            continue
        nonpositive_medical_evidence = (
            is_ai_extraction
            and has_medical_concept
            and extraction.assertion in {"absent", "uncertain"}
        )
        if nonpositive_medical_evidence:
            if extraction.confidence < ACCEPT_THRESHOLD:
                _log_ai_apply_decision(case, extraction, False, "confidence_too_low")
                continue
            _record_semantic_extraction(case, extraction)
            accepted_extractions.append(extraction)
            _log_ai_apply_decision(case, extraction, True, None)
            continue
        if is_ai_extraction and not accepted(extraction.confidence, extraction.semantic_status):
            reason = (
                "confidence_too_low"
                if extraction.semantic_status not in {"unknown", "ambiguous"}
                else "status_requires_clarification"
            )
            if not _field_has_value(case, extraction.field):
                _record_semantic_extraction(case, extraction)
                case.conversation_state.clarification_reasons[extraction.field] = (
                    "AI extraction confidence below threshold"
                    if reason == "confidence_too_low"
                    else "AI extraction requires clarification"
                )
            _log_ai_apply_decision(case, extraction, False, reason)
            continue
        _record_semantic_extraction(case, extraction)
        _apply_semantic_extraction(case, extraction)
        accepted_extractions.append(extraction)
        if is_ai_extraction:
            _log_ai_apply_decision(case, extraction, True, None)
    if accepted_extractions:
        case.semantic_extractions.extend(accepted_extractions)
        case.semantic_extractions = case.semantic_extractions[-50:]
    _refresh_collected_fields(case)


def _has_nonempty_medical_concept(extraction: SemanticExtraction) -> bool:
    if extraction.field == "symptom":
        return isinstance(extraction.normalized_value, str) and bool(extraction.normalized_value.strip())
    if extraction.field == "accompanying_symptoms":
        return isinstance(extraction.normalized_value, list) and any(
            isinstance(item, str) and item.strip() for item in extraction.normalized_value
        )
    return False


def _field_has_value(case: TriageCase, field_name: str) -> bool:
    if field_name in {"symptom", "body_part", "duration", "severity", "onset"}:
        return bool(getattr(case.patient_input, field_name))
    if field_name == "preferred_dates":
        return bool(case.availability.preferred_dates)
    if field_name == "preferred_days":
        return bool(case.availability.preferred_days)
    if field_name == "preferred_sessions":
        return bool(case.availability.preferred_sessions)
    return False


def _log_ai_apply_decision(
    case: TriageCase,
    extraction: SemanticExtraction,
    accepted_value: bool,
    reject_reason: str | None,
) -> None:
    logger.info(
        "[SEMANTIC_AI_DECISION] case_id=%s field=%s accepted=%s reject_reason=%s",
        case.case_id,
        extraction.field,
        str(accepted_value).lower(),
        reject_reason or "null",
    )


def _record_semantic_extraction(case: TriageCase, extraction: SemanticExtraction) -> None:
    state = case.conversation_state
    state.field_statuses[extraction.field] = extraction.semantic_status
    state.field_confidence[extraction.field] = extraction.confidence
    if extraction.follow_up_reason:
        state.clarification_reasons[extraction.field] = extraction.follow_up_reason


def _apply_semantic_extraction(case: TriageCase, extraction: SemanticExtraction) -> None:
    availability = case.availability
    if extraction.field in {"preferred_dates", "preferred_days", "preferred_sessions", "can_take_leave"}:
        availability.semantic_status[extraction.field] = extraction.semantic_status
        availability.confidence[extraction.field] = extraction.confidence

    if (
        extraction.extractor.startswith("ai")
        and extraction.field in _MEDICAL_ASSERTION_FIELDS
        and extraction.assertion != "present"
    ):
        return

    if extraction.semantic_status != "uncertain" and not accepted(
        extraction.confidence,
        extraction.semantic_status,
    ):
        return

    if extraction.field in {"symptom", "body_part", "duration"}:
        text = str(extraction.normalized_value or "").strip()
        if extraction.field == "symptom" and extraction.extractor.startswith("ai"):
            text = extraction.source_text.strip()
        if extraction.field == "body_part" and any(
            term in text for term in ("胃部", "腸胃", "胃")
        ):
            text = normalize_body_part(text) or text
        current = getattr(case.patient_input, extraction.field)
        more_specific_batch_body_part = (
            extraction.field == "body_part"
            and extraction.extractor == "deterministic_batch"
            and _is_more_specific_body_part(text, current)
        )
        if text and (
            case.conversation_state.revision_mode
            or not current
            or more_specific_batch_body_part
        ):
            setattr(case.patient_input, extraction.field, text)
            _consume_field(case, extraction.field, extraction.semantic_status, extraction.confidence)
    elif extraction.field == "onset":
        text = str(extraction.normalized_value or "").strip()
        if text and (case.conversation_state.revision_mode or not case.patient_input.onset):
            case.patient_input.onset = text
    elif extraction.field == "accompanying_symptoms":
        _merge_list(case.patient_input.accompanying_symptoms, _as_string_list(extraction.normalized_value))
    elif extraction.field == "preferred_dates":
        values = [
            value
            for value in _as_string_list(extraction.normalized_value)
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
        ]
        if case.conversation_state.revision_mode:
            availability.preferred_dates = values
        elif not availability.preferred_dates:
            _merge_list(availability.preferred_dates, values)
    elif extraction.field == "preferred_days":
        values = _as_string_list(extraction.normalized_value)
        if case.conversation_state.revision_mode:
            availability.preferred_days = values
        elif not availability.preferred_days:
            _merge_list(availability.preferred_days, values)
        _consume_field(case, extraction.field, extraction.semantic_status, extraction.confidence)
    elif extraction.field == "preferred_sessions":
        values = _as_string_list(extraction.normalized_value)
        if case.conversation_state.revision_mode:
            availability.preferred_sessions = values
        elif not availability.preferred_sessions:
            _merge_list(availability.preferred_sessions, values)
        _consume_field(case, extraction.field, extraction.semantic_status, extraction.confidence)
    elif extraction.field == "can_take_leave" and isinstance(extraction.normalized_value, bool):
        availability.can_take_leave = extraction.normalized_value
        _consume_field(case, extraction.field, extraction.semantic_status, extraction.confidence)
    elif extraction.field == "severity":
        if isinstance(extraction.normalized_value, dict):
            level = extraction.normalized_value.get("severity_level")
        else:
            level = extraction.normalized_value
        if level and (case.conversation_state.revision_mode or not case.patient_input.severity):
            case.patient_input.severity = str(level)
            _consume_field(case, extraction.field, extraction.semantic_status, extraction.confidence)


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _merge_list(target: list[str], values: list[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _looks_like_symptom_revision(text: str) -> bool:
    if _looks_like_time_preference_revision(text):
        return False
    if looks_like_department_request(text) and not has_symptom_semantics(text):
        return False
    if any(keyword in text for keyword in ["痛", "疼", "不舒服", "麻", "腫", "酸", "癢", "咳", "喘", "暈", "發燒"]):
        return True
    if any(keyword in text for keyword in ["膝", "腰", "背", "肩", "手", "腳", "頭", "胸", "腹", "胃", "喉", "皮膚"]):
        return True
    if any(keyword in text for keyword in ["症狀", "部位", "其實是", "不是"]):
        return True
    return False


def _looks_like_time_preference_revision(text: str) -> bool:
    return any(keyword in text for keyword in ["時段", "時間", "上午", "下午", "夜間", "晚上", "早上", "週", "星期", "禮拜"])


def _reset_symptom_dependent_fields(case: TriageCase) -> None:
    patient = case.patient_input
    patient.duration = None
    patient.severity = None
    patient.onset = None
    patient.accompanying_symptoms = []
    patient.department_context = None
    patient.red_flags = []
    patient.red_flags_checked = False
    patient.red_flags_status = "not_checked"
    patient.severity_normalized = type(patient.severity_normalized)()
    patient.urgency_normalized = type(patient.urgency_normalized)()
    case.ttas_evidence = []
    case.ttas_result = type(case.ttas_result)()
    for field in [RED_FLAG_QUESTION_KEY, "duration", "severity"]:
        if field in case.conversation_state.consumed_fields:
            case.conversation_state.consumed_fields.remove(field)
        if field in case.conversation_state.asked_fields:
            case.conversation_state.asked_fields.remove(field)
        case.conversation_state.question_attempts.pop(field, None)
        case.conversation_state.field_statuses.pop(field, None)
        case.conversation_state.field_confidence.pop(field, None)


def _mark_question_asked(case: TriageCase, question_key: str) -> None:
    state = case.conversation_state
    state.last_question_key = question_key
    state.question_attempts[question_key] = state.question_attempts.get(question_key, 0) + 1
    if question_key not in state.asked_fields:
        state.asked_fields.append(question_key)


def _consume_field(
    case: TriageCase,
    field: str,
    status: str | None = None,
    confidence: float | None = None,
) -> None:
    state = case.conversation_state
    if field not in state.consumed_fields:
        state.consumed_fields.append(field)
    if status:
        state.field_statuses[field] = status
    if confidence is not None:
        state.field_confidence[field] = confidence


def _sync_consumed_fields(case: TriageCase) -> None:
    patient = case.patient_input
    availability = case.availability
    if patient.symptom:
        _consume_field(case, "symptom", case.conversation_state.field_statuses.get("symptom"))
    if patient.body_part:
        _consume_field(case, "body_part", case.conversation_state.field_statuses.get("body_part"))
    if patient.duration:
        _consume_field(case, "duration", case.conversation_state.field_statuses.get("duration"))
    if patient.severity:
        _consume_field(case, "severity", case.conversation_state.field_statuses.get("severity"))
    if patient.red_flags_checked:
        red_flag_status = {
            "uncertain": "uncertain",
            "positive_specific": "available",
            "negative": "unavailable",
        }.get(patient.red_flags_status, "available" if patient.red_flags else "unavailable")
        _consume_field(
            case,
            RED_FLAG_QUESTION_KEY,
            red_flag_status,
            case.conversation_state.field_confidence.get(RED_FLAG_QUESTION_KEY),
        )
    if availability.preferred_days or case.conversation_state.field_statuses.get(PREFERRED_DAYS_KEY) == "unavailable":
        _consume_field(
            case,
            PREFERRED_DAYS_KEY,
            availability.semantic_status.get(PREFERRED_DAYS_KEY) or case.conversation_state.field_statuses.get(PREFERRED_DAYS_KEY),
            availability.confidence.get(PREFERRED_DAYS_KEY) or case.conversation_state.field_confidence.get(PREFERRED_DAYS_KEY),
        )
    if availability.preferred_sessions or case.conversation_state.field_statuses.get(PREFERRED_SESSIONS_KEY) == "unavailable":
        _consume_field(
            case,
            PREFERRED_SESSIONS_KEY,
            availability.semantic_status.get(PREFERRED_SESSIONS_KEY) or case.conversation_state.field_statuses.get(PREFERRED_SESSIONS_KEY),
            availability.confidence.get(PREFERRED_SESSIONS_KEY) or case.conversation_state.field_confidence.get(PREFERRED_SESSIONS_KEY),
        )


def _extract_body_part(text: str) -> Optional[str]:
    text = strip_negated_red_flags(text)
    if looks_like_department_request(text) and not has_symptom_semantics(text):
        return None
    return normalize_body_part(text)


def _is_more_specific_body_part(candidate: str, current: str | None) -> bool:
    """Allow a precise child location to refine, but never degrade, a broad one."""
    refinement_parents = {
        "大腿": {"腿"},
        "小腿": {"腿"},
        "左腿": {"腿"},
        "右腿": {"腿"},
        "左大腿": {"腿", "左腿", "大腿"},
        "右大腿": {"腿", "右腿", "大腿"},
        "左小腿": {"腿", "左腿", "小腿"},
        "右小腿": {"腿", "右腿", "小腿"},
        "腳踝": {"腳"},
    }
    normalized_candidate = str(candidate or "").strip()
    normalized_current = str(current or "").strip()
    return bool(
        normalized_candidate
        and normalized_current
        and normalized_current in refinement_parents.get(normalized_candidate, set())
    )


def _extract_duration(text: str) -> Optional[str]:
    if "一年半" in text:
        return "18個月"
    if "半年" in text:
        return "6個月"
    match = re.search(r"(\d+\s*(?:天|週|周|個月|年))", text)
    if match:
        return match.group(1).replace(" ", "")
    match = re.search(r"(\d+)\s*(?:個)?(?:禮拜|星期)", text)
    if match:
        return f"{int(match.group(1))}週"
    if "好幾天" in text:
        return "好幾天"
    if "幾天" in text:
        return "幾天"
    if "半天" in text:
        return "半天"
    if re.search(r"好幾(?:個)?(?:禮拜|星期)", text):
        return "好幾週"
    if re.search(r"幾(?:個)?(?:禮拜|星期)", text):
        return "幾週"
    match = re.search(r"([一二兩三四五六七八九十]+)\s*(?:個)?(?:禮拜|星期)", text)
    if match:
        amount = _parse_chinese_duration_number(match.group(1))
        if amount is not None:
            return f"{amount}週"
    match = re.search(r"([一二兩三四五六七八九十]+)\s*(天|週|周|個月|年)", text)
    if match:
        amount = _parse_chinese_duration_number(match.group(1))
        if amount is not None:
            unit = match.group(2)
            if unit == "周":
                unit = "週"
            return f"{amount}{unit}"
    onset_match = DURATION_ONSET_PATTERN.search(text)
    if onset_match:
        matched = onset_match.group(0)
        if "昨天" in matched:
            return "1天"
        if "今天" in matched:
            return "1天內"
        if "前天" in matched:
            return "2天"
        if "上週" in matched:
            return "從上週開始"
        if "上個月" in matched:
            return "從上個月開始"
        for daypart in ("早上", "上午", "中午", "下午", "晚上", "半夜"):
            if daypart in matched:
                return f"從{daypart}開始"
    return None


def _parse_chinese_duration_number(value: str) -> Optional[int]:
    if value in CHINESE_DURATION_NUMBERS:
        return CHINESE_DURATION_NUMBERS[value]
    if value == "十":
        return 10
    if "十" in value:
        tens_text, ones_text = value.split("十", 1)
        tens = CHINESE_DURATION_NUMBERS.get(tens_text, 1 if tens_text == "" else None)
        ones = CHINESE_DURATION_NUMBERS.get(ones_text, 0 if ones_text == "" else None)
        if tens is not None and ones is not None:
            return tens * 10 + ones
    return None


def _extract_severity(text: str) -> Optional[str]:
    if any(
        word in text
        for word in [
            "沒有影響",
            "正常生活",
            "正常作息",
            "不影響作息",
            "不影響日常生活",
            "不影響睡眠",
            "還能正常上班",
            "還能正常走路",
            "沒什麼影響",
            "沒有很嚴重",
        ]
    ):
        return "輕微"
    if any(word in text for word in ["很痛", "劇痛", "嚴重", "影響睡覺", "影響睡眠", "睡不著", "痛醒", "無法工作", "無法走路"]):
        return "很痛 / 影響明顯"
    if any(word in text for word in ["明顯", "很不舒服", "走路困難", "爬樓梯很吃力"]):
        return "明顯"
    if any(word in text for word in ["中等", "中度", "還可以忍"]):
        return "中度"
    if any(word in text for word in ["輕微", "有點痛", "一點痛", "不太影響", "還好但"]):
        return "輕微"
    return None


def _extract_onset(text: str) -> Optional[str]:
    if any(word in text for word in ["突然", "突發", "一下子"]):
        return "突發"
    if any(word in text for word in ["慢慢", "漸進", "越來越"]):
        return "漸進"
    return None


def _extract_accompanying_symptoms(text: str) -> list[str]:
    symptoms = []
    symptom_text = strip_negated_red_flags(text)
    candidates = ["爬樓梯吃力", "走路困難", "發燒", "紅腫熱痛", "頭暈", "噁心", "麻", "無力"]
    for candidate in candidates:
        if candidate in symptom_text and not is_negated_keyword(symptom_text, candidate):
            symptoms.append(candidate)
    return symptoms


def _normalize_weekday_text(text: str) -> str:
    normalized = text
    replacements = {
        "星期一": "週一",
        "星期二": "週二",
        "星期三": "週三",
        "星期四": "週四",
        "星期五": "週五",
        "星期六": "週六",
        "星期日": "週日",
        "星期天": "週日",
        "禮拜一": "週一",
        "禮拜二": "週二",
        "禮拜三": "週三",
        "禮拜四": "週四",
        "禮拜五": "週五",
        "禮拜六": "週六",
        "禮拜日": "週日",
        "禮拜天": "週日",
        "周一": "週一",
        "周二": "週二",
        "周三": "週三",
        "周四": "週四",
        "周五": "週五",
        "周六": "週六",
        "周日": "週日",
        "周天": "週日",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    if "明天" in text:
        tomorrow = _taipei_today() + timedelta(days=1)
        normalized = f"{normalized} {ALL_WEEKDAYS[tomorrow.weekday()]}"
    if "大後天" in text:
        three_days_later = _taipei_today() + timedelta(days=3)
        normalized = f"{normalized} {ALL_WEEKDAYS[three_days_later.weekday()]}"
    if "後天" in text.replace("大後天", ""):
        after_tomorrow = _taipei_today() + timedelta(days=2)
        normalized = f"{normalized} {ALL_WEEKDAYS[after_tomorrow.weekday()]}"
    return normalized


def _is_any_days_answer(text: str, last_question_key: Optional[str]) -> bool:
    if any(term in text for term in ["每天", "整週", "全週", *ANY_DAY_TERMS]):
        return True
    return (
        last_question_key == PREFERRED_DAYS_KEY
        and any(term in text for term in ["都可以", "都有空", "都行", "皆可"])
        and not any(term in text for term in ["前半", "後半"])
    )


def _taipei_today():
    return datetime.now(ZoneInfo("Asia/Taipei")).date()


def _weekday_range(start: str, end: str) -> list[str]:
    if start not in ALL_WEEKDAYS or end not in ALL_WEEKDAYS:
        return []
    start_index = ALL_WEEKDAYS.index(start)
    end_index = ALL_WEEKDAYS.index(end)
    if start_index <= end_index:
        return ALL_WEEKDAYS[start_index : end_index + 1]
    return [*ALL_WEEKDAYS[start_index:], *ALL_WEEKDAYS[: end_index + 1]]


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _refresh_collected_fields(case: TriageCase) -> None:
    patient = case.patient_input
    fields = []
    if patient.symptom:
        fields.append("symptom")
    if patient.body_part:
        fields.append("body_part")
    if patient.duration:
        fields.append("duration")
    if patient.severity:
        fields.append("severity")
    if patient.onset:
        fields.append("onset")
    if patient.accompanying_symptoms:
        fields.append("accompanying_symptoms")
    if patient.red_flags:
        fields.append("red_flags")
    if patient.red_flags_checked:
        fields.append("red_flags_checked")
    if case.availability.preferred_days:
        fields.append("preferred_days")
    if case.availability.preferred_dates:
        fields.append("preferred_dates")
    if case.availability.preferred_sessions:
        fields.append("preferred_sessions")
    patient.collected_fields = fields
