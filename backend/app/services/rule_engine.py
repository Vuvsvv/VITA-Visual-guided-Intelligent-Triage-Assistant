from __future__ import annotations

import math
import re
import logging
from typing import Any

from app.schemas import Message, SemanticExtraction, TriageCase
from app.services.confidence_scoring import ACCEPT_THRESHOLD, accepted
from app.services.department_knowledge import normalize_concept
from app.services.field_acceptance import ai_normalized_value_valid, normalized_value_valid
from app.services.question_specs import QUESTION_SPECS

logger = logging.getLogger(__name__)


RED_FLAG_QUESTION_KEY = "red_flags"
QUESTION_TEXTS = {spec.state_field: spec.canonical_text for spec in QUESTION_SPECS}
CHECKLIST_FIELD_ORDER = tuple(spec.state_field for spec in QUESTION_SPECS)
_EVIDENCE_ASSERTIONS = {"present", "absent", "uncertain"}
_MEDICAL_ASSERTION_FIELDS = {"symptom", "accompanying_symptoms"}
_SEMANTIC_STATUSES = {"available", "unavailable", "unknown", "partial", "ambiguous", "uncertain"}

PREFERRED_DAYS_KEY = "preferred_days"
PREFERRED_DATES_KEY = "preferred_dates"
PREFERRED_SESSIONS_KEY = "preferred_sessions"


def apply_user_message(
    case: TriageCase,
    message: str,
    *,
    semantic_first: bool = False,
    record_history: bool = True,
    apply_legacy_safety: bool = False,
) -> TriageCase:
    """Record input only; all free-text understanding belongs to the Interpreter."""
    text = message.strip()
    if text and record_history:
        case.history_records.append(Message(role="user", content=text))
    return case


def mark_questions_asked(case: TriageCase, fields: list[str]) -> None:
    for field in fields:
        if field not in CHECKLIST_FIELD_ORDER:
            raise ValueError(f"Unsupported checklist field: {field}")
        _mark_question_asked(case, field)


def apply_semantic_extractions(
    case: TriageCase,
    extractions: list[SemanticExtraction],
    *,
    allow_red_flag_completion: bool = True,
    replace_existing_fields: set[str] | None = None,
) -> None:
    """Apply schema-validated extraction while preserving deterministic gates."""
    accepted_extractions: list[SemanticExtraction] = []
    replacement_fields = (replace_existing_fields or set()) & {
        "symptom", "body_part", "duration", "severity", "onset", "accompanying_symptoms",
    }
    for extraction in extractions:
        is_ai_extraction = extraction.extractor.startswith("ai")
        replace_existing = bool(
            is_ai_extraction and extraction.field in replacement_fields
            and type(extraction.confidence) in {int, float} and math.isfinite(extraction.confidence)
            and ACCEPT_THRESHOLD <= extraction.confidence <= 1
            and extraction.semantic_status in {"available", "partial", "unavailable"}
            and extraction.source_text.strip()
            and any(extraction.source_text in item.content for item in case.history_records if item.role == "user")
        )
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
        duplicate_candidate = bool(
            is_ai_extraction
            and not case.conversation_state.revision_mode
            and not replace_existing
            and extraction.semantic_status in {"available", "partial"}
            and extraction.field != "accompanying_symptoms"
            and not (
                extraction.field == "symptom"
                and extraction.assertion in {"absent", "uncertain"}
            )
            and _field_has_value(case, extraction.field)
        )
        # A further present symptom concept never replaces the stored symptom,
        # but it stays retrievable evidence instead of being discarded.
        additional_symptom = bool(
            duplicate_candidate
            and extraction.field == "symptom"
            and extraction.assertion == "present"
            and _is_new_symptom_concept(case, extraction, accepted_extractions)
        )
        # A further stated body part never replaces the stored one either; it stays evidence
        # so a later symptom's own body part is not mistaken for the first one.
        additional_body_part = bool(
            duplicate_candidate
            and extraction.field == "body_part"
            and _is_new_body_part(case, extraction, accepted_extractions)
        )
        if duplicate_candidate and not (additional_symptom or additional_body_part):
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
            _apply_semantic_extraction(case, extraction, replace_existing=replace_existing)
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
        if additional_symptom or additional_body_part:
            accepted_extractions.append(extraction)
            _log_ai_apply_decision(case, extraction, True, None)
            continue
        _record_semantic_extraction(case, extraction)
        _apply_semantic_extraction(case, extraction, replace_existing=replace_existing)
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


def _is_new_symptom_concept(
    case: TriageCase, extraction: SemanticExtraction, pending: list[SemanticExtraction],
) -> bool:
    """Compare structured concepts only; the stored symptom keeps the AI source text."""
    concept = normalize_concept(str(extraction.normalized_value))
    known = {normalize_concept(case.patient_input.symptom or "")}
    known.update(
        normalize_concept(str(item.normalized_value))
        for item in (*case.semantic_extractions, *pending)
        if item.field == "symptom" and item.assertion == "present"
    )
    return bool(concept) and concept not in known


def _is_new_body_part(
    case: TriageCase, extraction: SemanticExtraction, pending: list[SemanticExtraction],
) -> bool:
    value = str(extraction.normalized_value or "").strip()
    known = {(case.patient_input.body_part or "").strip()}
    known.update(
        str(item.normalized_value or "").strip()
        for item in (*case.semantic_extractions, *pending) if item.field == "body_part"
    )
    return bool(value) and value not in known


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


def _apply_semantic_extraction(case: TriageCase, extraction: SemanticExtraction, *, replace_existing: bool = False) -> None:
    availability = case.availability
    if extraction.field in {"preferred_dates", "preferred_days", "preferred_sessions", "can_take_leave"}:
        availability.semantic_status[extraction.field] = extraction.semantic_status
        availability.confidence[extraction.field] = extraction.confidence

    if (
        extraction.extractor.startswith("ai")
        and extraction.field in _MEDICAL_ASSERTION_FIELDS
        and extraction.assertion != "present"
    ):
        if replace_existing and extraction.field == "accompanying_symptoms" and extraction.assertion == "absent":
            removed = set(_as_string_list(extraction.normalized_value))
            case.patient_input.accompanying_symptoms = [
                concept for concept in case.patient_input.accompanying_symptoms if concept not in removed
            ]
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
        current = getattr(case.patient_input, extraction.field)
        if text and (
            case.conversation_state.revision_mode
            or replace_existing
            or not current
        ):
            setattr(case.patient_input, extraction.field, text)
            _consume_field(case, extraction.field, extraction.semantic_status, extraction.confidence)
    elif extraction.field == "onset":
        text = str(extraction.normalized_value or "").strip()
        if text and (replace_existing or case.conversation_state.revision_mode or not case.patient_input.onset):
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
        if level and (replace_existing or case.conversation_state.revision_mode or not case.patient_input.severity):
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
