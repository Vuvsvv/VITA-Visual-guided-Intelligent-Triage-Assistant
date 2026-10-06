from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.schemas import TriageCase
from app.services.confidence_scoring import ACCEPT_THRESHOLD
from app.services.field_acceptance import (
    has_strong_uncertainty,
    plausible_semantic_target,
    requires_semantic_refinement,
)


_SUPPORTED_FIELDS = (
    "symptom",
    "body_part",
    "duration",
    "severity",
    "red_flags",
    "preferred_days",
    "preferred_sessions",
)
_UNRELIABLE_STATUSES = {"unknown", "ambiguous"}
_VAGUE_DURATION_VALUES = {"幾天", "好幾天", "unknown"}


@dataclass(frozen=True)
class DeterministicParseSnapshot:
    last_question_key: str | None
    values: dict[str, Any]
    consumed_fields: frozenset[str]
    field_statuses: dict[str, str]
    field_confidence: dict[str, float]


@dataclass(frozen=True)
class SemanticRefinementDecision:
    should_call: bool
    reason: str
    target_field: str | None
    reliable_fields: tuple[str, ...] = ()


def capture_deterministic_parse_snapshot(case: TriageCase) -> DeterministicParseSnapshot:
    state = case.conversation_state
    return DeterministicParseSnapshot(
        last_question_key=state.last_question_key,
        values={field: _field_value(case, field) for field in _SUPPORTED_FIELDS},
        consumed_fields=frozenset(state.consumed_fields),
        field_statuses=dict(state.field_statuses),
        field_confidence=dict(state.field_confidence),
    )


def decide_semantic_refinement(
    *,
    before: DeterministicParseSnapshot,
    after: TriageCase,
    user_text: str,
    request_confirmed: bool = False,
) -> SemanticRefinementDecision:
    """Decide whether deterministic parsing needs semantic AI assistance.

    The decision is based on the field requested before parsing, field changes
    made by this turn, and deterministic confidence/status metadata. It does not
    alter triage state or let AI choose the next question.
    """
    target = before.last_question_key if before.last_question_key in _SUPPORTED_FIELDS else None
    text = user_text.strip()
    if request_confirmed:
        return SemanticRefinementDecision(False, "request_confirmation_is_explicit", target)
    if not text:
        return SemanticRefinementDecision(False, "no_user_text", target)
    if target == "red_flags":
        return SemanticRefinementDecision(False, "red_flags_are_deterministic_only", target)
    if target is None and not any(
        plausible_semantic_target(field, text)
        for field in _SUPPORTED_FIELDS
        if field != "red_flags"
    ):
        return SemanticRefinementDecision(False, "no_supported_semantic_content", None)

    changed_fields = tuple(
        field for field in _SUPPORTED_FIELDS if _field_changed(before, after, field)
    )
    conflicting_fields = tuple(
        field for field in changed_fields if _field_conflicts(before, after, field)
    )
    if conflicting_fields:
        return SemanticRefinementDecision(True, "deterministic_value_conflict", target)

    if has_strong_uncertainty(text):
        return SemanticRefinementDecision(True, "strongly_ambiguous_user_text", target)

    reliable_fields = tuple(
        field
        for field in changed_fields
        if _field_is_reliable(after, field, text)
    )
    if target is not None:
        if target in reliable_fields:
            return SemanticRefinementDecision(
                False,
                "requested_field_resolved_deterministically",
                target,
                reliable_fields,
            )
        return SemanticRefinementDecision(
            True,
            "requested_field_not_resolved_deterministically",
            target,
            reliable_fields,
        )

    if reliable_fields:
        return SemanticRefinementDecision(
            False,
            "new_reliable_fields_resolved_deterministically",
            None,
            reliable_fields,
        )
    return SemanticRefinementDecision(
        True,
        "no_reliable_deterministic_extraction",
        None,
    )


def _field_changed(
    before: DeterministicParseSnapshot,
    after: TriageCase,
    field: str,
) -> bool:
    state = after.conversation_state
    return (
        before.values[field] != _field_value(after, field)
        or (field not in before.consumed_fields and field in state.consumed_fields)
        or before.field_statuses.get(field) != state.field_statuses.get(field)
        or before.field_confidence.get(field) != state.field_confidence.get(field)
    )


def _field_conflicts(
    before: DeterministicParseSnapshot,
    after: TriageCase,
    field: str,
) -> bool:
    previous = before.values[field]
    current = _field_value(after, field)
    return _has_value(previous, field) and _has_value(current, field) and previous != current


def _field_is_reliable(case: TriageCase, field: str, user_text: str) -> bool:
    state = case.conversation_state
    status = state.field_statuses.get(field)
    confidence = state.field_confidence.get(field)
    if status in _UNRELIABLE_STATUSES:
        return False
    if confidence is not None and confidence < ACCEPT_THRESHOLD:
        return False

    value = _field_value(case, field)
    if not _has_value(value, field):
        return False

    if field != "red_flags" and requires_semantic_refinement(user_text):
        return False

    if field == "duration" and value in _VAGUE_DURATION_VALUES:
        return False
    if field == "severity":
        normalized = case.patient_input.severity_normalized
        if normalized.semantic_status in _UNRELIABLE_STATUSES:
            return False
        if normalized.confidence and normalized.confidence < ACCEPT_THRESHOLD:
            return False
    return True


def _field_value(case: TriageCase, field: str) -> Any:
    patient = case.patient_input
    availability = case.availability
    if field == "symptom":
        return patient.symptom
    if field == "body_part":
        return patient.body_part
    if field == "duration":
        return patient.duration
    if field == "severity":
        return patient.severity
    if field == "red_flags":
        return patient.red_flags_checked, tuple(patient.red_flags)
    if field == "preferred_days":
        return tuple(availability.preferred_days)
    if field == "preferred_sessions":
        return tuple(availability.preferred_sessions)
    raise ValueError(f"Unsupported semantic refinement field: {field}")


def _has_value(value: Any, field: str) -> bool:
    if field == "red_flags":
        checked, _ = value
        return bool(checked)
    return bool(value)
