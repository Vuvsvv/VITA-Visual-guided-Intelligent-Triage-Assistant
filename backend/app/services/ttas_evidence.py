from __future__ import annotations

import math
from typing import Any

from app.schemas import TTASEvidence, TriageCase
from app.services.confidence_scoring import ACCEPT_THRESHOLD
from app.services.ttas_rule_loader import TTASRuleLoadError, load_ttas_rules


_ITEM_KEYS = {"field", "value", "semantic_status", "confidence", "source_text"}
_STATUSES = {"available", "unknown", "ambiguous"}


def validate_ttas_evidence_items(
    value: object,
    *,
    user_sources: list[str],
) -> list[TTASEvidence]:
    if not isinstance(value, list):
        return []
    try:
        field_specs = load_ttas_rules().evidence_fields
    except TTASRuleLoadError:
        return []

    accepted_items: list[TTASEvidence] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != _ITEM_KEYS:
            continue
        field = item.get("field")
        status = item.get("semantic_status")
        source = item.get("source_text")
        confidence = item.get("confidence")
        if (
            not isinstance(field, str)
            or field not in field_specs
            or status not in _STATUSES
            or not isinstance(source, str)
            or not source.strip()
            or not any(source.strip() in text for text in user_sources)
            or type(confidence) not in {int, float}
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or confidence < ACCEPT_THRESHOLD
            or not _value_matches_spec(item.get("value"), field_specs[field], status)
        ):
            continue
        accepted_items.append(
            TTASEvidence(
                field=field,
                value=item.get("value"),
                semantic_status=status,
                confidence=float(confidence),
                source_text=source.strip(),
            )
        )
    return accepted_items


def apply_ttas_evidence(case: TriageCase, evidence: list[TTASEvidence]) -> None:
    for item in evidence:
        case.ttas_evidence.append(item)
        if item.semantic_status != "available":
            continue
        if item.field == "age_years" and type(item.value) in {int, float}:
            case.patient_input.age_years = float(item.value)
        elif item.field == "age_months" and type(item.value) in {int, float}:
            case.patient_input.age_months = float(item.value)


def effective_ttas_evidence(case: TriageCase) -> dict[str, TTASEvidence]:
    current: dict[str, TTASEvidence] = {}
    for item in case.ttas_evidence:
        if _stored_evidence_valid(case, item):
            current[item.field] = item
    for field, value in (
        ("age_years", case.patient_input.age_years),
        ("age_months", case.patient_input.age_months),
    ):
        if field not in current and type(value) in {int, float} and math.isfinite(value) and value >= 0:
            current[field] = TTASEvidence(
                field=field,
                value=value,
                semantic_status="available",
                confidence=1.0,
                source_text="trusted_profile",
            )
    return current


def _stored_evidence_valid(case: TriageCase, item: TTASEvidence) -> bool:
    if (
        item.confidence < ACCEPT_THRESHOLD
        or not math.isfinite(item.confidence)
        or item.semantic_status not in _STATUSES
    ):
        return False
    if item.source_text == "trusted_profile":
        return item.field in {"age_years", "age_months"}
    return any(
        message.role == "user" and item.source_text in message.content
        for message in case.history_records
    )


def _value_matches_spec(value: Any, spec: dict[str, Any], status: str) -> bool:
    type_spec = spec.get("type")
    if not isinstance(type_spec, str):
        return False
    choices = type_spec.split("|")
    if value is None:
        return "null" in choices and status != "available"
    if "number" in choices:
        if type(value) not in {int, float} or not math.isfinite(value):
            return False
        number = float(value)
        if number < 0 and spec.get("range") is None:
            return False
        bounds = spec.get("range")
        return not (
            isinstance(bounds, list)
            and len(bounds) == 2
            and not float(bounds[0]) <= number <= float(bounds[1])
        )
    if "integer" in choices:
        if type(value) is not int:
            return False
        bounds = spec.get("range")
        return not (
            isinstance(bounds, list)
            and len(bounds) == 2
            and not int(bounds[0]) <= value <= int(bounds[1])
        )
    if "true" in choices or "false" in choices:
        if type(value) is bool:
            return True
        return value == "unknown" and "unknown" in choices and status != "available"
    return isinstance(value, str) and value in choices and (value != "unknown" or status != "available")
