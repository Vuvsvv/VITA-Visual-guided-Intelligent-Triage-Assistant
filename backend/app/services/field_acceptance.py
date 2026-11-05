from __future__ import annotations

from datetime import date
from typing import Any


def normalized_value_valid(field: str, value: Any, status: str) -> bool:
    """Validate structured values, never classify patient language."""
    if status in {"unknown", "ambiguous"} and value is None:
        return True
    if field in {"symptom", "body_part", "duration", "onset"}:
        limit = {"symptom": 24, "body_part": 30, "duration": 40, "onset": 40}[field]
        return isinstance(value, str) and 0 < len(value.strip()) <= limit and not any(
            character in value for character in "\r\n"
        )
    if field == "accompanying_symptoms":
        return isinstance(value, list) and 0 < len(value) <= 10 and all(
            isinstance(item, str) and 0 < len(item.strip()) <= 30 for item in value
        )
    if field == "severity":
        return isinstance(value, str) and value in {"mild", "moderate", "severe"}
    if field in {"preferred_days", "preferred_sessions"}:
        options = (
            {"週一", "週二", "週三", "週四", "週五", "週六", "週日"}
            if field == "preferred_days" else {"上午", "下午", "夜間"}
        )
        return isinstance(value, list) and (bool(value) or status == "unavailable") and all(
            isinstance(item, str) and item in options for item in value
        )
    if field == "preferred_dates":
        if not isinstance(value, list) or not value:
            return False
        try:
            return all(isinstance(item, str) and date.fromisoformat(item).isoformat() == item for item in value)
        except ValueError:
            return False
    if field == "can_take_leave":
        return isinstance(value, bool)
    return False


def ai_normalized_value_rejection_reason(
    field: str, value: Any, status: str, source_text: str,
) -> str | None:
    if not normalized_value_valid(field, value, status):
        return "invalid_normalized_value"
    # Open-ended anatomy/onset must remain literal. Semantic canonicalization
    # cannot be proven by a dictionary or guessed from the patient's wording.
    if field in {"body_part", "onset"} and isinstance(value, str) and value.strip() not in source_text:
        return "normalized_value_not_supported_by_source"
    return None


def ai_normalized_value_valid(field: str, value: Any, status: str, source_text: str) -> bool:
    return ai_normalized_value_rejection_reason(field, value, status, source_text) is None
