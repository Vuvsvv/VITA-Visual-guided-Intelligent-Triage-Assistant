from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any, Iterable

from pydantic import ValidationError

from app.schemas import Availability, TimePreference
from app.services.confidence_scoring import ACCEPT_THRESHOLD
from app.services.schedule_filter import (
    normalize_session,
    open_rows,
    parse_date,
    row_matches_availability,
    select_feasible_rows,
    weekday_label,
)


WEEKDAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
SESSION_VALUE = {
    "morning": "上午",
    "afternoon": "下午",
    "evening": "晚上",
}
DATE_KINDS = frozenset({
    "date", "date_range", "relative_day", "relative_week", "relative_weekday", "weekday", "weekday_group",
})
_PROPOSAL_KEYS = {
    "kind",
    "relation",
    "priority",
    "source_text",
    "confidence",
    "date_value",
    "start_date",
    "end_date",
    "day_offset",
    "week_offset",
    "weekday",
    "weekdays",
    "session",
}


def validate_and_resolve_time_preferences(
    value: object,
    *,
    user_sources: list[str],
    reference_date: date,
) -> list[TimePreference]:
    """Validate untrusted AI proposals and freeze relative dates immediately."""
    if not isinstance(value, list):
        return []
    accepted: list[TimePreference] = []
    for raw in value:
        if not isinstance(raw, dict) or not set(raw) <= _PROPOSAL_KEYS:
            continue
        source = raw.get("source_text")
        confidence = raw.get("confidence")
        if (
            not isinstance(source, str)
            or not source.strip()
            or not any(source.strip() in text for text in user_sources)
            or type(confidence) not in {int, float}
            or not math.isfinite(float(confidence))
            or float(confidence) < ACCEPT_THRESHOLD
            or not 0.0 <= float(confidence) <= 1.0
        ):
            continue
        try:
            proposal = TimePreference.model_validate(raw)
            resolved_dates = _resolve_dates(proposal, reference_date)
            accepted.append(proposal.model_copy(update={
                "source_text": source.strip(),
                "confidence": float(confidence),
                "reference_date": reference_date.isoformat(),
                "resolved_dates": [item.isoformat() for item in resolved_dates],
            }))
        except (TypeError, ValueError, ValidationError):
            continue
    return accepted


def apply_time_preferences(availability: Availability, preferences: Iterable[TimePreference]) -> None:
    """Merge current validated facts while allowing a later fact to supersede the same target."""
    current = {_preference_identity(item): item for item in availability.time_preferences}
    for preference in preferences:
        current[_preference_identity(preference)] = preference
    availability.time_preferences = list(current.values())


def filter_hard_exclusions(
    rows: Iterable[dict[str, Any]],
    availability: Availability,
) -> list[dict[str, Any]]:
    exclusions = [item for item in availability.time_preferences if item.relation == "exclude"]
    return [row for row in rows if not any(_matches(row, item) for item in exclusions)]


def has_new_date_dimension(availability: Availability) -> bool:
    return any(item.kind in DATE_KINDS for item in availability.time_preferences)


def has_new_session_dimension(availability: Availability) -> bool:
    return any(item.kind == "session" for item in availability.time_preferences)


def select_effective_feasible_rows(
    rows: Iterable[dict[str, Any]],
    availability: Availability,
    doctor_preference: str = "",
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Apply new/legacy availability per dimension without relaxing new exclusions."""
    rows = list(rows)
    if not availability.time_preferences:
        return select_feasible_rows(rows, availability, doctor_preference, now=now)

    candidates = filter_hard_exclusions(open_rows(rows, now=now), availability)
    doctor = str(doctor_preference or "").strip()
    has_doctor = bool(doctor and doctor != "不限")
    doctor_rows = [
        row for row in candidates if doctor in str(row.get("doctor") or "")
    ] if has_doctor else candidates
    preferred_source = doctor_rows if doctor_rows else candidates

    legacy_dimensions = Availability(
        preferred_dates=([] if has_new_date_dimension(availability) else availability.preferred_dates),
        preferred_days=([] if has_new_date_dimension(availability) else availability.preferred_days),
        preferred_sessions=([] if has_new_session_dimension(availability) else availability.preferred_sessions),
        can_take_leave=availability.can_take_leave,
    )
    has_legacy_constraint = bool(
        legacy_dimensions.preferred_dates
        or legacy_dimensions.preferred_days
        or legacy_dimensions.preferred_sessions
    )
    if not has_legacy_constraint:
        return preferred_source, False

    matched = [
        row for row in preferred_source
        if row_matches_availability(row, legacy_dimensions)
    ]
    if matched:
        return matched, False
    if availability.can_take_leave:
        return preferred_source, True
    return [], False


def time_preference_vectors(
    rows: Iterable[dict[str, Any]],
    availability: Availability,
) -> dict[str, tuple[int, int]]:
    rows = list(rows)
    new_date_dimension = has_new_date_dimension(availability)
    new_session_dimension = has_new_session_dimension(availability)
    date_preferences = [
        item for item in availability.time_preferences
        if item.relation != "exclude" and item.kind in DATE_KINDS
    ] if new_date_dimension else []
    session_preferences = [
        item for item in availability.time_preferences
        if item.relation != "exclude" and item.kind == "session"
    ] if new_session_dimension else []
    max_date = max((item.priority or 0 for item in date_preferences), default=-1)
    max_session = max((item.priority or 0 for item in session_preferences), default=-1)
    vectors: dict[str, tuple[int, int]] = {}
    for index, row in enumerate(rows):
        date_rank = (
            _dimension_rank(row, date_preferences, max_date)
            if new_date_dimension
            else _legacy_date_rank(row, availability)
        )
        session_rank = (
            _dimension_rank(row, session_preferences, max_session)
            if new_session_dimension
            else _legacy_session_rank(row, availability)
        )
        vectors[_row_identity(row, index)] = (date_rank, session_rank)
    return vectors


def pareto_time_tiers(
    rows: Iterable[dict[str, Any]],
    availability: Availability,
) -> dict[str, int]:
    """Assign stable non-dominated layers without inventing date/session weights."""
    rows = list(rows)
    vectors = time_preference_vectors(rows, availability)
    remaining = list(vectors)
    tiers: dict[str, int] = {}
    tier = 0
    while remaining:
        front = [
            key for key in remaining
            if not any(_dominates(vectors[other], vectors[key]) for other in remaining if other != key)
        ]
        for key in front:
            tiers[key] = tier
        remaining = [key for key in remaining if key not in front]
        tier += 1
    return tiers


def row_time_identity(row: dict[str, Any], fallback_index: int = 0) -> str:
    return _row_identity(row, fallback_index)


def search_horizon_days(availability: Availability, *, reference_date: date, default: int = 21) -> int:
    resolved = [
        parsed
        for item in availability.time_preferences
        for value in item.resolved_dates
        if (parsed := parse_date(value)) is not None and parsed >= reference_date
    ]
    if not resolved:
        return default
    return max(default, min(90, (max(resolved) - reference_date).days + 1))


def describe_time_match(row: dict[str, Any], availability: Availability) -> str:
    matched = [
        item for item in availability.time_preferences
        if item.relation != "exclude" and _matches(row, item)
    ]
    if not matched:
        return "未命中明確時間順位，依真實可掛時間排序"
    best = min(item.priority or 99 for item in matched)
    return f"符合使用者時間偏好第 {best} 順位"


def _resolve_dates(preference: TimePreference, reference_date: date) -> list[date]:
    if preference.kind == "date":
        return [date.fromisoformat(preference.date_value or "")]
    if preference.kind == "date_range":
        start = date.fromisoformat(preference.start_date or "")
        end = date.fromisoformat(preference.end_date or "")
        return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    if preference.kind == "relative_day":
        return [reference_date + timedelta(days=preference.day_offset or 0)]
    if preference.kind in {"relative_week", "relative_weekday"}:
        monday = reference_date - timedelta(days=reference_date.weekday())
        week_start = monday + timedelta(weeks=preference.week_offset or 0)
        if preference.kind == "relative_weekday":
            return [week_start + timedelta(days=WEEKDAY_INDEX[preference.weekday or "monday"])]
        return [week_start + timedelta(days=offset) for offset in range(7)]
    return []


def _dimension_rank(
    row: dict[str, Any],
    preferences: list[TimePreference],
    max_priority: int,
) -> int:
    if not preferences:
        return 0
    matches = [item.priority for item in preferences if _matches(row, item) and item.priority]
    return min(matches) if matches else max_priority + 1


def _legacy_date_rank(row: dict[str, Any], availability: Availability) -> int:
    preferred_dates = {str(value).strip() for value in availability.preferred_dates if str(value).strip()}
    preferred_days = {str(value).strip() for value in availability.preferred_days if str(value).strip()}
    if not preferred_dates and not preferred_days:
        return 0
    row_date = parse_date(row.get("date"))
    if preferred_dates:
        return 0 if row_date and row_date.isoformat() in preferred_dates else 1
    return 0 if weekday_label(row.get("date")) in preferred_days else 1


def _legacy_session_rank(row: dict[str, Any], availability: Availability) -> int:
    preferred = {
        normalize_session(value) for value in availability.preferred_sessions if normalize_session(value)
    }
    if not preferred:
        return 0
    return 0 if normalize_session(row.get("session")) in preferred else 1


def preference_matches_row(row: dict[str, Any], preference: TimePreference) -> bool:
    return _matches(row, preference)


def _matches(row: dict[str, Any], preference: TimePreference) -> bool:
    row_date = parse_date(row.get("date"))
    if preference.kind == "session":
        return normalize_session(row.get("session")) == SESSION_VALUE.get(preference.session or "")
    if row_date is None:
        return False
    if preference.resolved_dates:
        return row_date.isoformat() in set(preference.resolved_dates)
    if preference.kind == "weekday":
        return row_date.weekday() == WEEKDAY_INDEX[preference.weekday or "monday"]
    if preference.kind == "weekday_group":
        return row_date.weekday() in {WEEKDAY_INDEX[item] for item in preference.weekdays}
    return False


def _preference_identity(preference: TimePreference) -> tuple[Any, ...]:
    return (
        preference.kind,
        preference.date_value,
        preference.start_date,
        preference.end_date,
        preference.day_offset,
        preference.week_offset,
        preference.weekday,
        tuple(preference.weekdays),
        preference.session,
    )


def _row_identity(row: dict[str, Any], fallback_index: int) -> str:
    schedule_id = str(row.get("schedule_id") or "").strip()
    if schedule_id:
        return schedule_id
    return "|".join((
        str(row.get("doctor_id") or ""),
        str(row.get("date") or ""),
        str(row.get("session") or ""),
        str(fallback_index),
    ))


def _dominates(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))
