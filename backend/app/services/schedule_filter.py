from __future__ import annotations

from datetime import date, datetime, time
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from app.schemas import Availability

WEEKDAY_LABELS = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]

SESSION_TIME_RANGES: dict[str, tuple[time, time]] = {
    "上午": (time(8, 30), time(12, 0)),
    "下午": (time(13, 30), time(17, 0)),
    "晚上": (time(18, 0), time(21, 0)),
    "夜診": (time(18, 0), time(21, 0)),
}

SESSION_ALIASES = {
    "早上": "上午",
    "上午": "上午",
    "morning": "上午",
    "下午": "下午",
    "午診": "下午",
    "afternoon": "下午",
    "晚上": "晚上",
    "晚診": "晚上",
    "夜診": "晚上",
    "夜間": "晚上",
    "evening": "晚上",
}

TAIPEI_ZONE = ZoneInfo("Asia/Taipei")
SESSION_RECOMMENDATION_CUTOFFS: dict[str, time] = {
    "上午": time(10, 0),
    "下午": time(15, 0),
    "晚上": time(19, 0),
}


def normalize_session(session: Any) -> str:
    text = str(session or "").strip()
    if not text:
        return ""
    if text in SESSION_ALIASES:
        return SESSION_ALIASES[text]
    for key, normalized in SESSION_ALIASES.items():
        if key in text:
            return normalized
    return text


def session_range(session: Any) -> str:
    normalized = normalize_session(session)
    times = SESSION_TIME_RANGES.get(normalized)
    if not times:
        return ""
    start, end = times
    return f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}"


def weekday_label(value: Any) -> str:
    parsed = parse_date(value)
    if parsed is None:
        return ""
    return WEEKDAY_LABELS[parsed.weekday()]


def parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def availability_slots(availability: Availability) -> list[tuple[str, str]]:
    days = [day for day in availability.preferred_days if day]
    sessions = [normalize_session(session) for session in availability.preferred_sessions if session]
    sessions = [session for session in sessions if session]
    if not days and not sessions:
        return []
    if not days:
        return [("", session) for session in sessions]
    if not sessions:
        return [(day, "") for day in days]
    return [(day, session) for day in days for session in sessions]


def availability_from_project_smart_slots(slots: Iterable[Any], can_take_leave: bool = False) -> Availability:
    days: list[str] = []
    sessions: list[str] = []
    for slot in slots:
        if isinstance(slot, dict):
            day = str(slot.get("day") or "").strip()
            session = normalize_session(slot.get("session"))
        else:
            day = str(getattr(slot, "day", "") or "").strip()
            session = normalize_session(getattr(slot, "session", ""))
        if day and day not in days:
            days.append(day)
        if session and session not in sessions:
            sessions.append(session)
    return Availability(
        preferred_days=days,
        preferred_sessions=sessions,
        can_take_leave=can_take_leave,
    )


def row_matches_availability(row: dict[str, Any], availability: Availability) -> bool:
    preferred_dates = {
        str(value or "").strip()
        for value in availability.preferred_dates
        if str(value or "").strip()
    }
    row_date = parse_date(row.get("date"))
    if preferred_dates and (row_date is None or row_date.isoformat() not in preferred_dates):
        return False

    slots = availability_slots(availability)
    if not slots:
        return True

    row_day = weekday_label(row.get("date"))
    row_session = normalize_session(row.get("session"))
    for day, session in slots:
        day_ok = not day or day == row_day
        session_ok = not session or session == row_session
        if day_ok and session_ok:
            return True
    return False


def session_is_open(row: dict[str, Any], now: datetime | None = None) -> bool:
    local_now = _taipei_now(now)
    row_date = parse_date(row.get("date"))
    if row_date is None:
        return True
    if row_date != local_now.date():
        return True

    normalized = normalize_session(row.get("session"))
    cutoff = SESSION_RECOMMENDATION_CUTOFFS.get(normalized)
    if cutoff is None:
        return True
    return local_now.time().replace(tzinfo=None) < cutoff


def _taipei_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(TAIPEI_ZONE)
    if now.tzinfo is None:
        return now.replace(tzinfo=TAIPEI_ZONE)
    return now.astimezone(TAIPEI_ZONE)


def row_is_available(row: dict[str, Any]) -> bool:
    status = str(row.get("status") or "").strip().lower()
    if status and status not in {"open", "available", "可掛號"}:
        return False
    if bool(row.get("is_placeholder", False)):
        return False
    if bool(row.get("is_leave", False)) and not row.get("substitute_doctor"):
        return False
    return True


def open_rows(rows: Iterable[dict[str, Any]], now: datetime | None = None) -> list[dict[str, Any]]:
    return [row for row in rows if row_is_available(row) and session_is_open(row, now=now)]


def filter_rows_by_availability(
    rows: Iterable[dict[str, Any]],
    availability: Availability,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    candidates = open_rows(rows, now=now)
    if not availability_slots(availability):
        return candidates

    matched = [row for row in candidates if row_matches_availability(row, availability)]
    if matched or not availability.can_take_leave:
        return matched
    return candidates


def select_feasible_rows(
    rows: Iterable[dict[str, Any]],
    availability: Availability,
    doctor_preference: str = "",
    now: datetime | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    candidates = open_rows(rows, now=now)
    doctor = str(doctor_preference or "").strip()
    has_doctor = bool(doctor and doctor != "不限")
    doctor_rows = [row for row in candidates if doctor in str(row.get("doctor", ""))] if has_doctor else candidates

    preferred_source = doctor_rows if doctor_rows else candidates
    preferred = [row for row in preferred_source if row_matches_availability(row, availability)]
    if preferred:
        return preferred, False

    if has_doctor and doctor_rows and availability.can_take_leave:
        return doctor_rows, True

    if availability.can_take_leave:
        return preferred_source, True

    return [], False
