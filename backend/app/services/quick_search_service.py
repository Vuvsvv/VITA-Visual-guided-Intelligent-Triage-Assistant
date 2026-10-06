from __future__ import annotations

import asyncio
from datetime import date
from typing import Any

from app.db import QUICK_SEARCH_VISIT_TYPE, fetch_quick_search_slot_by_id, fetch_quick_search_slots
from app.schemas import QuickSearchPeriod, RecommendationItem
from app.services.schedule_filter import normalize_session, row_is_available, session_is_open, session_range


async def search_quick_schedules(
    *,
    case_id: str,
    department: dict[str, Any],
    target_date: date,
    period: QuickSearchPeriod,
) -> list[RecommendationItem]:
    """Return deterministic, DB-backed schedule matches without AI or recommendation logic."""
    dept_id = int(department["dept_id"])
    rows = await asyncio.to_thread(
        fetch_quick_search_slots,
        department_id=dept_id,
        target_date=target_date,
        period=period.value,
    )

    expected_session = normalize_session(period.value)
    expected_date = target_date.isoformat()
    candidates = [
        row
        for row in rows
        if str(row.get("dept_id") or "").strip() == str(dept_id)
        and _date_to_text(row.get("date")) == expected_date
        and normalize_session(row.get("session")) == expected_session
        and bool(row.get("supports_followup"))
        and row_is_available(row)
        and session_is_open(row)
    ]
    candidates.sort(
        key=lambda row: (
            str(row.get("doctor") or ""),
            0 if str(row.get("visit_type") or "").strip() == QUICK_SEARCH_VISIT_TYPE else 1,
            _schedule_id_sort_key(row.get("schedule_id")),
        )
    )
    logical_slots: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in candidates:
        key = (
            str(row.get("doctor_id") or "").strip(),
            str(row.get("dept_id") or "").strip(),
            _date_to_text(row.get("date")),
            normalize_session(row.get("session")),
            str(row.get("room") or row.get("slot") or "").strip(),
        )
        logical_slots.setdefault(key, row)
    candidates = list(logical_slots.values())

    results: list[RecommendationItem] = []
    for index, row in enumerate(candidates, start=1):
        schedule_id = str(row.get("schedule_id") or "").strip()
        session = normalize_session(row.get("session")) or str(row.get("session") or "").strip()
        results.append(
            RecommendationItem(
                recommendation_id=f"qs_{case_id}_{schedule_id or index}",
                parentDept=str(department.get("parent_dept") or "").strip(),
                childDept=str(department.get("child_dept") or "").strip(),
                doctor=str(row.get("doctor") or "").strip(),
                doctor_id=str(row.get("doctor_id") or "").strip() or None,
                schedule_id=schedule_id or None,
                dept_id=dept_id,
                date=expected_date,
                session=session,
                session_time=session_range(session) or None,
                slot=str(row.get("slot") or row.get("room") or "").strip(),
                room=str(row.get("room") or row.get("slot") or "").strip() or None,
                visit_type=str(row.get("visit_type") or "").strip() or None,
                specialty_tags=str(row.get("specialty_tags") or "").strip() or None,
                score=0.0,
                reasons=["符合指定科別、日期與時段的正式可掛號班表"],
                rank=index,
                is_best_match=False,
                match_reason="快速查詢條件完全符合",
            )
        )
    return results


def revalidate_schedule(
    recommendation: RecommendationItem,
    *,
    require_followup_evidence: bool = False,
) -> RecommendationItem:
    """Reload any selected schedule and return only DB-authoritative identity fields."""
    if not recommendation.schedule_id or not recommendation.doctor_id or recommendation.dept_id is None:
        raise ValueError("班表缺少 schedule_id、doctor_id 或 dept_id。")
    try:
        target_date = date.fromisoformat(recommendation.date)
    except ValueError as exc:
        raise ValueError("班表日期格式無效。") from exc

    row = fetch_quick_search_slot_by_id(recommendation.schedule_id)
    if row is None:
        raise ValueError("指定班表已不存在或目前不可掛號。")
    if require_followup_evidence and not bool(row.get("supports_followup")):
        raise ValueError("指定班表沒有可供複診使用的正式資料依據。")
    if str(row.get("schedule_id") or "").strip() != recommendation.schedule_id:
        raise ValueError("班表識別碼資料不一致。")
    if str(row.get("doctor_id") or "").strip() != recommendation.doctor_id:
        raise ValueError("班表醫師資料已變更。")
    if str(row.get("dept_id") or "").strip() != str(recommendation.dept_id):
        raise ValueError("班表科別資料已變更。")
    if _date_to_text(row.get("date")) != recommendation.date:
        raise ValueError("班表日期資料已變更。")
    if normalize_session(row.get("session")) != normalize_session(recommendation.session):
        raise ValueError("班表時段資料已變更。")
    if not row_is_available(row) or not session_is_open(row):
        raise ValueError("指定班表目前不可掛號。")

    session = normalize_session(row.get("session")) or str(row.get("session") or "").strip()
    room = str(row.get("room") or row.get("slot") or "").strip()
    return recommendation.model_copy(
        update={
            "parentDept": str(row.get("parent_dept") or "").strip(),
            "childDept": str(row.get("child_dept") or "").strip(),
            "doctor": str(row.get("doctor") or "").strip(),
            "doctor_id": str(row.get("doctor_id") or "").strip(),
            "schedule_id": str(row.get("schedule_id") or "").strip(),
            "dept_id": int(row["dept_id"]),
            "date": _date_to_text(row.get("date")),
            "session": session,
            "session_time": session_range(session) or None,
            "room": room or None,
            "slot": room,
            "visit_type": str(row.get("visit_type") or "").strip() or None,
            "specialty_tags": str(row.get("specialty_tags") or "").strip() or None,
        }
    )


def revalidate_quick_schedule(recommendation: RecommendationItem) -> RecommendationItem:
    return revalidate_schedule(recommendation, require_followup_evidence=True)


def _date_to_text(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value or "").strip()[:10]


def _schedule_id_sort_key(value: Any) -> tuple[int, int | str]:
    text = str(value or "").strip()
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)
