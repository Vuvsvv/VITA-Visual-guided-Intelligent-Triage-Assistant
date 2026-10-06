from __future__ import annotations

import asyncio
import logging

from app.db import fetch_active_departments, fetch_reference_doctors, fetch_return_visit_slots
from app.schemas import (
    DepartmentResult,
    FollowupRecommendRequest,
    FollowupRecommendResponse,
    RecommendationItem,
)
from app.services.schedule_filter import normalize_session, row_matches_availability, select_feasible_rows, session_range


logger = logging.getLogger(__name__)


class ReturnVisitResolutionError(ValueError):
    """Return-visit identity is not canonical in the live Department/Doctor data."""


async def recommend_followup(req: FollowupRecommendRequest) -> FollowupRecommendResponse:
    departments = await asyncio.to_thread(fetch_active_departments)
    matches = [
        item for item in departments
        if req.dept_id is not None and str(item.get("dept_id")) == str(req.dept_id)
    ] if req.dept_id is not None else [
        item for item in departments
        if str(item.get("child_dept") or "").strip() == str(req.childDept or "").strip()
        and (not req.parentDept or str(item.get("parent_dept") or "").strip() == req.parentDept.strip())
    ]
    if len(matches) != 1:
        raise ReturnVisitResolutionError("回診科別無法對應唯一的正式科別。")
    canonical = matches[0]
    child = str(canonical.get("child_dept") or "").strip()
    parent = str(canonical.get("parent_dept") or "").strip()
    if req.childDept and req.childDept.strip() != child:
        raise ReturnVisitResolutionError("回診科別名稱與正式科別資料不一致。")
    if req.parentDept and req.parentDept.strip() != parent:
        raise ReturnVisitResolutionError("回診科別分類與正式科別資料不一致。")

    doctors = await asyncio.to_thread(
        fetch_reference_doctors,
        child,
        department_id=int(canonical["dept_id"]),
    )
    doctor_matches = [
        item for item in doctors
        if (req.original_doctor_id is None or str(item.get("doctor_id")) == str(req.original_doctor_id))
        and str(item.get("name") or "").strip() == req.original_doctor.strip()
    ]
    if len(doctor_matches) != 1:
        raise ReturnVisitResolutionError("原醫師身分或其與科別的關聯無法由正式資料確認。")
    canonical_doctor = doctor_matches[0]
    try:
        doctor_id = int(str(canonical_doctor["doctor_id"]))
    except (TypeError, ValueError) as exc:
        raise ReturnVisitResolutionError("正式醫師識別碼格式無效。") from exc
    if doctor_id <= 0:
        raise ReturnVisitResolutionError("正式醫師識別碼格式無效。")
    original_doctor = str(canonical_doctor["name"]).strip()

    rows = await asyncio.to_thread(
        fetch_return_visit_slots,
        department_name=child,
        doctor_name=original_doctor,
        preferred_dates=req.availability.preferred_dates,
        preferred_sessions=req.availability.preferred_sessions,
        department_id=int(canonical["dept_id"]),
        doctor_id=doctor_id,
    )

    rows = [
        row
        for row in rows
        if _matches_preferred_dates(row, req.availability.preferred_dates)
    ]
    original_rows = [
        row
        for row in rows
        if str(row.get("doctor") or "").strip() == original_doctor
        and str(row.get("child_dept") or row.get("childDept") or "").strip() == child
        and (
            str(row.get("doctor_id") or "").strip() == str(doctor_id)
        )
        and (
            str(row.get("dept_id") or "").strip() == str(canonical["dept_id"])
        )
    ]
    strict_availability = req.availability.model_copy(update={"can_take_leave": False})
    candidates, relaxed = select_feasible_rows(
        original_rows,
        strict_availability,
        original_doctor,
    )

    department = DepartmentResult(
        dept_id=int(canonical["dept_id"]),
        parentDept=parent,
        childDept=child,
        confidence=1.0,
        reason=["回診科別與原醫師已由正式主資料驗證"],
    )
    recommendations = _build_followup_items(
        req.case_id or "followup",
        candidates,
        department,
        req,
        relaxed,
    )

    return FollowupRecommendResponse(
        case_id=req.case_id or "followup",
        department=department,
        recommendations=recommendations,
        fallback_departments=[],
        total_count=len(recommendations),
    )


def _build_followup_items(
    case_id: str,
    rows: list[dict],
    department: DepartmentResult,
    req: FollowupRecommendRequest,
    relaxed: bool,
) -> list[RecommendationItem]:
    sorted_rows = sorted(
        _dedupe(rows),
        key=lambda row: (
            str(row.get("date")),
            _session_rank(row.get("session", "")),
            str(row.get("doctor", "")),
        ),
    )
    items = []
    for row in sorted_rows[:5]:
        time_match = row_matches_availability(row, req.availability)
        reasons = []
        reasons.append("符合指定原醫師回診")
        reasons.append("符合偏好時段" if time_match else "偏好時段無班，提供近期同科可掛號")
        if relaxed:
            reasons.append("已放寬時間限制以尋找可掛號複診時段")
        items.append(
            RecommendationItem(
                recommendation_id=f"fu_{case_id}_{len(items) + 1:03d}",
                parentDept=department.parentDept,
                childDept=department.childDept,
                doctor=str(row.get("doctor") or ""),
                doctor_id=str(row.get("doctor_id") or row.get("doctor") or ""),
                schedule_id=str(row.get("schedule_id") or ""),
                dept_id=department.dept_id,
                date=_date_to_text(row.get("date")),
                session=normalize_session(row.get("session")) or str(row.get("session") or ""),
                session_time=session_range(row.get("session")),
                slot=str(row.get("slot") or row.get("room") or ""),
                room=str(row.get("room") or row.get("slot") or ""),
                visit_type=str(row.get("visit_type") or ""),
                specialty_tags=str(row.get("specialty_tags") or ""),
                specialty_score=1.0,
                time_score=1.0 if time_match else 0.5,
                match_reason=reasons[0],
                score=95.0 - len(items) * 3,
                reasons=reasons,
                rank=len(items) + 1,
                is_best_match=len(items) == 0,
            )
        )
    return items


def _dedupe(rows: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for row in rows:
        key = (row.get("child_dept"), row.get("doctor"), row.get("date"), row.get("session"))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _matches_preferred_dates(row: dict, preferred_dates: list[str]) -> bool:
    requested = {str(value or "").strip() for value in preferred_dates if str(value or "").strip()}
    if not requested:
        return True
    value = row.get("date")
    actual = value.isoformat() if hasattr(value, "isoformat") else str(value or "").strip()[:10]
    return actual in requested


def _session_rank(session: str) -> int:
    normalized = normalize_session(session)
    return {"上午": 0, "下午": 1, "晚上": 2}.get(normalized, 3)


def _date_to_text(value) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value or "")
