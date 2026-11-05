from __future__ import annotations

from datetime import date, datetime
import logging

from app.db import fetch_active_departments, fetch_available_slots
from app.schemas import (
    DepartmentResult,
    RecommendationColumns,
    RecommendationItem,
    RecommendationResult,
    TriageCase,
    VisitType,
)
from app.services.department_preference_service import resolve_requested_department
from app.services.schedule_filter import (
    TAIPEI_ZONE,
    normalize_session,
    parse_date,
    row_is_available,
    row_matches_availability,
    session_is_open,
    session_range,
    weekday_label as schedule_weekday_label,
)
from app.services.patient_reason_text import department_reason, time_reason
from app.services.specialty_scoring import SpecialtyScore, score_doctor_specialties
from app.services.time_preference_service import (
    describe_time_match,
    pareto_time_tiers,
    row_time_identity,
    search_horizon_days,
    select_effective_feasible_rows,
)

logger = logging.getLogger(__name__)

TIME_FIRST_LIMIT = 5
SPECIALTY_DOCTOR_LIMIT = 5
SPECIALTY_SLOTS_PER_DOCTOR = 3

class DepartmentResolutionError(ValueError):
    """The selected department cannot be tied to one canonical DB record."""


def resolve_department_result(
    selected: DepartmentResult,
    departments: list[dict] | None = None,
) -> DepartmentResult | None:
    """Resolve a selection to exactly one DB candidate; never invent or remap an ID."""
    candidates = _canonical_departments(departments if departments is not None else fetch_active_departments())
    if selected.dept_id is not None:
        matches = [item for item in candidates if item["dept_id"] == selected.dept_id]
        if len(matches) != 1:
            return None
        item = matches[0]
        if selected.childDept and selected.childDept != item["child_dept"]:
            return None
        if selected.parentDept and selected.parentDept != item["parent_dept"]:
            return None
    else:
        matches = [
            item
            for item in candidates
            if item["child_dept"] == selected.childDept
            and (not selected.parentDept or item["parent_dept"] == selected.parentDept)
        ]
        if len(matches) != 1:
            return None
        item = matches[0]
    return DepartmentResult(
        dept_id=item["dept_id"],
        parentDept=item["parent_dept"],
        childDept=item["child_dept"],
        confidence=selected.confidence,
        reason=list(selected.reason),
    )


async def recommend_appointments(
    case: TriageCase,
    visit_type: VisitType | str | None = None,
) -> RecommendationResult:
    active_departments = _canonical_departments(fetch_active_departments())
    requested_department = None
    if case.patient_input.requested_department_id is not None or case.patient_input.requested_department_name:
        requested_department = resolve_requested_department(
            case,
            active_departments,
        )
        if requested_department is None:
            raise DepartmentResolutionError("使用者指定科別無法對應唯一的正式科別。")
    department = requested_department or case.department_result
    if department is None:
        raise DepartmentResolutionError("尚無已驗證的正式科別結果；不執行舊版科別猜測。")
    department = resolve_department_result(department, active_departments)
    if department is None or department.dept_id is None:
        raise DepartmentResolutionError("案件科別識別與目前正式科別主資料不一致。")
    case.department_result = department

    canonical_visit_type = normalize_visit_type(visit_type or case.visit_type)
    schedule_visit_type = schedule_visit_type_for(canonical_visit_type)
    current_taipei_datetime = datetime.now(TAIPEI_ZONE)
    logger.info(
        "[RECOMMEND_INPUT] case_id=%s visit_type=%s department_id=%s "
        "has_date_preferences=%s has_day_preferences=%s has_session_preferences=%s",
        case.case_id,
        canonical_visit_type.value,
        department.dept_id,
        bool(case.availability.preferred_dates),
        bool(case.availability.preferred_days),
        bool(case.availability.preferred_sessions),
    )
    primary_slots = fetch_available_slots(
        department.childDept,
        search_days=search_horizon_days(
            case.availability,
            reference_date=current_taipei_datetime.date(),
        ),
        max_slots=None,
        schedule_visit_type=schedule_visit_type,
        department_id=department.dept_id,
    )
    slots = _filter_slots_by_department(primary_slots, department.childDept, department.dept_id)
    slots = _filter_slots_by_visit_type(slots, canonical_visit_type)
    slots = _formal_schedule_rows(slots, department.dept_id)

    doctor_preference = case.preferences.doctor_preference or "不限"
    feasible_slots, relaxed_by_date = select_effective_feasible_rows(
        slots,
        case.availability,
        doctor_preference,
        now=current_taipei_datetime,
    )
    _log_schedule_filter_counts(
        primary_slots=primary_slots,
        visit_slots=slots,
        case=case,
        final_slots=feasible_slots,
        now=current_taipei_datetime,
    )
    ranking_slots = feasible_slots
    time_tiers = pareto_time_tiers(ranking_slots, case.availability)
    specialty_scores = await score_doctor_specialties(case, department, ranking_slots)

    specialty_first = _build_recommendations(
        case=case,
        case_id=case.case_id,
        slots=ranking_slots,
        prefix="rec_s",
        specialty_first=True,
        specialty_scores=specialty_scores,
        time_tiers=time_tiers,
        relaxed_by_date=relaxed_by_date,
    )
    time_first = _build_recommendations(
        case=case,
        case_id=case.case_id,
        slots=ranking_slots,
        prefix="rec_t",
        specialty_first=False,
        specialty_scores=specialty_scores,
        time_tiers=time_tiers,
        relaxed_by_date=relaxed_by_date,
    )

    total_count = len(specialty_first) + len(time_first)
    return RecommendationResult(
        case_id=case.case_id,
        recommendations=RecommendationColumns(
            specialty_first=specialty_first,
            time_first=time_first,
        ),
        fallback_departments=[],
        total_count=total_count,
    )


def _filter_slots_by_visit_type(slots: list[dict], visit_type: str) -> list[dict]:
    requested = schedule_visit_type_for(normalize_visit_type(visit_type))
    return [
        slot
        for slot in slots
        if requested in _schedule_visit_type_tokens(slot.get("visit_type"))
    ]


def _log_schedule_filter_counts(
    *,
    primary_slots: list[dict],
    visit_slots: list[dict],
    case: TriageCase,
    final_slots: list[dict],
    now: datetime,
) -> None:
    preferred_days = [day for day in case.availability.preferred_days if day]
    preferred_sessions = [
        normalized
        for session in case.availability.preferred_sessions
        if (normalized := normalize_session(session))
    ]
    day_filtered = [
        row
        for row in visit_slots
        if not preferred_days or schedule_weekday_label(row.get("date")) in preferred_days
    ]
    session_filtered = [
        row
        for row in day_filtered
        if not preferred_sessions or normalize_session(row.get("session")) in preferred_sessions
    ]
    cutoff_filtered = [row for row in session_filtered if session_is_open(row, now=now)]
    active_filtered = [row for row in cutoff_filtered if row_is_available(row)]
    logger.info(
        "[SCHEDULE_FILTER] before_count=%s after_visit_type=%s after_day_filter=%s "
        "after_session_filter=%s after_cutoff=%s after_doctor_active=%s final_count=%s",
        len(primary_slots),
        len(visit_slots),
        len(day_filtered),
        len(session_filtered),
        len(cutoff_filtered),
        len(active_filtered),
        len(final_slots),
    )


def _filter_slots_by_department(slots: list[dict], child_dept: str, dept_id: int | None = None) -> list[dict]:
    del child_dept
    if dept_id is None:
        return []
    return [
        slot
        for slot in slots
        if _optional_int(slot.get("dept_id")) == dept_id
    ]


def _formal_schedule_rows(slots: list[dict], dept_id: int) -> list[dict]:
    """Only SQL rows with complete canonical navigation identity may be recommended."""
    return [
        slot for slot in slots
        if _optional_int(slot.get("dept_id")) == dept_id
        and bool(str(slot.get("doctor_id") or "").strip())
        and bool(str(slot.get("schedule_id") or "").strip())
        and parse_date(slot.get("date")) is not None
        and bool(normalize_session(slot.get("session")))
    ]


def normalize_visit_type(value: VisitType | str | None) -> VisitType:
    if isinstance(value, VisitType):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"", "unknown"}:
        raise ValueError("visit_type is required and must be initial, followup, quick_search, or return_visit")
    try:
        return VisitType(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported visit_type: {normalized}") from exc


def schedule_visit_type_for(value: VisitType | str | None) -> str:
    canonical = normalize_visit_type(value)
    if canonical == VisitType.QUICK_SEARCH:
        raise ValueError("quick_search does not use the recommendation pipeline")
    return {
        VisitType.INITIAL: "初診",
        VisitType.FOLLOWUP: "複診",
        VisitType.RETURN_VISIT: "複診",
    }[canonical]


def _schedule_visit_type_tokens(value: object) -> set[str]:
    return {
        token.strip()
        for token in str(value or "").split("/")
        if token.strip()
    }


def _canonical_departments(departments: list[dict]) -> list[dict]:
    canonical: list[dict] = []
    for item in departments:
        try:
            dept_id = int(item.get("dept_id"))
        except (TypeError, ValueError):
            continue
        parent = str(item.get("parent_dept") or "").strip()
        child = str(item.get("child_dept") or "").strip()
        if not child:
            continue
        canonical.append({"dept_id": dept_id, "parent_dept": parent, "child_dept": child})
    return canonical


def _build_recommendations(
    case: TriageCase,
    case_id: str,
    slots: list[dict],
    prefix: str,
    specialty_first: bool,
    specialty_scores: dict[str, SpecialtyScore] | None = None,
    time_tiers: dict[str, int] | None = None,
    relaxed_by_date: bool = False,
) -> list[RecommendationItem]:
    specialty_scores = specialty_scores or {}
    time_tiers = time_tiers or pareto_time_tiers(slots, case.availability)
    sorted_slots = sorted(
        slots,
        key=lambda slot: _ranking_key(slot, specialty_scores, time_tiers, specialty_first),
    )
    if specialty_first:
        sorted_slots = _doctor_grouped_slots(sorted_slots)
    items = []
    seen = set()
    for slot in sorted_slots:
        key = str(slot.get("schedule_id") or "")
        if key in seen:
            continue
        seen.add(key)

        specialty = _score_for_slot(slot, specialty_scores)
        time_tier = time_tiers.get(row_time_identity(slot), 0)
        time_score = 1.0 if time_tier == 0 else 0.0
        score = weighted_score(specialty.score, time_score, specialty_first)
        if specialty_first:
            reasons = _score_explanation_reasons(
                case=case,
                slot=slot,
                specialty=specialty,
                time_tier=time_tier,
                time_score=time_score,
                total_score=score,
                specialty_first=True,
            )
        else:
            reasons = _score_explanation_reasons(
                case=case,
                slot=slot,
                specialty=specialty,
                time_tier=time_tier,
                time_score=time_score,
                total_score=score,
                specialty_first=False,
            )
        reasons.insert(0, f"推薦理由：{_compact_recommendation_reason(case, slot)}")
        if relaxed_by_date and not row_matches_availability(slot, case.availability):
            reasons.append("您方便的時段近期無號，改推薦時間最接近的門診")
        if slot.get("visit_type"):
            reasons.append(f"掛號別：{slot.get('visit_type')}")

        item = RecommendationItem(
            recommendation_id=f"{prefix}_{case_id}_{len(items) + 1:03d}",
            parentDept=slot.get("parent_dept", ""),
            childDept=slot.get("child_dept", ""),
            doctor=slot.get("doctor", ""),
            date=_date_to_text(slot["date"]),
            session=normalize_session(slot.get("session", "")) or str(slot.get("session", "")),
            slot=str(slot.get("slot") or slot.get("room") or ""),
            doctor_id=str(slot.get("doctor_id") or ""),
            schedule_id=str(slot.get("schedule_id") or ""),
            session_time=session_range(slot.get("session", "")),
            room=str(slot.get("room") or slot.get("slot") or ""),
            visit_type=str(slot.get("visit_type") or ""),
            specialty_tags=str(slot.get("specialty_tags") or ""),
            specialty_score=round(specialty.score, 2),
            time_score=round(time_score, 2),
            match_reason=None,
            score=score,
            reasons=reasons,
            department_reason=department_reason(case, slot, specialty),
            time_reason=time_reason(case, slot, relaxed_by_date),
            rank=len(items) + 1,
            is_best_match=specialty_first and len(items) == 0,
            dept_id=_optional_int(slot.get("dept_id")) or (
                case.department_result.dept_id if case.department_result else None
            ),
        )
        items.append(item)
        if not specialty_first and len(items) >= TIME_FIRST_LIMIT:
            break
    return items


def _doctor_grouped_slots(sorted_slots: list[dict]) -> list[dict]:
    """Specialty column: doctors first, then slots.

    Takes the first SPECIALTY_DOCTOR_LIMIT doctors in ranking order and up to
    SPECIALTY_SLOTS_PER_DOCTOR of each one's slots, best time preference first; the
    doctor's slots stay together so each doctor's first slot is their best one.
    """
    groups: dict[str, list[dict]] = {}
    seen = set()
    for slot in sorted_slots:
        schedule_key = str(slot.get("schedule_id") or "")
        if schedule_key in seen:
            continue
        seen.add(schedule_key)
        doctor_key = str(slot.get("doctor_id") or "")
        if doctor_key not in groups:
            if len(groups) >= SPECIALTY_DOCTOR_LIMIT:
                continue
            groups[doctor_key] = []
        if len(groups[doctor_key]) < SPECIALTY_SLOTS_PER_DOCTOR:
            groups[doctor_key].append(slot)
    return [slot for group in groups.values() for slot in group]


def _ranking_key(
    slot: dict,
    specialty_scores: dict[str, SpecialtyScore],
    time_tiers: dict[str, int],
    specialty_first: bool,
) -> tuple:
    specialty = _score_for_slot(slot, specialty_scores)
    time_tier = time_tiers.get(row_time_identity(slot), 0)
    date_key = _date_sort_key(slot.get("date"))
    session_key = _session_rank(slot.get("session", ""))
    doctor_key = str(slot.get("doctor_id") or "")
    schedule_key = str(slot.get("schedule_id") or "")
    if specialty_first:
        return (
            -specialty.score,
            time_tier,
            date_key,
            session_key,
            doctor_key,
            schedule_key,
        )
    return (
        time_tier,
        date_key,
        session_key,
        -specialty.score,
        doctor_key,
        schedule_key,
    )


def weighted_score(
    specialty_score: float,
    time_score: float,
    specialty_first: bool,
) -> float:
    """Compatibility display value only; recommendation order never uses this value."""
    display = specialty_score if specialty_first else time_score
    return round(display * 100.0, 2)


def _score_for_slot(slot: dict, specialty_scores: dict[str, SpecialtyScore]) -> SpecialtyScore:
    key = str(slot.get("doctor_id") or "")
    if key in specialty_scores:
        return specialty_scores[key]
    return SpecialtyScore(
        doctor_id=key,
        doctor=str(slot.get("doctor") or ""),
        childDept=str(slot.get("child_dept") or ""),
        score=0.5,
        reason="AI 專長語意評分未完整可用，專長分數使用中性值 0.50",
        source="neutral",
    )


def _score_explanation_reasons(
    case: TriageCase,
    slot: dict,
    specialty: SpecialtyScore,
    time_tier: int,
    time_score: float,
    total_score: float,
    specialty_first: bool,
) -> list[str]:
    return [
        _department_basis_reason(case, slot),
        _time_basis_reason(case, slot, time_tier),
        _status_basis_reason(slot),
        _specialty_basis_reason(slot, specialty),
        _sorting_basis_reason(specialty, time_tier, total_score, specialty_first),
    ]


def _compact_recommendation_reason(case: TriageCase, slot: dict) -> str:
    if case.availability.time_preferences:
        return describe_time_match(slot, case.availability)
    if (case.availability.preferred_days or case.availability.preferred_sessions) and row_matches_availability(
        slot, case.availability
    ):
        return "符合您的方便看診時段"
    return "符合目前推薦科別"


def _department_basis_reason(case: TriageCase, slot: dict) -> str:
    child_dept = str(slot.get("child_dept") or "").strip()
    if not child_dept and case.department_result:
        child_dept = case.department_result.childDept
    symptom = _short_text(case.patient_input.symptom or case.patient_input.body_part or "問診內容")
    evidence = [f"使用者症狀「{symptom}」"]
    if case.patient_input.body_part:
        evidence.append(f"不適部位「{_short_text(case.patient_input.body_part)}」")
    if case.patient_input.duration:
        evidence.append(f"持續時間「{_short_text(case.patient_input.duration)}」")
    department_reason = ""
    if case.department_result and case.department_result.reason:
        department_reason = f"；科別判斷理由：{case.department_result.reason[0]}"
    return f"科別依據：{'；'.join(evidence)}，目前建議科別為 {child_dept or '目前科別'}{department_reason}"


def _time_basis_reason(case: TriageCase, slot: dict, time_tier: int) -> str:
    preferred = _availability_text(case)
    slot_time = _slot_time_text(slot)
    if case.availability.time_preferences:
        return f"時間依據：{describe_time_match(slot, case.availability)}；Schedule 為 {slot_time}；Pareto preference tier={time_tier}"
    if not case.availability.preferred_dates and not case.availability.preferred_days and not case.availability.preferred_sessions:
        return f"時間依據：使用者未指定日期/時段偏好，Schedule 為 {slot_time}"
    if row_matches_availability(slot, case.availability):
        return f"時間依據：使用者偏好 {preferred}；Schedule 為 {slot_time}，符合偏好"
    return f"時間依據：使用者偏好 {preferred}；Schedule 為 {slot_time}，未完全符合偏好"


def _status_basis_reason(slot: dict) -> str:
    raw_status = str(slot.get("status") or "").strip()
    status_text = raw_status or "空白/NULL"
    availability = "可掛號" if row_is_available(slot) else "不可掛號"
    return f"狀態依據：Schedule.status={status_text}，判斷為{availability}"


def _specialty_basis_reason(slot: dict, specialty: SpecialtyScore) -> str:
    tags = str(slot.get("specialty_tags") or slot.get("specialty") or "").strip()
    if not tags:
        return "專長依據：SQL specialty_tags 無資料，專長分數使用中性值 0.50"
    return (
        f"專長依據：SQL specialty_tags={_short_text(tags)}；"
        f"specialty semantic relevance={specialty.score:.2f}；{specialty.reason}"
    )


def _sorting_basis_reason(
    specialty: SpecialtyScore,
    time_tier: int,
    total_score: float,
    specialty_first: bool,
) -> str:
    if specialty_first:
        order = "specialty_score → time preference tier → datetime → doctor_id"
    else:
        order = "time preference tier → datetime → specialty_score → doctor_id"
    return (
        f"排序依據：{order}；specialty_score={specialty.score:.2f}；"
        f"time preference tier={time_tier}；相容顯示 score={total_score:.2f}，不作排序權重"
    )


def _availability_text(case: TriageCase) -> str:
    dates = [value for value in case.availability.preferred_dates if value]
    days = [day for day in case.availability.preferred_days if day]
    sessions = [normalize_session(session) for session in case.availability.preferred_sessions if session]
    parts = []
    if dates:
        parts.append("/".join(dates))
    elif days:
        parts.append("/".join(days))
    if sessions:
        parts.append("/".join(sessions))
    return " ".join(parts) if parts else "未指定"


def _slot_time_text(slot: dict) -> str:
    date_text = _date_to_text(slot.get("date"))
    weekday = schedule_weekday_label(slot.get("date"))
    session = normalize_session(slot.get("session", "")) or str(slot.get("session") or "")
    parts = [date_text]
    if weekday:
        parts.append(weekday)
    if session:
        parts.append(session)
    return " ".join(part for part in parts if part)


def _short_text(value: str, max_len: int = 48) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."


def _session_rank(session: str) -> int:
    if "上午" in session or "早" in session:
        return 0
    if "下午" in session or "午" in session:
        return 1
    if "夜" in session or "晚" in session:
        return 2
    return 3


def _date_to_text(value) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _date_sort_key(value) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
