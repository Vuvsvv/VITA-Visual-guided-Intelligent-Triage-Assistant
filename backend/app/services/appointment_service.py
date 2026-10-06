from __future__ import annotations

from datetime import date, datetime
import logging
import re

from app.db import fetch_active_departments, fetch_available_slots
from app.schemas import (
    DepartmentResult,
    RecommendationColumns,
    RecommendationItem,
    RecommendationResult,
    TriageCase,
    VisitType,
)
from app.services.project_smart_department_adapter import detect_department_with_project_smart_adapter
from app.services.department_preference_service import resolve_requested_department
from app.services.negation_utils import strip_negated_red_flags
from app.services.schedule_filter import (
    TAIPEI_ZONE,
    normalize_session,
    row_is_available,
    row_matches_availability,
    select_feasible_rows,
    session_is_open,
    session_range,
    weekday_label as schedule_weekday_label,
)
from app.services.specialty_scoring import SpecialtyScore, score_doctor_specialties

logger = logging.getLogger(__name__)

_COMPACT_SPECIALTY_CONCEPTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("頭痛", "頭暈", "眩暈"), ("頭痛", "神經", "腦", "眩暈")),
    (("鼻", "鼻塞", "流鼻水", "鼻炎", "鼻竇"), ("鼻", "鼻炎", "鼻塞", "鼻竇", "過敏")),
    (("耳", "耳痛", "耳鳴", "聽力"), ("耳", "耳鳴", "聽力", "中耳")),
    (("膝", "關節", "走路", "爬樓梯", "扭傷", "骨"), ("膝", "關節", "骨科", "運動傷害", "復健")),
    (("皮膚", "紅疹", "癢", "濕疹"), ("皮膚", "紅疹", "過敏", "濕疹", "蕁麻疹")),
    (("胸痛", "心悸"), ("心臟", "心律", "冠心", "心血管")),
    (("咳", "喘", "呼吸困難"), ("胸腔", "肺", "呼吸", "氣喘")),
    (("腹", "胃", "腸", "腹瀉", "嘔吐"), ("腸胃", "胃腸", "消化", "肝膽", "腹")),
)


class DepartmentResolutionError(ValueError):
    """The selected department cannot be tied to one canonical DB record."""


async def detect_department_result(case: TriageCase) -> DepartmentResult:
    departments = _canonical_departments(fetch_active_departments())
    logger.info(
        "[DEPARTMENT] detection_called=true candidate_count=%s selected_dept_id=null "
        "selected_parent=null selected_child=null validation_result=pending failure_reason=null",
        len(departments),
    )
    child_names = [item["child_dept"] for item in departments if item["child_dept"]]
    if not child_names:
        raise DepartmentResolutionError("目前無法取得正式科別主資料，未建立科別判斷。")

    project_smart_result = await detect_department_with_project_smart_adapter(case, departments)
    if project_smart_result:
        resolved = resolve_department_result(project_smart_result, departments)
        if resolved is not None:
            _log_department_result(resolved, "valid", None)
            return resolved

    try:
        result = _rule_based_department(case, departments)
    except DepartmentResolutionError as exc:
        logger.warning(
            "[DEPARTMENT] detection_called=true candidate_count=%s selected_dept_id=null "
            "selected_parent=null selected_child=null validation_result=unresolved failure_reason=%s",
            len(departments),
            type(exc).__name__,
        )
        raise
    _log_department_result(result, "valid_rule_fallback", None)
    return result


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
        search_days=21,
        max_slots=None,
        schedule_visit_type=schedule_visit_type,
        department_id=department.dept_id,
    )
    slots = _filter_slots_by_department(primary_slots, department.childDept, department.dept_id)
    slots = _filter_slots_by_visit_type(slots, canonical_visit_type)

    doctor_preference = case.preferences.doctor_preference or "不限"
    feasible_slots, relaxed_by_date = select_feasible_rows(slots, case.availability, doctor_preference)
    _log_schedule_filter_counts(
        primary_slots=primary_slots,
        visit_slots=slots,
        case=case,
        final_slots=feasible_slots,
        now=current_taipei_datetime,
    )
    ranking_slots = feasible_slots
    specialty_scores = await score_doctor_specialties(case, department, ranking_slots)

    specialty_first = _build_recommendations(
        case=case,
        case_id=case.case_id,
        slots=ranking_slots,
        prefix="rec_s",
        specialty_first=True,
        specialty_scores=specialty_scores,
        relaxed_by_date=relaxed_by_date,
    )
    time_first = _build_recommendations(
        case=case,
        case_id=case.case_id,
        slots=ranking_slots,
        prefix="rec_t",
        specialty_first=False,
        specialty_scores=specialty_scores,
        relaxed_by_date=relaxed_by_date,
    )

    total_count = len(specialty_first[:5]) + len(time_first[:5])
    return RecommendationResult(
        case_id=case.case_id,
        recommendations=RecommendationColumns(
            specialty_first=specialty_first[:5],
            time_first=time_first[:5],
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
    requested = str(child_dept or "").strip()
    return [
        slot
        for slot in slots
        if (
            str(slot.get("dept_id") or "").strip() == str(dept_id)
            if dept_id is not None and slot.get("dept_id") is not None
            else str(slot.get("child_dept") or slot.get("childDept") or "").strip() == requested
        )
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


def _rule_based_department(case: TriageCase, departments: list[dict]) -> DepartmentResult:
    text = strip_negated_red_flags(_case_text(case))
    available = {item["child_dept"] for item in departments}
    requested = ""
    if not case.patient_input.red_flags:
        requested = _stored_requested_department(case, departments) or _requested_department(text, available)
    if requested:
        return _result_for_child(
            requested,
            departments,
            confidence=0.82,
            reason=[f"使用者明確表示想看 {requested}，且未偵測到陽性急迫症狀，優先尊重科別偏好"],
        )

    patient = case.patient_input
    if (
        patient.body_part
        and "頭" in patient.body_part
        and re.search(r"(?:頭.{0,5}(?:痛|疼)|(?:痛|疼).{0,5}頭)", patient.symptom or text)
        and "一般內科" in available
    ):
        return _result_for_child(
            "一般內科",
            departments,
            confidence=0.78,
            reason=["症狀為頭部疼痛，依既有一般內科規則先行評估"],
        )

    candidates = [
        (["膝", "關節", "骨", "走路", "爬樓梯"], "一般骨科", ["症狀位置偏向骨科或關節問題"]),
        (["胸痛", "心悸", "心臟"], "心臟內科", ["症狀和胸痛或心臟相關"]),
        (["頭暈", "頭痛", "咳", "發燒", "喉嚨", "感冒"], "一般內科", ["症狀可先由一般內科評估"]),
        (["眼", "視力"], "眼科", ["症狀和眼部相關"]),
        (["皮膚", "疹", "癢"], "皮膚科", ["症狀和皮膚相關"]),
    ]
    for keywords, dept, reasons in candidates:
        if any(keyword in text for keyword in keywords) and (not available or dept in available):
            return _result_for_child(
                dept,
                departments,
                confidence=0.75,
                reason=reasons,
            )

    raise DepartmentResolutionError("目前問診資料無法唯一判定正式科別，未套用預設科別。")


def _log_department_result(
    result: DepartmentResult,
    validation_result: str,
    failure_reason: str | None,
) -> None:
    logger.info(
        "[DEPARTMENT] detection_called=true candidate_count=validated selected_dept_id=%s "
        "validation_result=%s failure_reason=%s",
        result.dept_id,
        validation_result,
        failure_reason,
    )


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


def _result_for_child(
    child: str,
    departments: list[dict],
    *,
    confidence: float,
    reason: list[str],
) -> DepartmentResult:
    matches = [item for item in departments if item["child_dept"] == child]
    if len(matches) != 1:
        raise DepartmentResolutionError(f"科別名稱「{child}」無法唯一對應正式 department_id。")
    item = matches[0]
    return DepartmentResult(
        dept_id=item["dept_id"],
        parentDept=item["parent_dept"],
        childDept=item["child_dept"],
        confidence=confidence,
        reason=reason,
    )


def _build_recommendations(
    case: TriageCase,
    case_id: str,
    slots: list[dict],
    prefix: str,
    specialty_first: bool,
    specialty_scores: dict[str, SpecialtyScore] | None = None,
    relaxed_by_date: bool = False,
) -> list[RecommendationItem]:
    specialty_scores = specialty_scores or {}
    sorted_slots = sorted(
        slots,
        key=lambda slot: _ranking_key(slot, case, specialty_scores, specialty_first),
    )
    items = []
    seen = set()
    for slot in sorted_slots:
        key = (slot.get("parent_dept"), slot.get("child_dept"), slot.get("doctor"), slot.get("date"), slot.get("session"))
        if key in seen:
            continue
        seen.add(key)

        specialty = _score_for_slot(slot, specialty_scores)
        time_score = _time_score(slot, case)
        score = weighted_score(specialty.score, time_score, specialty_first)
        if specialty_first:
            reasons = _score_explanation_reasons(
                case=case,
                slot=slot,
                specialty=specialty,
                time_score=time_score,
                total_score=score,
                specialty_first=True,
            )
        else:
            reasons = _score_explanation_reasons(
                case=case,
                slot=slot,
                specialty=specialty,
                time_score=time_score,
                total_score=score,
                specialty_first=False,
            )
        reasons.insert(0, f"推薦理由：{_compact_recommendation_reason(case, slot)}")
        if relaxed_by_date and not row_matches_availability(slot, case.availability):
            reasons.append("您方便的時段近期無號，改推薦時間最接近的門診")
        if slot.get("visit_type"):
            reasons.append(f"掛號別：{slot.get('visit_type')}")
        if slot.get("source") == "mock":
            fallback_reason = slot.get("fallback_reason") or "DB schedule unavailable"
            reasons.append(f"使用 mock 班表 fallback：{fallback_reason}")
            reasons.append("Azure SQL 查詢失敗或無資料，使用 mock 班表")

        item = RecommendationItem(
            recommendation_id=f"{prefix}_{case_id}_{len(items) + 1:03d}",
            parentDept=slot.get("parent_dept", ""),
            childDept=slot.get("child_dept", ""),
            doctor=slot.get("doctor", ""),
            date=_date_to_text(slot["date"]),
            session=normalize_session(slot.get("session", "")) or str(slot.get("session", "")),
            slot=str(slot.get("slot") or slot.get("room") or ""),
            doctor_id=str(slot.get("doctor_id") or slot.get("doctor") or ""),
            schedule_id=str(slot.get("schedule_id") or ""),
            session_time=session_range(slot.get("session", "")),
            room=str(slot.get("room") or slot.get("slot") or ""),
            visit_type=str(slot.get("visit_type") or ""),
            specialty_tags=str(slot.get("specialty_tags") or ""),
            specialty_score=round(specialty.score, 2),
            time_score=round(time_score, 2),
            match_reason=specialty.reason if specialty.source == "ai" else None,
            score=score,
            reasons=reasons,
            rank=len(items) + 1,
            is_best_match=specialty_first and len(items) == 0,
            dept_id=_optional_int(slot.get("dept_id")) or (
                case.department_result.dept_id if case.department_result else None
            ),
        )
        items.append(item)
        if len(items) >= 5:
            break
    return items


def _ranking_key(
    slot: dict,
    case: TriageCase,
    specialty_scores: dict[str, SpecialtyScore],
    specialty_first: bool,
) -> tuple:
    specialty = _score_for_slot(slot, specialty_scores)
    time_score = _time_score(slot, case)
    total_score = weighted_score(specialty.score, time_score, specialty_first)
    date_key = _date_sort_key(slot.get("date"))
    session_key = _session_rank(slot.get("session", ""))
    doctor_key = str(slot.get("doctor") or "")
    if specialty_first:
        return (
            -total_score,
            -specialty.score,
            -time_score,
            date_key,
            session_key,
            doctor_key,
        )
    return (
        -total_score,
        -time_score,
        -specialty.score,
        date_key,
        session_key,
        doctor_key,
    )


def weighted_score(
    specialty_score: float,
    time_score: float,
    specialty_first: bool,
) -> float:
    """Return the displayed and ranked 0-100 weighted score."""
    if specialty_first:
        total = (specialty_score * 0.70) + (time_score * 0.30)
    else:
        total = (time_score * 0.70) + (specialty_score * 0.30)
    return round(total * 100.0, 2)


def _score_for_slot(slot: dict, specialty_scores: dict[str, SpecialtyScore]) -> SpecialtyScore:
    key = str(slot.get("doctor_id") or slot.get("doctor") or "")
    if key in specialty_scores:
        return specialty_scores[key]
    return SpecialtyScore(
        doctor_id=key,
        doctor=str(slot.get("doctor") or ""),
        childDept=str(slot.get("child_dept") or ""),
        score=0.5,
        reason="醫師專長資料不足，不列為主要依據；專長分數使用中性值 0.50",
    )


def _time_score(slot: dict, case: TriageCase) -> float:
    dates = [value for value in case.availability.preferred_dates if value]
    days = [day for day in case.availability.preferred_days if day]
    sessions = [normalize_session(session) for session in case.availability.preferred_sessions if session]
    sessions = [session for session in sessions if session]
    if not dates and not days and not sessions:
        return 0.8

    row_date = _date_to_text(slot.get("date"))
    row_day = schedule_weekday_label(slot.get("date"))
    row_session = normalize_session(slot.get("session"))
    date_or_day_match = row_date in dates if dates else (bool(days) and row_day in days)
    session_match = bool(sessions) and row_session in sessions

    if (dates or days) and sessions:
        if date_or_day_match and session_match:
            return 1.0
        if date_or_day_match:
            return 0.72
        if session_match:
            return 0.64
        return 0.35
    if dates or days:
        return 1.0 if date_or_day_match else 0.45
    return 1.0 if session_match else 0.45


def _score_explanation_reasons(
    case: TriageCase,
    slot: dict,
    specialty: SpecialtyScore,
    time_score: float,
    total_score: float,
    specialty_first: bool,
) -> list[str]:
    return [
        _department_basis_reason(case, slot),
        _time_basis_reason(case, slot, time_score),
        _status_basis_reason(slot),
        _specialty_basis_reason(slot, specialty),
        _sorting_basis_reason(specialty, time_score, total_score, specialty_first),
    ]


def _compact_recommendation_reason(case: TriageCase, slot: dict) -> str:
    matched_specialties = _matched_specialty_items(case, slot)
    if matched_specialties:
        return f"醫師專長相符：{'、'.join(matched_specialties[:2])}"
    if (
        case.availability.preferred_days or case.availability.preferred_sessions
    ) and row_matches_availability(slot, case.availability):
        return "符合您的方便看診時段"
    return "符合目前推薦科別"


def _matched_specialty_items(case: TriageCase, slot: dict) -> list[str]:
    raw = str(slot.get("specialty_tags") or slot.get("specialty") or "").strip()
    if not raw:
        return []
    items = [item.strip() for item in re.split(r"[、,，;；/|\n]+", raw) if item.strip()]
    patient_text = _case_text(case)
    active_specialty_terms: set[str] = set()
    for symptom_terms, specialty_terms in _COMPACT_SPECIALTY_CONCEPTS:
        if any(term in patient_text for term in symptom_terms):
            active_specialty_terms.update(specialty_terms)
    if not active_specialty_terms:
        return []

    ranked: list[tuple[int, int, str]] = []
    for index, item in enumerate(items):
        score = sum(1 for term in active_specialty_terms if term in item)
        if score:
            ranked.append((-score, index, _compact_specialty_text(item)))
    ranked.sort()
    return list(dict.fromkeys(item for _, _, item in ranked if item))[:2]


def _compact_specialty_text(value: str, max_len: int = 22) -> str:
    text = re.sub(r"(?:之|的)?診斷與治療", "診療", str(value or "").strip())
    text = re.sub(r"\s+", "", text)
    return text if len(text) <= max_len else f"{text[:max_len]}…"


def _department_basis_reason(case: TriageCase, slot: dict) -> str:
    child_dept = str(slot.get("child_dept") or "").strip()
    if not child_dept and case.department_result:
        child_dept = case.department_result.childDept
    symptom = _short_text(strip_negated_red_flags(case.patient_input.symptom or case.patient_input.body_part or _case_text(case) or "問診內容"))
    evidence = [f"使用者症狀「{symptom}」"]
    if case.patient_input.body_part:
        evidence.append(f"不適部位「{_short_text(case.patient_input.body_part)}」")
    if case.patient_input.duration:
        evidence.append(f"持續時間「{_short_text(case.patient_input.duration)}」")
    department_reason = ""
    if case.department_result and case.department_result.reason:
        department_reason = f"；科別判斷理由：{case.department_result.reason[0]}"
    return f"科別依據：{'；'.join(evidence)}，目前建議科別為 {child_dept or '目前科別'}{department_reason}"


def _time_basis_reason(case: TriageCase, slot: dict, time_score: float) -> str:
    preferred = _availability_text(case)
    slot_time = _slot_time_text(slot)
    if not case.availability.preferred_dates and not case.availability.preferred_days and not case.availability.preferred_sessions:
        return f"時間依據：使用者未指定日期/時段偏好，Schedule 為 {slot_time}，時間分數使用可掛號中性值 {time_score:.2f}"
    if row_matches_availability(slot, case.availability):
        return f"時間依據：使用者偏好 {preferred}；Schedule 為 {slot_time}，符合偏好，時間分數 {time_score:.2f}"
    return f"時間依據：使用者偏好 {preferred}；Schedule 為 {slot_time}，未完全符合偏好，時間分數 {time_score:.2f}"


def _status_basis_reason(slot: dict) -> str:
    raw_status = str(slot.get("status") or "").strip()
    status_text = raw_status or "空白/NULL"
    availability = "可掛號" if row_is_available(slot) else "不可掛號"
    return f"狀態依據：Schedule.status={status_text}，判斷為{availability}"


def _specialty_basis_reason(slot: dict, specialty: SpecialtyScore) -> str:
    tags = str(slot.get("specialty_tags") or slot.get("specialty") or "").strip()
    if not tags:
        return "專長依據：doctor.specialty_tags 無資料，專長資料不足，不列為主要依據；專長分數使用中性值 0.50"
    if specialty.score <= 0.5:
        return f"專長依據：doctor.specialty_tags={_short_text(tags)}，未命中主要症狀，專長資料不列為主要依據；{specialty.reason}"
    return f"專長依據：doctor.specialty_tags={_short_text(tags)}；{specialty.reason}"


def _sorting_basis_reason(
    specialty: SpecialtyScore,
    time_score: float,
    total_score: float,
    specialty_first: bool,
) -> str:
    if specialty_first:
        formula = "specialty_score*70 + time_score*30"
        label = "專長優先欄"
    else:
        formula = "time_score*70 + specialty_score*30"
        label = "時間優先欄"
    return (
        f"排序依據：目前排序依科別、時間、專長分數加權；{label}使用 {formula}，"
        f"Specialty score={specialty.score:.2f}，Time score={time_score:.2f}，總分={total_score:.2f}，非醫學精準分數"
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


def _weekday_label(value) -> str:
    return schedule_weekday_label(value)


def _case_text(case: TriageCase) -> str:
    patient = case.patient_input
    parts = [
        patient.symptom,
        patient.body_part or "",
        patient.duration or "",
        patient.severity or "",
        patient.onset or "",
        patient.department_context or "",
        patient.requested_department_name or "",
        " ".join(patient.accompanying_symptoms),
        " ".join(patient.red_flags),
    ]
    return strip_negated_red_flags("；".join(part for part in parts if part))


def _requested_department(text: str, available: set[str]) -> str:
    if not available:
        return ""
    for child in sorted(available, key=len, reverse=True):
        if not child:
            continue
        if any(marker in text for marker in (f"想看{child}", f"想看 {child}", f"要看{child}", f"要看 {child}", f"看{child}", f"看 {child}")):
            return child
    return ""


def _stored_requested_department(case: TriageCase, departments: list[dict]) -> str:
    requested_id = case.patient_input.requested_department_id
    requested_name = case.patient_input.requested_department_name
    if requested_id is None:
        return ""
    matches = [
        item
        for item in departments
        if item["dept_id"] == requested_id
        and (not requested_name or item["child_dept"] == requested_name)
    ]
    return matches[0]["child_dept"] if len(matches) == 1 else ""


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
