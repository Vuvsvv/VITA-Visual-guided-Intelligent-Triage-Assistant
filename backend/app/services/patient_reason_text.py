"""Patient-facing recommendation reasons built from already validated case data.

No AI call happens here: every quoted phrase is the patient's own validated text,
an official knowledge concept, or a SQL schedule/specialty value.
"""
from __future__ import annotations

import re

from app.config import get_settings
from app.schemas import TimePreference, TriageCase
from app.services.pediatric_routing import DECISION_REASON as PEDIATRIC_DECISION_REASON
from app.services.schedule_filter import normalize_session, parse_date, row_matches_availability, weekday_label
from app.services.specialty_scoring import SpecialtyScore
from app.services.time_preference_service import DATE_KINDS, preference_matches_row

MAX_QUOTED = 3
_ORDINALS = ["第一", "第二", "第三", "第四", "第五"]
_TAG_SPLIT = re.compile(r"[、,，;；/\n]+")


def department_reason(case: TriageCase, slot: dict, specialty: SpecialtyScore) -> str:
    """Why this department, plus why this doctor when the AI specialty level is meaningful."""
    parts = [_department_sentence(case, slot)]
    doctor_sentence = _doctor_sentence(slot, specialty)
    if doctor_sentence:
        parts.append(doctor_sentence)
    return "".join(parts)


def time_reason(case: TriageCase, slot: dict, relaxed_by_date: bool = False) -> str:
    """Why this slot's time, compared with the patient's validated time wording."""
    slot_text = _slot_text(slot)
    availability = case.availability
    if availability.time_preferences:
        sentence = _time_preference_sentence(availability.time_preferences, slot, slot_text)
    elif availability.preferred_dates or availability.preferred_days or availability.preferred_sessions:
        wanted = _legacy_preference_text(case)
        if row_matches_availability(slot, availability):
            sentence = f"這個門診在{slot_text}，符合您方便的{wanted}。"
        else:
            sentence = f"這個門診在{slot_text}，不完全符合您方便的{wanted}。"
    else:
        sentence = f"您沒有指定看診時間，這個門診在{slot_text}。"
    if relaxed_by_date and not row_matches_availability(slot, availability):
        sentence += "您方便的時段近期沒有號，改推薦時間最接近的門診。"
    return sentence


def _department_sentence(case: TriageCase, slot: dict) -> str:
    result = case.department_result
    dept_id = result.dept_id if result else None
    child = str(slot.get("child_dept") or (result.childDept if result else "") or "").strip() or "目前科別"
    patient = case.patient_input
    if (patient.requested_department_id is not None and patient.requested_department_id == dept_id) or (
        patient.requested_department_name and patient.requested_department_name == child
    ):
        return f"依您指定的科別，安排{child}門診。"

    state = case.conversation_state
    said = _unique(
        evidence.patient_source_text
        for candidate in state.candidate_departments if candidate.dept_id == dept_id
        for evidence in candidate.supporting_evidence
    )
    confirmed = _unique(
        record.concept for record in state.clue_concepts_asked
        if record.answer == "present" and dept_id in record.dept_ids
    )
    if said or confirmed:
        clauses = []
        if said:
            clauses.append(f"您提到{_quote(said)}")
        if confirmed:
            clauses.append(f"{'並' if said else '您'}表示有{_quote(confirmed)}")
        subject = "這些症狀" if len(said) + len(confirmed) > 1 else "此症狀"
        return f"{'，'.join(clauses)}，官方衛教資料將{subject}列在{child}，因此建議{child}。"
    decision = state.clue_decision
    if decision is not None and decision.reason == PEDIATRIC_DECISION_REASON:
        return f"就診者未滿 18 歲，依醫院就醫指引，兒童的各種症狀可先看兒科，因此建議{child}。"
    if dept_id is not None and dept_id == get_settings().family_medicine_dept_id:
        return f"目前的症狀還無法明確分到單一專科，建議先看{child}，由醫師初步評估。"
    symptom = str(patient.symptom or patient.body_part or "").strip()
    if symptom:
        return f"依您描述的「{_short(symptom)}」，建議{child}。"
    return f"建議{child}。"


def _doctor_sentence(slot: dict, specialty: SpecialtyScore) -> str:
    if specialty.source != "ai" or specialty.score < 0.5:
        return ""
    tags = _unique(_TAG_SPLIT.split(str(slot.get("specialty_tags") or slot.get("specialty") or "")))
    if not tags:
        return ""
    doctor = str(slot.get("doctor") or "").strip()
    if doctor and not doctor.endswith("醫師"):
        doctor += "醫師"
    level = "高度相關" if specialty.score >= 0.75 else "相關"
    return f"{doctor or '這位醫師'}的專長包含「{'、'.join(tags)}」，和您的症狀{level}。"


def _time_preference_sentence(preferences: list[TimePreference], slot: dict, slot_text: str) -> str:
    wanted = [item for item in preferences if item.relation != "exclude"]
    excluded = [item for item in preferences if item.relation == "exclude"]
    matched_parts: list[str] = []
    missed: list[str] = []
    for is_date in (True, False):
        dimension = [item for item in wanted if (item.kind in DATE_KINDS) == is_date]
        if not dimension:
            continue
        hits = [item for item in dimension if preference_matches_row(slot, item)]
        if not hits:
            missed.extend(item.source_text for item in dimension)
            continue
        best = min(hits, key=lambda item: item.priority or 99)
        matched_parts.append(f"「{best.source_text}」{_ordinal_suffix(best, dimension)}")

    sentence = f"這個門診在{slot_text}"
    if matched_parts:
        sentence += f"，符合您說的{'和'.join(matched_parts)}"
        if missed:
            sentence += f"，但不是您說的{_quote(_unique(missed))}"
        return sentence + "。"
    if missed:
        sentence += f"，不是您優先的{_quote(_unique(missed))}"
        if excluded:
            sentence += f"，但依您說的{_quote(_unique(item.source_text for item in excluded))}，這個時間可以看診"
        return sentence + "。"
    if excluded:
        sentence += f"；依您說的{_quote(_unique(item.source_text for item in excluded))}，這個時間可以看診"
    return sentence + "。"


def _ordinal_suffix(best: TimePreference, dimension: list[TimePreference]) -> str:
    """Name the rank only when the patient gave more than one ranked choice in this dimension."""
    if best.relation != "prefer" or best.priority is None:
        return ""
    ranks = sorted({item.priority for item in dimension if item.relation == "prefer" and item.priority})
    if len(ranks) < 2:
        return ""
    index = ranks.index(best.priority)
    ordinal = _ORDINALS[index] if index < len(_ORDINALS) else f"第 {index + 1} "
    return f"，是您的{ordinal}選擇"


def _legacy_preference_text(case: TriageCase) -> str:
    availability = case.availability
    days = list(availability.preferred_dates) or list(availability.preferred_days)
    sessions = [normalize_session(item) for item in availability.preferred_sessions if normalize_session(item)]
    return "、".join(days) + ("" if not days or not sessions else " ") + "、".join(sessions) or "時段"


def _slot_text(slot: dict) -> str:
    parsed = parse_date(slot.get("date"))
    date_text = f"{parsed.month}月{parsed.day}日" if parsed else str(slot.get("date") or "")
    weekday = weekday_label(slot.get("date"))
    session = normalize_session(slot.get("session", "")) or str(slot.get("session") or "")
    return f"{date_text}{f'（{weekday}）' if weekday else ''}{session}"


def _quote(values: list[str]) -> str:
    return "".join(f"「{value}」" for value in values)


def _unique(values) -> list[str]:
    seen: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.append(text)
    return seen[:MAX_QUOTED]


def _short(value: str, max_len: int = 24) -> str:
    return value if len(value) <= max_len else f"{value[:max_len]}…"
