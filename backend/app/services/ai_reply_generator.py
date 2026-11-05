from __future__ import annotations

import logging
from datetime import date
from typing import Any

from app.config import get_settings
from app.schemas import TimePreference, TriageCase
from app.services.chat_perf import record_ai_reply_outcome
from app.services.pediatric_routing import DECISION_REASON as PEDIATRIC_DECISION_REASON
from app.services.schedule_filter import WEEKDAY_LABELS
from app.services.time_preference_service import DATE_KINDS, SESSION_VALUE, WEEKDAY_INDEX

logger = logging.getLogger(__name__)


def build_department_confirmation_reply(case: TriageCase) -> str:
    """Build a grounded department explanation from fields already stored on the case."""
    department = case.department_result.childDept.strip() if case.department_result else ""
    patient = case.patient_input
    evidence: list[str] = []

    symptom = _clean_case_text(patient.symptom)
    if symptom:
        evidence.append(f"症狀「{symptom}」")
    body_parts = _stated_body_parts(case)
    if body_parts:
        evidence.append(f"不適部位「{'、'.join(body_parts)}」")
    duration = _clean_case_text(patient.duration)
    if duration:
        evidence.append(f"持續時間「{duration}」")
    if patient.accompanying_symptoms:
        accompanying = "、".join(
            value
            for value in (_clean_case_text(item) for item in patient.accompanying_symptoms[:2])
            if value
        )
        if accompanying:
            evidence.append(f"伴隨情況「{accompanying}」")

    basis = f"根據您剛才提到的{'、'.join(evidence)}" if evidence else "根據目前已蒐集的問診結果"
    preference_summary = build_availability_preference_summary(case)
    requested_department = (
        patient.requested_department_name.strip()
        if (
            not patient.red_flags
            and patient.requested_department_id is not None
            and patient.requested_department_name
        )
        else ""
    )
    if requested_department:
        retained = (
            f"已保留您先前提供的{'、'.join(evidence)}。"
            if evidence
            else "已保留目前問診資料。"
        )
        if department and department != requested_department:
            reply = (
                f"{retained}系統原建議為{department}；您已指定改看{requested_department}，"
                f"後續將依{requested_department}查詢醫師與班表。"
            )
        else:
            reply = (
                f"{retained}已記錄您希望看{requested_department}，"
                f"後續將依{requested_department}查詢醫師與班表。"
            )
        return f"{reply}\n{preference_summary}" if preference_summary else reply
    if not department:
        reply = f"{basis}，目前仍無法確認建議科別，請補充症狀資訊。"
        return f"{reply}\n{preference_summary}" if preference_summary else reply
    decision = case.conversation_state.clue_decision
    if decision is not None and decision.kind == "family_medicine":
        # The configured wording is used verbatim, never rewritten by AI.
        reply = f"{get_settings().family_medicine_fallback_message}請確認後取得推薦掛號方案。"
        return f"{reply}\n{preference_summary}" if preference_summary else reply
    if decision is not None and decision.reason == PEDIATRIC_DECISION_REASON:
        reply = (
            f"{basis}。就診者未滿 18 歲，依醫院就醫指引，兒童的各種症狀可先看兒科，"
            f"因此目前建議您先看{department}。請確認後取得推薦掛號方案。"
        )
        return f"{reply}\n{preference_summary}" if preference_summary else reply
    reply = (
        f"{basis}，這些情況較符合{department}的診療範圍，"
        f"因此目前建議您先看{department}。請確認後取得推薦掛號方案。"
    )
    return f"{reply}\n{preference_summary}" if preference_summary else reply


MAX_BODY_PARTS = 3


def _stated_body_parts(case: TriageCase) -> list[str]:
    """Every body part the patient stated, in the order said (例「屁股、大腿」).

    patient_input.body_part keeps one value, which a later message can replace, so the
    confirmation once showed only 「大腿」 for 「我的屁股好痛」…「我的大腿有點酸」.
    """
    # Deferred: department_reasoning_service imports conversation_service, which this module serves.
    from app.services.department_reasoning_service import validated_semantic_evidence_history

    parts: list[str] = []
    for item in validated_semantic_evidence_history(case):
        if item["field"] == "body_part" and item["semantic_status"] in {"available", "partial"}:
            value = _clean_case_text(item["normalized_value"])
            if value and value not in parts:
                parts.append(value)
    stored = _clean_case_text(case.patient_input.body_part)
    if stored and stored not in parts:
        parts.insert(0, stored)
    return parts[:MAX_BODY_PARTS]


_RANK_WORDS = ("優先", "其次", "再來")
_WEEKDAYS_ONLY = frozenset({"monday", "tuesday", "wednesday", "thursday", "friday"})
_WEEKEND_ONLY = frozenset({"saturday", "sunday"})


def build_availability_preference_summary(case: TriageCase) -> str:
    """Echo only concrete availability already stored on the case.

    Validated ``time_preferences`` show the backend-resolved dates (例「下週三早上」→
    「11 月 11 日（週三）上午」), ranked choices and exclusions, so the patient can see
    what the system understood before confirming.
    """
    if case.availability.time_preferences:
        return _time_preference_summary(case)
    dates = [_format_preferred_date(value) for value in case.availability.preferred_dates]
    dates = [value for value in dates if value]
    days = [str(value).strip() for value in case.availability.preferred_days if str(value).strip()]
    sessions = [str(value).strip() for value in case.availability.preferred_sessions if str(value).strip()]

    if dates:
        target = "、".join(dates)
        return f"就醫時間偏好：{target}{'、'.join(sessions)}。" if sessions else f"就醫時間偏好：{target}。"
    if days:
        target = "、".join(days)
        return f"就醫時間偏好：{target}{'、'.join(sessions)}。" if sessions else f"就醫時間偏好：{target}。"
    if sessions:
        return f"就醫時間偏好：{'、'.join(sessions)}時段。"
    return ""


def _time_preference_summary(case: TriageCase) -> str:
    preferences = case.availability.time_preferences
    outcomes = case.conversation_state.availability_dimension_outcomes
    wanted_dates = [item for item in preferences if item.relation != "exclude" and item.kind in DATE_KINDS]
    wanted_sessions = [item for item in preferences if item.relation != "exclude" and item.kind not in DATE_KINDS]
    excluded = _unique_labels(item for item in preferences if item.relation == "exclude")
    has_date = any(item.kind in DATE_KINDS for item in preferences)
    has_session = any(item.kind not in DATE_KINDS for item in preferences)

    # One wanted date and one wanted session read as one time: 「11 月 11 日（週三）上午」.
    combined = (len(wanted_dates) == len(wanted_sessions) == 1
                and wanted_dates[0].relation == wanted_sessions[0].relation == "prefer")
    parts: list[str] = []
    if combined:
        parts.append(_time_label(wanted_dates[0]) + _time_label(wanted_sessions[0]))
    elif wanted_dates:
        parts.append(_ranked_time_text(wanted_dates))
    elif not has_date and outcomes.get("date") == "unrestricted":
        parts.append("日期不限")
    if excluded:
        parts.append(f"{'、'.join(excluded)}不方便")
    if wanted_sessions and not combined:
        parts.append(f"時段{_ranked_time_text(wanted_sessions)}")
    elif not has_session and outcomes.get("session") == "unrestricted":
        parts.append("時段不限")
    return f"就醫時間偏好：{'；'.join(parts)}。" if parts else ""


def _ranked_time_text(items: list[TimePreference]) -> str:
    """「週三」alone; 「週三優先，其次週二」 for ranked choices; 「也可以」 for acceptable ones."""
    preferred = [item for item in items if item.relation == "prefer"]
    acceptable = _unique_labels(item for item in items if item.relation == "acceptable")
    if len(items) == 1 and preferred:
        return _time_label(preferred[0])
    texts: list[str] = []
    for index, rank in enumerate(sorted({item.priority or 99 for item in preferred})):
        labels = "、".join(_unique_labels(item for item in preferred if (item.priority or 99) == rank))
        word = _RANK_WORDS[min(index, len(_RANK_WORDS) - 1)]
        texts.append(f"{labels}{word}" if index == 0 else f"{word}{labels}")
    if acceptable:
        texts.append(f"{'、'.join(acceptable)}也可以")
    return "，".join(texts)


def _unique_labels(items) -> list[str]:
    labels: list[str] = []
    for item in items:
        label = _time_label(item)
        if label and label not in labels:
            labels.append(label)
    return labels


def _time_label(item: TimePreference) -> str:
    """Backend-resolved wording; the patient's own words only when nothing was resolved."""
    if item.kind == "session":
        return SESSION_VALUE.get(item.session or "", "")
    if item.kind == "weekday":
        return WEEKDAY_LABELS[WEEKDAY_INDEX[item.weekday]] if item.weekday else f"「{item.source_text}」"
    if item.kind == "weekday_group":
        group = set(item.weekdays)
        if group == _WEEKDAYS_ONLY:
            return "平日"
        if group == _WEEKEND_ONLY:
            return "週末"
        return "、".join(WEEKDAY_LABELS[WEEKDAY_INDEX[day]]
                        for day in sorted(group, key=WEEKDAY_INDEX.__getitem__))
    dates = sorted(item.resolved_dates)
    if not dates:
        return f"「{item.source_text}」"
    first, last = _format_date_with_weekday(dates[0]), _format_date_with_weekday(dates[-1])
    return first if len(dates) == 1 else f"{first}到{last}"


def _format_date_with_weekday(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return ""
    return f"{parsed.month} 月 {parsed.day} 日（{WEEKDAY_LABELS[parsed.weekday()]}）"


def build_no_schedule_message(case: TriageCase) -> str:
    """Describe an empty exact-match result without inventing nearby availability."""
    dates = [_format_preferred_date(value) for value in case.availability.preferred_dates]
    dates = [value for value in dates if value]
    sessions = [str(value).strip() for value in case.availability.preferred_sessions if str(value).strip()]
    department = (
        case.patient_input.requested_department_name.strip()
        if case.patient_input.requested_department_id is not None
        and case.patient_input.requested_department_name
        else case.department_result.childDept.strip() if case.department_result else ""
    )
    department_text = f"{department}班表" if department else "班表"

    if len(dates) == 1:
        preference = f"{dates[0]}{'、'.join(sessions)}"
        return f"目前找不到 {preference}符合條件的{department_text}。"
    if len(dates) > 1:
        return f"目前找不到您指定日期／時段的相符{department_text}。"
    if sessions:
        return f"目前找不到您偏好{'、'.join(sessions)}時段的相符{department_text}。"
    return f"目前找不到符合條件的{department_text}。"


def _format_preferred_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(str(value or "").strip())
    except ValueError:
        return ""
    return f"{parsed.month} 月 {parsed.day} 日"


def _clean_case_text(value: Any, max_length: int = 48) -> str:
    text = " ".join(str(value or "").strip(" ，。；\n\t").split())
    return text if len(text) <= max_length else f"{text[:max_length]}…"

async def generate_triage_reply(
    *,
    case: TriageCase,
    next_question: str | None,
    fallback_reply: str,
) -> str:
    """Generate presentation text without changing deterministic triage state."""
    target = _reply_target(case, next_question)

    # The question is already chosen by the checklist or conversation service.
    # Presentation must not generate a second, competing question.
    if next_question is not None:
        record_ai_reply_outcome(
            fallback=True,
            reason="controlled_question_variant",
            validator_accepted=None,
            parser_invoked=False,
        )
        return fallback_reply

    record_ai_reply_outcome(
        fallback=True,
        reason="controlled_status_reply",
        validator_accepted=None,
        parser_invoked=False,
    )
    return _fallback(target, fallback_reply, "controlled_status_reply")


def _reply_target(case: TriageCase, next_question: str | None) -> str:
    if next_question:
        return case.conversation_state.last_question_key or "next_question"
    if case.conversation_state.awaiting_confirmation:
        return "confirmation"
    if case.conversation_state.confirmed:
        return "confirmed"
    return "status"


def _fallback(target: str, fallback_reply: str, reason: str) -> str:
    logger.info(
        "ai contextual reply fallback target=%s reason=%s",
        target,
        reason,
    )
    return fallback_reply
