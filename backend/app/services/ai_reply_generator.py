from __future__ import annotations

import logging
from datetime import date
from typing import Any

from app.config import get_settings  # Legacy test seam; runtime reply is deterministic.
from app.schemas import TriageCase
from app.services.chat_perf import record_ai_reply_outcome
from app.services.negation_utils import strip_negated_red_flags

logger = logging.getLogger(__name__)
complete_prompt: Any = None  # Legacy test seam; runtime status replies never call an LLM.


def build_department_confirmation_reply(case: TriageCase) -> str:
    """Build a grounded department explanation from fields already stored on the case."""
    department = case.department_result.childDept.strip() if case.department_result else ""
    patient = case.patient_input
    evidence: list[str] = []

    symptom = _clean_case_text(strip_negated_red_flags(patient.symptom))
    if symptom:
        evidence.append(f"症狀「{symptom}」")
    body_part = _clean_case_text(patient.body_part)
    if body_part:
        evidence.append(f"不適部位「{body_part}」")
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
    reply = (
        f"{basis}，這些情況較符合{department}的診療範圍，"
        f"因此目前建議您先看{department}。請確認後取得推薦掛號方案。"
    )
    return f"{reply}\n{preference_summary}" if preference_summary else reply


def build_availability_preference_summary(case: TriageCase) -> str:
    """Echo only concrete availability already stored on the case."""
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
