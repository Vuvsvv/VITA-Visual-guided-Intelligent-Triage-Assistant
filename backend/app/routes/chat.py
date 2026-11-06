import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Response

from app.config import get_settings
from app.schemas import CaseResumeResult, ChatRequest, ConversationStage, Message, TriageCase, TriageResult, VisitType
from app.services.ai_reply_generator import build_department_confirmation_reply, generate_triage_reply
from app.services.case_store import (
    case_is_urgent, clear_recommendations, create_case, get_case, peek_case, sanitize_untrusted_snapshot, save_case,
)
from app.services.case_resume_service import build_case_resume
from app.services.rate_limit import enforce_chat_rate_limit
from app.services.chat_perf import (
    ai_phase,
    begin_chat_perf,
    finish_chat_perf,
    record_chat_response_trace,
    record_chat_route_trace,
)
from app.services.conversation_service import (
    FAMILY_MEDICINE_UNAVAILABLE_REPLY,
    UNRESOLVED_REPLY,
    SAFETY_POSITIVE_REPLY,
    SAFETY_UNRESOLVED_REPLY,
    SAFETY_ANSWER_CAP,
    adopt_safety_net_family_medicine,
    advance_conversation,
    ask_revision_detail_question,
    awaiting_availability_answer,
    capture_pending_answer,
    capture_prior_answer_reference,
    capture_safety_screen_answer,
    clear_answered_pending,
    known_symptom_concepts,
    reopen_safety_screen_for_new_symptoms,
    companion_question_due,
    request_clarification,
    safety_net_family_medicine_reason,
    safety_screen_resolved,
    safety_unresolved_stop,
    settle_unanswered_clue_question,
    symptom_grounded,
    validated_dialogue_act,
)
from app.services.department_reasoning_service import accepted_semantic_evidence, reason_about_departments
from app.services.department_preference_service import resolve_requested_department
from app.services.rag_triage_adapter import refine_case_with_ai
from app.services.rule_engine import (
    QUESTION_TEXTS,
    apply_user_message,
)
from app.services.ttas_evaluator import apply_ttas_evaluation, ttas_requires_immediate_action
from app.services.ai_service import runtime_ai_available

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

DEPARTMENT_CLARIFICATION_KEY = "department_clarification"
MAX_MESSAGE_CHARS = 100
MESSAGE_TOO_LONG_REPLY = "您這次說的內容超過 100 個字，請分成幾次、每次 100 字以內再傳一次。"
URGENT_WARNING_FALLBACK = (
    "初步篩檢結果顯示您的狀況可能較危急，線上問診無法判斷嚴重程度，因此不會繼續掛號導引。"
    "請立即前往急診；情況危急請撥打 119。"
)


@router.get("/{case_id}/resume", response_model=CaseResumeResult)
async def resume_case(case_id: str, response: Response) -> CaseResumeResult:
    response.headers["Cache-Control"] = "no-store"
    case = peek_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="原問診已失效，請重新開始問診。")
    return build_case_resume(case)


@router.post("", response_model=TriageResult, dependencies=[Depends(enforce_chat_rate_limit)])
async def chat(req: ChatRequest) -> TriageResult:
    perf, perf_token = begin_chat_perf()
    with perf.measure("case_prepare"):
        case = _resolve_case(req)
        clarification_before = case.conversation_state.clarification_status
        pending_before = case.conversation_state.pending_clarification_intent
        department_before = case.conversation_state.department_status
        department_id_before = case.department_result.dept_id if case.department_result else None
        prior_extractions = tuple(item.model_copy(deep=True) for item in case.semantic_extractions)
        prior_symptoms = known_symptom_concepts(case)
        prior_medical_facts = _clinical_progress_facts(case)
        availability_pending_before = awaiting_availability_answer(case)
        prior_availability_facts = _availability_facts(case)
        requested_department_before = _requested_department(case)
        # The first message after the 「我想修改」 button (an auto revision carries its own content).
        explicit_revision_turn = case.conversation_state.last_question_key == "revision"
        prior_user_sources = [item.content for item in case.history_records if item.role == "user"]
        _apply_visit_type(case, req.visit_type)
        settings = get_settings()
        urgent_terminal_at_request_start = _is_urgent_terminal(case)
        safety_check_turn = case.conversation_state.clarification_status == "safety_check"
        free_text_request = _request_has_text_input(req)
        case.conversation_state.last_dialogue_act = None
    record_chat_route_trace(
        case_id=case.case_id,
        triage_branch="ai_first_free_text",
    )
    logger.info(
        "chat request case_id=%s confirmed=%s has_message=%s has_triage_case=%s history_len=%s stage=%s",
        case.case_id,
        req.confirmed,
        bool(req.message),
        req.triage_case is not None,
        len(case.history_records),
        case.conversation_state.stage,
    )

    # Keyed batch answers are no longer accepted; keep the existing 400 response.
    if req.answers:
        finish_chat_perf(perf, perf_token)
        raise HTTPException(status_code=400, detail="已不支援固定問卷作答，請直接以文字描述症狀。")

    if case.visit_type == VisitType.RETURN_VISIT:
        with perf.measure("case_store"):
            save_case(case)
        reply = "回診不進行完整症狀分診，請使用 /followup/recommend 提供原科別、原醫師與可看診時段。"
        response = TriageResult(
            case_id=case.case_id,
            triage_case=case,
            conversation_state=case.conversation_state,
            triage=case.triage,
            department_result=case.department_result,
            next_question=None,
            reply=reply,
            needMoreInfo=True,
            triage_reasons=case.triage.reasons,
            question_batch=[],
        )
        record_chat_response_trace(deterministic_next_question=None, final_reply=reply)
        finish_chat_perf(perf, perf_token)
        return response

    if case.visit_type == VisitType.QUICK_SEARCH:
        with perf.measure("case_store"):
            save_case(case)
        finish_chat_perf(perf, perf_token)
        raise HTTPException(
            status_code=400,
            detail="quick_search 不進行問診，請改呼叫 /schedules/search。",
        )

    # Over-long input is answered before any AI call and is not recorded, so a pasted essay
    # cannot burn AI quota. A safety stop still wins: that case keeps getting its stop reply.
    if _message_too_long(req) and not _stopped_for_safety(case):
        with perf.measure("case_store"):
            save_case(case)
        logger.info("chat message too long case_id=%s limit=%s", case.case_id, MAX_MESSAGE_CHARS)
        response = TriageResult(
            case_id=case.case_id,
            triage_case=case,
            conversation_state=case.conversation_state,
            triage=case.triage,
            department_result=case.department_result,
            next_question=MESSAGE_TOO_LONG_REPLY,
            reply=MESSAGE_TOO_LONG_REPLY,
            needMoreInfo=True,
            triage_reasons=case.triage.reasons,
        )
        record_chat_response_trace(
            deterministic_next_question=MESSAGE_TOO_LONG_REPLY,
            final_reply=MESSAGE_TOO_LONG_REPLY,
        )
        finish_chat_perf(perf, perf_token)
        return response

    # A safety stop is final: 「我想修改」 must not reopen it (a reported 「有」 could otherwise
    # be followed by 「沒有」 and continue to booking). The request falls through and gets the stop reply.
    if req.revision_requested and not _stopped_for_safety(case):
        with perf.measure("deterministic_parse"):
            _begin_revision(case)
        with perf.measure("case_store"):
            save_case(case)
        reply = "好的，我會延續這次問診紀錄重新整理。請補充想修改的症狀、科別、嚴重程度、看診日期或時段。"
        logger.info(
            "chat revision requested case_id=%s history_len=%s",
            case.case_id,
            len(case.history_records),
        )
        with perf.measure("response_build"):
            response = TriageResult(
                case_id=case.case_id,
                triage_case=case,
                conversation_state=case.conversation_state,
                triage=case.triage,
                department_result=case.department_result,
                next_question=reply,
                reply=reply,
                needMoreInfo=True,
                triage_reasons=case.triage.reasons,
            )
        record_chat_response_trace(
            deterministic_next_question=reply,
            final_reply=reply,
        )
        finish_chat_perf(perf, perf_token)
        return response

    with perf.measure("deterministic_parse"):
        ai_suggestion = None
        department_preference_name = None
        has_user_input = False
        ai_available = runtime_ai_available(settings)
        case.conversation_state.free_text_mode = True
        semantic_ai_allowed = (
            ai_available
            and case.conversation_state.clarification_status != "unresolved"
        )
        logger.info(
            "chat interpretation available=%s allowed=%s safety_turn=%s deterministic_fallback=%s",
            ai_available, semantic_ai_allowed, safety_check_turn, not ai_available,
        )
        user_text_parts: list[str] = []
        messages = req.messages or []
        should_auto_revision = _should_treat_user_input_as_revision(case, req)
        if should_auto_revision:
            _begin_revision(case)
            logger.info(
                "chat auto revision case_id=%s stage=%s history_len=%s",
                case.case_id,
                case.conversation_state.stage,
                len(case.history_records),
            )
        if req.message and req.message.strip():
            apply_user_message(
                case,
                req.message,
                apply_legacy_safety=False,
            )
            user_text_parts.append(req.message)
            has_user_input = True
        elif messages:
            for message in messages:
                if message.role == "user" and message.content.strip():
                    apply_user_message(
                        case,
                        message.content,
                        apply_legacy_safety=False,
                    )
                    user_text_parts.append(message.content)
                    has_user_input = True
        if has_user_input:
            logger.info(
                "chat patient updated case_id=%s collected_field_count=%s red_flags_checked=%s last_question_key=%s history_len=%s",
                case.case_id,
                len(case.patient_input.collected_fields),
                case.patient_input.red_flags_checked,
                case.conversation_state.last_question_key,
                len(case.history_records),
            )

        _sync_confirmation_flags(case)
        preserve_urgent_terminal = bool(
            urgent_terminal_at_request_start and not has_user_input
        )
    if has_user_input:
        if semantic_ai_allowed:
            with perf.measure("semantic_refinement"), ai_phase("semantic_refinement"):
                ai_suggestion = await refine_case_with_ai(case, user_sources=user_text_parts)
            case.conversation_state.revision_mode = False
        case.conversation_state.last_dialogue_act = validated_dialogue_act(
            getattr(ai_suggestion, "dialogue_act", None), user_text_parts,
        )
        if case.conversation_state.last_dialogue_act == "decline_further_questions":
            case.conversation_state.clue_user_declined = True
        if ai_suggestion is not None and getattr(ai_suggestion, "department_preference", None) is not None:
            department_preference_name = ai_suggestion.department_preference.department_name
        if free_text_request and not ai_available and not safety_check_turn:
            has_explicit_preference = (
                case.patient_input.requested_department_id is not None
                or bool(case.patient_input.requested_department_name)
            )
            case.department_result = resolve_requested_department(case) if has_explicit_preference else None
            case.conversation_state.department_status = (
                "resolved" if case.department_result is not None else "unresolved"
            )
            case.conversation_state.candidate_departments = []

    with perf.measure("rule_engine"):
        confirmation_only = bool(not has_user_input and case.conversation_state.is_complete)
        if not confirmation_only and not preserve_urgent_terminal:
            case.triage = apply_ttas_evaluation(case)
        safety_interpretation = (
            ai_suggestion.pending_answer
            if safety_check_turn and ai_suggestion is not None
            else None
        )
        capture_safety_screen_answer(
            case, safety_interpretation, user_text_parts,
            dialogue_act=getattr(ai_suggestion, "dialogue_act", None),
            current_extractions=(ai_suggestion.semantic_extractions or []) if ai_suggestion else [],
            # The AI call failed this turn: the reply was never judged and is not an unclear answer.
            # Without any configured AI the screen still ends after SAFETY_ANSWER_CAP replies.
            interpretation_unavailable=bool(
                has_user_input and semantic_ai_allowed
                and (ai_suggestion is None or getattr(ai_suggestion, "provider_failed", False))
            ),
        )
        safety_completed_this_turn = bool(
            safety_check_turn
            and has_user_input
            and safety_screen_resolved(case)
            and not case.patient_input.red_flags
        )
        if preserve_urgent_terminal:
            pass
        elif confirmation_only:
            pass
        else:
            ttas_immediate = ttas_requires_immediate_action(case)
            if ttas_immediate:
                case.patient_input.red_flags_checked = True
                case.patient_input.red_flags_status = "positive_ttas"
                if not safety_check_turn:
                    case.conversation_state.turn_count += len(user_text_parts)
                case.conversation_state.clarification_status = "urgent"
                case.conversation_state.is_complete = True
                case.conversation_state.stage = ConversationStage.DONE
                case.conversation_state.awaiting_confirmation = False
                case.conversation_state.confirmed = False
                case.confirmed = False
                case.triage.need_more_info = False
                case.triage.next_question = None
                case.triage.is_final = True
            else:
                suggestion = None
                pending_before_turn = case.conversation_state.pending_clarification_intent
                pending_interpretation = (
                    ai_suggestion.pending_answer
                    if ai_suggestion is not None and pending_before_turn
                    else None
                )
                turn_interpretation_complete = bool(
                    ai_suggestion is not None
                    and getattr(ai_suggestion, "interpretation_complete", True)
                )
                if not safety_check_turn:
                    prior_captured = capture_prior_answer_reference(
                        case, getattr(ai_suggestion, "prior_answer_reference", None),
                        getattr(ai_suggestion, "dialogue_act", None),
                        prior_extractions=prior_extractions, prior_user_sources=prior_user_sources,
                    )
                    if not prior_captured:
                        capture_pending_answer(case, pending_interpretation, user_text_parts)
                    clear_answered_pending(case)
                    if turn_interpretation_complete:
                        # A symptom first said after the safety screen opens the screen again
                        # before any department decision this turn.
                        reopen_safety_screen_for_new_symptoms(
                            case, ai_suggestion.semantic_extractions or [], user_text_parts,
                            prior_symptoms=prior_symptoms, prior_user_sources=prior_user_sources,
                        )
                # After 「我想修改」, a message that changes no symptom, time or department (or
                # could not be understood) asks what to change instead of re-confirming or retrying.
                revision_detail_due = bool(
                    explicit_revision_turn
                    and has_user_input
                    and case.conversation_state.last_dialogue_act != "decline_further_questions"
                    and _symptom_facts(_clinical_progress_facts(case)) == _symptom_facts(prior_medical_facts)
                    and _availability_facts(case) == prior_availability_facts
                    and _requested_department(case) == requested_department_before
                )
                department_reasoning_allowed = bool(
                    has_user_input
                    and turn_interpretation_complete
                    and (not safety_check_turn or safety_completed_this_turn)
                    and not revision_detail_due
                    # A reopened safety screen comes first; departments are recomputed once it is done.
                    and not case.conversation_state.safety_rescreen_sources
                )
                # An availability answer with no new symptom keeps the department already decided;
                # re-running convergence on unchanged evidence could reopen it.
                department_kept = bool(
                    availability_pending_before
                    and _symptom_facts(_clinical_progress_facts(case)) == _symptom_facts(prior_medical_facts)
                )
                if department_reasoning_allowed and not department_kept:
                    settle_unanswered_clue_question(case)
                    with perf.measure("department_detection"), ai_phase("department_detection"):
                        await reason_about_departments(case)
                # After department recomputation, so a decision this turn is not overwritten.
                safety_net_reason = (
                    None if revision_detail_due else safety_net_family_medicine_reason(case, user_text_parts)
                )
                if safety_net_reason:
                    adopt_safety_net_family_medicine(case, safety_net_reason)
                clarification_planning_allowed = bool(
                    has_user_input
                    and turn_interpretation_complete
                    and semantic_ai_allowed
                    and case.conversation_state.clarification_status != "unresolved"
                    and (not safety_check_turn or safety_completed_this_turn or case.conversation_state.safety_answer_attempts < SAFETY_ANSWER_CAP)
                    # The Backend acknowledges added symptoms with its own line; no planner call is needed.
                    and not (safety_check_turn and case.conversation_state.safety_answer_validation == "symptom_added")
                    # Backend re-asks its fixed availability question; no planner call is needed.
                    and not awaiting_availability_answer(case)
                    # Nor for the fixed accompanying-symptom question of a single-department case.
                    and not companion_question_due(case)
                    and not revision_detail_due
                    # The reopened safety screen is a fixed question.
                    and not case.conversation_state.safety_rescreen_sources
                    # 「我想看婦產科」 with no symptom: Backend asks its fixed symptom question.
                    and not (department_preference_name and not symptom_grounded(case))
                )
                if clarification_planning_allowed:
                    with perf.measure("conversation_clarification"), ai_phase("conversation_clarification"):
                        suggestion = await request_clarification(
                            case,
                            user_text_parts,
                            focused_answer_status=(
                                pending_interpretation.answer_status
                                if pending_interpretation is not None else None
                            ),
                            dialogue_act=getattr(ai_suggestion, "dialogue_act", None),
                        )
                if revision_detail_due:
                    ask_revision_detail_question(case, user_text_parts)
                else:
                    advance_conversation(
                        case,
                        suggestion,
                        user_sources=user_text_parts,
                        current_extractions=(ai_suggestion.semantic_extractions or []) if ai_suggestion else [],
                        made_progress=bool(_clinical_progress_facts(case) - prior_medical_facts),
                        availability_changed=_availability_facts(case) != prior_availability_facts,
                        department_requested_this_turn=department_preference_name is not None,
                    )
                if not has_user_input and not safety_check_turn and not case.history_records:
                    case.conversation_state.last_question_key = "symptom"
                    case.conversation_state.reply_source = "INITIAL_PROMPT"
                    case.triage.next_question = QUESTION_TEXTS["symptom"]
                if case.triage.next_question:
                    case.history_records.append(Message(role="assistant", content=case.triage.next_question))
    logger.info(
        "chat triage evaluated case_id=%s need_more_info=%s has_next_question=%s red_flag_count=%s red_flags_checked=%s last_question_key=%s consumed_fields=%s question_attempts=%s field_statuses=%s",
        case.case_id,
        case.triage.need_more_info,
        bool(case.triage.next_question),
        len(case.patient_input.red_flags),
        case.patient_input.red_flags_checked,
        case.conversation_state.last_question_key,
        case.conversation_state.consumed_fields,
        case.conversation_state.question_attempts,
        case.conversation_state.field_statuses,
    )

    with perf.measure("rule_engine"):
        if _is_urgent_terminal(case):
            case.confirmed = False
            case.conversation_state.stage = ConversationStage.DONE
            case.conversation_state.awaiting_confirmation = False
            case.conversation_state.confirmed = False
            case.conversation_state.is_complete = True
            case.triage.need_more_info = False
            case.triage.next_question = None
            case.triage.is_final = True
        elif (
            case.triage.need_more_info
            or not safety_screen_resolved(case)
            or case.conversation_state.department_status != "resolved"
            or case.department_result is None
        ):
            case.confirmed = False
            case.conversation_state.stage = ConversationStage.COLLECTING
            case.conversation_state.awaiting_confirmation = False
            case.conversation_state.confirmed = False
            case.conversation_state.is_complete = False
        elif req.confirmed:
            case.confirmed = True
            case.conversation_state.stage = ConversationStage.RECOMMENDING
            case.conversation_state.awaiting_confirmation = False
            case.conversation_state.confirmed = True
        elif case.confirmed or case.conversation_state.confirmed:
            case.confirmed = True
            case.conversation_state.stage = ConversationStage.RECOMMENDING
            case.conversation_state.awaiting_confirmation = False
            case.conversation_state.confirmed = True
        else:
            case.confirmed = False
            case.conversation_state.stage = ConversationStage.WAITING_CONFIRMATION
            case.conversation_state.awaiting_confirmation = True
            case.conversation_state.confirmed = False

    # Stored recommendations must not outlive an urgent result or a department change.
    if case_is_urgent(case) or (case.department_result.dept_id if case.department_result else None) != department_id_before:
        clear_recommendations(case.case_id)
    with perf.measure("case_store"):
        save_case(case)
    logger.info(
        "chat response case_id=%s stage=%s is_complete=%s awaiting_confirmation=%s confirmed=%s history_len=%s has_next_question=%s last_question_key=%s consumed_fields=%s question_attempts=%s field_statuses=%s red_flags_checked=%s",
        case.case_id,
        case.conversation_state.stage,
        case.conversation_state.is_complete,
        case.conversation_state.awaiting_confirmation,
        case.conversation_state.confirmed,
        len(case.history_records),
        bool(case.triage.next_question),
        case.conversation_state.last_question_key,
        case.conversation_state.consumed_fields,
        case.conversation_state.question_attempts,
        case.conversation_state.field_statuses,
        case.patient_input.red_flags_checked,
    )

    reply = case.triage.next_question
    if (
        department_preference_name
        and case.conversation_state.last_question_key == "symptom"
        and case.triage.next_question
    ):
        reply = _department_preference_acknowledgement(
            department_preference_name,
            case.triage.next_question,
        )
    urgent_terminal = _is_urgent_terminal(case)
    if urgent_terminal:
        case.conversation_state.reply_source = "URGENT_BACKEND"
        reply = case.triage.warning_message or URGENT_WARNING_FALLBACK
    elif case.conversation_state.clarification_status == "unresolved":
        decision = case.conversation_state.clue_decision
        if case.conversation_state.safety_positive_reported:
            case.conversation_state.reply_source = "SAFETY_POSITIVE_STOP"
        elif safety_unresolved_stop(case):
            # Android stops the chat on this marker, like an urgent stop.
            case.conversation_state.reply_source = "SAFETY_UNRESOLVED_STOP"
        reply = (
            SAFETY_POSITIVE_REPLY if case.conversation_state.safety_positive_reported else
            SAFETY_UNRESOLVED_REPLY if not safety_screen_resolved(case) else
            FAMILY_MEDICINE_UNAVAILABLE_REPLY if decision and decision.kind == "family_medicine_unavailable" else
            UNRESOLVED_REPLY
        )
    elif reply is None and case.conversation_state.awaiting_confirmation and case.department_result:
        case.conversation_state.reply_source = "CONFIRMATION"
        reply = build_department_confirmation_reply(case)
    elif reply is None and case.conversation_state.confirmed:
        case.conversation_state.reply_source = "CONFIRMED_BACKEND"
        reply = "已確認分診結果，可呼叫 /recommend 取得推薦掛號方案。"

    if not urgent_terminal:
        with perf.measure("ai_reply_total"), ai_phase("ai_reply"):
            reply = await generate_triage_reply(
                case=case,
                next_question=case.triage.next_question,
                fallback_reply=reply or "",
            )

    logger.info(
        "chat ownership case_id=%s clarification_before=%s clarification_after=%s "
        "safety_check_turn=%s safety_completed_this_turn=%s safety_answer_validation=%s "
        "safety_answer_attempts=%s dialogue_act=%s pending_before=%s pending_after=%s "
        "department_before=%s department_after=%s question_accepted=%s rejection_reason=%s "
        "repair_attempted=%s final_reply_source=%s",
        case.case_id, clarification_before, case.conversation_state.clarification_status,
        safety_check_turn, safety_completed_this_turn, case.conversation_state.safety_answer_validation,
        case.conversation_state.safety_answer_attempts,
        getattr(getattr(ai_suggestion, "dialogue_act", None), "act", "unknown"),
        pending_before, case.conversation_state.pending_clarification_intent,
        department_before, case.conversation_state.department_status,
        case.conversation_state.reply_source in {"AI_PLANNER", "AI_REPAIR"},
        case.conversation_state.question_rejection_reason,
        case.conversation_state.question_repair_attempted, case.conversation_state.reply_source,
    )
    with perf.measure("response_build"):
        response = TriageResult(
            case_id=case.case_id,
            triage_case=case,
            conversation_state=case.conversation_state,
            triage=case.triage,
            department_result=case.department_result,
            next_question=case.triage.next_question,
            reply=reply,
            needMoreInfo=case.triage.need_more_info,
            triage_reasons=case.triage.reasons,
            question_batch=[],
        )
    record_chat_response_trace(
        deterministic_next_question=case.triage.next_question,
        final_reply=reply,
    )
    finish_chat_perf(perf, perf_token)
    return response


def _is_urgent_terminal(case: TriageCase) -> bool:
    return bool(
        case.conversation_state.clarification_status == "urgent"
        and case.triage.warning_required
        and case.triage.is_final
    )


def _stopped_for_safety(case: TriageCase) -> bool:
    """TTAS urgency, a reported positive screen answer, or a screen that could not be answered."""
    state = case.conversation_state
    return bool(
        state.clarification_status == "urgent" or case.triage.warning_required
        or state.safety_positive_reported or safety_unresolved_stop(case)
    )


def _sync_confirmation_flags(case: TriageCase) -> None:
    if case.confirmed:
        case.conversation_state.confirmed = True
        case.conversation_state.awaiting_confirmation = False


def _department_preference_acknowledgement(name: str, symptom_question: str) -> str:
    return (
        f"收到，已記錄您希望看{name}。這是您指定的科別偏好；"
        f"為了協助確認是否適合，仍需要了解症狀。{symptom_question}"
    )


def _apply_visit_type(case: TriageCase, requested: VisitType | None) -> None:
    if requested is None:
        return
    if case.visit_type is not None and case.visit_type != requested:
        raise HTTPException(
            status_code=409,
            detail="visit_type 已在問診開始時確定，不可於同一 case 中變更。",
        )
    if case.visit_type is None:
        case.visit_type = requested


def _resolve_case(req: ChatRequest) -> TriageCase:
    supplied = req.triage_case
    if req.case_id and supplied is not None and supplied.case_id != req.case_id:
        raise HTTPException(status_code=422, detail="triage_case.case_id 與 request case_id 不一致。")
    lookup_id = req.case_id or (supplied.case_id if supplied else None)
    stored = get_case(lookup_id) if lookup_id else None
    if supplied is not None and stored is not None:
        if supplied.visit_type is not None and stored.visit_type is not None and supplied.visit_type != stored.visit_type:
            raise HTTPException(
                status_code=409,
                detail="triage_case.visit_type 與已保存案件不一致。",
            )
    if stored is not None:
        return stored
    if req.require_existing_case:
        raise HTTPException(status_code=404, detail="原問診已失效，請重新開始問診。")
    if supplied is not None:
        case = sanitize_untrusted_snapshot(supplied)
        save_case(case)
        return case
    return create_case(lookup_id)


def _message_too_long(req: ChatRequest) -> bool:
    texts = [req.message or ""] + [item.content for item in req.messages if item.role == "user"]
    return any(len(text.strip()) > MAX_MESSAGE_CHARS for text in texts)


def _request_has_text_input(req: ChatRequest) -> bool:
    if req.message and req.message.strip():
        return True
    return any(message.role == "user" and message.content.strip() for message in req.messages)


def _availability_facts(case: TriageCase) -> dict:
    """Accepted structured time data only; used to tell whether this turn produced any."""
    return case.availability.model_dump(
        include={"time_preferences", "preferred_dates", "preferred_days", "preferred_sessions"},
    )


def _clinical_progress_facts(case: TriageCase) -> set[tuple[str, ...]]:
    """Count accepted clinical facts, not model proposals or paraphrased sources."""
    facts = {
        ("medical", item["field"], json.dumps(item["normalized_value"], ensure_ascii=False, sort_keys=True),
         str(item["assertion"]), item["semantic_status"])
        for item in accepted_semantic_evidence(case)
    }
    facts.update(("answer", intent) for intent in case.conversation_state.clarification_evidence)
    if safety_screen_resolved(case):
        facts.add(("safety", "resolved"))
    return facts


def _requested_department(case: TriageCase) -> tuple:
    return (case.patient_input.requested_department_id, case.patient_input.requested_department_name)


def _symptom_facts(facts: set[tuple[str, ...]]) -> set[tuple[str, ...]]:
    """Accepted symptom evidence only; answers to workflow questions are left out."""
    return {fact for fact in facts if fact[0] == "medical"}


def _begin_revision(case: TriageCase) -> None:
    clear_recommendations(case.case_id)
    case.confirmed = False
    case.recommendation_generated = False
    case.script_generated = False
    case.selected_recommendation_id = None
    case.conversation_state.stage = ConversationStage.COLLECTING
    case.conversation_state.is_complete = False
    case.conversation_state.awaiting_confirmation = False
    case.conversation_state.confirmed = False
    case.conversation_state.revision_mode = True
    case.conversation_state.last_question_key = "revision"
    case.conversation_state.turn_count = 0
    case.conversation_state.consecutive_no_progress_turns = 0
    case.conversation_state.clarification_status = "collecting"
    # Never reached from a safety stop (_stopped_for_safety); a symptom first said in the
    # revision reopens the safety screen (reopen_safety_screen_for_new_symptoms).
    case.conversation_state.safety_positive_reported = False
    case.conversation_state.uncertainty_reasons = []
    case.conversation_state.next_information_needed = []
    case.conversation_state.pending_clarification_intent = None
    case.conversation_state.pending_clarification_question = None
    case.conversation_state.department_status = "unresolved"
    case.conversation_state.candidate_departments = []
    case.conversation_state.department_uncertainty_reason = None
    case.conversation_state.department_next_information_needed = None
    case.conversation_state.department_next_question_intent = None
    case.conversation_state.semantic_official_links = []
    # Revision continues the case: answered questions, asked intents, clue answers and asked
    # body parts are kept, so nothing already answered is asked again. Only the department
    # decision and the turn budgets restart.
    case.conversation_state.clue_tiebreak_used = False
    case.conversation_state.clue_user_declined = False
    case.conversation_state.department_clue_offer = []
    case.conversation_state.department_clue_candidates = []
    case.conversation_state.pending_clue_question_index = None
    case.conversation_state.clue_reask_count = 0
    case.conversation_state.clue_decision = None
    case.conversation_state.safety_net_asked = False
    case.conversation_state.safety_net_choice = None
    case.conversation_state.body_part_question_source = None
    case.department_result = None
    case.conversation_state.field_statuses.pop("department", None)
    case.conversation_state.clarification_reasons.pop("department", None)
    case.conversation_state.question_attempts.pop(DEPARTMENT_CLARIFICATION_KEY, None)


def _should_treat_user_input_as_revision(case: TriageCase, req: ChatRequest) -> bool:
    if req.confirmed or req.revision_requested or _stopped_for_safety(case):
        return False
    has_user_text = bool(req.message and req.message.strip()) or any(
        message.role == "user" and message.content.strip()
        for message in req.messages
    )
    if not has_user_text:
        return False
    return (
        case.confirmed
        or case.conversation_state.confirmed
        or case.conversation_state.awaiting_confirmation
        or case.conversation_state.stage in {
            ConversationStage.WAITING_CONFIRMATION,
            ConversationStage.RECOMMENDING,
            ConversationStage.SCRIPT_READY,
        }
    )
