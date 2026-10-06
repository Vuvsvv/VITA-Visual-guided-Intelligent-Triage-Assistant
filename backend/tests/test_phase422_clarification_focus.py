from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.schemas import Message, SemanticExtraction, TriageCase
from app.services.conversation_service import (
    ClarificationSuggestion,
    advance_conversation,
    capture_pending_answer,
    request_clarification,
)


INTERNAL_REASON = "已記錄本輪新資訊，上一個澄清面向仍需確認"
FLANK_QUESTION = "目前是否有腰側或背部靠近腰的位置疼痛？"
SECOND_USER_TEXT = (
    "只有一點點血絲，不是整泡尿都紅，而且尿尿時有灼熱感。"
    "我沒有胸痛、呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛"
)


def ambiguous_flank_case() -> TriageCase:
    case = TriageCase(case_id="phase422-flank-focus")
    case.history_records = [
        Message(role="user", content="我最近有血尿，已經兩天了"),
        Message(role="assistant", content="血尿的量或程度如何？"),
        Message(role="user", content=SECOND_USER_TEXT),
    ]
    case.patient_input.symptom = "血尿"
    case.patient_input.red_flags_checked = True
    case.semantic_extractions = [
        SemanticExtraction(
            field="duration",
            normalized_value="2天",
            source_text="兩天",
            confidence=0.95,
            semantic_status="available",
        ),
        SemanticExtraction(
            field="accompanying_symptoms",
            normalized_value=["排尿灼熱"],
            source_text="尿尿時有灼熱感",
            confidence=0.95,
            semantic_status="available",
        ),
    ]
    state = case.conversation_state
    state.turn_count = 2
    state.clarification_status = "clarifying"
    state.department_status = "ambiguous"
    state.department_next_question_intent = "flank_pain"
    state.pending_clarification_intent = "flank_pain"
    state.asked_clarification_intents = ["severity", "flank_pain"]
    state.uncertainty_reasons = [INTERNAL_REASON]
    state.next_information_needed = [INTERNAL_REASON]
    return case


def test_valid_same_intent_planner_followup_is_accepted():
    case = ambiguous_flank_case()
    proposal = {
        "status": "clarification_needed",
        "question": FLANK_QUESTION,
        "intent": "flank_pain",
        "reason": "需要區分目前官方候選證據",
        "answered_intent": None,
        "answer_source_text": None,
        "answer_status": None,
        "answer_confidence": None,
    }
    with patch(
        "app.services.conversation_service.complete_runtime_json",
        new=AsyncMock(return_value=json.dumps(proposal, ensure_ascii=False)),
    ):
        suggestion = asyncio.run(request_clarification(case, ["目前沒有補充"]))

    advance_conversation(
        case,
        suggestion,
        user_sources=["目前沒有補充"],
        current_extractions=[],
    )

    assert case.triage.next_question == FLANK_QUESTION
    assert case.conversation_state.pending_clarification_intent == "flank_pain"
    assert case.conversation_state.department_status == "ambiguous"
    assert case.conversation_state.next_information_needed == []


def test_invalid_provider_followup_uses_neutral_nonleaking_retry():
    case = ambiguous_flank_case()
    proposal = {
        "status": "clarification_needed",
        "question": "目前科別仍不確定，可以說明嗎？",
        "intent": "wrong_intent",
        "reason": INTERNAL_REASON,
    }
    with patch(
        "app.services.conversation_service.complete_runtime_json",
        new=AsyncMock(return_value=json.dumps(proposal, ensure_ascii=False)),
    ):
        suggestion = asyncio.run(request_clarification(case, [SECOND_USER_TEXT]))

    assert suggestion is not None
    assert suggestion.question is None
    advance_conversation(
        case,
        suggestion,
        user_sources=[SECOND_USER_TEXT],
        current_extractions=[case.semantic_extractions[-1]],
    )

    question = case.triage.next_question or ""
    assert question == "可以再補充和剛才問題相關的症狀細節嗎？"
    assert INTERNAL_REASON not in question
    assert "上一個澄清面向仍需確認" not in question
    assert "科別仍不確定" not in question
    assert case.conversation_state.next_information_needed == []
    assert case.conversation_state.department_status == "ambiguous"


def test_duplicate_provider_question_uses_nonduplicate_fallback():
    case = ambiguous_flank_case()
    case.history_records.append(Message(role="assistant", content=FLANK_QUESTION))
    suggestion = ClarificationSuggestion(
        "clarification_needed",
        FLANK_QUESTION,
        "flank_pain",
        "仍需同一面向",
    )

    advance_conversation(case, suggestion, user_sources=["不太確定"], current_extractions=[])

    assert case.triage.next_question != FLANK_QUESTION
    assert case.triage.next_question == "可以再補充和剛才問題相關的症狀細節嗎？"
    assert case.conversation_state.pending_clarification_intent == "flank_pain"


@pytest.mark.parametrize("answer_status", ["partial", "unclear"])
def test_partial_or_unclear_answer_keeps_pending_intent(answer_status):
    case = ambiguous_flank_case()
    text = "我不太確定"
    follow_up = "疼痛的位置比較靠近腰側，還是目前完全沒有這種疼痛？"
    suggestion = ClarificationSuggestion(
        "clarification_needed",
        follow_up,
        "flank_pain",
        "同一面向仍需釐清",
        "flank_pain",
        text,
        answer_status,
        0.9,
    )

    advance_conversation(case, suggestion, user_sources=[text], current_extractions=[])

    assert case.conversation_state.pending_clarification_intent == "flank_pain"
    assert case.conversation_state.clarification_evidence == {}
    assert case.triage.next_question == follow_up


def test_grounded_answer_clears_pending_without_loosening_grounding():
    case = ambiguous_flank_case()
    text = "腰側完全不會痛"
    suggestion = ClarificationSuggestion(
        "sufficient",
        None,
        None,
        "上一題已有明確回答",
        "flank_pain",
        "完全不會痛",
        "answered",
        0.95,
    )

    assert capture_pending_answer(case, suggestion, [text]) is True
    assert case.conversation_state.pending_clarification_intent is None
    assert case.conversation_state.clarification_evidence["flank_pain"] == "完全不會痛"
    assert case.conversation_state.next_information_needed == []


@pytest.mark.parametrize(
    "internal_reason",
    [
        "已記錄本輪新資訊，上一個澄清面向仍需確認",
        "上一個澄清問題尚未可靠回答",
        "目前仍需更多可靠的症狀描述",
    ],
)
def test_internal_workflow_reasons_never_enter_patient_information_need(internal_reason):
    case = ambiguous_flank_case()
    case.conversation_state.next_information_needed = [internal_reason]

    advance_conversation(case, None, user_sources=["我不太確定"], current_extractions=[])

    assert case.conversation_state.next_information_needed == []
    assert case.conversation_state.uncertainty_reasons
    assert internal_reason not in (case.triage.next_question or "")


def test_hard_turn_cap_still_stops_without_selecting_candidate():
    case = ambiguous_flank_case()
    case.conversation_state.turn_count = 7

    advance_conversation(case, None, user_sources=["還是不太確定"], current_extractions=[])

    assert case.conversation_state.turn_count == 8
    assert case.conversation_state.clarification_status == "unresolved"
    assert case.conversation_state.department_status == "ambiguous"
    assert case.department_result is None
    assert case.triage.next_question is None
