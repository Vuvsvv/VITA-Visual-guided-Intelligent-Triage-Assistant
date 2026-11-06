"""Three safety gaps found in the 2026-11-06 scan (AI proposals mocked; safety gates real).

1. A symptom first said after the safety screen (「我剛剛突然胸口很痛、喘不過氣」 in a revision)
   was never screened; it now reopens the screen once.
2. An AI timeout during the screen counted as an unclear answer, so three timeouts stopped a
   patient who said 「都沒有」 each time; it now asks to resend and counts nothing.
3. 「我想修改」 after a safety stop reopened the case; a stop is now final.
"""
from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ClueConceptRecord, ClueDecisionRecord, DepartmentResult, Message, SemanticExtraction, TriageCase, VisitType
from app.services import rag_triage_adapter
from app.services.case_store import get_case, save_case
from app.services.conversation_service import (
    SAFETY_POSITIVE_REPLY, TECHNICAL_CLARIFICATION_REPLY, ClarificationSuggestion,
)
from app.services.rule_engine import QUESTION_TEXTS, RED_FLAG_QUESTION_KEY

chat_route = importlib.import_module("app.routes.chat")
SETTINGS = SimpleNamespace(cerebras_api_key="test-key")
ENT = DepartmentResult(dept_id=1333, parentDept="耳鼻喉部", childDept="耳科", confidence=0.9)
SCREEN = QUESTION_TEXTS[RED_FLAG_QUESTION_KEY]


def extraction(field, value, source, assertion="present"):
    return SemanticExtraction(field=field, normalized_value=value, source_text=source, semantic_status="available",
                              assertion=assertion, confidence=0.95, extractor="ai")


def dizzy_case(case_id: str) -> TriageCase:
    case = TriageCase(case_id=case_id, visit_type=VisitType.INITIAL)
    case.patient_input.symptom = "頭暈"
    case.patient_input.duration = "3天"
    case.history_records = [Message(role="user", content="我頭暈三天了")]
    case.semantic_extractions = [extraction("symptom", "頭暈", "頭暈"),
                                 extraction("duration", "3天", "三天", assertion=None)]
    return case


def screened_case(case_id: str) -> TriageCase:
    case = dizzy_case(case_id)
    case.patient_input.red_flags_checked = True
    case.patient_input.red_flags_status = "negative"
    case.history_records += [Message(role="assistant", content=SCREEN), Message(role="user", content="都沒有")]
    return case


def interpretation(text, *, act="new_information", answer=None, extractions=()):
    return json.dumps(dict(
        semantic_extractions=list(extractions), pending_answer=answer, ttas_evidence=[], time_preferences=[],
        prior_answer_reference=None, department_preference=None,
        dialogue_act=dict(act=act, source_text=text, confidence=0.95),
    ), ensure_ascii=False)


def present(field, value, source):
    return dict(field=field, normalized_value=value, source_text=source, semantic_status="available",
                assertion="present", confidence=0.95)


async def decide_ent(case):
    case.conversation_state.department_status = "resolved"
    case.department_result = ENT
    case.conversation_state.clue_decision = ClueDecisionRecord(kind="decide", reason="single_candidate",
                                                               dept_id=ENT.dept_id)


def post(body, interpreter, reasoning=None):
    reasoning = reasoning or AsyncMock(side_effect=decide_ent)
    planner = AsyncMock(return_value=ClarificationSuggestion(
        "clarification_needed", "請問頭暈是天旋地轉的感覺嗎？", "clarify_vertigo", "了解頭暈",
        information_need="頭暈型態"))
    with patch.object(chat_route, "get_settings", return_value=SETTINGS), \
         patch.object(rag_triage_adapter, "get_settings", return_value=SETTINGS), \
         patch.object(rag_triage_adapter, "complete_prompt", new=interpreter), \
         patch.object(rag_triage_adapter, "active_department_candidates", return_value=[]), \
         patch.object(chat_route, "reason_about_departments", new=reasoning), \
         patch.object(chat_route, "request_clarification", new=planner):
        response = TestClient(app).post("/chat", json=body)
    assert response.status_code == 200
    return response.json(), reasoning, planner


# 1. New symptom after the screen ------------------------------------------------------------

def confirmed_case(case_id: str) -> TriageCase:
    case = screened_case(case_id)
    state = case.conversation_state
    state.availability_status = "answered"
    state.awaiting_confirmation = True
    state.stage = "waiting_confirmation"
    state.department_status = "resolved"
    state.is_complete = True
    case.department_result = ENT
    save_case(case)
    return case


def test_new_symptom_in_a_revision_reopens_the_safety_screen():
    confirmed_case("rescreen-revision")
    post({"case_id": "rescreen-revision", "revision_requested": True}, AsyncMock())
    text = "我剛剛突然胸口很痛、喘不過氣"
    body, reasoning, planner = post({"case_id": "rescreen-revision", "message": text}, AsyncMock(
        return_value=interpretation(text, extractions=[
            present("symptom", "胸痛", "胸口很痛"), present("accompanying_symptoms", ["呼吸困難"], "喘不過氣")])))
    state = body["conversation_state"]
    assert state["clarification_status"] == "safety_check" and state["stage"] == "collecting"
    assert state["reply_source"] == "SAFETY_RESCREEN"
    assert body["reply"] == (
        "已記下您說的「胸口很痛」、「喘不過氣」。繼續之前想先確認一件事，以下狀況如果都沒有，回答「沒有」就可以。\n"
        + SCREEN)
    # No department decision or planner question before the screen is answered again.
    reasoning.assert_not_awaited()
    planner.assert_not_awaited()
    assert get_case("rescreen-revision").patient_input.red_flags_checked is False


def test_reopened_screen_answered_yes_stops_and_no_continues():
    for case_id, answer_text, assertion in (("rescreen-yes", "有，胸口還在痛", "present"),
                                            ("rescreen-no", "都沒有", "absent")):
        case = screened_case(case_id)
        case.conversation_state.clarification_status = "clarifying"
        save_case(case)
        text = "對了我今天開始胸口很痛"
        post({"case_id": case_id, "message": text}, AsyncMock(return_value=interpretation(
            text, extractions=[present("symptom", "胸痛", "胸口很痛")])))
        answer = dict(answered_intent="safety_screen", answer_status="answered", answer_assertion=assertion,
                      answer_source_text=answer_text, answer_confidence=0.95)
        body, _, _ = post({"case_id": case_id, "message": answer_text}, AsyncMock(
            return_value=interpretation(answer_text, act="answer", answer=answer)))
        if assertion == "present":
            assert body["reply"] == SAFETY_POSITIVE_REPLY
            assert body["conversation_state"]["reply_source"] == "SAFETY_POSITIVE_STOP"
        else:
            assert get_case(case_id).patient_input.red_flags_checked is True
            assert body["conversation_state"]["clarification_status"] != "safety_check"


def test_restated_old_symptom_or_clue_answer_does_not_reopen_the_screen():
    case = screened_case("rescreen-not-new")
    state = case.conversation_state
    state.clarification_status = "clarifying"
    state.clue_concepts_asked = [ClueConceptRecord(concept="耳鳴", dept_id=ENT.dept_id, question_index=1)]
    state.clue_questions_asked = 1
    save_case(case)
    # 「頭暈」 was said before; 「耳鳴」 is the official concept our clue question asked.
    text = "還是頭暈，也有耳鳴"
    body, _, _ = post({"case_id": "rescreen-not-new", "message": text}, AsyncMock(return_value=interpretation(
        text, extractions=[present("symptom", "頭暈", "頭暈"), present("accompanying_symptoms", ["耳鳴"], "有耳鳴")])))
    assert body["conversation_state"]["clarification_status"] != "safety_check"
    assert get_case("rescreen-not-new").patient_input.red_flags_checked is True


def test_new_symptom_before_the_first_screen_is_not_a_rescreen():
    case = dizzy_case("rescreen-first")
    case.conversation_state.clarification_status = "clarifying"
    save_case(case)
    text = "還會想吐"
    body, _, _ = post({"case_id": "rescreen-first", "message": text}, AsyncMock(return_value=interpretation(
        text, extractions=[present("accompanying_symptoms", ["噁心"], "想吐")])))
    assert body["conversation_state"]["reply_source"] != "SAFETY_RESCREEN"
    assert body["reply"] == SCREEN


# 2. AI failure during the screen ----------------------------------------------------------------

def safety_case(case_id: str) -> TriageCase:
    case = dizzy_case(case_id)
    case.history_records.append(Message(role="assistant", content=SCREEN))
    case.conversation_state.clarification_status = "safety_check"
    case.conversation_state.last_question_key = RED_FLAG_QUESTION_KEY
    save_case(case)
    return case


def test_ai_timeouts_during_the_screen_ask_to_resend_without_counting():
    safety_case("screen-timeout")
    for _ in range(4):
        body, _, _ = post({"case_id": "screen-timeout", "message": "都沒有"}, AsyncMock(side_effect=TimeoutError()))
        state = body["conversation_state"]
        assert body["reply"] == TECHNICAL_CLARIFICATION_REPLY
        assert state["reply_source"] == "TECHNICAL_FALLBACK"
        assert state["clarification_status"] == "safety_check" and state["safety_answer_attempts"] == 0
    # The next understood 「都沒有」 completes the screen.
    answer = dict(answered_intent="safety_screen", answer_status="answered", answer_assertion="absent",
                  answer_source_text="都沒有", answer_confidence=0.95)
    post({"case_id": "screen-timeout", "message": "都沒有"},
         AsyncMock(return_value=interpretation("都沒有", act="answer", answer=answer)))
    assert get_case("screen-timeout").patient_input.red_flags_checked is True


def test_failed_repair_call_does_not_count_but_a_broken_answer_still_does():
    safety_case("screen-repair-timeout")
    broken = json.dumps({"semantic_extractions": []})
    body, _, _ = post({"case_id": "screen-repair-timeout", "message": "都沒有"},
                      AsyncMock(side_effect=[broken, TimeoutError()]))
    assert body["conversation_state"]["safety_answer_attempts"] == 0
    assert body["reply"] == TECHNICAL_CLARIFICATION_REPLY

    safety_case("screen-broken")
    body, _, _ = post({"case_id": "screen-broken", "message": "還好吧"}, AsyncMock(side_effect=[broken, broken]))
    assert body["conversation_state"]["safety_answer_attempts"] == 1


def test_resend_prompt_keeps_the_screen_question_for_the_interpreter():
    safety_case("screen-pending")
    post({"case_id": "screen-pending", "message": "都沒有"}, AsyncMock(side_effect=TimeoutError()))
    prompt = rag_triage_adapter._build_symptom_collection_prompt(get_case("screen-pending"), ["都沒有"])
    assert f'"pending_question": "{SCREEN}"' in prompt


# 3. A safety stop is final -----------------------------------------------------------------------

def test_revision_after_a_positive_screen_keeps_the_stop():
    case = dizzy_case("stop-revision")
    state = case.conversation_state
    state.clarification_status = "unresolved"
    state.safety_positive_reported = True
    state.safety_answer_attempts = 1
    state.reply_source = "SAFETY_POSITIVE_STOP"
    save_case(case)
    for body_extra in ({"revision_requested": True}, {"message": "沒有啦我說錯了，我要掛號"}):
        body, _, _ = post({"case_id": "stop-revision", **body_extra}, AsyncMock(
            return_value=interpretation("沒有啦我說錯了", act="correction")))
        assert body["reply"] == SAFETY_POSITIVE_REPLY
        assert body["conversation_state"]["clarification_status"] == "unresolved"
        assert body["conversation_state"]["reply_source"] == "SAFETY_POSITIVE_STOP"
    assert get_case("stop-revision").conversation_state.safety_positive_reported is True


def test_revision_after_an_unanswerable_screen_keeps_the_stop():
    case = dizzy_case("stop-unresolved")
    case.conversation_state.clarification_status = "unresolved"
    case.conversation_state.safety_answer_attempts = 3
    save_case(case)
    body, _, _ = post({"case_id": "stop-unresolved", "revision_requested": True}, AsyncMock())
    assert body["conversation_state"]["reply_source"] == "SAFETY_UNRESOLVED_STOP"
    assert body["conversation_state"]["clarification_status"] == "unresolved"
