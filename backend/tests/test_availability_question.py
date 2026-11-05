"""6-H: ask appointment availability after the department converged, before confirmation."""
from __future__ import annotations

import importlib
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import (
    DepartmentResult, Message, SemanticExtraction, TimePreference, TriageCase, UrgencyResult, VisitType,
)
from app.services import rag_triage_adapter
from app.services.case_store import get_case, save_case
from app.services.conversation_service import (
    AVAILABILITY_DATE_QUESTION, AVAILABILITY_PENDING_INTENT, AVAILABILITY_QUESTION, AVAILABILITY_SESSION_QUESTION,
    ClarificationSuggestion,
)

chat_route = importlib.import_module("app.routes.chat")
SETTINGS = SimpleNamespace(cerebras_api_key="test-key")


def ready_case(case_id: str) -> TriageCase:
    """Symptom, detail and safety are settled; the department resolves on the next turn."""
    case = TriageCase(case_id=case_id, visit_type=VisitType.INITIAL)
    case.patient_input.symptom = "膝蓋痛"
    case.patient_input.red_flags_checked = True
    case.patient_input.red_flags_status = "negative"
    case.history_records = [Message(role="user", content="膝蓋痛三天")]
    case.semantic_extractions = [SemanticExtraction(
        field="duration", normalized_value="3天", source_text="三天", semantic_status="available",
        confidence=0.97, extractor="ai",
    )]
    save_case(case)
    return case


def payload(text: str, *, act: str = "new_information", extractions=(), pending=None, times=()) -> str:
    return json.dumps(dict(
        semantic_extractions=list(extractions), pending_answer=pending, ttas_evidence=[],
        time_preferences=list(times), prior_answer_reference=None, department_preference=None,
        dialogue_act=dict(act=act, source_text=text, confidence=0.99),
    ), ensure_ascii=False)


def symptom(text: str) -> dict:
    return dict(field="symptom", normalized_value=text, source_text=text, semantic_status="available",
                assertion="present", confidence=0.97)


def availability_answer(source: str) -> dict:
    return dict(answered_intent=AVAILABILITY_PENDING_INTENT, answer_status="answered",
                answer_source_text=source, answer_confidence=0.95)


def next_tuesday(source: str, relation: str = "prefer") -> dict:
    return dict(kind="relative_weekday", relation=relation, priority=1 if relation != "exclude" else None,
                week_offset=1, weekday="tuesday", source_text=source, confidence=0.95)


def afternoon(source: str) -> dict:
    return dict(kind="session", relation="prefer", priority=1, session="afternoon", source_text=source,
                confidence=0.95)


async def converge(case):
    case.conversation_state.department_status = "resolved"
    case.department_result = DepartmentResult(dept_id=1298, parentDept="外科系", childDept="一般骨科", confidence=0.9)


def urgent_ttas(case):
    """Stand-in for a matched TTAS level 1 result; the real rules are covered by Phase 5 tests."""
    case.ttas_result.status = "matched"
    case.ttas_result.level_candidate = 1
    return UrgencyResult(urgency_level="level_1", warning_required=True, warning_message="請立即就醫。",
                         need_more_info=False, is_final=True)


def turn(case_id: str, message: str, interpretation: str, *, urgent: bool = False, department=converge):
    planner = AsyncMock(return_value=ClarificationSuggestion("sufficient", None, None, "資訊足夠"))
    ttas = {"new": urgent_ttas} if urgent else {"new": chat_route.apply_ttas_evaluation}
    with patch.object(chat_route, "get_settings", return_value=SETTINGS), patch.object(
        rag_triage_adapter, "get_settings", return_value=SETTINGS,
    ), patch.object(rag_triage_adapter, "complete_prompt", new=AsyncMock(return_value=interpretation)), patch.object(
        chat_route, "reason_about_departments", new=department,
    ), patch.object(chat_route, "request_clarification", new=planner), patch.object(
        chat_route, "generate_triage_reply", new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"]),
    ), patch.object(chat_route, "ttas_requires_immediate_action", return_value=urgent), patch.object(
        chat_route, "apply_ttas_evaluation", **ttas,
    ):
        response = TestClient(app).post("/chat", json={"case_id": case_id, "message": message})
    assert response.status_code == 200
    return response.json(), planner


def ask(case_id: str):
    body, _ = turn(case_id, "膝蓋痛", payload("膝蓋痛", extractions=[symptom("膝蓋痛")]))
    return body


def state_of(body):
    return body["conversation_state"]


def test_converged_department_asks_availability_before_confirmation():
    body = ask(ready_case("avail-ask").case_id)
    state = state_of(body)
    assert body["reply"] == AVAILABILITY_QUESTION and state["reply_source"] == "AVAILABILITY_QUESTION"
    assert state["pending_clarification_intent"] == AVAILABILITY_PENDING_INTENT
    assert (state["availability_status"], state["availability_question_attempts"]) == ("asked", 1)
    assert state["is_complete"] is False and state["stage"] == "collecting"
    assert body["department_result"]["dept_id"] == 1298


def test_specific_time_answer_goes_to_confirmation():
    # 6-H-1: a session-only answer now asks for the date; this answer covers both dimensions.
    case_id = ready_case("avail-specific").case_id
    ask(case_id)
    text = "下週二下午"
    body, planner = turn(case_id, text, payload(text, act="answer", pending=availability_answer(text), times=[
        next_tuesday("下週二"), afternoon("下午")]))
    state = state_of(body)
    assert state["availability_status"] == "answered" and state["stage"] == "waiting_confirmation"
    assert state["is_complete"] is True and state["reply_source"] == "CONFIRMATION"
    assert [item["session"] for item in body["triage_case"]["availability"]["time_preferences"]] == [None, "afternoon"]


def test_anytime_answer_confirms_without_time_conditions():
    case_id = ready_case("avail-anytime").case_id
    ask(case_id)
    body, _ = turn(case_id, "都可以", payload("都可以", act="answer", pending=availability_answer("都可以")))
    state = state_of(body)
    assert state["availability_status"] == "answered" and state["stage"] == "waiting_confirmation"
    availability = body["triage_case"]["availability"]
    assert availability["time_preferences"] == [] and availability["preferred_sessions"] == []


def test_anytime_answer_keeps_the_decided_department():
    # The convergence AI may judge unchanged evidence differently; the availability answer adds no symptom.
    case_id = ready_case("avail-keep-department").case_id
    ask(case_id)
    reopen = AsyncMock(side_effect=lambda case: setattr(case.conversation_state, "department_status", "ambiguous"))
    body, _ = turn(case_id, "都可以", payload("都可以", act="answer", pending=availability_answer("都可以")),
                   department=reopen)
    state = state_of(body)
    reopen.assert_not_awaited()
    assert state["department_status"] == "resolved" and body["department_result"]["dept_id"] == 1298
    assert state["reply_source"] == "CONFIRMATION" and state["stage"] == "waiting_confirmation"


def test_new_symptom_with_the_availability_answer_recomputes_the_department():
    case_id = ready_case("avail-new-symptom").case_id
    ask(case_id)
    department = AsyncMock(side_effect=converge)
    text = "都可以，另外也會頭暈"
    turn(case_id, text, payload(text, act="answer", pending=availability_answer("都可以"),
                                extractions=[symptom("頭暈")]), department=department)
    department.assert_awaited_once()


def test_time_already_given_is_not_asked_again():
    # 6-H-1: both dimensions must be known up front; a session alone now asks for the date.
    case = ready_case("avail-known")
    case.availability.preferred_days = ["星期二"]
    case.availability.preferred_sessions = ["下午"]
    save_case(case)
    state = state_of(ask(case.case_id))
    assert state["availability_status"] == "answered" and state["availability_question_attempts"] == 0
    assert state["stage"] == "waiting_confirmation" and state["reply_source"] == "CONFIRMATION"


def test_two_off_topic_answers_become_unspecified_and_confirm():
    case_id = ready_case("avail-off-topic").case_id
    ask(case_id)
    off_topic = payload("今天天氣很好", act="off_topic")
    body, planner = turn(case_id, "今天天氣很好", off_topic)
    state = state_of(body)
    assert body["reply"] == AVAILABILITY_QUESTION and state["availability_question_attempts"] == 2
    assert state["availability_status"] == "asked"
    planner.assert_not_awaited()
    body, _ = turn(case_id, "今天天氣很好", off_topic)
    state = state_of(body)
    assert state["availability_status"] == "unspecified" and state["stage"] == "waiting_confirmation"
    assert state["is_complete"] is True


def test_urgent_symptom_while_answering_time_takes_urgent_flow():
    case_id = ready_case("avail-urgent").case_id
    ask(case_id)
    text = "下午可以，但現在胸口突然很痛"
    body, _ = turn(case_id, text, payload(text, act="answer", pending=availability_answer("下午可以"),
                                          extractions=[symptom("胸口突然很痛")]), urgent=True)
    state = state_of(body)
    assert state["clarification_status"] == "urgent" and state["stage"] == "done"
    assert state["availability_status"] == "asked"


def test_revision_does_not_ask_availability_again():
    case_id = ready_case("avail-revision").case_id
    ask(case_id)
    turn(case_id, "都可以", payload("都可以", act="answer", pending=availability_answer("都可以")))
    revision = TestClient(app).post("/chat", json={"case_id": case_id, "revision_requested": True})
    assert state_of(revision.json())["availability_status"] == "answered"
    body, _ = turn(case_id, "其實是右膝痛", payload("其實是右膝痛", extractions=[symptom("右膝痛")]))
    state = state_of(body)
    assert state["availability_status"] == "answered" and state["availability_question_attempts"] == 1
    assert state["stage"] == "waiting_confirmation" and body["reply"] != AVAILABILITY_QUESTION


def test_recommend_is_blocked_while_availability_is_open():
    case_id = ready_case("avail-recommend").case_id
    ask(case_id)
    case = get_case(case_id)
    case.confirmed = True
    case.conversation_state.confirmed = True
    save_case(case)
    response = TestClient(app).post("/recommend", json={"case_id": case_id})
    assert response.status_code == 400 and "尚未完成" in response.json()["detail"]


# 6-H-1: date and session are settled as two dimensions.

def with_time(case_id: str, *preferences: dict) -> str:
    case = ready_case(case_id)
    case.availability.time_preferences = [TimePreference(**item) for item in preferences]
    save_case(case)
    return case.case_id


def only_afternoon() -> list[dict]:
    return [dict(kind="session", relation="exclude", session=value, source_text="只能下午", confidence=0.95)
            for value in ("morning", "evening")]


def test_session_only_opening_asks_only_the_date():
    case_id = with_time("avail-dim-session-first", *only_afternoon())
    body = ask(case_id)
    state = state_of(body)
    assert body["reply"] == AVAILABILITY_DATE_QUESTION and state["reply_source"] == "AVAILABILITY_QUESTION"
    assert state["pending_clarification_intent"] == AVAILABILITY_PENDING_INTENT
    assert state["availability_pending_dimensions"] == ["date"]
    assert (state["availability_date_attempts"], state["availability_session_attempts"]) == (1, 0)
    body, _ = turn(case_id, "下週二", payload("下週二", act="answer", pending=availability_answer("下週二"),
                                             times=[next_tuesday("下週二")]))
    state = state_of(body)
    assert state["availability_status"] == "answered" and state["stage"] == "waiting_confirmation"


def test_date_only_answer_asks_the_session():
    case_id = ready_case("avail-dim-date-answer").case_id
    ask(case_id)
    text = "下週二比較方便"
    body, planner = turn(case_id, text, payload(text, act="answer", pending=availability_answer(text),
                                                 times=[next_tuesday("下週二")]))
    state = state_of(body)
    assert body["reply"] == AVAILABILITY_SESSION_QUESTION and state["availability_status"] == "asked"
    assert state["availability_pending_dimensions"] == ["session"] and state["is_complete"] is False
    assert (state["availability_date_attempts"], state["availability_session_attempts"]) == (1, 2)
    body, _ = turn(case_id, "下午", payload("下午", act="answer", pending=availability_answer("下午"),
                                           times=[afternoon("下午")]))
    state = state_of(body)
    assert state["availability_status"] == "answered" and state["stage"] == "waiting_confirmation"
    assert state["is_complete"] is True


def test_anytime_answer_settles_both_dimensions_as_unrestricted():
    case_id = ready_case("avail-dim-anytime").case_id
    ask(case_id)
    body, _ = turn(case_id, "都可以", payload("都可以", act="answer", pending=availability_answer("都可以")))
    state = state_of(body)
    assert state["availability_dimension_outcomes"] == {"date": "unrestricted", "session": "unrestricted"}
    assert state["availability_status"] == "answered" and state["availability_question_attempts"] == 1


def test_excluded_date_only_asks_the_session():
    case_id = ready_case("avail-dim-exclude").case_id
    ask(case_id)
    text = "週四不行"
    body, _ = turn(case_id, text, payload(text, act="answer", pending=availability_answer(text), times=[dict(
        kind="weekday", relation="exclude", weekday="thursday", source_text="週四不行", confidence=0.95)]))
    state = state_of(body)
    assert body["reply"] == AVAILABILITY_SESSION_QUESTION and state["availability_pending_dimensions"] == ["session"]


def test_two_off_topic_session_answers_leave_session_unrestricted():
    case_id = with_time("avail-dim-session-limit", next_tuesday("下週二"))
    assert ask(case_id)["reply"] == AVAILABILITY_SESSION_QUESTION
    off_topic = payload("今天天氣很好", act="off_topic")
    body, _ = turn(case_id, "今天天氣很好", off_topic)
    state = state_of(body)
    assert body["reply"] == AVAILABILITY_SESSION_QUESTION and state["availability_session_attempts"] == 2
    body, _ = turn(case_id, "今天天氣很好", off_topic)
    state = state_of(body)
    assert state["availability_dimension_outcomes"] == {"session": "exhausted"}
    assert state["availability_status"] == "answered" and state["stage"] == "waiting_confirmation"
    assert state["availability_date_attempts"] == 0


def test_two_off_topic_full_questions_exhaust_both_dimensions():
    case_id = ready_case("avail-dim-full-limit").case_id
    ask(case_id)
    off_topic = payload("今天天氣很好", act="off_topic")
    turn(case_id, "今天天氣很好", off_topic)
    body, _ = turn(case_id, "今天天氣很好", off_topic)
    state = state_of(body)
    assert state["availability_dimension_outcomes"] == {"date": "exhausted", "session": "exhausted"}
    assert state["availability_status"] == "unspecified" and state["stage"] == "waiting_confirmation"


def test_urgent_symptom_while_asking_session_takes_urgent_flow():
    case_id = with_time("avail-dim-urgent", next_tuesday("下週二"))
    assert ask(case_id)["reply"] == AVAILABILITY_SESSION_QUESTION
    text = "下午可以，但現在胸口突然很痛"
    body, _ = turn(case_id, text, payload(text, act="answer", pending=availability_answer("下午可以"),
                                          extractions=[symptom("胸口突然很痛")]), urgent=True)
    state = state_of(body)
    assert state["clarification_status"] == "urgent" and state["stage"] == "done"


def test_revision_does_not_ask_the_settled_dimension_again():
    case_id = with_time("avail-dim-revision", next_tuesday("下週二"))
    assert ask(case_id)["reply"] == AVAILABILITY_SESSION_QUESTION
    revision = TestClient(app).post("/chat", json={"case_id": case_id, "revision_requested": True})
    assert revision.status_code == 200
    body, _ = turn(case_id, "其實是右膝痛", payload("其實是右膝痛", extractions=[symptom("右膝痛")]))
    state = state_of(body)
    assert body["reply"] == AVAILABILITY_SESSION_QUESTION and state["availability_pending_dimensions"] == ["session"]
    assert (state["availability_date_attempts"], state["availability_session_attempts"]) == (0, 2)


def test_interpreter_receives_the_asked_dimension_via_pending_context():
    case_id = with_time("avail-dim-context", next_tuesday("下週二"))
    ask(case_id)
    prompt = rag_triage_adapter._build_symptom_collection_prompt(get_case(case_id), ["下午"])
    assert re.search(r'"pending_availability_dimensions": \[\s*"session"\s*\]', prompt)
    assert f'"pending_question": "{AVAILABILITY_SESSION_QUESTION}"' in prompt
