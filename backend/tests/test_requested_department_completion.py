"""A requested department completes without planner questions once the safety screen is done.

Real 310 conversation (2026-11-03): 「我的子宮很痛」→ safety → 「我想要直接掛婦產科」→ time →
confirmation → 我想修改 → the planner then asked 「下週二上午還是下週三上午？」 and pain severity,
and 「直接讓我掛號」 had no effect.
"""
from __future__ import annotations

import importlib
import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import DepartmentResult, DialogueActInterpretation, Message, SemanticExtraction, TriageCase, VisitType
from app.services import rag_triage_adapter
from app.services.case_store import save_case
from app.services.conversation_service import REVISION_DETAIL_QUESTION, ClarificationSuggestion
from app.services.time_preference_service import validate_and_resolve_time_preferences

chat_route = importlib.import_module("app.routes.chat")
SETTINGS = SimpleNamespace(cerebras_api_key="test-key")
GYN = DepartmentResult(dept_id=1260, parentDept="婦女醫學部", childDept="婦產科", confidence=0.9)


def confirmed_case(case_id: str) -> TriageCase:
    case = TriageCase(case_id=case_id, visit_type=VisitType.INITIAL)
    case.patient_input.symptom = "痛"
    case.patient_input.body_part = "子宮"
    case.patient_input.red_flags_checked = True
    case.patient_input.red_flags_status = "negative"
    case.patient_input.requested_department_id = GYN.dept_id
    case.patient_input.requested_department_name = GYN.childDept
    case.history_records = [Message(role="user", content="我的子宮很痛"), Message(role="user", content="都沒有"),
                            Message(role="user", content="我想要直接掛婦產科")]
    case.semantic_extractions = [SemanticExtraction(
        field="symptom", normalized_value="痛", source_text="痛", semantic_status="available",
        assertion="present", confidence=0.95, extractor="ai",
    ), SemanticExtraction(
        field="body_part", normalized_value="子宮", source_text="子宮", semantic_status="available",
        confidence=0.95, extractor="ai",
    )]
    state = case.conversation_state
    state.clarification_evidence = {"safety_screen": "都沒有"}
    state.availability_status = "answered"
    state.awaiting_confirmation = True
    state.stage = "waiting_confirmation"
    state.department_status = "resolved"
    case.department_result = GYN
    save_case(case)
    return case


def payload(text: str, *, act: str = "correction", times=()) -> str:
    return json.dumps(dict(
        semantic_extractions=[], pending_answer=None, ttas_evidence=[], time_preferences=list(times),
        prior_answer_reference=None, department_preference=None,
        dialogue_act=dict(act=act, source_text=text, confidence=0.99),
    ), ensure_ascii=False)


async def requested(case):
    # What reason_about_departments does for a validated requested department.
    case.conversation_state.clue_decision = None
    case.conversation_state.department_status = "resolved"
    case.department_result = GYN


def turn(case_id: str, *, message: str | None = None, interpretation: str = "", revision: bool = False):
    planner = AsyncMock(return_value=ClarificationSuggestion(
        "clarification_needed", "請問疼痛的程度是輕微、適中，還是劇烈？", "pain_severity", "了解嚴重度"))
    with patch.object(chat_route, "get_settings", return_value=SETTINGS), patch.object(
        rag_triage_adapter, "get_settings", return_value=SETTINGS,
    ), patch.object(rag_triage_adapter, "complete_prompt", new=AsyncMock(return_value=interpretation)), patch.object(
        chat_route, "reason_about_departments", new=requested,
    ), patch.object(chat_route, "request_clarification", new=planner), patch.object(
        chat_route, "generate_triage_reply", new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"]),
    ), patch.object(chat_route, "ttas_requires_immediate_action", return_value=False):
        body = {"case_id": case_id, "revision_requested": True} if revision else {"case_id": case_id, "message": message}
        response = TestClient(app).post("/chat", json=body)
    assert response.status_code == 200
    return response.json(), planner


def test_revised_time_for_a_requested_department_goes_back_to_confirmation():
    case_id = confirmed_case("requested-revision").case_id
    turn(case_id, revision=True)
    text = "我想要下週三不然就下週二但是下週一不行然後這週都不可以"
    body, planner = turn(case_id, message=text, interpretation=payload(text, times=[
        dict(kind="relative_weekday", relation="prefer", priority=1, week_offset=1, weekday="wednesday",
             source_text="下週三", confidence=0.95),
        dict(kind="relative_weekday", relation="prefer", priority=2, week_offset=1, weekday="tuesday",
             source_text="下週二", confidence=0.95),
    ]))
    state = body["conversation_state"]
    assert state["stage"] == "waiting_confirmation" and state["reply_source"] == "CONFIRMATION"
    assert "婦產科" in body["reply"] and "疼痛的程度" not in body["reply"]


def test_declining_after_revision_confirms_the_requested_department():
    case_id = confirmed_case("requested-decline").case_id
    turn(case_id, revision=True)
    text = "直接讓我掛號"
    body, _ = turn(case_id, message=text, interpretation=payload(text, act="decline_further_questions"))
    state = body["conversation_state"]
    assert state["stage"] == "waiting_confirmation" and state["reply_source"] == "CONFIRMATION"


def test_time_only_correction_is_a_complete_interpretation_and_empty_one_is_not():
    text = "改成下週三"
    times = validate_and_resolve_time_preferences(
        [dict(kind="relative_weekday", relation="prefer", priority=1, week_offset=1, weekday="wednesday",
              source_text="下週三", confidence=0.95)],
        user_sources=[text], reference_date=date(2026, 11, 3),
    )

    def reject(time_preferences):
        return rag_triage_adapter._turn_interpretation_rejection_reason(
            json.loads(payload(text)), pending_intent=None, user_sources=[text],
            dialogue_act=DialogueActInterpretation(act="correction", source_text=text, confidence=0.99),
            semantic_extractions=[], pending_answer=None, ttas_evidence=[],
            time_preferences=time_preferences, department_preference=None,
        )

    assert reject(times) is None
    assert reject([]) == "correction_without_grounded_evidence"


WED_TUE = [
    dict(kind="relative_weekday", relation="prefer", priority=1, week_offset=1, weekday="wednesday",
         source_text="下週三", confidence=0.95),
    dict(kind="relative_weekday", relation="prefer", priority=2, week_offset=1, weekday="tuesday",
         source_text="下週二", confidence=0.95),
]


def test_revision_without_anything_to_change_asks_what_to_change():
    # 310：「我想要更改日期跟時段」曾回「系統剛剛沒有回應」。
    case_id = confirmed_case("revision-detail").case_id
    turn(case_id, revision=True)
    text = "我想要更改日期跟時段"
    body, planner = turn(case_id, message=text, interpretation=payload(text))
    state = body["conversation_state"]
    assert body["reply"] == REVISION_DETAIL_QUESTION and state["reply_source"] == "REVISION_DETAIL_QUESTION"
    assert state["stage"] == "collecting" and planner.await_count == 0
    text = "我想要下週三不然就下週二"
    body, _ = turn(case_id, message=text, interpretation=payload(text, times=WED_TUE))
    assert body["conversation_state"]["reply_source"] == "CONFIRMATION"
    assert "就醫時間偏好：" in body["reply"] and "週三）優先，其次" in body["reply"]


def test_revision_detail_question_is_asked_only_once():
    case_id = confirmed_case("revision-detail-once").case_id
    turn(case_id, revision=True)
    text = "我想要更改日期跟時段"
    turn(case_id, message=text, interpretation=payload(text))
    body, _ = turn(case_id, message="嗯", interpretation=payload("嗯", act="unknown"))
    assert body["reply"] != REVISION_DETAIL_QUESTION


def test_revision_with_new_content_is_not_asked_what_to_change():
    case_id = confirmed_case("revision-content").case_id
    turn(case_id, revision=True)
    text = "改成下週三不然就下週二"
    body, _ = turn(case_id, message=text, interpretation=payload(text, times=WED_TUE))
    assert body["conversation_state"]["reply_source"] == "CONFIRMATION"
