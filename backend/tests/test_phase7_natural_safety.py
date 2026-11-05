"""AI proposals are mocked; safety state, evidence and TTAS gates stay real."""
import importlib
import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import DepartmentResult
from app.services import ai_service, conversation_service, rag_triage_adapter
from app.services.case_store import get_case
from app.services.rule_engine import QUESTION_TEXTS, RED_FLAG_QUESTION_KEY

route = importlib.import_module("app.routes.chat")


def evidence(field, value, source, assertion="present"):
    return dict(field=field, normalized_value=value, source_text=source,
                semantic_status="available", confidence=0.98, assertion=assertion)


def pending(text, status="answered", assertion="absent", confidence=0.98):
    return dict(answered_intent="safety_screen", answer_status=status,
                answer_assertion=assertion, answer_source_text=text, answer_confidence=confidence)


@pytest.fixture
def runtime():
    interpreter = AsyncMock()
    planner = AsyncMock(return_value=json.dumps(dict(
        status="clarification_needed", question="這次的不舒服是在什麼情況開始的？",
        intent="clarify_onset", reason="需要了解情況", information_needs=["發作的情境"])))
    department = AsyncMock()
    with ExitStack() as stack:
        stack.enter_context(patch.object(route, "get_settings", return_value=SimpleNamespace()))
        stack.enter_context(patch.object(route, "runtime_ai_available", return_value=True))
        stack.enter_context(patch.object(rag_triage_adapter, "_ai_available", return_value=True))
        stack.enter_context(patch.object(rag_triage_adapter, "complete_prompt", interpreter))
        stack.enter_context(patch.object(conversation_service, "complete_runtime_json", planner))
        stack.enter_context(patch.object(route, "reason_about_departments", department))
        yield TestClient(app), interpreter, planner, department


def respond(interpreter, extractions=(), answer=None, ttas=()):
    interpreter.side_effect = None
    interpreter.return_value = json.dumps(dict(semantic_extractions=list(extractions),
        pending_answer=answer, ttas_evidence=list(ttas), time_preferences=[],
        prior_answer_reference=None, department_preference=None,
        dialogue_act=dict(act="answer" if answer else "new_information",
            source_text=answer["answer_source_text"] if answer else extractions[0]["source_text"] if extractions else "",
            confidence=0.98)))


@pytest.mark.parametrize("text,value,span", [
    ("我現在頭很痛", "頭痛", "頭很痛"),
    ("最近頭暈，感覺房子會轉", "眩暈", "房子會轉"),
    ("頭暈而且有耳鳴", "耳鳴", "耳鳴"),
    ("上禮拜踩空，右腳腳背腫痛", "腳背腫痛", "右腳腳背腫痛"),
    ("我想看皮膚科，手一直癢", "搔癢", "手一直癢"),
])
def test_free_text_always_interpreted_and_safety_first(runtime, text, value, span):
    client, interpreter, _, _ = runtime
    respond(interpreter, [evidence("symptom", value, span)])
    result = client.post("/chat", json=dict(visit_type="initial", message=text)).json()
    interpreter.assert_awaited_once()
    assert result["question_batch"] == []
    assert result["conversation_state"]["clarification_status"] in {"clarifying", "safety_check"}
    assert result["triage_case"]["semantic_extractions"][0]["normalized_value"] == value


def test_empty_start_creates_natural_case_even_with_batch_flag(runtime):
    client, interpreter, _, _ = runtime
    result = client.post("/chat", json=dict(visit_type="followup")).json()
    assert result["case_id"]
    assert result["question_batch"] == []
    assert result["next_question"] == QUESTION_TEXTS["symptom"]
    interpreter.assert_not_awaited()


@pytest.mark.parametrize("reply_flag", [False, True])
def test_legacy_reply_flag_does_not_disable_natural_question_planner(runtime, reply_flag):
    client, interpreter, planner, _ = runtime
    respond(interpreter, [evidence("symptom", "頭暈", "頭暈")])
    with patch.object(route, "get_settings", return_value=SimpleNamespace(
        ai_reply_generation_enabled=reply_flag,
    )):
        result = client.post("/chat", json=dict(message="頭暈")).json()
    planner.assert_awaited_once()
    assert result["reply"] == "這次的不舒服是在什麼情況開始的？"
    assert result["question_batch"] == []


def test_live_loop_restates_the_checklist_and_is_bounded_without_invented_negative(runtime):
    client, interpreter, planner, department = runtime
    respond(interpreter, [evidence("symptom", "頭痛", "頭很痛"), evidence("body_part", "頭", "頭", None)])
    first = client.post("/chat", json=dict(message="我現在頭很痛")).json()
    case_id = first["case_id"]
    department.reset_mock()
    question = "這次頭痛是突然開始的嗎？"
    planner.return_value = json.dumps(dict(status="clarification_needed", question=question,
        intent="safety_screen", reason="安全面向尚未確認", information_needs=["頭痛是否突然開始"]))
    texts = ["我覺得有點暈但是其他症狀都沒有", "你為什麼要跟我跳針", "神經"]
    for index, text in enumerate(texts):
        respond(interpreter, [evidence("accompanying_symptoms", ["頭暈"], "有點暈")] if index == 0 else [],
                pending(text, "partial" if index == 0 else "unclear", "uncertain"))
        result = client.post("/chat", json=dict(case_id=case_id, message=text)).json()
        assert not result["triage_case"]["patient_input"]["red_flags_checked"]
        assert result["triage_case"]["patient_input"]["red_flags"] == []
        assert result["conversation_state"]["confirmed"] is False
        if index < 2:
            # The planner's focused question is never shown; the full checklist is restated,
            # after a warning on the last retry.
            assert result["next_question"] == (
                QUESTION_TEXTS[RED_FLAG_QUESTION_KEY] if index == 0
                else f"{conversation_service.SAFETY_LAST_ATTEMPT_REPLY}\n{QUESTION_TEXTS[RED_FLAG_QUESTION_KEY]}"
            )
    assert interpreter.await_count == 4
    assert result["conversation_state"]["safety_answer_attempts"] == 3
    assert result["conversation_state"]["clarification_status"] == "unresolved"
    assert result["next_question"] is None
    assert "不能繼續一般掛號" in result["reply"]
    assert {e["normalized_value"][0] for e in result["triage_case"]["semantic_extractions"] if e["field"] == "accompanying_symptoms"} == {"頭暈"}
    department.assert_not_awaited()
    assert client.post("/recommend", json=dict(case_id=case_id, visit_type="initial")).status_code == 400


def test_denial_after_uncertain_answers_completes_the_screen(runtime):
    # 310 report 2026-10-30: focused single-item denials were partial and could never complete the
    # screen, so a patient denying everything still hit the three-attempt cap.
    client, interpreter, planner, department = runtime
    respond(interpreter, [evidence("symptom", "頭暈", "頭腦昏沉"), evidence("onset", "最近", "最近")])
    first = client.post("/chat", json=dict(message="我最近頭腦昏沉食慾不振")).json()
    assert first["conversation_state"]["clarification_status"] == "safety_check"
    case_id = first["case_id"]
    planner.return_value = json.dumps(dict(status="clarification_needed", question="請問最近是否有劇烈頭痛？",
        intent="safety_screen", reason="安全面向尚未確認", information_needs=["是否突然發生"],
        explanation="「意識不清」指叫不醒或認不得人，不是頭昏，「突發」指突然出現。"))
    for text in ("最近意識昏沉，因為頭暈，這算嗎", "不是突然出現的，就是這幾天都這樣"):
        respond(interpreter, answer=pending(text, "partial", "uncertain"))
        result = client.post("/chat", json=dict(case_id=case_id, message=text)).json()
        assert result["next_question"].endswith(QUESTION_TEXTS[RED_FLAG_QUESTION_KEY])
        assert result["next_question"].startswith("「意識不清」指叫不醒")
    department.reset_mock()
    respond(interpreter, answer=pending("沒有"))
    result = client.post("/chat", json=dict(case_id=case_id, message="沒有")).json()
    assert result["triage_case"]["patient_input"]["red_flags_status"] == "negative"
    assert result["conversation_state"]["clarification_status"] != "unresolved"
    department.assert_awaited_once()


def test_complete_grounded_denial_reconnects_department(runtime):
    client, interpreter, _, department = runtime
    respond(interpreter, [evidence("symptom", "頭痛", "頭很痛"), evidence("body_part", "頭", "頭", None)])
    first = client.post("/chat", json=dict(message="我現在頭很痛")).json()
    text = "都沒有，我沒有剛才提到的那些急迫症狀。"
    respond(interpreter, answer=pending(text))
    department.reset_mock()
    result = client.post("/chat", json=dict(case_id=first["case_id"], message=text)).json()
    assert result["triage_case"]["patient_input"]["red_flags_status"] == "negative"
    department.assert_awaited_once()


@pytest.mark.parametrize("failure", [TimeoutError(), ValueError("invalid response"), "{broken"])
def test_interpretation_failure_bounded_no_medical_guess(runtime, failure):
    client, interpreter, _, _ = runtime
    respond(interpreter, [evidence("symptom", "頭痛", "頭很痛"), evidence("body_part", "頭", "頭", None)])
    first = client.post("/chat", json=dict(message="我現在頭很痛")).json()
    if isinstance(failure, Exception):
        interpreter.side_effect = failure
    else:
        interpreter.return_value = failure
    for _ in range(3):
        result = client.post("/chat", json=dict(case_id=first["case_id"], message="不知道")).json()
    assert result["conversation_state"]["clarification_status"] == "unresolved"
    assert not result["triage_case"]["patient_input"]["red_flags_checked"]


def test_ungrounded_evidence_rejected(runtime):
    client, interpreter, _, _ = runtime
    respond(interpreter, [evidence("symptom", "耳鳴", "耳朵嗡嗡叫")])
    result = client.post("/chat", json=dict(message="我頭暈")).json()
    assert result["triage_case"]["semantic_extractions"] == []


def test_real_availability_requires_initialization_not_just_key():
    settings = SimpleNamespace(cerebras_api_key="not-a-real-key")
    with patch.object(ai_service, "_cerebras_client", None), patch.object(ai_service, "_initialize_cerebras", side_effect=ImportError()):
        assert ai_service.runtime_ai_available(settings) is False
    with patch.object(ai_service, "_cerebras_client", None), patch.object(ai_service, "_initialize_cerebras"):
        assert ai_service.runtime_ai_available(settings) is False


@pytest.mark.parametrize("answer", [
    pending("不知道", "unclear", "uncertain"),
    pending("沒有胸痛", "partial", "uncertain"),
    pending("都沒有", confidence=0.2),
    pending("這段不在本輪原文"),
    {**pending("都沒有"), "answered_intent": "different_question"},
])
def test_invalid_or_incomplete_safety_answer_never_clears_screen(runtime, answer):
    client, interpreter, _, _ = runtime
    respond(interpreter, [evidence("symptom", "頭痛", "頭痛"), evidence("duration", "3天", "3天", None)])
    first = client.post("/chat", json=dict(message="頭痛3天")).json()
    respond(interpreter, answer=answer)
    result = client.post("/chat", json=dict(case_id=first["case_id"], message="都沒有，沒有胸痛，不知道")).json()
    assert not result["triage_case"]["patient_input"]["red_flags_checked"]
    assert result["conversation_state"]["safety_answer_attempts"] == 1
    assert not result["conversation_state"]["confirmed"]


def test_unavailable_safety_turn_is_bounded_without_legacy_parser(runtime):
    client, interpreter, _, department = runtime
    respond(interpreter, [evidence("symptom", "頭痛", "頭痛"), evidence("duration", "3天", "3天", None)])
    first = client.post("/chat", json=dict(message="頭痛3天")).json()
    interpreter.reset_mock()
    department.reset_mock()
    from app.services import rule_engine
    assert not hasattr(rule_engine, "normalize_message")
    with patch.object(route, "runtime_ai_available", return_value=False):
        for text in ("都沒有", "不知道", "你為什麼跳針"):
            result = client.post("/chat", json=dict(case_id=first["case_id"], message=text)).json()
    assert result["conversation_state"]["clarification_status"] == "unresolved"
    assert not result["triage_case"]["patient_input"]["red_flags_checked"]
    interpreter.assert_not_awaited()
    department.assert_not_awaited()


def test_time_only_turn_uses_same_interpreter_and_preserves_relative_facts(runtime):
    client, interpreter, _, _ = runtime
    interpreter.return_value = json.dumps(dict(semantic_extractions=[], pending_answer=None,
        ttas_evidence=[], prior_answer_reference=None, department_preference=None, dialogue_act=dict(act="new_information", source_text="只能下週二下午看診", confidence=0.99), time_preferences=[
            dict(kind="relative_weekday", relation="prefer", priority=1, week_offset=1,
                 weekday="tuesday", source_text="下週二", confidence=0.99),
            dict(kind="session", relation="exclude", session="morning", priority=None,
                 source_text="只能下週二下午看診", confidence=0.99),
            dict(kind="session", relation="exclude", session="evening", priority=None,
                 source_text="只能下週二下午看診", confidence=0.99),
        ]))
    result = client.post("/chat", json=dict(message="我只能下週二下午看診")).json()
    interpreter.assert_awaited_once()
    preferences = result["triage_case"]["availability"]["time_preferences"]
    assert preferences[0]["reference_date"]
    assert preferences[0]["resolved_dates"]
    assert {p["session"] for p in preferences if p["relation"] == "exclude"} == {"morning", "evening"}
    assert not result["conversation_state"]["is_complete"]


def test_new_symptoms_and_revision_stay_grounded_across_turns(runtime):
    from app.services.department_reasoning_service import accepted_semantic_evidence
    client, interpreter, _, _ = runtime
    respond(interpreter, [evidence("symptom", "頭暈", "頭暈")])
    first = client.post("/chat", json=dict(message="我頭暈")).json()
    case_id = first["case_id"]
    respond(interpreter, [evidence("accompanying_symptoms", ["耳鳴"], "有耳鳴")])
    client.post("/chat", json=dict(case_id=case_id, message="現在還有耳鳴"))
    respond(interpreter, [evidence("accompanying_symptoms", ["耳鳴"], "沒有耳鳴", "absent")])
    client.post("/chat", json=dict(case_id=case_id, message="現在沒有耳鳴"))
    current = accepted_semantic_evidence(get_case(case_id))
    assert next(e for e in current if e["normalized_value"] == "耳鳴")["assertion"] == "absent"
    assert any(e["normalized_value"] == "頭暈" for e in current)
    assert interpreter.await_count == 3


def test_relative_days_order_across_week_boundary():
    from datetime import date
    from app.services.time_preference_service import validate_and_resolve_time_preferences
    facts = [dict(kind="date", relation="acceptable", priority=1, date_value=day,
                  source_text="後天跟大後天都可以", confidence=0.99)
             for day in ("2026-10-11", "2026-10-12")]
    result = validate_and_resolve_time_preferences(facts, user_sources=["後天跟大後天都可以"], reference_date=date(2026, 10, 9))
    assert [date.fromisoformat(p.resolved_dates[0]).weekday() for p in result] == [6, 0]
