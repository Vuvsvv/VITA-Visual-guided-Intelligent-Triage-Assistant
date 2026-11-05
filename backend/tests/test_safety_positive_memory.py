"""1B: an accepted positive safety answer is never overridden by a later denial.

2026-10-30: the patient's own "yes" (answered or partial present) stops the screen at once
with SAFETY_POSITIVE_REPLY instead of asking the remaining items.

AI proposals are mocked; safety state, TTAS evaluation and the /recommend and
/generate_script gates stay real. Replays follow benchmark c19/c20 turn by turn.
"""
import importlib
import json
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app
from app.services import conversation_service, rag_triage_adapter
from app.services.case_store import get_case
from app.services.conversation_service import SAFETY_POSITIVE_REPLY
from app.services.rule_engine import QUESTION_TEXTS, RED_FLAG_QUESTION_KEY

route = importlib.import_module("app.routes.chat")
script_route = importlib.import_module("app.routes.generate_script")


def evidence(field, value, source, assertion="present"):
    return dict(field=field, normalized_value=value, source_text=source,
                semantic_status="available", confidence=0.98, assertion=assertion)


def pending(text, status="answered", assertion="absent", confidence=0.98):
    return dict(answered_intent="safety_screen", answer_status=status,
                answer_assertion=assertion, answer_source_text=text, answer_confidence=confidence)


def safety_question(question):
    return json.dumps(dict(status="clarification_needed", question=question, intent="safety_screen",
                           reason="安全面向尚未確認", information_needs=["是否有這項急迫症狀"]))


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


def start_safety(client, interpreter, text, extractions):
    respond(interpreter, extractions)
    first = client.post("/chat", json=dict(visit_type="initial", message=text)).json()
    assert first["conversation_state"]["clarification_status"] == "safety_check"
    assert first["next_question"] == QUESTION_TEXTS[RED_FLAG_QUESTION_KEY]
    return first["case_id"]


def say(client, interpreter, planner, case_id, text, answer, *, next_question=None, extractions=(), ttas=()):
    respond(interpreter, extractions, answer, ttas)
    if next_question:
        planner.return_value = safety_question(next_question)
    return client.post("/chat", json=dict(case_id=case_id, message=text)).json()


def assert_safety_stop(client, case_id, result, department):
    state = result["conversation_state"]
    assert state["clarification_status"] == "unresolved"
    assert state["safety_positive_reported"] is True
    assert state["stage"] != "waiting_confirmation" and state["awaiting_confirmation"] is False
    assert state["clue_questions_asked"] == 0 and state["availability_status"] == "not_asked"
    assert result["reply"] == SAFETY_POSITIVE_REPLY and result["next_question"] is None
    patient = result["triage_case"]["patient_input"]
    assert patient["red_flags_checked"] is False and patient["red_flags_status"] != "negative"
    # Urgency still only comes from TTAS; the stop never asserts a TTAS level.
    assert result["triage_case"]["ttas_result"]["level_candidate"] is None
    department.assert_not_awaited()
    assert client.post("/recommend", json=dict(case_id=case_id, visit_type="initial")).status_code == 400
    with pytest.raises(HTTPException) as blocked:
        script_route._require_recommendable(get_case(case_id))
    assert blocked.value.status_code == 400


CHEST = [evidence("symptom", "胸痛", "胸口很痛"), evidence("onset", "突然", "突然")]
WEAKNESS = [evidence("symptom", "手腳無力", "手腳沒力"), evidence("body_part", "右側手腳", "右邊手腳", None),
            evidence("onset", "突然", "突然")]


@pytest.mark.parametrize("opening,extractions,text,answer", [
    ("突然胸口很痛", CHEST, "有，胸口很痛", "有，胸口很痛"),
    ("突然右邊手腳沒力", WEAKNESS, "有半邊無力", "有半邊無力"),
    # Benchmark c20 run 2: a partial denial (prompt rule 15a) that still reports one item.
    ("突然右邊手腳沒力", WEAKNESS, "有半邊無力，其他沒有。", "有半邊無力"),
], ids=["chest_pain", "hemiparesis", "partial_positive"])
def test_patient_positive_answer_stops_at_once_without_further_safety_questions(
        runtime, opening, extractions, text, answer):
    client, interpreter, planner, department = runtime
    case_id = start_safety(client, interpreter, opening, extractions)
    department.reset_mock()
    status = "partial" if answer != text else "answered"
    planner.reset_mock()
    result = say(client, interpreter, planner, case_id, text, pending(answer, status=status, assertion="present"),
                 next_question="請問是否有突發劇烈頭痛？")
    assert result["conversation_state"]["safety_answer_validation"] == "patient_reported_positive"
    assert result["conversation_state"]["reply_source"] == "SAFETY_POSITIVE_STOP"
    assert_safety_stop(client, case_id, result, department)
    planner.assert_not_awaited()
    # A later denial or any other reply stays on the stop.
    result = say(client, interpreter, planner, case_id, "沒有", pending("沒有"))
    assert result["conversation_state"]["clarification_status"] == "unresolved"
    assert result["reply"] == SAFETY_POSITIVE_REPLY


def test_ttas_immediate_level_still_takes_the_urgent_path_over_the_positive_stop(runtime):
    client, interpreter, planner, _ = runtime
    case_id = start_safety(client, interpreter, "突然胸口很痛", CHEST)
    text = "有，化學藥劑濺到我的眼睛"
    eye = dict(field="chemical_eye_injury", value=True, semantic_status="available", confidence=0.95,
               source_text="化學藥劑濺到我的眼睛")
    result = say(client, interpreter, planner, case_id, text, pending(text, assertion="present"), ttas=[eye])
    assert result["conversation_state"]["clarification_status"] == "urgent"
    assert result["reply"] != SAFETY_POSITIVE_REPLY


def test_validated_ttas_true_evidence_counts_as_positive(runtime):
    client, interpreter, planner, department = runtime
    case_id = start_safety(client, interpreter, "突然右邊手腳沒力", WEAKNESS)
    text = "說話有點不清楚"
    stroke = dict(field="stroke_symptoms", value=True, semantic_status="available", confidence=0.95, source_text=text)
    result = say(client, interpreter, planner, case_id, text, pending(text, status="partial", assertion="uncertain"),
                 next_question="請問是否有突發劇烈頭痛？", ttas=[stroke])
    # This evidence alone does not reach a TTAS immediate level, so the screen stays open.
    assert result["conversation_state"]["clarification_status"] == "safety_check"
    assert result["conversation_state"]["safety_positive_reported"] is True
    department.reset_mock()
    result = say(client, interpreter, planner, case_id, "沒有", pending("沒有"))
    assert_safety_stop(client, case_id, result, department)


def test_all_negative_screen_still_reaches_department(runtime):
    client, interpreter, planner, department = runtime
    case_id = start_safety(client, interpreter, "突然胸口很痛", CHEST)
    department.reset_mock()
    result = say(client, interpreter, planner, case_id, "沒有", pending("沒有"))
    state = result["conversation_state"]
    assert state["safety_positive_reported"] is False
    assert state["clarification_status"] != "unresolved"
    assert result["triage_case"]["patient_input"]["red_flags_status"] == "negative"
    department.assert_awaited_once()


def test_revision_after_positive_allows_a_new_screen(runtime):
    client, interpreter, planner, department = runtime
    case_id = start_safety(client, interpreter, "突然胸口很痛", CHEST)
    text = "有，胸口很痛"
    say(client, interpreter, planner, case_id, text, pending(text, assertion="present"),
        next_question="請問您現在是否有意識不清的情況？")
    stopped = say(client, interpreter, planner, case_id, "沒有", pending("沒有"))
    assert stopped["conversation_state"]["clarification_status"] == "unresolved"
    revised = client.post("/chat", json=dict(case_id=case_id, revision_requested=True)).json()
    assert revised["conversation_state"]["safety_positive_reported"] is False
    correction = "我說錯了，不是胸痛，是膝蓋痛三天"
    respond(interpreter, [evidence("symptom", "膝蓋痛", "膝蓋痛"), evidence("duration", "三天", "三天")])
    rescreen = client.post("/chat", json=dict(case_id=case_id, message=correction)).json()
    assert rescreen["conversation_state"]["clarification_status"] == "safety_check"
    assert rescreen["next_question"] == QUESTION_TEXTS[RED_FLAG_QUESTION_KEY]
    department.reset_mock()
    result = say(client, interpreter, planner, case_id, "都沒有", pending("都沒有"))
    assert result["triage_case"]["patient_input"]["red_flags_status"] == "negative"
    assert result["conversation_state"]["clarification_status"] != "unresolved"
    department.assert_awaited_once()


# Benchmark baseline turns (docs/BENCHMARK_BASELINE_20261028.md): patient text, answer status/assertion,
# and the safety follow-up the system asked next. The final denial used to complete the screen;
# now the first positive answer stops it.
C19_RUN1 = [
    ("有，胸口很痛，也喘不過氣。", "answered", "present", "請問您的呼吸困難是否已達到說不出話或感到呼吸急促的程度？"),
    ("有，已經說不出話", "answered", "present", "請問您現在是否有意識不清的情況？"),
    ("沒有", "answered", "absent", None),
]
C19_RUN2 = [
    ("有，胸口突然疼，喘不過氣，冷汗很多，沒有意識不清、流血、半邊無力或劇烈頭痛。", "answered", "present",
     "請問呼吸困難是否嚴重到無法自行呼吸或需要他人協助？"),
    ("需要他人協助。", "answered", "present", "請問胸痛是否同時伴隨心悸（感覺心跳很快或不規則）？"),
    ("沒有", "answered", "absent", None),
]
C20_RUN1 = [
    ("有半邊無力，其他沒有。", "answered", "present", "請問無力是否已影響到說話或吞嚥？"),
    ("有，說話有點不清楚，吞嚥還好。", "answered", "present", "請問是否有突發劇烈頭痛？"),
    ("沒有", "answered", "absent", None),
]
C20_RUN2 = [
    ("有，半邊無力。", "answered", "present", "請問是否有突發胸痛？"),
    ("沒有胸痛。", "partial", "absent", "請問是否有嚴重呼吸困難？"),
    ("沒有。", "answered", "absent", None),
]


@pytest.mark.parametrize("opening,extractions,turns", [
    ("突然胸口很痛", CHEST, C19_RUN1),
    ("突然胸口很痛", CHEST, C19_RUN2),
    ("突然右邊手腳沒力", WEAKNESS, C20_RUN1),
    ("突然右邊手腳沒力", WEAKNESS, C20_RUN2),
], ids=["c19_run1", "c19_run2", "c20_run1", "c20_run2"])
def test_benchmark_urgent_replays_stop_before_general_triage(runtime, opening, extractions, turns):
    client, interpreter, planner, department = runtime
    case_id = start_safety(client, interpreter, opening, extractions)
    department.reset_mock()
    text, status, assertion, follow_up = turns[0]
    result = say(client, interpreter, planner, case_id, text, pending(text, status, assertion),
                 next_question=follow_up)
    # The first "yes" already stops; the system no longer asks the follow-ups it asked in the baseline.
    assert_safety_stop(client, case_id, result, department)
    for text, status, assertion, _ in turns[1:]:
        result = say(client, interpreter, planner, case_id, text, pending(text, status, assertion))
        assert result["conversation_state"]["clarification_status"] == "unresolved"
        assert result["reply"] == SAFETY_POSITIVE_REPLY
    department.assert_not_awaited()
