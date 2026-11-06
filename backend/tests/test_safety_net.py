"""Step 11 safety net: one turn-8 choice and the turn-20 family medicine cap; all AI and SQL are mocked."""
import json
from unittest.mock import patch

import pytest

from app.config import get_settings
from app.schemas import DepartmentResult, DialogueActInterpretation, Message, TriageCase
from app.services import conversation_service as cs, department_reasoning_service as reasoning
from app.services import rag_triage_adapter as adapter
from app.services.case_store import save_case
from app.services.rule_engine import QUESTION_TEXTS, RED_FLAG_QUESTION_KEY
from test_phase7_natural_safety import runtime, respond, evidence, pending, route

SETTINGS = get_settings()
ASK, CAP, FAMILY = SETTINGS.safety_net_ask_turn, SETTINGS.hard_turn_cap, SETTINGS.family_medicine_dept_id
ACTIVE = [dict(dept_id=FAMILY, parent_dept="內科", child_dept="家庭醫學科(一般門診/戒菸)")]
STALL = "還是頭暈"


def ambiguous_case(turns, *, safety_done=True, case_id="safety-net-test"):
    case = TriageCase(case_id=case_id)
    case.patient_input.symptom = "頭暈"
    case.patient_input.red_flags_checked = safety_done
    case.patient_input.red_flags_status = "negative" if safety_done else "not_checked"
    case.history_records = [Message(role="user", content="頭暈"), Message(role="user", content="昨天開始")]
    state = case.conversation_state
    state.turn_count = turns
    state.clarification_status = "clarifying" if safety_done else "safety_check"
    state.department_status = "ambiguous"
    state.clarification_evidence = {"onset_time": "昨天開始"}
    # Time was already answered; these tests cover the safety net, not 6-H availability.
    state.availability_status = "answered"
    save_case(case)
    return case


def post(client, case, text=None, active=ACTIVE, **extra):
    body = dict(case_id=case.case_id, **extra)
    if text is not None:
        body["message"] = text
    with patch.object(reasoning, "fetch_active_departments", return_value=active):
        return client.post("/chat", json=body).json()


def stall(client, interpreter, case):
    respond(interpreter, [evidence("symptom", "頭暈", "頭暈")])
    return post(client, case, STALL)


def choice(interpreter, source, value, *, status="answered", confidence=0.95, extractions=()):
    respond(interpreter, list(extractions), answer=dict(
        answered_intent=cs.SAFETY_NET_PENDING_INTENT, answer_status=status, answer_source_text=source,
        answer_confidence=confidence, answer_choice=value))


def assert_family_medicine_confirmation(result, reason):
    state = result["conversation_state"]
    assert state["clue_decision"]["kind"] == "family_medicine"
    assert state["clue_decision"]["reason"] == reason
    assert result["department_result"]["dept_id"] == FAMILY
    assert state["stage"] == "waiting_confirmation" and state["awaiting_confirmation"] is True
    assert state["confirmed"] is False
    assert result["reply"].startswith(SETTINGS.family_medicine_fallback_message)


def test_turn_eight_asks_once_and_continue_runs_to_the_turn_twenty_family_medicine(runtime):
    client, interpreter, _, _ = runtime
    case = ambiguous_case(ASK - 1)
    asked = stall(client, interpreter, case)
    assert asked["conversation_state"]["turn_count"] == ASK
    assert asked["next_question"] == cs.SAFETY_NET_QUESTION
    assert asked["conversation_state"]["reply_source"] == "SAFETY_NET_QUESTION"
    assert asked["conversation_state"]["pending_clarification_intent"] == cs.SAFETY_NET_PENDING_INTENT

    choice(interpreter, "繼續描述", "continue")
    result = post(client, case, "我想繼續描述")
    assert result["conversation_state"]["safety_net_choice"] == "continue"
    assert result["conversation_state"]["pending_clarification_intent"] != cs.SAFETY_NET_PENDING_INTENT
    for _ in range(ASK + 2, CAP):
        result = stall(client, interpreter, case)
        assert result["next_question"] != cs.SAFETY_NET_QUESTION
        assert result["conversation_state"]["clarification_status"] == "clarifying"
    assert result["conversation_state"]["turn_count"] == CAP - 1

    final = stall(client, interpreter, case)
    assert final["conversation_state"]["turn_count"] == CAP
    assert_family_medicine_confirmation(final, "safety_net_turn_cap")


def test_choosing_family_medicine_asks_for_confirmation_and_keeps_new_symptoms(runtime):
    client, interpreter, _, department = runtime
    case = ambiguous_case(ASK - 1)
    stall(client, interpreter, case)
    calls = department.await_count
    choice(interpreter, "先掛家醫科", "family_medicine", extractions=[evidence("symptom", "頭痛", "頭痛")])
    rescreen = post(client, case, "先掛家醫科，另外有點頭痛")
    # 「頭痛」 is first said after the safety screen, so the screen is asked again first
    # (test_safety_gaps.py); the family medicine choice is kept for the turn after it.
    assert rescreen["conversation_state"]["reply_source"] == "SAFETY_RESCREEN"
    assert rescreen["conversation_state"]["safety_net_choice"] == "family_medicine"
    respond(interpreter, answer=pending("都沒有"))
    result = post(client, case, "都沒有")
    assert_family_medicine_confirmation(result, "safety_net_choice")
    assert department.await_count == calls + 1  # Departments are recomputed once the screen is done.
    assert any(item["normalized_value"] == "頭痛" for item in result["triage_case"]["semantic_extractions"])

    confirmed = post(client, case, confirmed=True)
    assert confirmed["conversation_state"]["stage"] == "recommending"
    assert confirmed["conversation_state"]["confirmed"] is True


def test_family_medicine_choice_without_exactly_one_live_row_invents_nothing(runtime):
    client, interpreter, _, _ = runtime
    case = ambiguous_case(ASK - 1)
    stall(client, interpreter, case)
    choice(interpreter, "先掛家醫科", "family_medicine")
    result = post(client, case, "先掛家醫科", active=ACTIVE * 2)
    assert result["conversation_state"]["clarification_status"] == "unresolved"
    assert result["conversation_state"]["clue_decision"]["kind"] == "family_medicine_unavailable"
    assert result["reply"] == cs.FAMILY_MEDICINE_UNAVAILABLE_REPLY
    assert result["department_result"] is None


@pytest.mark.parametrize("source,value,status,confidence", [
    ("今天天氣很好", None, "unclear", 0.9),
    ("先掛家醫科", "family_medicine", "answered", 0.2),
    ("先掛家醫科", "family_medicine", "partial", 0.9),
])
def test_unjudgeable_reply_continues_and_is_never_asked_again(runtime, source, value, status, confidence):
    client, interpreter, _, _ = runtime
    case = ambiguous_case(ASK - 1)
    stall(client, interpreter, case)
    choice(interpreter, source, value, status=status, confidence=confidence)
    result = post(client, case, source)
    state = result["conversation_state"]
    assert state["safety_net_choice"] == "continue"
    assert state["clue_decision"] is None and result["department_result"] is None
    assert result["next_question"] != cs.SAFETY_NET_QUESTION
    assert stall(client, interpreter, case)["next_question"] != cs.SAFETY_NET_QUESTION


def test_open_safety_screen_never_asks_or_routes_to_family_medicine(runtime):
    client, interpreter, _, _ = runtime
    case = ambiguous_case(ASK - 1, safety_done=False)
    result = stall(client, interpreter, case)
    assert result["next_question"] != cs.SAFETY_NET_QUESTION
    assert result["conversation_state"]["safety_net_asked"] is False

    case = ambiguous_case(CAP - 1, safety_done=False, case_id="safety-net-open-screen")
    result = stall(client, interpreter, case)
    assert result["conversation_state"]["clarification_status"] == "unresolved"
    assert result["reply"] == cs.SAFETY_UNRESOLVED_REPLY
    assert result["conversation_state"]["reply_source"] == "SAFETY_UNRESOLVED_STOP"
    assert result["department_result"] is None and result["conversation_state"]["clue_decision"] is None


def test_urgent_symptom_on_turn_eight_takes_the_urgent_path(runtime):
    client, interpreter, _, _ = runtime
    case = ambiguous_case(ASK - 1)
    respond(interpreter, [evidence("symptom", "眼睛灼傷", "眼睛")], ttas=[dict(
        field="chemical_eye_injury", value=True, semantic_status="available", confidence=0.95,
        source_text="化學藥劑濺到我的眼睛")])
    result = post(client, case, "化學藥劑濺到我的眼睛")
    assert result["conversation_state"]["clarification_status"] == "urgent"
    assert result["conversation_state"]["safety_net_asked"] is False
    assert result["reply"] != cs.SAFETY_NET_QUESTION


def test_pending_availability_question_is_not_interrupted(runtime):
    client, interpreter, _, _ = runtime
    case = ambiguous_case(ASK - 1)
    state = case.conversation_state
    case.department_result = DepartmentResult(dept_id=1, parentDept="測試", childDept="測試", confidence=0.99)
    state.department_status = "resolved"
    state.availability_status = "asked"
    state.availability_date_attempts = state.availability_session_attempts = 1
    state.availability_pending_dimensions = ["date", "session"]
    state.pending_clarification_intent = cs.AVAILABILITY_PENDING_INTENT
    state.pending_clarification_question = cs.AVAILABILITY_QUESTION
    save_case(case)
    respond(interpreter, answer=dict(answered_intent=cs.AVAILABILITY_PENDING_INTENT, answer_status="answered",
                                    answer_source_text="都可以", answer_confidence=0.95))
    result = post(client, case, "都可以")
    assert result["conversation_state"]["turn_count"] == ASK
    assert result["conversation_state"]["safety_net_asked"] is False
    assert result["conversation_state"]["stage"] == "waiting_confirmation"
    assert result["reply"] != cs.SAFETY_NET_QUESTION


def test_revision_resets_turns_and_the_safety_net_state():
    case = ambiguous_case(ASK + 3)
    case.conversation_state.safety_net_asked = True
    case.conversation_state.safety_net_choice = "continue"
    route._begin_revision(case)
    assert case.conversation_state.turn_count == 0
    assert case.conversation_state.safety_net_asked is False
    assert case.conversation_state.safety_net_choice is None


def test_old_snapshots_load_without_safety_net_fields():
    payload = ambiguous_case(3).model_dump()
    for key in ("safety_net_asked", "safety_net_choice"):
        payload["conversation_state"].pop(key)
    state = TriageCase.model_validate(payload).conversation_state
    assert state.safety_net_asked is False and state.safety_net_choice is None


def test_turn_interpreter_choice_is_kept_only_for_the_safety_net_question():
    value = dict(answered_intent=cs.SAFETY_NET_PENDING_INTENT, answer_status="answered",
                 answer_source_text="先掛家醫科", answer_confidence=0.9, answer_choice="family_medicine")
    parsed = adapter._pending_answer_from_ai(value, pending_intent=cs.SAFETY_NET_PENDING_INTENT,
                                             user_sources=["先掛家醫科"])
    assert parsed.answer_choice == "family_medicine"
    assert adapter._pending_answer_from_ai(dict(value, answer_choice="maybe"), pending_intent=cs.SAFETY_NET_PENDING_INTENT,
                                           user_sources=["先掛家醫科"]) is None
    other = adapter._pending_answer_from_ai(dict(value, answered_intent="clarify_onset"), pending_intent="clarify_onset",
                                            user_sources=["先掛家醫科"])
    assert other.answer_choice is None


def test_prior_answer_reference_cannot_answer_the_safety_net_question():
    act = DialogueActInterpretation(act="claims_already_answered", source_text="我說過了", confidence=0.9)
    reference = dict(answered_intent=cs.SAFETY_NET_PENDING_INTENT, answer_source_text="頭暈", answer_confidence=0.9)
    assert cs.validate_prior_answer_reference(reference, act, cs.SAFETY_NET_PENDING_INTENT, (), ["頭暈"]) is None


def test_declining_further_questions_with_no_candidate_department_routes_to_family_medicine(runtime):
    client, interpreter, _, _ = runtime
    case = ambiguous_case(3)
    respond(interpreter)
    payload = json.loads(interpreter.return_value)
    payload["dialogue_act"] = dict(act="decline_further_questions", source_text="我說不出更多了", confidence=0.95)
    interpreter.return_value = json.dumps(payload, ensure_ascii=False)
    result = post(client, case, "我說不出更多了")
    assert_family_medicine_confirmation(result, "user_declined_no_candidates")


def test_declining_before_the_safety_screen_is_complete_never_routes_to_family_medicine(runtime):
    client, interpreter, _, _ = runtime
    case = ambiguous_case(3, safety_done=False, case_id="safety-net-decline-open-screen")
    respond(interpreter)
    payload = json.loads(interpreter.return_value)
    payload["dialogue_act"] = dict(act="decline_further_questions", source_text="不想回答", confidence=0.95)
    interpreter.return_value = json.dumps(payload, ensure_ascii=False)
    result = post(client, case, "不想回答")
    assert result["conversation_state"]["clue_decision"] is None and result["department_result"] is None


def test_declining_routes_to_family_medicine_even_when_the_symptom_is_not_verbatim(runtime):
    # The patient said 刺刺的有點痛; the case holds the normalized 子宮刺痛, which is not in any message.
    client, interpreter, _, _ = runtime
    case = ambiguous_case(10, case_id="safety-net-decline-normalized")
    case.patient_input.symptom = "子宮刺痛"
    case.conversation_state.safety_net_asked = True
    case.conversation_state.safety_net_choice = "continue"
    save_case(case)
    respond(interpreter)
    payload = json.loads(interpreter.return_value)
    payload["dialogue_act"] = dict(act="decline_further_questions", source_text="我要先掛號", confidence=0.95)
    interpreter.return_value = json.dumps(payload, ensure_ascii=False)
    result = post(client, case, "我說我要先掛號")
    assert_family_medicine_confirmation(result, "user_declined_no_candidates")


def test_stuck_turn_after_choosing_continue_asks_an_open_question_not_a_retry(runtime):
    client, interpreter, planner, _ = runtime
    case = ambiguous_case(ASK + 1)
    case.conversation_state.safety_net_asked = True
    case.conversation_state.safety_net_choice = "continue"
    save_case(case)
    planner.return_value = "{broken"
    result = stall(client, interpreter, case)
    assert result["next_question"] == cs.STUCK_OPEN_QUESTION
    assert result["conversation_state"]["reply_source"] == "STUCK_FALLBACK"


def test_registration_requests_are_described_to_the_turn_interpreter():
    prompt = adapter._build_symptom_collection_prompt(ambiguous_case(3, case_id="safety-net-prompt"), ["我要先掛號"])
    assert "「先決定掛號」" in prompt and "「我要先掛號」" in prompt


def reasoning_outcome(department, *, reason=None, candidates=()):
    """Mimic one department reasoning turn: its uncertainty reason and clue candidates."""
    async def outcome(case, *args, **kwargs):
        state = case.conversation_state
        state.department_uncertainty_reason = reason
        state.department_clue_candidates = list(candidates)
    department.side_effect = outcome


def test_no_candidate_at_all_asks_on_turn_three_and_only_once(runtime):
    # 2026-11-04: 「舌頭痛」「屁股疼」 matched nothing in the knowledge base and were asked about for eight turns.
    client, interpreter, _, department = runtime
    reasoning_outcome(department, reason=cs.NO_OFFICIAL_EVIDENCE_REASON)
    case = ambiguous_case(SETTINGS.safety_net_no_candidate_turn - 1)
    asked = stall(client, interpreter, case)
    assert asked["conversation_state"]["turn_count"] == SETTINGS.safety_net_no_candidate_turn
    assert asked["next_question"] == cs.SAFETY_NET_QUESTION

    choice(interpreter, "繼續描述", "continue")
    post(client, case, "我想繼續描述")
    for _ in range(SETTINGS.safety_net_no_candidate_turn + 2, ASK + 2):
        assert stall(client, interpreter, case)["next_question"] != cs.SAFETY_NET_QUESTION


def test_family_medicine_as_the_only_candidate_also_asks_on_turn_three(runtime):
    client, interpreter, _, department = runtime
    reasoning_outcome(department, candidates=[FAMILY])
    case = ambiguous_case(SETTINGS.safety_net_no_candidate_turn - 1)
    assert stall(client, interpreter, case)["next_question"] == cs.SAFETY_NET_QUESTION


@pytest.mark.parametrize("reason,candidates", [
    (None, [1274, FAMILY]),  # candidates that do not converge still wait for turn eight
    ("正式科別主資料目前無法取得", []),  # SQL unavailable says nothing about the knowledge base
    ("科別證據比對暫時無法完成", []),  # neither does a failed reasoning turn
])
def test_turn_three_is_only_for_a_knowledge_base_that_matched_nothing(runtime, reason, candidates):
    client, interpreter, _, department = runtime
    reasoning_outcome(department, reason=reason, candidates=candidates)
    case = ambiguous_case(SETTINGS.safety_net_no_candidate_turn - 1)
    result = stall(client, interpreter, case)
    assert result["conversation_state"]["turn_count"] == SETTINGS.safety_net_no_candidate_turn
    assert result["next_question"] != cs.SAFETY_NET_QUESTION
