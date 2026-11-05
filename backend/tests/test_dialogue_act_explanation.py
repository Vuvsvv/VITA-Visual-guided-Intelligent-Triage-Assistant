"""Validated dialogue act diagnostics and planner explanations; all AI responses are mocked."""
import ast
import asyncio
import json
import pathlib
from unittest.mock import AsyncMock, patch

import pytest

from app.config import get_settings
from app.schemas import DialogueActInterpretation, Message, TriageCase
from app.services import conversation_service as cs
from app.services.case_store import save_case
from app.services.rule_engine import QUESTION_TEXTS, RED_FLAG_QUESTION_KEY
from test_natural_chat_state_repair import (
    FOCUSED, INTENT, SAFETY_EXPLANATION, SAFETY_EXPLANATION_QUESTION, act, active_safety_case, clinical_case, plan,
    safety_explanation_plan,
)
from test_phase7_natural_safety import runtime  # noqa: F401


def interpret(interpreter, dialogue_act):
    interpreter.return_value = json.dumps(dict(
        semantic_extractions=[], pending_answer=None, ttas_evidence=[], time_preferences=[],
        prior_answer_reference=None, department_preference=None, dialogue_act=dialogue_act,
    ), ensure_ascii=False)


def test_validated_dialogue_act_is_exposed_in_chat_response(runtime):
    client, interpreter, planner, _ = runtime
    case = clinical_case()
    save_case(case)
    interpret(interpreter, act("我聽不懂", "request_explanation"))
    planner.return_value = json.dumps(plan(), ensure_ascii=False)
    result = client.post("/chat", json=dict(case_id=case.case_id, message="我聽不懂")).json()
    state = result["conversation_state"]
    assert state["last_dialogue_act"] == "request_explanation"
    assert "我聽不懂" not in json.dumps(state, ensure_ascii=False)


@pytest.mark.parametrize("dialogue_act", [
    {**act("我聽不懂", "request_explanation"), "confidence": 0.1},
    act("本輪沒有這段原文", "request_explanation"),
])
def test_low_confidence_or_ungrounded_act_is_not_recorded(runtime, dialogue_act):
    client, interpreter, planner, _ = runtime
    case = clinical_case()
    save_case(case)
    interpret(interpreter, dialogue_act)
    planner.return_value = json.dumps(plan(), ensure_ascii=False)
    result = client.post("/chat", json=dict(case_id=case.case_id, message="我聽不懂")).json()
    assert "last_dialogue_act" in result["conversation_state"]
    assert result["conversation_state"]["last_dialogue_act"] is None


def test_previous_turn_act_is_cleared_when_this_turn_has_none(runtime):
    client, _, _, _ = runtime
    case = clinical_case()
    case.conversation_state.last_dialogue_act = "request_explanation"
    save_case(case)
    result = client.post("/chat", json=dict(case_id=case.case_id)).json()
    assert result["conversation_state"]["last_dialogue_act"] is None


@pytest.mark.parametrize("dialogue_act,sources", [
    (None, ["我聽不懂"]),
    (DialogueActInterpretation(act="request_explanation", source_text="我聽不懂", confidence=0.1), ["我聽不懂"]),
    (DialogueActInterpretation(act="request_explanation", source_text="原文不存在", confidence=0.99), ["我聽不懂"]),
    (DialogueActInterpretation(act="request_explanation", source_text=" ", confidence=0.99), ["我聽不懂"]),
])
def test_validated_dialogue_act_rejects_unreliable_acts(dialogue_act, sources):
    assert cs.validated_dialogue_act(dialogue_act, sources) is None


def test_validated_dialogue_act_accepts_grounded_confident_act():
    value = DialogueActInterpretation(act="request_explanation", source_text="聽不懂", confidence=0.99)
    assert cs.validated_dialogue_act(value, ["我聽不懂"]) == "request_explanation"


EXPLANATION = "頭暈的感受是指暈的時候周圍或身體的感覺，這能幫助分辨不舒服的類型。"
MULTI_SYMPTOM_EXPLANATION = "暈的感受包括天旋地轉、眼前發黑，還有是否走路不穩，這些能幫助了解暈的類型。"


def explained(explanation=EXPLANATION, question=FOCUSED, intent=INTENT):
    return json.dumps({**plan(question, intent), "explanation": explanation}, ensure_ascii=False)


def ask_explanation(client, interpreter, case):
    save_case(case)
    interpret(interpreter, act("我聽不懂", "request_explanation"))
    return client.post("/chat", json=dict(case_id=case.case_id, message="我聽不懂，這是什麼意思？")).json()


def test_explanation_is_shown_before_question_and_pending_keeps_question_only(runtime):
    client, interpreter, planner, _ = runtime
    planner.return_value = explained()
    result = ask_explanation(client, interpreter, clinical_case())
    state = result["conversation_state"]
    assert result["reply"] == f"{EXPLANATION}\n{FOCUSED}"
    assert result["next_question"] == f"{EXPLANATION}\n{FOCUSED}"
    assert state["pending_clarification_question"] == FOCUSED
    assert result["triage_case"]["history_records"][-1]["content"] == f"{EXPLANATION}\n{FOCUSED}"
    assert state["reply_source"] == "AI_PLANNER"
    assert state["question_rejection_reason"] is None
    planner.assert_awaited_once()


def test_missing_explanation_triggers_one_repair_that_can_add_it(runtime):
    client, interpreter, planner, _ = runtime
    planner.side_effect = [json.dumps(plan(), ensure_ascii=False), explained()]
    result = ask_explanation(client, interpreter, clinical_case())
    state = result["conversation_state"]
    assert planner.await_count == 2
    assert "missing_or_invalid_explanation" in planner.await_args_list[1].args[0]
    assert result["reply"] == f"{EXPLANATION}\n{FOCUSED}"
    assert state["pending_clarification_question"] == FOCUSED
    assert state["reply_source"] == "AI_REPAIR"
    assert state["question_rejection_reason"] == "missing_or_invalid_explanation"


def test_repair_without_explanation_degrades_to_valid_question(runtime):
    client, interpreter, planner, _ = runtime
    planner.return_value = json.dumps(plan(), ensure_ascii=False)
    result = ask_explanation(client, interpreter, clinical_case())
    state = result["conversation_state"]
    assert planner.await_count == 2
    assert result["reply"] == FOCUSED
    assert state["pending_clarification_question"] == FOCUSED
    assert state["question_rejection_reason"] == "explanation_degraded"
    assert state["reply_source"] == "AI_REPAIR"
    assert result["reply"] != cs.TECHNICAL_CLARIFICATION_REPLY


def test_explanation_may_list_several_symptoms(runtime):
    client, interpreter, planner, _ = runtime
    assert cs._MULTI_DIMENSION_QUESTION.search(MULTI_SYMPTOM_EXPLANATION)
    planner.return_value = explained(MULTI_SYMPTOM_EXPLANATION)
    result = ask_explanation(client, interpreter, clinical_case())
    assert result["reply"] == f"{MULTI_SYMPTOM_EXPLANATION}\n{FOCUSED}"
    planner.assert_awaited_once()


@pytest.mark.parametrize("explanation", [
    "這和耳鼻喉科的評估有關，能幫助了解暈的類型。",
    "頭暈的感受是什麼意思呢？這能幫助分辨類型。",
    "這能幫助分辨類型；也能確認安全。",
    "從描述看來你得了眩暈症，所以需要確認感受。",
    "確診前需要先確認頭暈的感受。",
    "這需要醫師判斷，所以想先確認感受。",
    "已記錄本輪新資訊，需要再確認感受。",
    "短",
    "說明" * 76,
    None,
    ["頭暈的感受"],
])
def test_invalid_explanation_is_rejected(runtime, explanation):
    client, interpreter, planner, _ = runtime
    assert not cs._valid_explanation(explanation)
    planner.side_effect = [explained(explanation), explained()]
    result = ask_explanation(client, interpreter, clinical_case())
    assert "missing_or_invalid_explanation" in planner.await_args_list[1].args[0]
    assert result["reply"] == f"{EXPLANATION}\n{FOCUSED}"


def test_explanation_is_dropped_when_act_is_not_request_explanation(runtime):
    client, interpreter, planner, _ = runtime
    case = clinical_case()
    state = case.conversation_state
    state.pending_clarification_intent = None
    state.asked_clarification_intents = []
    save_case(case)
    interpret(interpreter, act("天旋地轉", "answer"))
    planner.return_value = explained()
    result = client.post("/chat", json=dict(case_id=case.case_id, message="天旋地轉")).json()
    assert result["conversation_state"]["last_dialogue_act"] == "answer"
    assert result["reply"] == FOCUSED
    assert EXPLANATION not in json.dumps(result, ensure_ascii=False)
    planner.assert_awaited_once()


def test_unvalidated_request_explanation_does_not_keep_explanation():
    case = clinical_case()
    unreliable = DialogueActInterpretation(act="request_explanation", source_text="原文不存在", confidence=0.99)
    with patch.object(cs, "complete_runtime_json", AsyncMock(return_value=explained())) as planner:
        suggestion = asyncio.run(cs.request_clarification(case, ["我聽不懂"], dialogue_act=unreliable))
    planner.assert_awaited_once()
    assert suggestion.question == FOCUSED
    assert suggestion.explanation is None


def test_duplicate_check_compares_question_inside_explained_reply():
    case = clinical_case()
    case.history_records.append(Message(role="assistant", content=f"{EXPLANATION}\n{FOCUSED}"))
    rejection = cs._question_rejection(case, json.loads(explained()), INTENT, False)
    assert rejection == "duplicate_question"


def test_safety_explanation_keeps_attempts_and_shows_explanation(runtime):
    client, interpreter, planner, department = runtime
    planner.return_value = json.dumps(safety_explanation_plan(), ensure_ascii=False)
    result = ask_explanation(client, interpreter, active_safety_case())
    state = result["conversation_state"]
    assert state["safety_answer_attempts"] == 1
    assert state["safety_answer_validation"] == "request_explanation"
    assert state["clarification_status"] == "safety_check"
    assert result["reply"] == f"{SAFETY_EXPLANATION}\n{CHECKLIST}"
    assert result["reply"] != cs.SAFETY_RETRY_REPLY
    assert state["reply_source"] == "AI_PLANNER"
    assert not result["triage_case"]["patient_input"]["red_flags_checked"]
    department.assert_not_awaited()


def test_safety_explanation_degrades_without_technical_fallback(runtime):
    client, interpreter, planner, _ = runtime
    planner.return_value = json.dumps({**safety_explanation_plan(), "explanation": None}, ensure_ascii=False)
    result = ask_explanation(client, interpreter, active_safety_case())
    state = result["conversation_state"]
    assert state["safety_answer_attempts"] == 1
    assert result["reply"] == CHECKLIST
    assert state["question_rejection_reason"] == "explanation_degraded"
    assert state["reply_source"] == "AI_REPAIR"
    assert result["reply"] not in {cs.SAFETY_RETRY_REPLY, cs.TECHNICAL_CLARIFICATION_REPLY}


def test_safety_explanation_restates_the_checklist_after_earlier_turns(runtime):
    client, interpreter, planner, _ = runtime
    case = active_safety_case()
    case.history_records.append(
        Message(role="assistant", content=f"{SAFETY_EXPLANATION}\n{SAFETY_EXPLANATION_QUESTION}"))
    case.history_records.append(Message(role="assistant", content=CHECKLIST))
    planner.return_value = json.dumps(safety_explanation_plan(), ensure_ascii=False)
    result = ask_explanation(client, interpreter, case)
    # The safety screen always restates the full checklist, so repeats are expected there.
    assert result["conversation_state"]["question_rejection_reason"] is None
    assert result["reply"] == f"{SAFETY_EXPLANATION}\n{CHECKLIST}"


CHECKLIST = QUESTION_TEXTS[RED_FLAG_QUESTION_KEY]
EARLIER = "你頭暈時有什麼感受？"


def restate_checklist(explanation=SAFETY_EXPLANATION):
    return json.dumps({**safety_explanation_plan(), "question": CHECKLIST, "explanation": explanation},
                      ensure_ascii=False)


def test_safety_explanation_may_restate_explained_checklist(runtime):
    client, interpreter, planner, _ = runtime
    planner.return_value = restate_checklist()
    result = ask_explanation(client, interpreter, active_safety_case())
    state = result["conversation_state"]
    assert result["reply"] == f"{SAFETY_EXPLANATION}\n{CHECKLIST}"
    assert state["safety_answer_attempts"] == 1
    assert state["safety_answer_validation"] == "request_explanation"
    assert state["reply_source"] == "AI_PLANNER"
    assert state["question_rejection_reason"] is None
    planner.assert_awaited_once()


def test_restated_checklist_without_explanation_degrades_to_the_checklist(runtime):
    client, interpreter, planner, _ = runtime
    planner.return_value = restate_checklist(None)
    result = ask_explanation(client, interpreter, active_safety_case())
    state = result["conversation_state"]
    assert planner.await_count == 2
    assert "missing_or_invalid_explanation" in planner.await_args_list[1].args[0]
    assert state["question_rejection_reason"] == "explanation_degraded"
    assert result["reply"] == CHECKLIST


def test_restated_checklist_repair_may_add_explanation(runtime):
    client, interpreter, planner, _ = runtime
    planner.side_effect = [restate_checklist(None), restate_checklist()]
    result = ask_explanation(client, interpreter, active_safety_case())
    state = result["conversation_state"]
    assert result["reply"] == f"{SAFETY_EXPLANATION}\n{CHECKLIST}"
    assert state["reply_source"] == "AI_REPAIR"
    assert state["safety_answer_attempts"] == 1


def explained_again_case():
    case = clinical_case()
    case.history_records += [Message(role="user", content="不太懂"), Message(role="assistant", content=FOCUSED)]
    case.conversation_state.pending_clarification_question = FOCUSED
    return case


def test_explained_earlier_question_is_still_duplicate(runtime):
    client, interpreter, planner, _ = runtime
    case = explained_again_case()
    assert cs._question_rejection(case, json.loads(explained(question=EARLIER)), INTENT, False,
                                  "request_explanation") == "duplicate_question"
    planner.return_value = explained(question=EARLIER)
    result = ask_explanation(client, interpreter, case)
    assert result["conversation_state"]["question_rejection_reason"] == "duplicate_question"
    assert result["reply"] == cs.SAFETY_NET_QUESTION


@pytest.mark.parametrize("dialogue_act", ["answer", None])
def test_restating_last_question_without_request_explanation_is_duplicate(dialogue_act):
    proposal = json.loads(explained(question=EARLIER))
    assert cs._question_rejection(clinical_case(), proposal, INTENT, False, dialogue_act) == "duplicate_question"


def test_clarifying_explanation_may_restate_pending_question(runtime):
    client, interpreter, planner, _ = runtime
    planner.return_value = explained()
    result = ask_explanation(client, interpreter, explained_again_case())
    state = result["conversation_state"]
    assert result["reply"] == f"{EXPLANATION}\n{FOCUSED}"
    assert state["pending_clarification_question"] == FOCUSED
    assert state["pending_clarification_intent"] == INTENT
    assert state["reply_source"] == "AI_PLANNER"
    assert state["question_rejection_reason"] is None
    planner.assert_awaited_once()


def test_repeated_safety_explanations_still_stop_at_hard_turn_cap(runtime):
    client, interpreter, planner, _ = runtime
    planner.return_value = restate_checklist()
    case = active_safety_case()
    save_case(case)
    interpret(interpreter, act("我聽不懂", "request_explanation"))
    cap = get_settings().hard_turn_cap
    for turn in range(1, cap + 1):
        result = client.post("/chat", json=dict(case_id=case.case_id, message="我聽不懂，這是什麼意思？")).json()
        state = result["conversation_state"]
        assert state["safety_answer_attempts"] == 1
        if turn < cap:
            assert state["clarification_status"] == "safety_check"
            assert result["reply"] == f"{SAFETY_EXPLANATION}\n{CHECKLIST}"
    assert state["clarification_status"] == "unresolved"
    assert state["reply_source"] == "SAFETY_UNRESOLVED_STOP"
    assert not result["triage_case"]["patient_input"]["red_flags_checked"]


SIMPLIFIED_EXPLANATION = "这些症状指的是突然出现的胸部疼痛、呼吸困难、意识改变，确认是否有这些情况可以帮助我们判断是否需要紧急处理。"
SIMPLIFIED_QUESTION = "我想确认头晕时是周围在旋转，还是快要昏倒？"
# Characters a simplified sample may share with Taiwan Big5 and so must never be in the guard.
BIG5_LOOKALIKES = "么确体种异"


def test_simplified_guard_only_holds_characters_outside_taiwan_big5():
    characters = cs._SIMPLIFIED_ONLY.pattern.strip("[]")
    assert characters
    for character in characters:
        with pytest.raises(UnicodeEncodeError):
            character.encode("cp950")
    for character in BIG5_LOOKALIKES:
        character.encode("cp950")
        assert character not in characters


def test_existing_traditional_samples_are_not_flagged():
    samples = [FOCUSED, EXPLANATION, MULTI_SYMPTOM_EXPLANATION, SAFETY_EXPLANATION, SAFETY_EXPLANATION_QUESTION,
               CHECKLIST, EARLIER, *QUESTION_TEXTS.values()]
    for path in pathlib.Path(__file__).parent.glob("test_*.py"):
        if path.name == pathlib.Path(__file__).name:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        samples += [node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert len(samples) > 500
    assert [text for text in samples if cs._SIMPLIFIED_ONLY.search(text)] == []
    for explanation in (EXPLANATION, MULTI_SYMPTOM_EXPLANATION, SAFETY_EXPLANATION):
        assert cs._valid_explanation(explanation)
    for question in (FOCUSED, SAFETY_EXPLANATION_QUESTION, CHECKLIST, EARLIER):
        assert cs._question_rejection(
            TriageCase(case_id="traditional"), json.loads(explained(question=question)), INTENT, False,
        ) is None


def test_simplified_explanation_is_repaired_into_traditional(runtime):
    client, interpreter, planner, _ = runtime
    planner.side_effect = [explained(SIMPLIFIED_EXPLANATION), explained()]
    result = ask_explanation(client, interpreter, clinical_case())
    state = result["conversation_state"]
    assert planner.await_count == 2
    assert "simplified_chinese_output" in planner.await_args_list[1].args[0]
    assert result["reply"] == f"{EXPLANATION}\n{FOCUSED}"
    assert state["question_rejection_reason"] == "simplified_chinese_output"
    assert state["reply_source"] == "AI_REPAIR"


def test_simplified_explanation_after_repair_degrades_to_traditional_question(runtime):
    client, interpreter, planner, _ = runtime
    planner.return_value = explained(SIMPLIFIED_EXPLANATION)
    result = ask_explanation(client, interpreter, clinical_case())
    state = result["conversation_state"]
    assert planner.await_count == 2
    assert result["reply"] == FOCUSED
    assert state["question_rejection_reason"] == "explanation_degraded"
    assert not cs._SIMPLIFIED_ONLY.search(json.dumps(result, ensure_ascii=False))


def test_simplified_question_is_rejected(runtime):
    client, interpreter, planner, _ = runtime
    proposal = json.loads(explained(question=SIMPLIFIED_QUESTION))
    assert cs._question_rejection(clinical_case(), proposal, INTENT, False) == "simplified_chinese_output"
    planner.return_value = explained(question=SIMPLIFIED_QUESTION)
    result = ask_explanation(client, interpreter, clinical_case())
    state = result["conversation_state"]
    assert planner.await_count == 2
    assert state["question_rejection_reason"] == "simplified_chinese_output"
    assert result["reply"] == cs.SAFETY_NET_QUESTION
