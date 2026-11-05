"""Safety screen turns where the patient keeps describing symptoms instead of answering."""
import json

from app.schemas import TriageCase
from app.services import conversation_service as cs
from app.services.case_store import save_case
from test_phase7_natural_safety import runtime, evidence, pending, respond


CHECKLIST = cs.QUESTION_TEXTS[cs.RED_FLAG_QUESTION_KEY]


def _safety_case(case_id):
    case = TriageCase(case_id=case_id)
    case.patient_input.symptom = "喉嚨痛"
    cs._enter_safety_check(case)
    save_case(case)
    return case


def _off_topic(interpreter, text):
    interpreter.side_effect = None
    interpreter.return_value = json.dumps(dict(semantic_extractions=[], pending_answer=None,
        ttas_evidence=[], time_preferences=[], prior_answer_reference=None, department_preference=None,
        dialogue_act=dict(act="off_topic", source_text=text, confidence=0.98)))


def _post(client, case, text):
    return client.post("/chat", json={"case_id": case.case_id, "message": text}).json()


def test_added_symptoms_do_not_count_and_are_acknowledged(runtime):
    client, interpreter, planner, department = runtime
    case = _safety_case("safety-added-ack")
    for text, spans, added in [("我喉嚨也有點痛，還會咳嗽", ["喉嚨也有點痛", "還會咳嗽"], ["咳嗽"]),
                               ("咳到晚上都睡不好，還流鼻水", ["咳到晚上都睡不好", "流鼻水"], ["流鼻水"])]:
        respond(interpreter, [evidence("symptom", "咳嗽", spans[0]),
                              evidence("accompanying_symptoms", added, spans[1])])
        result = _post(client, case, text)
        state = result["conversation_state"]
        assert state["safety_answer_attempts"] == 0
        assert state["clarification_status"] == "safety_check"
        assert state["safety_answer_validation"] == "symptom_added"
        assert result["reply"] == f"{cs.safety_symptom_acknowledgement(spans)}\n{CHECKLIST}"
    # The Backend owns the acknowledgement; neither the planner nor department reasoning runs.
    planner.assert_not_awaited()
    department.assert_not_awaited()
    respond(interpreter, answer=pending("沒有"))
    result = _post(client, case, "沒有")
    assert result["triage_case"]["patient_input"]["red_flags_checked"] is True


def test_off_topic_counts_warns_on_last_retry_and_stops(runtime):
    client, interpreter, planner, department = runtime
    case = _safety_case("safety-off-topic")
    planner.return_value = "{}"
    replies = []
    for text in ["今天天氣好冷喔", "你是機器人嗎", "哈哈"]:
        _off_topic(interpreter, text)
        replies.append(_post(client, case, text))
    assert [r["conversation_state"]["safety_answer_attempts"] for r in replies] == [1, 2, 3]
    assert replies[0]["reply"] == f"{cs.SAFETY_RETRY_REPLY}\n{CHECKLIST}"
    assert replies[1]["reply"] == f"{cs.SAFETY_LAST_ATTEMPT_REPLY}\n{CHECKLIST}"
    assert replies[2]["reply"] == cs.SAFETY_UNRESOLVED_REPLY
    assert replies[2]["conversation_state"]["reply_source"] == "SAFETY_UNRESOLVED_STOP"
    # The stop must not read as an emergency to Android's 119 check.
    assert "119" not in cs.SAFETY_UNRESOLVED_REPLY
    department.assert_not_awaited()


def test_mixed_turns_count_only_unclear_answers(runtime):
    client, interpreter, planner, department = runtime
    case = _safety_case("safety-mixed")
    planner.return_value = "{}"
    respond(interpreter, [evidence("symptom", "腹痛", "肚子也痛")])
    assert _post(client, case, "我肚子也痛")["conversation_state"]["safety_answer_attempts"] == 0
    respond(interpreter, answer=pending("不知道欸", "unclear", "uncertain"))
    assert _post(client, case, "不知道欸")["conversation_state"]["safety_answer_attempts"] == 1
    respond(interpreter, [evidence("accompanying_symptoms", ["腹瀉"], "拉肚子")])
    result = _post(client, case, "還有拉肚子")
    assert result["conversation_state"]["safety_answer_attempts"] == 1
    assert result["reply"].startswith("已記下您說的「拉肚子」")
    respond(interpreter, answer=pending("沒有"))
    assert _post(client, case, "沒有")["triage_case"]["patient_input"]["red_flags_checked"] is True


def test_added_symptom_with_a_full_answer_still_completes_the_screen(runtime):
    client, interpreter, planner, department = runtime
    case = _safety_case("safety-added-and-answered")
    text = "喉嚨痛，其他都沒有"
    respond(interpreter, [evidence("symptom", "喉嚨痛", "喉嚨痛")], pending("其他都沒有"))
    interpreter.return_value = interpreter.return_value.replace('"answer"', '"new_information"')
    result = _post(client, case, text)
    assert result["triage_case"]["patient_input"]["red_flags_checked"] is True
    assert result["conversation_state"]["safety_answer_validation"] == "accepted_complete_negative"


def test_added_symptoms_stay_bounded_by_the_hard_turn_cap(runtime):
    client, interpreter, planner, department = runtime
    case = _safety_case("safety-added-cap")
    case.conversation_state.turn_count = 19
    save_case(case)
    respond(interpreter, [evidence("symptom", "咳嗽", "一直咳嗽")])
    result = _post(client, case, "一直咳嗽")
    assert result["conversation_state"]["safety_answer_attempts"] == 0
    assert result["conversation_state"]["reply_source"] == "SAFETY_UNRESOLVED_STOP"
    assert result["reply"] == cs.SAFETY_UNRESOLVED_REPLY
