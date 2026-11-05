"""Under-18 patients go to pediatrics first.

The AI says who the patient is (patient_profile); code validates it, reads the age and decides.
"""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.schemas import ConversationState, Message, SemanticExtraction, TriageCase
from app.services import conversation_service as conversation_flow
from app.services import department_reasoning_service as service
from app.services import pediatric_routing as pediatric
from app.services import rag_triage_adapter as adapter
from app.services.ai_reply_generator import build_department_confirmation_reply
from app.services.department_knowledge import load_department_knowledge
from app.services.rule_engine import apply_semantic_extractions, apply_user_message

_, RECORDS = load_department_knowledge()
ACTIVE = [
    dict(dept_id=record["canonical_dept_id"], parent_dept=record["canonical_parent_dept"],
         child_dept=record["canonical_child_dept"])
    for record in {item["canonical_dept_id"]: item for item in RECORDS}.values()
]
INFECTION, NOSE = 1234, 1334


def profile(subject, relation, age, source, confidence=0.95):
    return dict(subject=subject, relation_text=relation, age_text=age, source_text=source, confidence=confidence)


def state_with(subject=None, relation=None, age=None, answer=None):
    return ConversationState(patient_subject=subject, patient_relation_text=relation,
                             patient_age_text=age, patient_age_answer=answer)


@pytest.mark.parametrize(("state", "expected"), [
    (state_with("other", "女兒", "8歲"), "child"),
    (state_with("other", "兒子", "十七歲"), "child"),
    (state_with("other", "兒子", "30歲"), "adult"),
    (state_with("other", "我家寶寶"), "child"),
    (state_with("other", "女兒", "國小三年級"), "child"),
    (state_with("other", "兒子", "高二"), "child"),
    (state_with("other", "寶寶", "6個月大"), "child"),
    (state_with("self", None, "15歲"), "child"),
    (state_with("self"), "adult"),
    (state_with(), "adult"),
    (state_with("other", "我媽"), "adult"),
    (state_with("other", "小女兒"), "ask"),
    (state_with("other", "小女兒", answer="7"), "child"),
    (state_with("other", "小女兒", answer="七歲"), "child"),
    (state_with("other", "小女兒", answer="二十五歲"), "adult"),
    (state_with("other", "小女兒", answer="不知道"), "adult"),
])
def test_patient_status_is_decided_by_code_from_validated_fields(state, expected):
    assert pediatric.patient_status(state) == expected


@pytest.mark.parametrize(("value", "accepted"), [
    (profile("other", "小女兒", None, "我小女兒感冒"), True),
    (profile("self", None, None, "我是幼稚園老師"), True),
    (profile("other", "小女兒", "7歲", "我小女兒感冒"), False),   # age not in the source
    (profile("other", "小女兒", None, "我小兒子感冒"), False),    # source not said
    (profile("self", "女兒", None, "我女兒幫我掛號"), False),     # self has no relation
    (profile("other", "小女兒", None, "我小女兒感冒", 0.5), False),
    (profile("parent", "小女兒", None, "我小女兒感冒"), False),
    (dict(subject="other", source_text="我小女兒感冒", confidence=0.9), False),
])
def test_profile_must_be_grounded_and_confident(value, accepted):
    assert (pediatric.validate_patient_profile(value, ["我小女兒感冒很多天", "我是幼稚園老師，喉嚨痛"])
            is not None) == accepted


def test_same_person_keeps_an_earlier_age_and_a_new_person_replaces_it():
    state = ConversationState()
    pediatric.apply_patient_profile(state, dict(subject="other", relation_text="女兒", age_text="8歲"))
    pediatric.apply_patient_profile(state, dict(subject="other", relation_text="女兒", age_text=None))
    assert state.patient_age_text == "8歲"
    pediatric.apply_patient_profile(state, dict(subject="self", relation_text=None, age_text=None))
    assert (state.patient_subject, state.patient_age_text) == ("self", None)


def test_age_question_names_the_person_described():
    assert pediatric.age_question(state_with("other", "小女兒")) == "請問您小女兒今年幾歲？"
    assert pediatric.age_question(state_with("other", "我孫子")) == "請問您孫子今年幾歲？"


def test_pediatric_subspecialty_reached_by_evidence_is_kept():
    assert pediatric.pediatric_dept_id({INFECTION, NOSE}) == pediatric.PEDIATRICS_DEPT_ID
    assert pediatric.pediatric_dept_id({1322, NOSE}) == 1322


def interpret(text, patient_profile, symptom, symptom_source):
    case = TriageCase(case_id="pediatric-profile")
    extraction = dict(field="symptom", normalized_value=symptom, source_text=symptom_source,
                      semantic_status="available", confidence=0.98, assertion="present")
    data = dict(semantic_extractions=[extraction], pending_answer=None, ttas_evidence=[], time_preferences=[],
                prior_answer_reference=None, department_preference=None, patient_profile=patient_profile,
                dialogue_act=dict(act="new_information", source_text=text, confidence=0.99))
    with patch.object(adapter, "_ai_available", return_value=True), \
            patch.object(adapter, "complete_prompt", AsyncMock(return_value=json.dumps(data, ensure_ascii=False))):
        apply_user_message(case, text)
        asyncio.run(adapter.refine_case_with_ai(case, user_sources=[text]))
    return case.conversation_state


def test_kindergarten_teacher_is_not_a_child():
    # The word 幼稚園 alone would have made a child; the AI says the teacher is the patient.
    state = interpret("我是幼稚園老師，喉嚨痛", profile("self", None, None, "我是幼稚園老師"), "喉嚨痛", "喉嚨痛")
    assert state.patient_subject == "self"
    assert pediatric.patient_status(state) == "adult"


def test_turn_interpreter_stores_the_daughter_and_asks_her_age():
    state = interpret("我小女兒感冒很多天都還沒好", profile("other", "小女兒", None, "我小女兒感冒很多天"),
                      "感冒", "感冒很多天都還沒好")
    assert (state.patient_subject, state.patient_relation_text) == ("other", "小女兒")
    assert pediatric.patient_status(state) == "ask"


def test_turn_interpreter_without_patient_profile_still_parses():
    state = interpret("我頭痛", None, "頭痛", "頭痛")
    assert state.patient_subject is None


def item(field, value, source):
    if field == "accompanying_symptoms":
        value = [value]
    return SemanticExtraction(field=field, normalized_value=value, source_text=source, semantic_status="available",
                              assertion="present", confidence=.99, extractor="ai")


def cold_case():
    """2026-11-03 on 310: 「我小女兒感冒…咳嗽流鼻水…喉嚨痛」 ended at 感染科."""
    case = TriageCase(case_id="pediatric-cold")
    for text, extractions in [
        ("我小女兒感冒很多天都還沒好", [item("symptom", "感冒", "感冒很多天都還沒好")]),
        ("有咳嗽跟流鼻水其他沒有", [item("accompanying_symptoms", "咳嗽", "咳嗽"),
                                   item("accompanying_symptoms", "流鼻水", "流鼻水")]),
    ]:
        case.history_records.append(Message(role="user", content=text))
        apply_semantic_extractions(case, extractions, allow_red_flag_completion=False)
    pediatric.apply_patient_profile(case.conversation_state,
                                    dict(subject="other", relation_text="小女兒", age_text=None))
    case.patient_input.red_flags_checked = True
    return case


def reason(case):
    async def provider(prompt, *, purpose):
        if purpose == "department_convergence":
            return json.dumps({"status": "ambiguous", "candidates": []})
        return json.dumps({"matches": [], "judgments": []})

    with patch.object(service, "fetch_active_departments", return_value=ACTIVE), \
            patch.object(service, "complete_runtime_json", new=provider):
        asyncio.run(service.reason_about_departments(case))


def test_age_less_daughter_is_asked_her_age_once_after_the_safety_screen():
    case = cold_case()
    reason(case)
    assert case.department_result is None or case.department_result.dept_id != pediatric.PEDIATRICS_DEPT_ID
    assert conversation_flow._age_question_due(case)
    conversation_flow._ask_age_question(case)
    state = case.conversation_state
    assert case.triage.next_question == "請問您小女兒今年幾歲？"
    assert state.reply_source == "AGE_QUESTION"
    assert conversation_flow.capture_pending_answer(case, None, ["7歲"]) is False
    assert state.pending_clarification_intent is None
    assert state.patient_age_answer == "7歲"
    assert not conversation_flow._age_question_due(case)


def test_child_goes_to_pediatrics_instead_of_infection_or_nose():
    case = cold_case()
    case.conversation_state.patient_age_question_asked = True
    case.conversation_state.patient_age_answer = "7"
    reason(case)
    state = case.conversation_state
    assert state.department_status == "resolved"
    assert case.department_result.dept_id == pediatric.PEDIATRICS_DEPT_ID
    assert state.clue_decision.reason == pediatric.DECISION_REASON
    assert "未滿 18 歲" in build_department_confirmation_reply(case)


def test_unknown_age_stays_on_the_ordinary_route():
    case = cold_case()
    case.conversation_state.patient_age_question_asked = True
    case.conversation_state.patient_age_answer = "不知道"
    reason(case)
    assert case.department_result is None or case.department_result.dept_id != pediatric.PEDIATRICS_DEPT_ID


def test_requested_department_is_never_overridden_or_asked():
    case = cold_case()
    case.patient_input.requested_department_name = "鼻科"
    assert not conversation_flow._age_question_due(case)


def test_volunteered_age_alone_is_new_information():
    case = TriageCase(case_id="pediatric-age-only")
    text = "我女兒今年7歲"
    data = dict(semantic_extractions=[], pending_answer=None, ttas_evidence=[], time_preferences=[],
                prior_answer_reference=None, department_preference=None,
                patient_profile=profile("other", "女兒", "7歲", "我女兒今年7歲"),
                dialogue_act=dict(act="new_information", source_text=text, confidence=0.99))
    with patch.object(adapter, "_ai_available", return_value=True), \
            patch.object(adapter, "complete_prompt", AsyncMock(return_value=json.dumps(data, ensure_ascii=False))):
        apply_user_message(case, text)
        suggestion = asyncio.run(adapter.refine_case_with_ai(case, user_sources=[text]))
    assert suggestion.interpretation_complete
    assert pediatric.patient_status(case.conversation_state) == "child"


def test_a_later_turn_can_fill_the_patient_missed_on_the_first_turn():
    # 2026-11-03 on 310: 「我兒子咳嗽」 asked the age on one run and not on another.
    case = TriageCase(case_id="pediatric-late-profile")
    first, second = "我兒子咳嗽", "有流鼻水"
    case.history_records.append(Message(role="user", content=first))
    data = dict(semantic_extractions=[dict(field="accompanying_symptoms", normalized_value=["流鼻水"],
                                           source_text="流鼻水", semantic_status="available", confidence=0.98,
                                           assertion="present")],
                pending_answer=None, ttas_evidence=[], time_preferences=[], prior_answer_reference=None,
                department_preference=None, patient_profile=profile("other", "兒子", "", "我兒子咳嗽"),
                dialogue_act=dict(act="new_information", source_text=second, confidence=0.99))
    with patch.object(adapter, "_ai_available", return_value=True), \
            patch.object(adapter, "complete_prompt", AsyncMock(return_value=json.dumps(data, ensure_ascii=False))):
        apply_user_message(case, second)
        asyncio.run(adapter.refine_case_with_ai(case, user_sources=[second]))
    assert pediatric.patient_status(case.conversation_state) == "ask"


def test_prompt_shows_the_recorded_patient_and_a_filled_example():
    case = TriageCase(case_id="pediatric-prompt")
    pediatric.apply_patient_profile(case.conversation_state,
                                    dict(subject="other", relation_text="兒子", age_text=None))
    with patch.object(adapter, "active_department_candidates", return_value=[]):
        prompt = adapter._build_symptom_collection_prompt(case, ["咳嗽"])
    assert 'Backend 目前記錄的就診者：{"subject": "other", "relation_text": "兒子", "age_text": null}' in prompt
    assert '"relation_text": "兒子"' in prompt
