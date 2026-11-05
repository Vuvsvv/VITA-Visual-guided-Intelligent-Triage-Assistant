import asyncio
import json
from unittest.mock import AsyncMock, patch

from app.schemas import CandidateDepartment, ClueOffer, Message, TriageCase
from app.services import conversation_service
from app.services.conversation_service import NO_CANDIDATE_PROMPT

FREE = dict(status="clarification_needed", question="頭腦昏沉比較像頭暈，還是想睡、沒精神？", intent="symptom_quality",
            reason="釐清症狀", explanation=None, answered_intent=None, answer_source_text=None, answer_status=None,
            answer_confidence=None, information_needs=["昏沉的感覺"])


def screened_case():
    case = TriageCase(case_id="no-candidate-planner")
    case.history_records = [Message(role="user", content="最近頭腦昏沉")]
    case.patient_input.symptom = "頭腦昏沉"
    case.patient_input.red_flags_checked = True
    case.patient_input.red_flags_status = "negative"
    case.conversation_state.department_status = "unresolved"
    return case


def prompt_of(case, response=FREE):
    provider = AsyncMock(return_value=json.dumps(response, ensure_ascii=False))
    with patch.object(conversation_service, "complete_runtime_json", new=provider):
        asyncio.run(conversation_service.request_clarification(case, ["最近頭腦昏沉"]))
    return provider.await_args_list[0].args[0]


def test_no_candidates_asks_planner_to_clarify_vague_symptom_first():
    prompt = prompt_of(screened_case())

    assert NO_CANDIDATE_PROMPT in prompt
    assert "想睡、沒精神" in prompt and "不要用持續多久、嚴重程度、部位這類問題填補" in prompt


def test_candidates_without_clue_mode_omit_no_candidate_guidance():
    case = screened_case()
    state = case.conversation_state
    state.department_status = "ambiguous"
    state.department_clue_candidates = [1232]
    state.candidate_departments = [CandidateDepartment(dept_id=1232, parentDept="內科部", childDept="一般內科",
                                                       confidence=.8, supporting_evidence=[])]

    assert NO_CANDIDATE_PROMPT not in prompt_of(case)


def test_clue_mode_omits_no_candidate_guidance():
    case = screened_case()
    state = case.conversation_state
    state.department_status = "ambiguous"
    state.department_clue_candidates = [1232, 1274]
    state.department_clue_offer = [ClueOffer(concept="頭暈", dept_ids=[1232, 1274])]
    response = {**FREE, "question": "請問您有沒有頭暈？", "intent": "clue_question_1", "clue_concepts": ["頭暈"]}

    assert NO_CANDIDATE_PROMPT not in prompt_of(case, response)


def test_safety_planning_omits_no_candidate_guidance():
    case = screened_case()
    case.patient_input.red_flags_checked = False
    case.patient_input.red_flags_status = "not_checked"
    case.conversation_state.clarification_status = "safety_check"
    response = {**FREE, "question": "現在有沒有胸痛？", "intent": "safety_screen"}

    assert NO_CANDIDATE_PROMPT not in prompt_of(case, response)
