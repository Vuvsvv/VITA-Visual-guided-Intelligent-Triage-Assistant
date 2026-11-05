"""Knowledge-based clue questions: planner clue mode, concept answers and clue-answer evidence."""
import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import (
    ClueConceptRecord, ClueDecisionRecord, ClueOffer, ConversationState, DialogueActInterpretation, Message,
    PendingAnswerInterpretation, SemanticExtraction, TriageCase,
)
from app.services import conversation_service, rag_triage_adapter
from app.services import department_clues as clues
from app.services import department_reasoning_service as reasoning
from app.services.case_store import save_case
from app.services.department_knowledge import load_department_knowledge, lookup_concept

chat_route = importlib.import_module("app.routes.chat")
ENT, NEURO, CARDIO = 1333, 1274, 1241
OFFER = [
    ClueOffer(concept="耳鳴", dept_ids=[ENT]), ClueOffer(concept="聽力變化", dept_ids=[ENT]),
    ClueOffer(concept="心悸", dept_ids=[CARDIO]), ClueOffer(concept="噁心", dept_ids=[ENT, NEURO]),
    ClueOffer(concept="頭痛", dept_ids=[ENT, NEURO]), ClueOffer(concept="耳朵悶塞", dept_ids=[ENT]),
]
FIRST = clues.clue_intent(1)


def clue_case(case_id="clue-case"):
    case = TriageCase(case_id=case_id)
    case.history_records = [Message(role="user", content="最近一直頭暈")]
    case.patient_input.symptom = "頭暈"
    case.patient_input.red_flags_checked = True
    case.patient_input.red_flags_status = "negative"
    state = case.conversation_state
    state.department_status = "ambiguous"
    state.department_clue_offer = list(OFFER)
    state.department_clue_candidates = [ENT, NEURO, CARDIO]
    return case


def pending_clue_case(concepts=("耳鳴", "聽力變化"), case_id="clue-pending"):
    case = clue_case(case_id)
    state = case.conversation_state
    question = "請問您有沒有" + "、".join(concepts) + "？"
    state.clue_concepts_asked = clues.append_clue_records(
        [], list(concepts), ENT, 1, dept_ids={item.concept: item.dept_ids for item in OFFER},
    )
    state.clue_questions_asked = 1
    state.pending_clue_question_index = 1
    state.pending_clarification_intent = FIRST
    state.pending_clarification_question = question
    state.asked_clarification_intents = [FIRST]
    case.history_records.append(Message(role="assistant", content=question))
    return case


def proposal(question, concepts, intent=FIRST, **extra):
    base = dict(status="clarification_needed", question=question, intent=intent, reason="區分候選",
                explanation=None, answered_intent=None, answer_source_text=None, answer_status=None,
                answer_confidence=None, information_needs=None, clue_concepts=concepts)
    return {**base, **extra}


def plan(case, *responses, user_sources=("還是很暈",), act=None):
    provider = AsyncMock(side_effect=[
        item if isinstance(item, Exception) else json.dumps(item, ensure_ascii=False) for item in responses
    ])
    with patch.object(conversation_service, "complete_runtime_json", new=provider):
        suggestion = asyncio.run(conversation_service.request_clarification(
            case, list(user_sources), dialogue_act=act,
        ))
    return suggestion, provider


def payload_of(provider, call=0):
    return json.loads(provider.await_args_list[call].args[0].split("資料：", 1)[1].split("\n", 1)[0])


def apply(case, suggestion, user_sources=("還是很暈",)):
    conversation_service.advance_conversation(
        case, suggestion, user_sources=list(user_sources), current_extractions=[], made_progress=True,
    )


# --- Asking -------------------------------------------------------------------------------------

def test_planner_receives_offer_and_backend_intent_and_question_is_recorded():
    case = clue_case()
    suggestion, provider = plan(case, proposal("請問您有沒有耳鳴（耳朵嗡嗡叫）或聽力變化？", ["耳鳴", "聽力變化"]))
    payload = payload_of(provider)
    assert payload["clue_mode"] is True and payload["required_next_question_intent"] == FIRST
    assert payload["clue_offer"][0] == {"concept": "耳鳴", "dept_ids": [ENT]}
    assert "parentDept" not in json.dumps(payload["clue_offer"])
    assert (suggestion.reply_source, suggestion.clue_concepts, suggestion.clue_dept_id) == (
        "AI_PLANNER", ("耳鳴", "聽力變化"), ENT)
    apply(case, suggestion)
    state = case.conversation_state
    assert case.triage.next_question == "請問您有沒有耳鳴（耳朵嗡嗡叫）或聽力變化？"
    assert (state.clue_questions_asked, state.pending_clue_question_index, state.pending_clarification_intent) == (
        1, 1, FIRST)
    assert [(item.concept, item.answer, item.dept_ids) for item in state.clue_concepts_asked] == [
        ("耳鳴", "unanswered", [ENT]), ("聽力變化", "unanswered", [ENT])]


def test_concept_outside_offer_is_rejected_then_repaired():
    case = clue_case()
    suggestion, provider = plan(
        case, proposal("請問您有沒有眼睛痛？", ["眼睛痛"]), proposal("請問您有沒有耳鳴？", ["耳鳴"]),
    )
    assert provider.await_count == 2
    assert (suggestion.rejection_reason, suggestion.reply_source, suggestion.clue_concepts) == (
        "invalid_clue_concepts", "AI_REPAIR", ("耳鳴",))


def test_cross_department_combination_without_common_department_is_rejected():
    case = clue_case()
    suggestion, _ = plan(case, proposal("請問您有沒有耳鳴、心悸？", ["耳鳴", "心悸"]),
                         proposal("請問您有沒有噁心、頭痛？", ["噁心", "頭痛"]))
    assert suggestion.rejection_reason == "invalid_clue_concepts"
    assert (suggestion.clue_concepts, suggestion.clue_dept_id) == (("噁心", "頭痛"), min(ENT, NEURO))


def test_more_than_three_concepts_are_rejected():
    case = clue_case()
    four = ["耳鳴", "聽力變化", "噁心", "頭痛"]
    suggestion, _ = plan(case, proposal("請問您有沒有耳鳴、聽力變化、噁心、頭痛？", four),
                         proposal("請問您有沒有耳鳴、聽力變化、噁心？", four[:3]))
    assert suggestion.rejection_reason == "invalid_clue_concepts" and len(suggestion.clue_concepts) == 3


def test_question_must_quote_every_official_term():
    case = clue_case()
    suggestion, _ = plan(case, proposal("請問您耳朵會不會嗡嗡叫？", ["耳鳴"]),
                         proposal("請問您有沒有耳鳴（耳朵嗡嗡叫）？", ["耳鳴"]))
    assert suggestion.rejection_reason == "clue_concept_not_in_question"
    assert suggestion.question == "請問您有沒有耳鳴（耳朵嗡嗡叫）？"


def test_department_name_and_sufficient_are_rejected_in_clue_mode():
    case = clue_case()
    sufficient = dict(proposal(None, []), status="sufficient")
    suggestion, _ = plan(case, sufficient, proposal("耳鼻喉科會問：您有沒有耳鳴？", ["耳鳴"]))
    assert suggestion.reply_source == "CLUE_TEMPLATE"
    assert suggestion.rejection_reason == "unsafe_or_internal_question"


def test_repair_failure_uses_backend_template_and_counts_one_question():
    case = clue_case()
    suggestion, provider = plan(case, proposal("請問您有沒有眼睛痛？", ["眼睛痛"]), "not json")
    assert provider.await_count == 2
    assert suggestion.reply_source == "CLUE_TEMPLATE"
    assert suggestion.question == "請問有沒有以下情況，或以前曾被告知有：耳鳴、聽力變化、噁心？"
    assert (suggestion.intent, suggestion.clue_dept_id) == (FIRST, ENT)
    apply(case, suggestion)
    state = case.conversation_state
    assert state.reply_source == "CLUE_TEMPLATE" and state.clue_questions_asked == 1
    assert [item.concept for item in state.clue_concepts_asked] == ["耳鳴", "聽力變化", "噁心"]
    assert state.clue_concepts_asked[2].dept_ids == [ENT, NEURO]


def test_templates_cover_current_and_previously_told_and_pass_question_guards():
    concepts = ["耳鳴", "聽力變化", "噁心"]
    template = clues.clue_template_question(concepts)
    reask = clues.clue_template_question(concepts[:2], reask=True)
    assert template == "請問有沒有以下情況，或以前曾被告知有：耳鳴、聽力變化、噁心？"
    assert reask == "想再確認一次，請問有沒有以下情況，或以前曾被告知有：耳鳴、聽力變化？"
    # Proposed verbatim by the planner, both pass every existing question guard unchanged.
    suggestion, provider = plan(clue_case(), proposal(template, concepts))
    assert provider.await_count == 1 and suggestion.rejection_reason is None and suggestion.question == template
    suggestion, provider = plan(pending_clue_case(), proposal(reask, concepts[:2]), user_sources=("嗯",))
    assert provider.await_count == 1 and suggestion.rejection_reason is None and suggestion.question == reask


def test_planner_prompt_asks_current_or_previously_told_without_doctor_words():
    _, provider = plan(clue_case(), proposal("請問您有沒有耳鳴？", ["耳鳴"]))
    prompt = provider.await_args.args[0]
    assert "以前曾被告知有" in prompt and "不得暗示患者可能罹患該病" in prompt


def test_provider_timeout_keeps_technical_fallback_and_counts_nothing():
    case = clue_case()
    suggestion, _ = plan(case, TimeoutError())
    assert (suggestion.reply_source, suggestion.rejection_reason) == ("TECHNICAL_FALLBACK", "provider_timeout")
    apply(case, suggestion)
    state = case.conversation_state
    assert case.triage.next_question == conversation_service.TECHNICAL_CLARIFICATION_REPLY
    assert (state.clue_questions_asked, state.clue_concepts_asked, state.pending_clue_question_index) == (0, [], None)


def test_only_ask_clue_or_tiebreak_decisions_ask_a_new_clue_question():
    case = clue_case()
    case.conversation_state.clue_questions_asked = 5
    case.conversation_state.clue_decision = ClueDecisionRecord(kind="tiebreak", reason="question_limit")
    sixth = clues.clue_intent(6)
    suggestion, provider = plan(case, proposal("請問您有沒有耳鳴？", ["耳鳴"], intent=sixth))
    assert payload_of(provider)["required_next_question_intent"] == sixth and suggestion.clue_concepts == ("耳鳴",)
    apply(case, suggestion)
    assert case.conversation_state.clue_tiebreak_used and case.conversation_state.clue_questions_asked == 6
    for kind in ("decide", "family_medicine", "defer_to_safety_net"):
        case = clue_case()
        case.conversation_state.clue_decision = ClueDecisionRecord(kind=kind)
        free = dict(proposal("頭暈時會不會覺得天旋地轉？", None, intent="dizziness_type"),
                    information_needs=["頭暈型態"])
        _, provider = plan(case, free)
        assert "clue_mode" not in payload_of(provider)


def test_safety_planning_never_enters_clue_mode():
    case = clue_case()
    case.patient_input.red_flags_checked = False
    case.conversation_state.clarification_status = "safety_check"
    safety = dict(proposal("請問現在有沒有胸痛或呼吸困難？", None, intent="safety_screen"),
                  information_needs=["急迫症狀"])
    suggestion, provider = plan(case, safety)
    payload = payload_of(provider)
    assert "clue_mode" not in payload and payload["required_next_question_intent"] == "safety_screen"
    assert suggestion.intent == "safety_screen" and suggestion.clue_concepts == ()


def test_explanation_request_restates_pending_clue_without_counting_a_miss():
    case = pending_clue_case()
    question = case.conversation_state.pending_clarification_question
    act = DialogueActInterpretation(act="request_explanation", source_text="什麼意思", confidence=0.95)
    explained = proposal(question, ["耳鳴", "聽力變化"], explanation="耳鳴是指耳朵裡聽到嗡嗡聲")
    suggestion, provider = plan(case, explained, user_sources=("什麼意思",), act=act)
    payload = payload_of(provider)
    assert payload["clue_reask"] is True and payload["required_next_question_intent"] == FIRST
    assert suggestion.explanation == "耳鳴是指耳朵裡聽到嗡嗡聲" and suggestion.question == question
    assert case.conversation_state.clue_reask_count == 0
    case.conversation_state.last_dialogue_act = "request_explanation"
    apply(case, suggestion, ("什麼意思",))
    assert case.triage.next_question == "耳鳴是指耳朵裡聽到嗡嗡聲\n" + question
    assert case.conversation_state.clue_questions_asked == 1


# --- Answers ------------------------------------------------------------------------------------

def concept(name, assertion, source, confidence=0.95):
    return dict(concept=name, answer_assertion=assertion, source_text=source, confidence=confidence)


def interpret(value, sources, concepts=("耳鳴", "聽力變化")):
    return rag_triage_adapter._pending_answer_from_ai(
        value, pending_intent=FIRST, user_sources=list(sources), clue_concepts=list(concepts),
    )


def pending_answer(sources, *answers, status="answered"):
    return dict(answered_intent=FIRST, answer_status=status, answer_source_text=sources[0],
                answer_confidence=0.95, concept_answers=list(answers))


def test_present_absent_and_uncertain_answers_update_records():
    case = pending_clue_case(("耳鳴", "聽力變化", "耳朵悶塞"))
    text = ["耳鳴有，聽力沒變，悶塞不確定"]
    answer = interpret(pending_answer(text, concept("耳鳴", "present", "耳鳴有"),
                                      concept("聽力變化", "absent", "聽力沒變"),
                                      concept("耳朵悶塞", "uncertain", "悶塞不確定")),
                       text, ("耳鳴", "聽力變化", "耳朵悶塞"))
    assert conversation_service.capture_pending_answer(case, answer, text)
    state = case.conversation_state
    assert [(item.concept, item.answer, item.answer_source_text) for item in state.clue_concepts_asked] == [
        ("耳鳴", "present", "耳鳴有"), ("聽力變化", "absent", "聽力沒變"), ("耳朵悶塞", "uncertain", "悶塞不確定")]
    assert state.pending_clarification_intent is None and state.pending_clue_question_index is None
    assert state.clarification_evidence[FIRST] == text[0]


def test_previously_told_by_doctor_answer_is_recorded_present():
    case = pending_clue_case(("高血壓", "心悸"))
    text = ["醫師說過我有高血壓"]
    answer = interpret(pending_answer(text, concept("高血壓", "present", "醫師說過我有高血壓"), status="partial"),
                       text, ("高血壓", "心悸"))
    assert conversation_service.capture_pending_answer(case, answer, text)
    assert [(item.concept, item.answer, item.answer_source_text)
            for item in case.conversation_state.clue_concepts_asked] == [
        ("高血壓", "present", "醫師說過我有高血壓"), ("心悸", "unanswered", None)]
    prompt = rag_triage_adapter._build_symptom_collection_prompt(pending_clue_case(), text)
    assert "醫師說過我有 ○○" in prompt and "以前被說有 ○○" in prompt


def test_bare_yes_grounds_and_unanswered_concepts_stay_unanswered():
    case = pending_clue_case()
    answer = interpret(pending_answer(["有"], concept("耳鳴", "present", "有"), status="partial"), ["有"])
    assert conversation_service.capture_pending_answer(case, answer, ["有"])
    assert [(item.concept, item.answer) for item in case.conversation_state.clue_concepts_asked] == [
        ("耳鳴", "present"), ("聽力變化", "unanswered")]


def test_unasked_ungrounded_and_low_confidence_concept_answers_are_dropped():
    answer = interpret(pending_answer(["有耳鳴"], concept("心悸", "present", "有耳鳴"),
                                      concept("耳鳴", "present", "嗡嗡"),
                                      concept("聽力變化", "absent", "有耳鳴", 0.3),
                                      concept("耳鳴", "maybe", "有耳鳴")), ["有耳鳴"])
    assert answer is not None and answer.concept_answers == []
    case = pending_clue_case()
    assert not conversation_service.capture_pending_answer(case, answer, ["有耳鳴"])
    assert case.conversation_state.pending_clarification_intent == FIRST


def test_duplicate_concept_answer_keeps_first_only():
    answer = interpret(pending_answer(["有，沒有"], concept("耳鳴", "present", "有"),
                                      concept("耳鳴", "absent", "沒有")), ["有，沒有"])
    assert [(item.concept, item.answer_assertion) for item in answer.concept_answers] == [("耳鳴", "present")]


def test_answered_clue_without_concept_answers_triggers_interpreter_repair():
    data = dict(semantic_extractions=[], pending_answer=pending_answer(["有"]), ttas_evidence=[],
                time_preferences=[], prior_answer_reference=None, department_preference=None,
                dialogue_act=dict(act="answer", source_text="有", confidence=0.95))
    answer = interpret(data["pending_answer"], ["有"])
    reason = rag_triage_adapter._turn_interpretation_rejection_reason(
        data, pending_intent=FIRST, user_sources=["有"],
        dialogue_act=DialogueActInterpretation(**data["dialogue_act"]), semantic_extractions=[],
        pending_answer=answer, ttas_evidence=[], time_preferences=[], department_preference=None,
        clue_concepts=["耳鳴", "聽力變化"],
    )
    assert reason == "answer_without_concept_answers"


def test_interpreter_prompt_lists_pending_clue_concepts():
    case = pending_clue_case()
    prompt = rag_triage_adapter._build_symptom_collection_prompt(case, ["有"])
    assert '"pending_clue_concepts": [\n    "耳鳴",\n    "聽力變化"\n  ]' in prompt
    assert "concept_answers" in prompt


def test_first_miss_reasks_same_concepts_without_counting():
    case = pending_clue_case()
    reask = proposal("想再確認，您會不會耳鳴或覺得聽力變化？", ["耳鳴", "聽力變化"])
    conversation_service.settle_unanswered_clue_question(case)
    suggestion, provider = plan(case, reask, user_sources=("昨天沒睡好",))
    assert payload_of(provider)["clue_reask"] is True
    assert case.conversation_state.clue_reask_count == 1
    apply(case, suggestion, ("昨天沒睡好",))
    state = case.conversation_state
    assert (state.pending_clarification_intent, state.clue_questions_asked, len(state.clue_concepts_asked)) == (
        FIRST, 1, 2)
    assert case.triage.next_question == "想再確認，您會不會耳鳴或覺得聽力變化？"


def test_second_miss_marks_uncertain_and_asks_a_new_question():
    case = pending_clue_case()
    case.conversation_state.clue_reask_count = 1
    second = clues.clue_intent(2)
    conversation_service.settle_unanswered_clue_question(case)
    suggestion, provider = plan(case, proposal("請問您有沒有噁心或頭痛？", ["噁心", "頭痛"], intent=second),
                                user_sources=("不知道你在問什麼",))
    assert payload_of(provider)["required_next_question_intent"] == second
    state = case.conversation_state
    assert [item.answer for item in state.clue_concepts_asked] == ["uncertain", "uncertain"]
    apply(case, suggestion, ("不知道你在問什麼",))
    assert (state.clue_questions_asked, state.pending_clue_question_index, state.clue_reask_count) == (2, 2, 0)
    assert state.pending_clarification_intent == second


# --- Clue-answer evidence (real knowledge base) -------------------------------------------------

_, RECORDS = load_department_knowledge()
ACTIVE = [
    dict(dept_id=record["canonical_dept_id"], parent_dept=record["canonical_parent_dept"],
         child_dept=record["canonical_child_dept"])
    for record in {item["canonical_dept_id"]: item for item in RECORDS}.values()
]


def extraction(value, source, assertion="present"):
    return SemanticExtraction(field="symptom", normalized_value=value,
                              semantic_status="available" if assertion == "present" else "unavailable",
                              assertion=assertion, confidence=0.95, source_text=source, extractor="ai")


def kb_case(*records):
    case = TriageCase(case_id="clue-evidence")
    case.history_records = [Message(role="user", content="最近一直頭暈"), Message(role="user", content="有"),
                            Message(role="user", content="沒有")]
    case.semantic_extractions = [extraction("頭暈", "頭暈")]
    case.conversation_state.clue_concepts_asked = list(records)
    return case


def record(name, index, answer, source, dept_ids=(ENT,)):
    return ClueConceptRecord(concept=name, dept_id=dept_ids[0], question_index=index, answer=answer,
                             answer_source_text=source, dept_ids=list(dept_ids))


def converge(case):
    payloads = []

    async def provider(prompt, *, purpose):
        payloads.append(json.loads(prompt.split("\n資料：", 1)[1]))
        return json.dumps({"status": "ambiguous", "candidates": []})

    with patch.object(reasoning, "fetch_active_departments", return_value=ACTIVE), patch.object(
        reasoning, "complete_runtime_json", new=provider,
    ):
        asyncio.run(reasoning.reason_about_departments(case))
    return payloads[-1]


def test_present_answer_becomes_marked_official_evidence_and_absent_does_not():
    case = kb_case(record("耳鳴", 1, "present", "有"), record("心悸", 2, "absent", "沒有", (CARDIO,)))
    payload = converge(case)
    marked = [item for item in payload["retrieved_official_evidence"]
              if item.get("evidence_origin") == clues.CLUE_ANSWER_ORIGIN]
    assert {item["knowledge_concept"] for item in marked} == {"耳鳴"}
    assert {item["dept_id"] for item in marked} == {r["canonical_dept_id"] for r in lookup_concept("耳鳴", RECORDS)}
    assert all(item["patient_source_text"] == "有" for item in marked)
    assert payload["clue_answers"] == [{"concept": "耳鳴", "answer": "present"},
                                       {"concept": "心悸", "answer": "absent"}]
    offered = {item.concept for item in case.conversation_state.department_clue_offer}
    assert offered and not offered & {"耳鳴", "心悸"}
    assert ENT in case.conversation_state.department_clue_candidates


def test_contradicting_semantic_evidence_suppresses_clue_answer_evidence():
    case = kb_case(record("耳鳴", 1, "present", "有"))
    case.history_records.append(Message(role="user", content="其實沒有耳鳴"))
    case.semantic_extractions.append(extraction("耳鳴", "沒有耳鳴", assertion="absent"))
    payload = converge(case)
    assert not any(item.get("evidence_origin") for item in payload["retrieved_official_evidence"])


def test_two_bare_yes_answers_score_two_for_the_department():
    case = kb_case(record("耳鳴", 1, "present", "有"), record("耳痛", 2, "present", "有"))
    retrieved = converge(case)["retrieved_official_evidence"]
    scores = clues.department_scores(retrieved, clues.candidate_dept_ids(retrieved),
                                     case.conversation_state.clue_concepts_asked)
    assert scores[ENT] == 1 + 2


def test_clue_answer_also_extracted_as_a_symptom_scores_once():
    case = kb_case(record("耳鳴", 1, "present", "有耳鳴"))
    case.history_records[1] = Message(role="user", content="有耳鳴")
    case.semantic_extractions.append(extraction("耳鳴", "耳鳴"))
    retrieved = converge(case)["retrieved_official_evidence"]
    assert any(item["patient_source_text"] == "耳鳴" and not item.get("evidence_origin") for item in retrieved)
    scores = clues.department_scores(retrieved, clues.candidate_dept_ids(retrieved),
                                     case.conversation_state.clue_concepts_asked, RECORDS)
    assert scores[ENT] == 1 + 1


def test_specialty_only_kb_concept_is_not_offered():
    assert {r["evidence_type"] for r in RECORDS if r["concept"] == "臉型雕塑"} == {"department_specialty_evidence"}
    holder = next(r["canonical_dept_id"] for r in RECORDS if r["concept"] == "臉型雕塑")
    offered = {clue.concept for clue in clues.clue_concepts(RECORDS, {holder, ENT}, set(), [])}
    assert offered and "臉型雕塑" not in offered


def test_kb_concept_with_direct_and_specialty_records_is_offered_for_every_holder():
    holders = {r["canonical_dept_id"]: r["evidence_type"] for r in lookup_concept("頭痛", RECORDS)}
    specialty = {dept for dept, kind in holders.items() if kind == "department_specialty_evidence"}
    direct = {dept for dept, kind in holders.items() if kind == "direct_routing_evidence"}
    assert specialty and direct and ENT not in holders
    candidates = {min(specialty), min(direct), ENT}
    clue = next(item for item in clues.clue_concepts(RECORDS, candidates, set(), []) if item.concept == "頭痛")
    assert clue.dept_ids == {min(specialty), min(direct)}


def test_specialty_evidence_still_counts_in_retrieved_and_scores():
    case = kb_case()
    # The chief complaint: a department that does not hold it is no clue candidate.
    case.history_records = [Message(role="user", content="臉型雕塑")]
    case.semantic_extractions = [extraction("臉型雕塑", "臉型雕塑")]
    retrieved = converge(case)["retrieved_official_evidence"]
    holder = next(r["canonical_dept_id"] for r in RECORDS if r["concept"] == "臉型雕塑")
    assert any(item["knowledge_concept"] == "臉型雕塑" and item["dept_id"] == holder for item in retrieved)
    scores = clues.department_scores(retrieved, clues.candidate_dept_ids(retrieved), [])
    assert scores[holder] == 1
    assert holder in case.conversation_state.department_clue_candidates


def test_cross_department_present_concept_credits_both_and_does_not_confirm_single():
    asked = [record("噁心", 1, "present", "有", (ENT, NEURO))]
    candidates = {ENT, NEURO, CARDIO}
    assert clues.department_scores([], candidates, asked) == {ENT: 1, NEURO: 1, CARDIO: 0}
    confirmed = clues.clue_confirmed_departments(asked, candidates)
    assert confirmed == {ENT, NEURO}
    askable = clues.clue_concepts(RECORDS, candidates, set(), asked)
    decision = clues.decide(ai_resolved_dept_id=None, candidates=candidates, scores={}, askable=askable,
                            questions_asked=1, tiebreak_used=False, user_declined=False,
                            clue_confirmed_dept_ids=confirmed, question_limit=5)
    assert decision.kind == "ask_clue"


# --- Old snapshots ------------------------------------------------------------------------------

def test_old_snapshots_load_with_clue_defaults():
    legacy = ClueConceptRecord.model_validate({"concept": "耳鳴", "dept_id": ENT, "question_index": 1})
    assert (legacy.dept_ids, legacy.answer_source_text) == ([ENT], None)
    state = ConversationState.model_validate({"turn_count": 2, "clue_concepts_asked": [
        {"concept": "耳鳴", "dept_id": ENT, "question_index": 1, "answer": "present"}]})
    assert state.clue_concepts_asked[0].dept_ids == [ENT]
    assert (state.department_clue_offer, state.department_clue_candidates, state.pending_clue_question_index,
            state.clue_reask_count) == ([], [], None, 0)
    old = PendingAnswerInterpretation(answered_intent="x", answer_status="answered",
                                      answer_source_text="有", answer_confidence=0.9)
    assert old.concept_answers == []


# --- Chat route: full clue cycle and urgent path ------------------------------------------------

def chat(case, message, extractions, plan_response, pending=None, act="answer", ttas=None):
    save_case(case)
    settings = SimpleNamespace(cerebras_api_key="test-key")
    semantic = AsyncMock(return_value=json.dumps(dict(
        semantic_extractions=extractions, pending_answer=pending, ttas_evidence=ttas or [],
        time_preferences=[], prior_answer_reference=None, department_preference=None,
        dialogue_act=dict(act=act, source_text=message, confidence=0.99),
    ), ensure_ascii=False))
    planner = AsyncMock(return_value=json.dumps(plan_response, ensure_ascii=False))
    with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
        rag_triage_adapter, "get_settings", return_value=settings,
    ), patch.object(rag_triage_adapter, "complete_prompt", new=semantic), patch.object(
        conversation_service, "complete_runtime_json", new=planner,
    ), patch.object(chat_route, "reason_about_departments", new=AsyncMock()), patch.object(
        chat_route, "generate_triage_reply", new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"]),
    ):
        response = TestClient(app).post("/chat", json=dict(
            case_id=case.case_id, message=message, triage_case=case.model_dump(mode="json"),
        ))
    assert response.status_code == 200
    return response.json(), planner


def symptom(value, source):
    return dict(field="symptom", normalized_value=value, source_text=source, confidence=0.95,
                semantic_status="available", assertion="present")


def test_chat_asks_clue_then_records_bare_yes_answer():
    case = clue_case("clue-chat")
    first, _ = chat(case, "還是很暈", [symptom("頭暈", "很暈")],
                    proposal("請問您有沒有噁心或頭痛？", ["噁心", "頭痛"]), act="new_information")
    state = first["conversation_state"]
    assert first["next_question"] == "請問您有沒有噁心或頭痛？"
    assert (state["clue_questions_asked"], state["pending_clarification_intent"]) == (1, FIRST)
    assert [item["dept_ids"] for item in state["clue_concepts_asked"]] == [[ENT, NEURO], [ENT, NEURO]]

    after = TriageCase.model_validate(first["triage_case"])
    after.conversation_state.department_clue_offer = list(OFFER[:3])
    answer = pending_answer(["有"], concept("噁心", "present", "有"), status="partial")
    second, _ = chat(after, "有", [], proposal("請問您有沒有耳鳴？", ["耳鳴"], intent=clues.clue_intent(2)),
                     pending=answer)
    asked = second["conversation_state"]["clue_concepts_asked"]
    assert [(item["concept"], item["answer"], item["answer_source_text"]) for item in asked[:2]] == [
        ("噁心", "present", "有"), ("頭痛", "unanswered", None)]
    assert second["conversation_state"]["clue_questions_asked"] == 2


def test_chat_off_target_answer_reasks_then_settles():
    case = pending_clue_case(case_id="clue-chat-miss")
    miss = pending_answer(["昨天沒睡好"], status="unclear")
    first, _ = chat(case, "昨天沒睡好", [], proposal("想再確認，您會不會耳鳴或聽力變化？", ["耳鳴", "聽力變化"]),
                    pending=miss)
    state = first["conversation_state"]
    assert (state["pending_clarification_intent"], state["clue_questions_asked"], state["clue_reask_count"]) == (
        FIRST, 1, 1)
    again = pending_answer(["還是不懂"], status="unclear")
    second, _ = chat(TriageCase.model_validate(first["triage_case"]), "還是不懂", [],
                     proposal("請問您有沒有心悸？", ["心悸"], intent=clues.clue_intent(2)), pending=again)
    state = second["conversation_state"]
    assert [item["answer"] for item in state["clue_concepts_asked"][:2]] == ["uncertain", "uncertain"]
    assert (state["pending_clarification_intent"], state["clue_questions_asked"]) == (clues.clue_intent(2), 2)
    assert second["next_question"] == "請問您有沒有心悸？"


def test_chat_urgent_path_skips_clue_planning():
    case = pending_clue_case(case_id="clue-chat-urgent")
    result, planner = chat(case, "化學藥劑濺到我的眼睛", [symptom("眼睛灼傷", "眼睛")], {}, act="new_information",
                           ttas=[dict(field="chemical_eye_injury", value=True, semantic_status="available",
                                      confidence=0.95, source_text="化學藥劑濺到我的眼睛")])
    planner.assert_not_awaited()
    state = result["conversation_state"]
    assert state["clarification_status"] == "urgent"
    assert (state["clue_questions_asked"], [item["answer"] for item in state["clue_concepts_asked"]]) == (
        1, ["unanswered", "unanswered"])
