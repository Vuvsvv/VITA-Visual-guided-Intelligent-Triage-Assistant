"""Commit 5: department_clues.decide() wired into the chat flow, with the family medicine fallback."""
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.schemas import (
    CandidateDepartment, ClueConceptRecord, ClueDecisionRecord, ConversationState, DepartmentResult, Message, TriageCase,
)
from app.services import conversation_service, rag_triage_adapter
from app.services import department_clues as clues
from app.services import department_reasoning_service as reasoning
from app.services.case_store import save_case
from app.services.department_knowledge import load_department_knowledge, lookup_concept

chat_route = importlib.import_module("app.routes.chat")
_, RECORDS = load_department_knowledge()
ACTIVE = [
    dict(dept_id=record["canonical_dept_id"], parent_dept=record["canonical_parent_dept"],
         child_dept=record["canonical_child_dept"])
    for record in {item["canonical_dept_id"]: item for item in RECORDS}.values()
]
FAMILY = get_settings().family_medicine_dept_id
ENT = 1333
DIZZY = {record["canonical_dept_id"] for record in lookup_concept("頭暈", RECORDS)}
CONFIRM_PROMPT = "請確認後取得推薦掛號方案。"


def symptom(value, source, field="symptom"):
    return dict(field=field, normalized_value=[value] if field == "accompanying_symptoms" else value,
                source_text=source, confidence=0.95, semantic_status="available", assertion="present")


class Convergence:
    """Fake convergence AI: ambiguous with no candidates, or resolves the first ENT evidence."""

    def __init__(self, resolve_ent=False):
        self.resolve_ent = resolve_ent

    async def __call__(self, prompt, *, purpose):
        if not self.resolve_ent:
            return json.dumps({"status": "ambiguous", "candidates": []})
        retrieved = json.loads(prompt.split("\n資料：", 1)[1])["retrieved_official_evidence"]
        item = next(entry for entry in retrieved if entry["dept_id"] == ENT)
        support = {key: item[key] for key in ("patient_source_text", "knowledge_source_id", "knowledge_concept")}
        return json.dumps({"status": "resolved", "candidates": [
            {"dept_id": ENT, "confidence": 0.95, "supporting_evidence": [support]}]}, ensure_ascii=False)


def new_case(case_id):
    case = TriageCase(case_id=case_id)
    case.patient_input.red_flags_checked = True
    case.patient_input.red_flags_status = "negative"
    # Time was already answered; this test covers the medical confirmation gate, not 6-H availability.
    case.conversation_state.availability_status = "answered"
    return case


def turn(case, message, extractions=(), assertion=None, *, active=ACTIVE, convergence=None, ttas=None, act=None,
         answer=None):
    """One /chat turn with real department reasoning; the planner always fails, so clues use the template."""
    concepts = clues.pending_clue_concepts(case.conversation_state)
    pending = None
    if concepts and assertion:
        pending = dict(
            answered_intent=case.conversation_state.pending_clarification_intent, answer_status="answered",
            answer_source_text=message, answer_confidence=0.95,
            concept_answers=[dict(concept=name, answer_assertion=assertion(name) if callable(assertion) else assertion,
                                  source_text=message, confidence=0.95) for name in concepts],
        )
    pending = answer or pending
    save_case(case)
    settings = SimpleNamespace(cerebras_api_key="test-key")
    semantic = AsyncMock(return_value=json.dumps(dict(
        semantic_extractions=list(extractions), pending_answer=pending, ttas_evidence=ttas or [],
        time_preferences=[], prior_answer_reference=None, department_preference=None,
        dialogue_act=dict(act=act or ("answer" if pending else "new_information"), source_text=message, confidence=0.99),
    ), ensure_ascii=False))
    with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
        rag_triage_adapter, "get_settings", return_value=settings,
    ), patch.object(rag_triage_adapter, "complete_prompt", new=semantic), patch.object(
        conversation_service, "complete_runtime_json", new=AsyncMock(return_value="not json"),
    ), patch.object(reasoning, "fetch_active_departments", return_value=active), patch.object(
        reasoning, "complete_runtime_json", new=convergence or Convergence(),
    ), patch.object(chat_route, "generate_triage_reply", new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"])):
        response = TestClient(app).post("/chat", json=dict(
            case_id=case.case_id, message=message, triage_case=case.model_dump(mode="json"),
        ))
    assert response.status_code == 200
    body = response.json()
    return TriageCase.model_validate(body["triage_case"]), body


def assert_waiting_confirmation(body, dept_id):
    state = body["conversation_state"]
    assert (state["stage"], state["awaiting_confirmation"], state["confirmed"]) == (
        "waiting_confirmation", True, False)
    assert body["triage_case"]["confirmed"] is False
    assert body["department_result"]["dept_id"] == dept_id
    assert body["next_question"] is None


def dizzy_start(case_id):
    return turn(new_case(case_id), "最近一直頭暈", [symptom("頭暈", "頭暈")])


# --- Flows on the real knowledge base -----------------------------------------------------------

def test_present_answer_to_unique_clue_asks_confirmation_of_that_department():
    case, body = dizzy_start("decide-unique")
    assert body["conversation_state"]["reply_source"] == "CLUE_TEMPLATE"
    assert body["conversation_state"]["clue_decision"]["kind"] == "ask_clue"
    # Confirmation follows the concept's current official holders, not the snapshot taken when asked.
    first = next(record for record in case.conversation_state.clue_concepts_asked
                 if len({item["canonical_dept_id"] for item in lookup_concept(record.concept, RECORDS)}) == 1)
    assert len(first.dept_ids) == 1
    case, body = turn(case, "有", assertion=lambda name: "present" if name == first.concept else "absent")
    # 2026-11-03: two or more target departments need clue_min_questions (3) before deciding.
    assert body["conversation_state"]["clue_decision"]["kind"] == "ask_clue"
    for _ in range(2):
        case, body = turn(case, "沒有", assertion="absent")
    assert case.conversation_state.clue_questions_asked == 3
    decision = body["conversation_state"]["clue_decision"]
    assert (decision["kind"], decision["reason"], decision["dept_id"]) == (
        "decide", "clue_confirmed_single_department", first.dept_ids[0])
    assert_waiting_confirmation(body, first.dept_ids[0])
    assert body["reply"].endswith(CONFIRM_PROMPT) and body["department_result"]["childDept"] in body["reply"]


def test_three_absent_answers_decide_the_leading_department():
    # 2026-11-03: a leader strictly ahead settles at clue_min_questions once the challenger had no present answer.
    case, _ = turn(new_case("decide-limit"), "頭暈還有耳鳴",
                   [symptom("頭暈", "頭暈"), symptom("耳鳴", "耳鳴", "accompanying_symptoms")])
    for _ in range(2):
        case, body = turn(case, "沒有", assertion="absent")
        assert body["conversation_state"]["clue_decision"]["kind"] == "ask_clue"
    assert case.conversation_state.clue_questions_asked == 3
    case, body = turn(case, "沒有", assertion="absent")
    decision = body["conversation_state"]["clue_decision"]
    assert (decision["kind"], decision["reason"], decision["dept_id"]) == ("decide", "leader_confirmed", ENT)
    assert_waiting_confirmation(body, ENT)


def test_unknown_answers_lead_to_tiebreak_then_family_medicine_confirmation():
    case, _ = dizzy_start("decide-unknown")
    asked = clues.pending_clue_concepts(case.conversation_state)
    case, body = turn(case, "不知道", assertion="uncertain")
    state = case.conversation_state
    assert [record.answer for record in state.clue_concepts_asked[:len(asked)]] == ["uncertain"] * len(asked)
    assert (state.clue_reask_count, state.clue_questions_asked) == (0, 2)
    assert state.pending_clarification_intent == clues.clue_intent(2)
    for _ in range(3):
        case, body = turn(case, "不知道", assertion="uncertain")
    # 2026-10-28: the convergence AI ranks no candidate here, so the tie goes straight to
    # family medicine at the 5-question limit; no 6th tiebreak question is asked.
    case, body = turn(case, "不知道", assertion="uncertain")
    state = case.conversation_state
    assert (state.clue_questions_asked, state.clue_tiebreak_used) == (5, False)
    assert set(body["conversation_state"]["clue_decision"]["tied_dept_ids"]) == DIZZY
    decision = body["conversation_state"]["clue_decision"]
    assert (decision["kind"], decision["dept_id"]) == ("family_medicine", FAMILY)
    assert_waiting_confirmation(body, FAMILY)
    assert body["reply"] == get_settings().family_medicine_fallback_message + CONFIRM_PROMPT


def test_family_medicine_unavailable_is_terminal_and_invents_nothing():
    case, _ = dizzy_start("decide-family-missing")
    # 2026-10-28: the tie is decided at the 5th answer (no 6th tiebreak question).
    for _ in range(4):
        case, _ = turn(case, "不知道", assertion="uncertain")
    without_family = [row for row in ACTIVE if row["dept_id"] != FAMILY]
    case, body = turn(case, "不知道", assertion="uncertain", active=without_family)
    state = body["conversation_state"]
    assert state["clue_decision"]["kind"] == "family_medicine_unavailable"
    assert (state["clarification_status"], state["reply_source"], state["stage"]) == (
        "unresolved", "FAMILY_MEDICINE_UNAVAILABLE", "collecting")
    assert body["reply"] == conversation_service.FAMILY_MEDICINE_UNAVAILABLE_REPLY
    assert body["department_result"] is None and body["next_question"] is None


def test_ai_resolved_candidate_waits_for_the_minimum_clue_questions():
    # 2026-11-03: the convergence AI's resolution no longer skips the clue questions.
    convergence = Convergence(resolve_ent=True)
    case, body = turn(new_case("decide-ai"), "頭暈還有耳鳴",
                      [symptom("頭暈", "頭暈"), symptom("耳鳴", "耳鳴", "accompanying_symptoms")],
                      convergence=convergence)
    state = body["conversation_state"]
    assert (state["clue_decision"]["kind"], state["department_status"]) == ("ask_clue", "ambiguous")
    assert body["department_result"] is None and state["clue_questions_asked"] == 1
    for _ in range(2):
        case, body = turn(case, "沒有", assertion="absent", convergence=convergence)
        assert body["conversation_state"]["clue_decision"]["kind"] == "ask_clue"
    case, body = turn(case, "沒有", assertion="absent", convergence=convergence)
    assert body["conversation_state"]["clue_decision"]["reason"] == "ai_resolved"
    assert body["conversation_state"]["clue_questions_asked"] == 3
    assert_waiting_confirmation(body, ENT)


def test_urgent_turn_never_runs_decide():
    case, _ = dizzy_start("decide-urgent")
    _, body = turn(case, "化學藥劑濺到我的眼睛", [symptom("眼睛灼傷", "眼睛")], ttas=[dict(
        field="chemical_eye_injury", value=True, semantic_status="available", confidence=0.95,
        source_text="化學藥劑濺到我的眼睛")])
    state = body["conversation_state"]
    assert state["clarification_status"] == "urgent"
    assert state["clue_decision"]["kind"] == "ask_clue"  # unchanged from the previous turn
    assert state["clue_questions_asked"] == 1 and body["department_result"] is None


def test_blanket_unknown_marks_all_three_concepts_uncertain_and_ends_the_question():
    case = new_case("decide-unknown-three")
    names = ["耳鳴", "耳痛", "重聽"]
    state = case.conversation_state
    state.clue_concepts_asked = clues.append_clue_records([], names, ENT, 1)
    state.clue_questions_asked, state.pending_clue_question_index = 1, 1
    state.pending_clarification_intent = clues.clue_intent(1)
    answer = rag_triage_adapter._pending_answer_from_ai(dict(
        answered_intent=clues.clue_intent(1), answer_status="answered", answer_source_text="不知道",
        answer_confidence=0.9, concept_answers=[
            dict(concept=name, answer_assertion="uncertain", source_text="不知道", confidence=0.9) for name in names]),
        pending_intent=clues.clue_intent(1), user_sources=["不知道"], clue_concepts=names)
    assert conversation_service.capture_pending_answer(case, answer, ["不知道"])
    conversation_service.settle_unanswered_clue_question(case)
    assert [record.answer for record in state.clue_concepts_asked] == ["uncertain"] * 3
    assert (state.pending_clarification_intent, state.clue_reask_count, state.clue_questions_asked) == (None, 0, 1)
    prompt = rag_triage_adapter._build_symptom_collection_prompt(case, ["不知道"])
    assert "籠統回答「不知道」「不清楚」「不確定」" in prompt


# --- apply_clue_decision units ------------------------------------------------------------------

A, B, C = 1232, 1241, 1274


def hit(dept_id, source):
    return dict(dept_id=dept_id, patient_source_text=source, knowledge_concept="頭暈")


def clue(name, dept_ids, priority=1):
    ids = frozenset(dept_ids)
    return clues.ClueConcept(name, ids, len(ids) == 1, True, priority, False)


def decision_case(candidates=(A, B), asked=5):
    case = TriageCase(case_id="apply")
    case.patient_input.red_flags_checked = True
    state = case.conversation_state
    state.department_status = "ambiguous"
    state.department_clue_candidates = list(candidates)
    state.clue_questions_asked = asked
    # Time was already answered; this test covers the medical confirmation gate, not 6-H availability.
    state.availability_status = "answered"
    return case


TIED = [hit(A, "頭暈"), hit(B, "頭暈")]


def test_tie_without_tied_unique_clue_goes_directly_to_family_medicine():
    case = decision_case()
    reasoning.apply_clue_decision(case, [clue("咳嗽", {A, B})], TIED, ACTIVE)
    assert case.conversation_state.clue_decision.kind == "family_medicine"
    assert case.department_result.dept_id == FAMILY and case.conversation_state.department_status == "resolved"


def test_family_medicine_requires_exactly_one_live_row_for_configured_id():
    family_row = next(row for row in ACTIVE if row["dept_id"] == FAMILY)
    for rows, kind in (([], "family_medicine_unavailable"), ([family_row], "family_medicine"),
                       ([family_row, dict(family_row)], "family_medicine_unavailable")):
        case = decision_case()
        reasoning.apply_clue_decision(case, [], TIED, [row for row in ACTIVE if row["dept_id"] != FAMILY] + rows)
        assert case.conversation_state.clue_decision.kind == kind
        if kind == "family_medicine":
            assert (case.department_result.parentDept, case.department_result.childDept) == (
                family_row["parent_dept"], family_row["child_dept"])
        else:
            assert case.department_result is None and case.conversation_state.department_status == "ambiguous"


def test_family_medicine_id_comes_from_settings():
    other = next(row for row in ACTIVE if row["dept_id"] not in {FAMILY, A, B})
    with patch.object(reasoning, "get_settings", return_value=Settings(_env_file=None, FAMILY_MEDICINE_DEPT_ID=other["dept_id"])):
        case = decision_case()
        reasoning.apply_clue_decision(case, [], TIED, ACTIVE)
    assert case.department_result.dept_id == other["dept_id"]


def test_tiebreak_uses_every_askable_clue_not_only_the_top_offer():
    case = decision_case(candidates=(A, B, C))
    askable = [clue(f"C線索{index:02d}", {C}) for index in range(10)] + [clue("A與C線索", {A, C}, priority=3)]
    assert "A與C線索" not in {item.concept for item in clues.rank_clue_concepts(askable, 10)}
    reasoning.apply_clue_decision(case, askable, [*TIED, hit(A, "暈"), hit(B, "暈"), hit(C, "頭暈")], ACTIVE)
    state = case.conversation_state
    # 2026-10-28: the decision flow no longer asks a tiebreak question; the helper still
    # searches every askable clue, not only the planner's top-ranked offer.
    assert state.clue_decision.kind == "family_medicine" and set(state.clue_decision.tied_dept_ids) == {A, B}
    assert [item.concept for item in clues.tiebreak_clues(askable, {A, B})] == ["A與C線索"]


def test_ai_resolved_decides_even_while_a_clue_question_is_pending():
    case = decision_case(asked=1)
    state = case.conversation_state
    state.department_status = "resolved"
    case.department_result = DepartmentResult(dept_id=A, childDept="一般內科")
    state.clue_concepts_asked = [ClueConceptRecord(concept="咳嗽", dept_id=A, question_index=1)]
    state.pending_clue_question_index, state.pending_clarification_intent = 1, clues.clue_intent(1)
    reasoning.apply_clue_decision(case, [], TIED, ACTIVE)
    assert (state.clue_decision.kind, state.clue_decision.reason) == ("decide", "ai_resolved")
    assert state.pending_clarification_intent is None and state.pending_clue_question_index is None


def test_decide_is_skipped_before_safety_while_pending_and_on_explanation_requests():
    unscreened = decision_case()
    unscreened.patient_input.red_flags_checked = False
    pending = decision_case()
    pending.conversation_state.clue_concepts_asked = [ClueConceptRecord(concept="咳嗽", dept_id=A, question_index=5)]
    pending.conversation_state.pending_clue_question_index = 5
    pending.conversation_state.pending_clarification_intent = clues.clue_intent(5)
    explaining = decision_case()
    explaining.conversation_state.last_dialogue_act = "request_explanation"
    for case in (unscreened, pending, explaining):
        reasoning.apply_clue_decision(case, [], TIED, ACTIVE)
        assert case.conversation_state.clue_decision is None and case.department_result is None


def test_backend_decision_counts_as_sufficient_but_ai_cannot_confirm():
    case = decision_case()
    case.history_records = [Message(role="user", content="頭暈"), Message(role="user", content="沒有")]
    case.patient_input.symptom = "頭暈"
    case.conversation_state.clarification_evidence = {clues.clue_intent(1): "沒有"}
    reasoning.apply_clue_decision(case, [], TIED, ACTIVE)
    planner_question = conversation_service.ClarificationSuggestion(
        "clarification_needed", "請問頭暈多久了？", "duration", "仍想追問")
    conversation_service.advance_conversation(case, planner_question, user_sources=["沒有"], current_extractions=[])
    state = case.conversation_state
    assert state.is_complete and state.reply_source == "CONFIRMATION" and case.triage.next_question is None
    assert not state.confirmed and not case.confirmed


# --- Revision -----------------------------------------------------------------------------------

def test_revision_keeps_asked_clue_concepts_and_never_repeats_the_clue_question():
    case, body = dizzy_start("decide-revision")
    question = body["next_question"]
    asked = list(case.conversation_state.clue_concepts_asked)
    chat_route._begin_revision(case)
    state = case.conversation_state
    assert (state.clue_questions_asked, state.clue_concepts_asked) == (1, asked)
    assert (state.clue_tiebreak_used, state.clue_decision, state.department_clue_offer,
            state.pending_clue_question_index, state.clue_reask_count) == (False, None, [], None, 0)
    case, body = turn(case, "還是頭暈", [symptom("頭暈", "頭暈")])
    assert body["next_question"] != question
    assert {record.concept for record in asked} <= {
        record["concept"] for record in body["conversation_state"]["clue_concepts_asked"]}


# --- Old snapshots ------------------------------------------------------------------------------

def test_old_snapshots_load_without_decision_fields():
    # revision_history_start was dropped; snapshots that still carry it must load.
    state = ConversationState.model_validate(
        {"turn_count": 4, "clue_questions_asked": 2, "revision_history_start": 3})
    assert state.clue_decision is None
    record = ClueDecisionRecord.model_validate({"kind": "decide"})
    assert (record.reason, record.dept_id, record.tied_dept_ids) == ("", None, [])


# --- Clue answers use current knowledge-base holders and current candidates ---------------------

ORTHO, RHEUMA = 1298, 1255
JOINT_HOLDERS = {record["canonical_dept_id"] for record in lookup_concept("關節痛", RECORDS)}


class BackPainConvergence:
    """Fake convergence AI ranking orthopedics, neurosurgery and rehabilitation as in 310."""

    async def __call__(self, prompt, *, purpose):
        retrieved = json.loads(prompt.split("\n資料：", 1)[1])["retrieved_official_evidence"]
        candidates = []
        for dept_id, confidence in ((ORTHO, 0.7), (1244, 0.6), (1265, 0.5)):
            item = next(entry for entry in retrieved if entry["dept_id"] == dept_id)
            support = {key: item[key] for key in ("patient_source_text", "knowledge_source_id", "knowledge_concept")}
            candidates.append({"dept_id": dept_id, "confidence": confidence, "supporting_evidence": [support]})
        return json.dumps({"status": "ambiguous", "candidates": candidates}, ensure_ascii=False)


def test_present_clue_holder_that_misses_the_chief_complaint_stays_out():
    # 「背痛」, not 「腰酸背痛」: 新光醫院 lists 腰酸背痛 under 過敏風濕科 too since 2026-11-04.
    case, body = turn(new_case("decide-joint-pain"), "周一開始背痛", [symptom("背痛", "背痛")],
                      convergence=BackPainConvergence())
    state = case.conversation_state
    assert RHEUMA not in state.department_clue_candidates
    # Not 「關節痛」: Taipei VGH 復健醫學部 doctors list it too since 2026-11-04, so it no longer singles out 骨科.
    asked = next(record for record in state.clue_concepts_asked if record.concept == "晨起關節僵硬")
    assert asked.dept_ids == [ORTHO]

    case, body = turn(case, "早上起來關節很僵硬",
                      assertion=lambda name: "present" if name == "晨起關節僵硬" else "absent",
                      convergence=BackPainConvergence())
    state = case.conversation_state
    # 「晨起關節僵硬」 is also a 風濕 concept, but 風濕 does not hold 「背痛」 (2026-11-03 chief complaint rule).
    assert ORTHO in state.department_clue_candidates and RHEUMA not in state.department_clue_candidates
    assert state.clue_decision.kind == "ask_clue"
    assert state.department_status != "resolved" and body["department_result"] is None
    assert clues.pending_clue_concepts(state) and state.clue_questions_asked == 2


def test_present_concept_with_one_current_candidate_holder_still_confirms():
    asked = [ClueConceptRecord(concept="關節痛", dept_id=ORTHO, dept_ids=[ORTHO], question_index=1, answer="present")]
    candidates = {ORTHO, ENT}
    confirmed = clues.clue_confirmed_departments(asked, candidates, RECORDS)
    assert confirmed == {ORTHO}
    decision = clues.decide(
        ai_resolved_dept_id=None, candidates=candidates, scores={ORTHO: 2, ENT: 1}, askable=[clue("耳鳴", {ENT})],
        questions_asked=1, tiebreak_used=False, user_declined=False, clue_confirmed_dept_ids=confirmed,
        question_limit=5, non_target_dept_ids={FAMILY},
    )
    assert (decision.kind, decision.dept_id, decision.reason) == ("decide", ORTHO, "clue_confirmed_single_department")


def test_present_concept_scores_every_current_holder_once_per_question():
    candidates = {ORTHO, RHEUMA, ENT}
    asked = [
        ClueConceptRecord(concept="關節痛", dept_id=ORTHO, dept_ids=[ORTHO], question_index=1, answer="present"),
        ClueConceptRecord(concept="膝關節疼痛", dept_id=ORTHO, dept_ids=[ORTHO], question_index=1, answer="present"),
    ]
    assert JOINT_HOLDERS & candidates == {ORTHO, RHEUMA}
    assert clues.department_scores([], candidates, asked, RECORDS) == {ORTHO: 1, RHEUMA: 1, ENT: 0}
    assert clues.clue_confirmed_departments(asked, candidates, RECORDS) == {ORTHO, RHEUMA}
    # Without records only the snapshot taken when the question was asked is available.
    assert clues.department_scores([], candidates, asked) == {ORTHO: 1, RHEUMA: 0, ENT: 0}


def test_concept_no_longer_in_knowledge_base_falls_back_to_snapshot():
    record = ClueConceptRecord(concept="已移除概念", dept_id=ENT, dept_ids=[ENT], question_index=1, answer="present")
    assert clues.answered_concept_departments(record, RECORDS) == {ENT}


# --- 2026-10-28: a tied top score is broken by the convergence AI's ranking ---------------------

ORTHO, REHAB, RHEUM, NEURO, NEURO_SURG, CARDIO, META, PSY = 1298, 1265, 1255, 1274, 1244, 1241, 1237, 1251


def ranked(case, order):
    """Store validated convergence candidates in the AI's order (confidence descending)."""
    rows = {row["dept_id"]: row for row in ACTIVE}
    case.conversation_state.candidate_departments = [
        CandidateDepartment(dept_id=dept_id, parentDept=rows[dept_id]["parent_dept"],
                            childDept=rows[dept_id]["child_dept"], confidence=0.6 - 0.1 * index)
        for index, dept_id in enumerate(order)
    ]
    return case


def test_decide_breaks_a_tie_with_the_ai_rank():
    common = dict(ai_resolved_dept_id=None, askable=[clue("咳嗽", {A, B})], questions_asked=5, tiebreak_used=False,
                  user_declined=False, clue_confirmed_dept_ids=set(), question_limit=5, non_target_dept_ids={FAMILY})
    tied = clues.decide(candidates={A, B, C}, scores={A: 2, B: 2, C: 1}, ai_order=[C, B, A], **common)
    assert (tied.kind, tied.dept_id, tied.reason, tied.tied_dept_ids) == ("decide", B, "tie_broken_by_ai_rank", (A, B))
    unranked = clues.decide(candidates={A, B, C}, scores={A: 2, B: 2, C: 1}, ai_order=[C], **common)
    assert (unranked.kind, unranked.reason, unranked.tied_dept_ids) == ("family_medicine", "question_limit", (A, B))
    single = clues.decide(candidates={A, B, C}, scores={A: 1, B: 2, C: 1}, ai_order=[A, C], **common)
    assert (single.kind, single.dept_id, single.reason) == ("decide", B, "question_limit")


def test_turn_limit_decides_only_a_single_top_target_department():
    # Both had present answers and B was asked less often, so the clue answers alone settle nothing.
    common = dict(ai_resolved_dept_id=None, askable=[clue("咳嗽", {A, B})], questions_asked=4, tiebreak_used=False,
                  user_declined=False, clue_confirmed_dept_ids={A, B}, question_limit=5, non_target_dept_ids={FAMILY},
                  ai_order=[B, A], asked_counts={A: 3, B: 1})
    before = clues.decide(candidates={A, B, FAMILY}, scores={A: 2, B: 1, FAMILY: 1}, **common)
    assert (before.kind, before.reason) == ("ask_clue", "continue")
    single = clues.decide(candidates={A, B, FAMILY}, scores={A: 2, B: 1, FAMILY: 1}, turn_limit_reached=True, **common)
    assert (single.kind, single.dept_id, single.reason) == ("decide", A, "turn_limit_top_score")
    # A tie is not broken by the AI rank here; the safety-net question is asked instead.
    tied = clues.decide(candidates={A, B, FAMILY}, scores={A: 2, B: 2, FAMILY: 1}, turn_limit_reached=True, **common)
    assert (tied.kind, tied.reason) == ("ask_clue", "continue")
    family = clues.decide(candidates={A, B, FAMILY}, scores={A: 1, B: 1, FAMILY: 2}, turn_limit_reached=True, **common)
    assert (family.kind, family.reason) == ("ask_clue", "continue")


def unsettled_answers():
    """A asked twice with one present answer, challenger B once with a present answer."""
    return [
        ClueConceptRecord(concept="甲", dept_id=A, dept_ids=[A], question_index=1, answer="present"),
        ClueConceptRecord(concept="乙", dept_id=A, dept_ids=[A], question_index=2, answer="absent"),
        ClueConceptRecord(concept="丙", dept_id=B, dept_ids=[B], question_index=3, answer="present"),
        ClueConceptRecord(concept="丁", dept_id=A, dept_ids=[A], question_index=4, answer="absent"),
    ]


def test_safety_net_turn_decides_the_top_department_instead_of_asking():
    settings = get_settings()
    retrieved = [hit(A, "頭暈"), hit(A, "頭痛"), hit(B, "頭暈")]
    case = decision_case(asked=4)
    case.conversation_state.clue_concepts_asked = unsettled_answers()
    case.conversation_state.turn_count = settings.safety_net_ask_turn - 1
    reasoning.apply_clue_decision(case, [clue("咳嗽", {A, B})], retrieved, ACTIVE)
    assert (case.conversation_state.clue_decision.kind, case.conversation_state.clue_decision.reason) == (
        "decide", "turn_limit_top_score")
    assert case.department_result.dept_id == A and case.conversation_state.department_status == "resolved"
    # Once the safety-net question was asked, the patient's choice to continue is kept.
    later = decision_case(asked=4)
    later.conversation_state.clue_concepts_asked = unsettled_answers()
    later.conversation_state.turn_count = settings.safety_net_ask_turn
    later.conversation_state.safety_net_asked = True
    reasoning.apply_clue_decision(later, [clue("咳嗽", {A, B})], retrieved, ACTIVE)
    assert later.conversation_state.clue_decision.kind == "ask_clue"


def test_tie_takes_the_ai_ranked_department_and_waits_for_confirmation():
    case = ranked(decision_case(), [C, B, A])
    reasoning.apply_clue_decision(case, [clue("咳嗽", {A, B})], TIED, ACTIVE)
    decision = case.conversation_state.clue_decision
    assert (decision.kind, decision.dept_id, decision.reason) == ("decide", B, "tie_broken_by_ai_rank")
    assert set(decision.tied_dept_ids) == {A, B}
    assert case.department_result.dept_id == B and case.conversation_state.department_status == "resolved"
    assert case.conversation_state.clue_tiebreak_used is False


def test_tie_outside_the_ai_ranking_still_falls_back_to_family_medicine():
    for order in ([C], []):
        case = ranked(decision_case(), order)
        reasoning.apply_clue_decision(case, [clue("咳嗽", {A, B})], TIED, ACTIVE)
        assert case.conversation_state.clue_decision.kind == "family_medicine"
        assert case.department_result.dept_id == FAMILY


def test_family_medicine_ranked_first_in_a_tie_uses_the_general_confirmation():
    case = ranked(decision_case(candidates=(A, FAMILY)), [FAMILY, A])
    # A single target department: the accompanying-symptom question was answered with nothing new.
    case.conversation_state.companion_questions_asked = 1
    case.conversation_state.companion_question_sources = ["頭暈"]
    case.history_records = [Message(role="user", content="頭暈"), Message(role="user", content="沒有")]
    case.patient_input.symptom = "頭暈"
    case.conversation_state.clarification_evidence = {clues.clue_intent(1): "沒有"}
    reasoning.apply_clue_decision(case, [], [hit(A, "頭暈"), hit(FAMILY, "頭暈")], ACTIVE)
    decision = case.conversation_state.clue_decision
    assert (decision.kind, decision.dept_id, decision.reason) == ("decide", FAMILY, "tie_broken_by_ai_rank")
    conversation_service.advance_conversation(case, None, user_sources=["沒有"], current_extractions=[])
    assert case.conversation_state.reply_source == "CONFIRMATION" and case.conversation_state.is_complete
    reply = chat_route.build_department_confirmation_reply(case)
    assert reply != get_settings().family_medicine_fallback_message + CONFIRM_PROMPT
    assert case.department_result.childDept in reply


class RankedConvergence:
    """Fake convergence AI: ambiguous, ranking the given departments in order."""

    def __init__(self, order):
        self.order = order

    async def __call__(self, prompt, *, purpose):
        retrieved = json.loads(prompt.split("\n資料：", 1)[1])["retrieved_official_evidence"]
        candidates = []
        for index, dept_id in enumerate(self.order):
            item = next(entry for entry in retrieved if entry["dept_id"] == dept_id)
            support = {key: item[key] for key in ("patient_source_text", "knowledge_source_id", "knowledge_concept")}
            candidates.append({"dept_id": dept_id, "confidence": 0.6 - 0.1 * index, "supporting_evidence": [support]})
        return json.dumps({"status": "ambiguous", "candidates": candidates}, ensure_ascii=False)


def test_tie_at_the_question_limit_never_asks_a_sixth_question():
    order = sorted(DIZZY - {FAMILY})[::-1][:2]
    convergence = RankedConvergence(order)
    case, body = turn(new_case("decide-ai-rank"), "最近一直頭暈", [symptom("頭暈", "頭暈")], convergence=convergence)
    # Absent answers never subtract, so every dizziness department stays tied at the limit.
    for _ in range(4):
        case, body = turn(case, "沒有", assertion="absent", convergence=convergence)
        assert body["conversation_state"]["clue_decision"]["kind"] == "ask_clue"
    case, body = turn(case, "沒有", assertion="absent", convergence=convergence)
    decision = body["conversation_state"]["clue_decision"]
    assert (decision["kind"], decision["dept_id"], decision["reason"]) == ("decide", order[0], "tie_broken_by_ai_rank")
    assert set(decision["tied_dept_ids"]) == DIZZY
    assert (case.conversation_state.clue_questions_asked, case.conversation_state.clue_tiebreak_used) == (5, False)
    assert_waiting_confirmation(body, order[0])
    assert body["reply"] != get_settings().family_medicine_fallback_message + CONFIRM_PROMPT


# Benchmark baseline (docs/BENCHMARK_BASELINE_20261028.md): validated convergence candidates, in AI
# order, on the turn the old flow fell back to family medicine after 6 clue questions. Every listed
# candidate is tied here; the decision must take the first AI-ranked one instead of family medicine.
BENCHMARK_FINAL_TURNS = {
    "c09_run1": [REHAB],
    "c09_run2": [RHEUM, ORTHO, REHAB],
    "c10_run1": [A, NEURO, NEURO_SURG],
    "c10_run2": [A, FAMILY, NEURO],
    "c11_run1": [CARDIO, A, META],
    "c11_run2": [CARDIO, A],
    "c16_run1": [A, PSY],
    "c16_run2": [A, PSY],
}


@pytest.mark.parametrize("name", sorted(BENCHMARK_FINAL_TURNS))
def test_benchmark_family_medicine_fallbacks_take_the_ai_ranked_department(name):
    order = BENCHMARK_FINAL_TURNS[name]
    tied = set(order) | {FAMILY}
    case = ranked(decision_case(candidates=tuple(sorted(tied))), order)
    # c09_run1 has a single target department, whose accompanying-symptom question is done.
    case.conversation_state.companion_questions_asked = 1
    case.conversation_state.companion_question_sources = ["症狀"]
    reasoning.apply_clue_decision(case, [], [hit(dept_id, "症狀") for dept_id in tied], ACTIVE)
    decision = case.conversation_state.clue_decision
    assert (decision.kind, decision.dept_id, decision.reason) == ("decide", order[0], "tie_broken_by_ai_rank")
    assert set(decision.tied_dept_ids) == tied
    assert case.department_result.dept_id == order[0]


# --- Commit 6: decline_further_questions ------------------------------------------------------------

def test_decline_decides_from_current_evidence_without_another_clue_question():
    case = decision_case(asked=1)
    case.conversation_state.clue_user_declined = True
    reasoning.apply_clue_decision(case, [clue("咳嗽", {A, B})], [*TIED, hit(A, "暈")], ACTIVE)
    decision = case.conversation_state.clue_decision
    assert (decision.kind, decision.dept_id, decision.reason) == ("decide", A, "user_declined")
    assert case.department_result.dept_id == A and case.conversation_state.department_status == "resolved"


def test_decline_tie_takes_the_ai_ranked_department_else_family_medicine():
    case = ranked(decision_case(asked=1), [C, B, A])
    case.conversation_state.clue_user_declined = True
    reasoning.apply_clue_decision(case, [clue("咳嗽", {A, B})], TIED, ACTIVE)
    decision = case.conversation_state.clue_decision
    assert (decision.kind, decision.dept_id, decision.reason) == ("decide", B, "tie_broken_by_ai_rank")

    case = ranked(decision_case(asked=1), [C])
    case.conversation_state.clue_user_declined = True
    reasoning.apply_clue_decision(case, [clue("咳嗽", {A, B})], TIED, ACTIVE)
    decision = case.conversation_state.clue_decision
    assert (decision.kind, decision.reason) == ("family_medicine", "user_declined")
    assert case.department_result.dept_id == FAMILY


def test_declining_a_pending_clue_question_skips_the_reask_and_asks_for_confirmation():
    case, body = dizzy_start("decline-pending")
    assert body["conversation_state"]["clue_decision"]["kind"] == "ask_clue"
    asked = case.conversation_state.clue_questions_asked
    # A completed safety screen implies grounded details were already collected.
    case.conversation_state.clarification_evidence = {"onset_time": "昨天開始"}
    case, body = turn(case, "我不想再回答了，直接幫我選科", act="decline_further_questions")
    state = body["conversation_state"]
    assert state["clue_decision"]["kind"] in {"decide", "family_medicine"}
    assert state["clue_decision"]["reason"] in {"user_declined", "tie_broken_by_ai_rank"}
    assert case.conversation_state.clue_questions_asked == asked
    assert case.conversation_state.pending_clue_question_index is None
    assert all(record.answer != "unanswered" for record in case.conversation_state.clue_concepts_asked)
    assert_waiting_confirmation(body, state["clue_decision"]["dept_id"])


def test_decline_is_remembered_so_a_later_turn_does_not_ask_clue_questions_again():
    # 310 report 2026-10-30: after "我不想回答了，直接給我結果" and the time question, the next
    # turn recomputed departments without the decline and asked another clue question.
    case, _ = dizzy_start("decline-remembered")
    case.conversation_state.clarification_evidence = {"onset_time": "昨天開始"}
    # As on 310: the time question comes after the decision, before confirmation.
    case.conversation_state.availability_status = "not_asked"
    case, body = turn(case, "我不想回答了，直接給我結果", act="decline_further_questions")
    assert case.conversation_state.clue_user_declined is True
    assert body["next_question"] == conversation_service.AVAILABILITY_QUESTION
    decided = body["conversation_state"]["clue_decision"]["dept_id"]
    case, body = turn(case, "都可以吧", act="answer", answer=dict(
        answered_intent=conversation_service.AVAILABILITY_PENDING_INTENT, answer_status="answered",
        answer_source_text="都可以吧", answer_confidence=0.95))
    assert body["conversation_state"]["last_dialogue_act"] == "answer"
    state = body["conversation_state"]
    assert (state["clue_decision"]["kind"], state["clue_decision"]["dept_id"]) == (
        body["conversation_state"]["clue_decision"]["kind"], decided)
    assert state["clue_decision"]["reason"] in {"user_declined", "tie_broken_by_ai_rank"}
    assert case.conversation_state.pending_clue_question_index is None
    assert state["reply_source"] != "CLUE_TEMPLATE"


def test_revision_clears_the_decline():
    case = decision_case()
    case.conversation_state.clue_user_declined = True
    chat_route._begin_revision(case)
    assert case.conversation_state.clue_user_declined is False
