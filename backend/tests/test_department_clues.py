"""Pure clue-question selection, scoring and decision helpers (no AI, no SQL)."""
from app.config import Settings
from app.schemas import ClueConceptRecord, ConversationState, TriageCase
from app.services import department_clues as clues

ENT, NEURO, CARDIO = 1333, 1241, 1244


def record(concept, dept_id, evidence_type="direct_routing_evidence", priority=1):
    return dict(concept=concept, canonical_dept_id=dept_id, evidence_type=evidence_type, source_priority=priority)


def hit(dept_id, source, concept="頭暈", **extra):
    return dict(dept_id=dept_id, patient_source_text=source, knowledge_concept=concept, **extra)


def asked(concept, dept_id, index, answer="unanswered"):
    return ClueConceptRecord(concept=concept, dept_id=dept_id, question_index=index, answer=answer)


RECORDS = [
    record("頭暈", ENT), record("頭暈", NEURO), record("頭暈", CARDIO),
    record("耳鳴", ENT), record("聽力變化", ENT, priority=2),
    record("肢體無力", NEURO, priority=2),
    record("心悸", CARDIO, priority=1),
    record("噁心", ENT, priority=2), record("噁心", NEURO, priority=2),
    record("國字臉", ENT, "department_specialty_evidence", 1),
]
CANDIDATES = {ENT, NEURO, CARDIO}


def decide(**overrides):
    params = dict(
        ai_resolved_dept_id=None, candidates=CANDIDATES, scores={ENT: 1, NEURO: 1, CARDIO: 1},
        askable=clues.clue_concepts(RECORDS, CANDIDATES, set(), []), questions_asked=1,
        tiebreak_used=False, user_declined=False, clue_confirmed_dept_ids=set(), question_limit=5,
    )
    params.update(overrides)
    return clues.decide(**params)


def test_candidates_are_all_retrieved_departments():
    retrieved = [hit(ENT, "頭暈"), hit(NEURO, "頭暈"), hit(CARDIO, "頭暈"), hit(ENT, "耳朵怪怪")]
    assert clues.candidate_dept_ids(retrieved) == CANDIDATES


def test_unique_concepts_rank_first_then_priority_then_text():
    ranked = clues.rank_clue_concepts(clues.clue_concepts(RECORDS, CANDIDATES, set(), []), 10)
    assert [clue.concept for clue in ranked] == ["心悸", "耳鳴", "聽力變化", "肢體無力", "噁心"]
    assert ranked[0].unique and ranked[0].dept_ids == {CARDIO}
    assert not ranked[-1].unique and ranked[-1].dept_ids == {ENT, NEURO}


def test_specialty_only_concept_is_never_a_clue():
    concepts = {clue.concept for clue in clues.clue_concepts(RECORDS, CANDIDATES, set(), [])}
    assert "國字臉" not in concepts


def test_concept_with_one_direct_record_is_asked_with_every_holding_candidate():
    records = RECORDS + [record("耳痛", ENT), record("耳痛", NEURO, "department_specialty_evidence", 3)]
    clue = next(item for item in clues.clue_concepts(records, CANDIDATES, set(), []) if item.concept == "耳痛")
    assert clue.dept_ids == {ENT, NEURO} and not clue.unique


def test_rank_respects_limit():
    assert len(clues.rank_clue_concepts(clues.clue_concepts(RECORDS, CANDIDATES, set(), []), 2)) == 2


def test_concept_shared_by_every_candidate_is_not_a_clue():
    concepts = {clue.concept for clue in clues.clue_concepts(RECORDS, CANDIDATES, set(), [])}
    assert "頭暈" not in concepts


def test_concepts_already_mentioned_are_excluded():
    retrieved = [hit(ENT, "耳朵嗡嗡叫", "耳鳴")]
    evidence = [dict(field="symptom", normalized_value="心悸", assertion="absent")]
    mentioned = clues.mentioned_concepts(retrieved, evidence)
    concepts = {clue.concept for clue in clues.clue_concepts(RECORDS, CANDIDATES, mentioned, [])}
    assert "耳鳴" not in concepts and "心悸" not in concepts and "肢體無力" in concepts


def test_absent_or_present_answers_are_not_asked_again_and_uncertain_moves_back():
    history = [asked("耳鳴", ENT, 1, "absent"), asked("心悸", CARDIO, 2, "present"),
               asked("聽力變化", ENT, 3, "uncertain")]
    ranked = clues.rank_clue_concepts(clues.clue_concepts(RECORDS, CANDIDATES, set(), history), 10)
    names = [clue.concept for clue in ranked]
    assert "耳鳴" not in names and "心悸" not in names
    assert names.index("肢體無力") < names.index("聽力變化")


def test_one_source_text_supporting_two_concepts_scores_once():
    retrieved = [hit(ENT, "天旋地轉", "頭暈"), hit(ENT, "天旋地轉", "眩暈"), hit(NEURO, "天旋地轉", "頭暈")]
    assert clues.department_scores(retrieved, {ENT, NEURO}, []) == {ENT: 1, NEURO: 1}


def test_absent_answers_do_not_subtract():
    retrieved = [hit(ENT, "頭暈"), hit(NEURO, "頭暈")]
    history = [asked("耳鳴", ENT, 1, "absent"), asked("肢體無力", NEURO, 2, "absent")]
    assert clues.department_scores(retrieved, {ENT, NEURO}, history) == {ENT: 1, NEURO: 1}


def test_two_present_questions_answered_with_same_text_score_twice():
    retrieved = [hit(ENT, "頭暈"), hit(ENT, "有", "耳鳴", evidence_origin=clues.CLUE_ANSWER_ORIGIN),
                 hit(ENT, "有", "聽力變化", evidence_origin=clues.CLUE_ANSWER_ORIGIN)]
    history = [asked("耳鳴", ENT, 1, "present"), asked("聽力變化", ENT, 2, "present")]
    assert clues.department_scores(retrieved, {ENT}, history) == {ENT: 1 + 2}


def test_three_present_concepts_in_one_question_score_once():
    history = [asked(name, ENT, 1, "present") for name in ("耳鳴", "聽力變化", "噁心")]
    assert clues.department_scores([hit(ENT, "頭暈")], {ENT}, history) == {ENT: 2}


def answered(concept, dept_id, index, source):
    return ClueConceptRecord(concept=concept, dept_id=dept_id, question_index=index, answer="present",
                             answer_source_text=source)


def test_symptom_restating_a_present_clue_answer_scores_once():
    # 病人答「有耳鳴」，Turn Interpreter 又把「耳鳴」抽成症狀：耳科只靠題號得 1 分。
    retrieved = [hit(ENT, "頭暈"), hit(ENT, "耳鳴", "耳鳴"),
                 hit(ENT, "有耳鳴", "耳鳴", evidence_origin=clues.CLUE_ANSWER_ORIGIN)]
    assert clues.department_scores(retrieved, {ENT}, [answered("耳鳴", ENT, 1, "有耳鳴")]) == {ENT: 1 + 1}


def test_synonym_inside_the_clue_answer_does_not_score_again():
    retrieved = [hit(ENT, "頭暈"), hit(ENT, "耳朵嗡嗡叫", "耳朵異音")]
    history = [answered("耳鳴", ENT, 1, "有，耳朵嗡嗡叫")]
    assert clues.department_scores(retrieved, {ENT}, history) == {ENT: 1 + 1}


def test_extra_symptom_in_a_clue_answer_still_scores():
    retrieved = [hit(ENT, "頭暈"), hit(ENT, "耳鳴", "耳鳴"), hit(ENT, "噁心", "噁心")]
    history = [answered("耳鳴", ENT, 1, "有耳鳴")]
    assert clues.department_scores(retrieved, {ENT}, history) == {ENT: 1 + 1 + 1}


def test_restated_answer_still_scores_for_a_department_the_question_did_not_credit():
    retrieved = [hit(ENT, "耳鳴", "耳鳴"), hit(NEURO, "耳鳴", "腦鳴")]
    history = [answered("耳鳴", ENT, 1, "耳鳴")]
    assert clues.department_scores(retrieved, {ENT, NEURO}, history) == {ENT: 1, NEURO: 1}


def test_joined_fragment_holding_a_restated_answer_does_not_score_again():
    retrieved = [hit(ENT, "頭暈"), hit(ENT, "耳鳴", "耳鳴"),
                 hit(ENT, "耳鳴頭脹", "耳鳴頭脹", patient_source_parts=["耳鳴", "頭脹"])]
    history = [answered("耳鳴", ENT, 1, "有耳鳴")]
    assert clues.department_scores(retrieved, {ENT}, history) == {ENT: 1 + 1}


def test_question_limit_selects_highest_score():
    result = decide(questions_asked=5, scores={ENT: 3, NEURO: 1, CARDIO: 1})
    assert (result.kind, result.dept_id, result.reason) == ("decide", ENT, "question_limit")


def test_below_limit_without_trigger_keeps_asking():
    assert decide().kind == "ask_clue"


def test_tie_without_tiebreak_asks_tiebreak_for_tied_departments():
    # 2026-10-28: a tie no longer asks a tiebreak question; without an AI ranking it goes to
    # family medicine. tiebreak_clues() is kept as a helper and still selects tied-unique clues.
    result = decide(questions_asked=5, scores={ENT: 2, NEURO: 2, CARDIO: 1})
    assert result.kind == "family_medicine" and result.tied_dept_ids == (NEURO, ENT)
    tiebreakers = clues.tiebreak_clues(clues.clue_concepts(RECORDS, CANDIDATES, set(), []), {ENT, NEURO})
    assert "噁心" not in {clue.concept for clue in tiebreakers}
    assert all(len(clue.dept_ids & {ENT, NEURO}) == 1 for clue in tiebreakers)


def test_tie_after_tiebreak_goes_to_family_medicine():
    result = decide(questions_asked=6, tiebreak_used=True, scores={ENT: 2, NEURO: 2, CARDIO: 1})
    assert result.kind == "family_medicine" and result.tied_dept_ids == (NEURO, ENT)


def test_tie_without_any_tied_unique_clue_goes_to_family_medicine():
    askable = [clue for clue in clues.clue_concepts(RECORDS, CANDIDATES, set(), []) if clue.concept == "噁心"]
    result = decide(askable=askable, questions_asked=5, scores={ENT: 2, NEURO: 2, CARDIO: 1})
    assert result.kind == "family_medicine"


def test_user_declined_with_tie_goes_directly_to_family_medicine():
    result = decide(user_declined=True, scores={ENT: 1, NEURO: 1, CARDIO: 1})
    assert (result.kind, result.reason) == ("family_medicine", "user_declined")


def test_user_declined_without_tie_decides_highest():
    result = decide(user_declined=True, scores={ENT: 1, NEURO: 2, CARDIO: 1})
    assert (result.kind, result.dept_id) == ("decide", NEURO)


def test_ai_resolved_candidate_is_adopted():
    result = decide(ai_resolved_dept_id=CARDIO, scores={ENT: 3, NEURO: 1, CARDIO: 1})
    assert (result.kind, result.dept_id, result.reason) == ("decide", CARDIO, "ai_resolved")


def test_ai_resolved_outside_candidates_is_ignored():
    assert decide(ai_resolved_dept_id=9999).kind == "ask_clue"


def test_clue_confirmed_single_department_decides():
    history = [asked("耳鳴", ENT, 1, "present"), asked("心悸", CARDIO, 2, "absent")]
    confirmed = clues.clue_confirmed_departments(history, CANDIDATES)
    assert confirmed == {ENT}
    scores = clues.department_scores([hit(dept, "頭暈") for dept in CANDIDATES], CANDIDATES, history)
    result = decide(questions_asked=2, clue_confirmed_dept_ids=confirmed, scores=scores)
    assert (result.kind, result.dept_id, result.reason) == ("decide", ENT, "clue_confirmed_single_department")


def test_present_clues_for_two_departments_do_not_trigger():
    history = [asked("耳鳴", ENT, 1, "present"), asked("心悸", CARDIO, 2, "present")]
    confirmed = clues.clue_confirmed_departments(history, CANDIDATES)
    assert decide(questions_asked=2, clue_confirmed_dept_ids=confirmed).kind == "ask_clue"


def test_no_askable_clues_decides_by_highest_score():
    result = decide(askable=[], questions_asked=1, scores={ENT: 1, NEURO: 3, CARDIO: 1})
    assert (result.kind, result.dept_id, result.reason) == ("decide", NEURO, "no_askable_clues")


def test_empty_candidates_defer_to_safety_net():
    result = decide(candidates=set(), scores={}, askable=[], questions_asked=5, user_declined=True)
    assert result.kind == "defer_to_safety_net"


def test_common_clue_departments_enforces_one_department_and_limit():
    offered = clues.rank_clue_concepts(clues.clue_concepts(RECORDS, CANDIDATES, set(), []), 10)
    assert clues.common_clue_departments(["耳鳴", "聽力變化"], offered, 3) == {ENT}
    assert clues.common_clue_departments(["耳鳴", "心悸"], offered, 3) == set()
    assert clues.common_clue_departments(["耳鳴", "聽力變化", "噁心", "頭暈"], offered, 3) == set()
    assert clues.common_clue_departments(["不存在"], offered, 3) == set()
    assert clues.common_clue_departments([], offered, 3) == set()


def test_append_clue_records_is_bounded_and_unanswered():
    history = clues.append_clue_records([], ["耳鳴", "聽力變化"], ENT, 1)
    assert [(item.concept, item.answer) for item in history] == [("耳鳴", "unanswered"), ("聽力變化", "unanswered")]
    assert len(clues.append_clue_records(history, ["x"] * 40, ENT, 2)) == clues.MAX_CLUE_CONCEPT_RECORDS


def test_old_snapshot_without_clue_fields_loads_with_defaults():
    case = TriageCase.model_validate({"case_id": "old", "conversation_state": {"turn_count": 3}})
    state = case.conversation_state
    assert (state.clue_questions_asked, state.clue_concepts_asked, state.clue_tiebreak_used) == (0, [], False)
    assert ConversationState.model_validate({}).clue_concepts_asked == []


def test_clue_settings_defaults():
    settings = Settings(_env_file=None)
    assert (settings.clue_question_limit, settings.clue_concepts_per_question,
            settings.clue_planner_concept_limit) == (5, 3, 10)
    assert settings.family_medicine_dept_id == 1233
    assert settings.family_medicine_fallback_message == "目前的症狀無法明確區分科別，系統建議先由家庭醫學科初步評估。"
