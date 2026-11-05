from app.config import get_settings
from app.schemas import CandidateDepartment, ClueConceptRecord, Message, SemanticExtraction, TriageCase
from app.services import department_clues as clues
from app.services import department_reasoning_service as reasoning
from app.services.conversation_service import _clue_prompt, ClueQuestionSpec
from app.services.department_knowledge import load_department_knowledge

FAMILY = get_settings().family_medicine_dept_id
ORTHO, NEUROSURG, REHAB, NEURO, URO, PAIN, PLASTIC, GYN = 1298, 1244, 1265, 1274, 1314, 1345, 1294, 1319
METAB, GENERAL, RHEUMA = 1237, 1232, 1255
A, B = 1101, 1102


def record(concept, dept_id, evidence_type="direct_routing_evidence", priority=1):
    return dict(concept=concept, canonical_dept_id=dept_id, evidence_type=evidence_type, source_priority=priority)


RECORDS = [
    record("家醫獨有", FAMILY), record("家醫與A", FAMILY), record("家醫與A", A),
    record("A獨有", A), record("B獨有", B), record("A與B共有", A), record("A與B共有", B),
]


def offered(askable):
    return [clue.concept for clue in askable]


def test_family_medicine_only_concept_is_not_a_clue_but_keeps_scoring():
    candidates = {FAMILY, A, B}
    askable = clues.clue_concepts(RECORDS, candidates, set(), [], {FAMILY})

    assert "家醫獨有" not in offered(askable)
    shared = next(clue for clue in askable if clue.concept == "家醫與A")
    assert shared.dept_ids == {FAMILY, A}
    assert "家醫獨有" in offered(clues.clue_concepts(RECORDS, candidates, set(), []))
    asked = [ClueConceptRecord(concept="家醫與A", dept_id=A, dept_ids=[FAMILY, A], question_index=1, answer="present")]
    assert clues.department_scores([], candidates, asked) == {FAMILY: 1, A: 1, B: 0}


def test_tiebreak_never_targets_family_medicine_and_tie_still_ends_there():
    candidates = {FAMILY, A, B}
    askable = clues.clue_concepts(RECORDS, candidates, set(), [], {FAMILY})

    # 家醫與A singles out only family medicine within the tie, so it is no tiebreaker.
    assert sorted(offered(clues.tiebreak_clues(askable, {FAMILY, B}, {FAMILY}))) == ["A與B共有", "B獨有"]
    assert "家醫與A" in offered(clues.tiebreak_clues(askable, {FAMILY, B}))
    only_family_targets = [clue for clue in askable if clue.concept == "家醫與A"]
    assert clues.tiebreak_clues(only_family_targets, {FAMILY, B}, {FAMILY}) == []
    decision = clues.decide(
        ai_resolved_dept_id=None, candidates=candidates, scores={FAMILY: 2, A: 1, B: 2},
        askable=only_family_targets, questions_asked=5, tiebreak_used=False, user_declined=False,
        clue_confirmed_dept_ids=set(), question_limit=5, non_target_dept_ids={FAMILY},
    )
    assert decision.kind == "family_medicine" and decision.tied_dept_ids == tuple(sorted({FAMILY, B}))


C = 1103
RANK_RECORDS = [record("A獨有", A), record("B獨有", B), record("C獨有", C), record("A與C共有", A), record("A與C共有", C)]


def test_higher_score_department_clues_come_before_other_unique_clues():
    askable = clues.clue_concepts(RANK_RECORDS, {A, B, C}, set(), [])

    assert offered(clues.rank_clue_concepts(askable, 10)) == ["A獨有", "B獨有", "C獨有", "A與C共有"]
    priority = clues.department_priority({A: 2, B: 1, C: 0}, [])
    assert offered(clues.rank_clue_concepts(askable, 10, priority)) == ["A獨有", "A與C共有", "B獨有", "C獨有"]


def test_equal_scores_follow_convergence_order_and_unlisted_departments_last():
    askable = clues.clue_concepts(RANK_RECORDS, {A, B, C}, set(), [])
    scores = {A: 1, B: 1, C: 1}

    assert offered(clues.rank_clue_concepts(askable, 10, clues.department_priority(scores, [B]))) == [
        "B獨有", "A獨有", "C獨有", "A與C共有",
    ]
    assert offered(clues.rank_clue_concepts(askable, 10, clues.department_priority(scores, [C, A]))) == [
        "C獨有", "A與C共有", "A獨有", "B獨有",
    ]


def test_family_medicine_gives_no_target_priority():
    priority = clues.department_priority({FAMILY: 5, A: 1}, [FAMILY, A], {FAMILY})

    assert FAMILY not in priority


def back_pain_case():
    case = TriageCase(case_id="back-pain-clues")
    case.history_records = [Message(role="user", content="周一開始腰酸背痛"), Message(role="user", content="腳會麻")]
    case.semantic_extractions = [
        SemanticExtraction(field="symptom", normalized_value="腰酸背痛", source_text="腰酸背痛",
                           semantic_status="available", assertion="present", confidence=.99),
        SemanticExtraction(field="accompanying_symptoms", normalized_value=["手腳會麻"], source_text="腳會麻",
                           semantic_status="available", assertion="present", confidence=.99),
    ]
    case.conversation_state.candidate_departments = [
        CandidateDepartment(dept_id=dept_id, parentDept="外科部", childDept="候選", confidence=.6,
                            supporting_evidence=[]) for dept_id in (ORTHO, NEUROSURG, REHAB)
    ]
    return case


def test_back_pain_first_clue_targets_likely_departments_not_urology_or_family_medicine():
    _, records = load_department_knowledge()
    resolutions = [dict(status="resolved", canonical_dept_id=r["canonical_dept_id"], db_dept_id=r["canonical_dept_id"],
                        db_parent_dept=r["canonical_parent_dept"], db_child_dept=r["canonical_child_dept"])
                   for r in records]
    case = back_pain_case()
    retrieved = reasoning.retrieve_official_evidence(case, records, resolutions)
    # 整形外科 joins through 「手腳會麻」 and 婦產科 through 「腰酸背痛」 on the Taichung symptom table (2026-11-03);
    # 新陳代謝科 joins through 「手腳會麻」 on the Pingtung VGH table, and 一般內科 and 過敏風濕科 through
    # 「腰酸背痛」 on the E-Da and Shin Kong tables (2026-11-04).
    assert clues.candidate_dept_ids(retrieved) == {
        ORTHO, NEUROSURG, REHAB, NEURO, URO, PAIN, PLASTIC, GYN, METAB, GENERAL, RHEUMA, FAMILY}

    askable = reasoning.store_clue_offer(case, records, retrieved)
    offer = case.conversation_state.department_clue_offer
    first = set(offer[0].dept_ids)
    assert first & {ORTHO, NEUROSURG, REHAB}
    assert first - {URO, GYN, METAB, GENERAL, RHEUMA, FAMILY}
    assert all(set(item.dept_ids) - {FAMILY} for item in offer)
    assert all(clue.dept_ids - {FAMILY} for clue in askable)
    selected, target = clues.template_clue_selection(offer, get_settings().clue_concepts_per_question)
    assert target in {ORTHO, NEUROSURG, REHAB} and selected


def test_clue_prompt_asks_for_a_full_sentence_starting_with_qing_wen():
    spec = ClueQuestionSpec(clues.clue_intent(1), (), False, 3)

    assert "以「請問」開頭（例如「請問您有沒有…」），不要只列概念加「嗎」" in _clue_prompt(spec)


def test_clue_prompt_asks_about_the_family_member_being_described():
    # 2026-11-02: 「我小女兒感冒」 got the clue 「請問您是否有嗅覺喪失」.
    spec = ClueQuestionSpec(clues.clue_intent(1), (), False, 3)

    assert "若使用者在描述家人或他人的狀況" in _clue_prompt(spec)
    assert "您" not in clues.clue_template_question(["嗅覺喪失"])
    assert "您" not in clues.clue_template_question(["嗅覺喪失"], reask=True)
