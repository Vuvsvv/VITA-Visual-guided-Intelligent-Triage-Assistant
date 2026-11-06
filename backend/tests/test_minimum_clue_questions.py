"""2026-11-03: at least clue_min_questions clue questions before deciding between departments,
leader and challenger asked in turn, and one open question when only one department fits."""
import pytest

from app.config import get_settings
from app.schemas import ClueConceptRecord, ClueDecisionRecord
from app.services import conversation_service
from app.services import department_clues as clues
from tests.test_clue_decision import ACTIVE, FAMILY, clue, decision_case, hit, new_case, ranked, symptom, turn

DERM, RHEUM, PLASTIC, DENTAL, COLORECTAL = 1252, 1255, 1294, 1248, 1245


def decide(**overrides):
    values = dict(
        ai_resolved_dept_id=None, candidates={DERM, RHEUM}, scores={DERM: 1, RHEUM: 1},
        askable=[clue("多汗", {DERM}), clue("關節痛", {RHEUM})], questions_asked=0, tiebreak_used=False,
        user_declined=False, clue_confirmed_dept_ids=set(), question_limit=5, non_target_dept_ids={FAMILY},
        ai_order=[DERM, RHEUM], min_questions=3, asked_counts={},
    )
    values.update(overrides)
    return clues.decide(**values)


# --- decide() --------------------------------------------------------------------------------------

def test_a_present_answer_alone_does_not_decide_before_the_minimum():
    early = decide(scores={DERM: 2, RHEUM: 1}, questions_asked=1, clue_confirmed_dept_ids={DERM},
                   asked_counts={DERM: 1})
    assert (early.kind, early.reason) == ("ask_clue", "continue")
    settled = decide(scores={DERM: 3, RHEUM: 1}, questions_asked=3, clue_confirmed_dept_ids={DERM},
                     asked_counts={DERM: 2, RHEUM: 1})
    assert (settled.kind, settled.dept_id, settled.reason) == ("decide", DERM, "clue_confirmed_single_department")


def test_ai_resolution_waits_for_the_minimum_unless_nothing_is_askable():
    assert decide(ai_resolved_dept_id=DERM, questions_asked=2).kind == "ask_clue"
    assert decide(ai_resolved_dept_id=DERM, questions_asked=3).reason == "ai_resolved"
    assert decide(ai_resolved_dept_id=DERM, askable=[]).reason == "ai_resolved"


def test_declining_still_decides_at_once():
    result = decide(scores={DERM: 2, RHEUM: 1}, user_declined=True)
    assert (result.kind, result.dept_id, result.reason) == ("decide", DERM, "user_declined")


def test_leader_with_a_present_challenger_waits_until_both_were_asked_equally():
    unbalanced = decide(scores={DERM: 3, RHEUM: 2}, questions_asked=3, clue_confirmed_dept_ids={DERM, RHEUM},
                        asked_counts={DERM: 2, RHEUM: 1})
    assert unbalanced.kind == "ask_clue"
    balanced = decide(scores={DERM: 3, RHEUM: 2}, questions_asked=4, clue_confirmed_dept_ids={DERM, RHEUM},
                      asked_counts={DERM: 2, RHEUM: 2})
    assert (balanced.kind, balanced.dept_id, balanced.reason) == ("decide", DERM, "leader_confirmed")


def test_all_present_tie_asked_equally_goes_to_the_ai_rank_else_family_medicine():
    common = dict(scores={DERM: 3, RHEUM: 3}, questions_asked=4, clue_confirmed_dept_ids={DERM, RHEUM},
                  asked_counts={DERM: 2, RHEUM: 2})
    ranked_tie = decide(ai_order=[RHEUM, DERM], **common)
    assert (ranked_tie.kind, ranked_tie.dept_id, ranked_tie.reason) == ("decide", RHEUM, "tie_broken_by_ai_rank")
    unranked = decide(ai_order=[], **common)
    assert (unranked.kind, unranked.reason) == ("family_medicine", "balanced_tie")


def test_all_absent_tie_keeps_asking_until_the_limit():
    assert decide(questions_asked=3, asked_counts={DERM: 2, RHEUM: 1}).kind == "ask_clue"
    assert decide(questions_asked=5, asked_counts={DERM: 3, RHEUM: 2}).reason == "tie_broken_by_ai_rank"


def test_single_target_asks_the_companion_question_first():
    common = dict(candidates={DENTAL, FAMILY}, scores={DENTAL: 2, FAMILY: 1}, askable=[clue("牙齦腫", {DENTAL})],
                  ai_order=[DENTAL])
    assert (decide(companion_due=True, **common).kind) == "ask_companion"
    result = decide(companion_due=False, **common)
    assert (result.kind, result.dept_id, result.reason) == ("decide", DENTAL, "single_candidate")
    # The convergence AI's resolution also waits for the companion question.
    assert decide(companion_due=True, ai_resolved_dept_id=DENTAL, **common).kind == "ask_companion"


# --- Who the next clue question asks about --------------------------------------------------------

def test_hair_loss_all_yes_alternates_and_ends_in_a_balanced_tie():
    """掉頭髮, every clue answered present: 皮膚、風濕、皮膚、風濕, then the AI rank breaks the tie."""
    scores, counts, confirmed, targets = {DERM: 1, RHEUM: 1}, {}, set(), []
    for question in range(1, 5):
        order = clues.ranked_targets({DERM, RHEUM}, scores, [DERM, RHEUM], {FAMILY})
        target = clues.clue_target_department(order, counts, confirmed)
        targets.append(target)
        scores[target] += 1
        counts[target] = counts.get(target, 0) + 1
        confirmed.add(target)
        result = decide(scores=scores, questions_asked=question, clue_confirmed_dept_ids=confirmed,
                        asked_counts=counts)
        if question < 4:
            assert result.kind == "ask_clue"
    assert targets == [DERM, RHEUM, DERM, RHEUM]
    assert (result.kind, result.dept_id, result.reason) == ("decide", DERM, "tie_broken_by_ai_rank")


def test_before_any_present_answer_ranked_departments_are_asked_in_turn():
    order = [DERM, RHEUM, PLASTIC]
    assert clues.clue_target_department(order, {}, set()) == DERM
    assert clues.clue_target_department(order, {DERM: 1}, set()) == RHEUM
    assert clues.clue_target_department(order, {DERM: 1, RHEUM: 1}, set()) == PLASTIC


def test_asked_counts_skip_unanswered_questions():
    asked = [
        ClueConceptRecord(concept="甲", dept_id=DERM, dept_ids=[DERM], question_index=1, answer="absent"),
        ClueConceptRecord(concept="乙", dept_id=DERM, dept_ids=[DERM, RHEUM], question_index=2, answer="present"),
        ClueConceptRecord(concept="丙", dept_id=RHEUM, dept_ids=[RHEUM], question_index=3),
    ]
    assert clues.asked_question_counts(asked, {DERM, RHEUM}) == {DERM: 2, RHEUM: 1}


def test_clue_offer_holds_only_the_target_departments_clues():
    from app.services import department_reasoning_service as reasoning
    case = ranked(decision_case(candidates=(DERM, RHEUM), asked=0), [DERM, RHEUM])
    retrieved = [dict(hit(DERM, "掉頭髮"), knowledge_concept="掉頭髮"),
                 dict(hit(RHEUM, "掉頭髮"), knowledge_concept="掉頭髮")]
    from app.services.department_knowledge import load_department_knowledge
    _, records = load_department_knowledge()
    reasoning.store_clue_offer(case, records, retrieved)
    offer = case.conversation_state.department_clue_offer
    assert offer and all(DERM in item.dept_ids for item in offer)
    # Concepts no other candidate holds come first.
    assert offer[0].dept_ids == [DERM]


# --- Chat flow -------------------------------------------------------------------------------------

def test_single_department_symptom_gets_one_open_question_before_confirmation():
    # 「牙痛」 also reaches 口腔顎面外科 since the 2026-11-03 Taichung table batch and 「痔瘡」 reaches
    # 一般外科 since the 2026-11-04 Fenglin branch table; 「肛裂」 is still one department.
    case = new_case("companion-anal-fissure")
    case.conversation_state.clarification_evidence = {"onset_time": "昨天開始"}
    case, body = turn(case, "我肛裂", [symptom("肛裂", "肛裂")])
    state = body["conversation_state"]
    assert state["clue_decision"]["kind"] == "ask_companion"
    assert state["reply_source"] == "COMPANION_QUESTION" and body["department_result"] is None
    assert body["next_question"] == "除了「肛裂」，最近還有沒有其他不舒服的地方？"
    assert state["pending_clarification_intent"] == conversation_service.COMPANION_PENDING_INTENT
    case, body = turn(case, "沒有了", **no_more("沒有了"))
    decision = body["conversation_state"]["clue_decision"]
    assert (decision["kind"], decision["reason"], decision["dept_id"]) == ("decide", "single_candidate", COLORECTAL)
    assert body["conversation_state"]["companion_questions_asked"] == 1
    assert body["conversation_state"]["stage"] == "waiting_confirmation"


def no_more(text):
    return dict(act="answer", answer=dict(
        answered_intent=conversation_service.COMPANION_PENDING_INTENT, answer_status="answered",
        answer_source_text=text, answer_confidence=0.95))


def test_companion_answer_with_a_new_scored_symptom_asks_once_more():
    case = new_case("companion-again")
    case.conversation_state.clarification_evidence = {"onset_time": "昨天開始"}
    case, _ = turn(case, "我肛裂", [symptom("肛裂", "肛裂")])
    # 「下腹痛」 is another official 直腸外科 concept, but it also brings other departments in.
    case, body = turn(case, "還有下腹痛", [symptom("肛裂", "肛裂"), symptom("下腹痛", "下腹痛")])
    # A symptom first said after the safety screen reopens it (test_safety_gaps.py).
    assert body["conversation_state"]["reply_source"] == "SAFETY_RESCREEN"
    case, body = turn(case, "都沒有", answer=dict(
        answered_intent="safety_screen", answer_status="answered", answer_assertion="absent",
        answer_source_text="都沒有", answer_confidence=0.95))
    state = body["conversation_state"]
    if len([dept for dept in state["department_clue_candidates"] if dept != FAMILY]) == 1:
        assert body["next_question"] == conversation_service.COMPANION_FOLLOWUP_QUESTION
    else:
        # More departments now fit, so the clue questions take over.
        assert state["clue_decision"]["kind"] == "ask_clue"


def test_stale_companion_decision_is_not_asked_again():
    case = new_case("companion-stale")
    state = case.conversation_state
    state.clue_decision = ClueDecisionRecord(kind="ask_companion", reason="single_candidate", scored_sources=["牙痛"])
    state.companion_questions_asked, state.companion_question_sources = 1, ["牙痛"]
    assert not conversation_service.companion_question_due(case)
    state.clue_decision.scored_sources = ["牙痛", "臉頰腫"]
    assert conversation_service.companion_question_due(case)
    state.companion_questions_asked = conversation_service.COMPANION_QUESTION_LIMIT
    assert not conversation_service.companion_question_due(case)


@pytest.mark.parametrize("asked, expected", [(0, "除了「頭暈」，"), (1, "還有其他")])
def test_companion_question_wording(asked, expected):
    assert conversation_service.companion_question("頭暈", asked).startswith(expected)
    assert not conversation_service._UNSAFE_QUESTION.search(conversation_service.companion_question("頭暈", asked))


def test_minimum_comes_from_settings():
    assert get_settings().clue_min_questions == 3
    assert {row["dept_id"] for row in ACTIVE} >= {DERM, RHEUM, DENTAL}
