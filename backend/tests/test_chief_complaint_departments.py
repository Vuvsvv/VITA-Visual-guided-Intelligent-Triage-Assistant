"""2026-11-03: the recommended department must cover the chief complaint.

「掉了好多頭髮」 then 「眼睛很乾」「眼睛有刺痛」 was decided as 眼科, which does not see hair loss.
"""
from app.services.department_knowledge import lookup_concept
from tests.test_clue_decision import RECORDS, new_case, symptom, turn

DERM, RHEUM, EYE = 1252, 1255, 1329
DRY_EYE = {"乾眼症", "眼睛乾澀", "眼乾"}
EYE_PAIN = {"刺眼", "眼睛疼痛", "流淚", "急性結膜炎"}
HAIR = symptom("掉頭髮", "掉了好多頭髮")
EYE_PART = dict(symptom("眼睛", "眼睛"), field="body_part")


def answer_like_the_patient(case):
    """Dry eyes when asked about them, stinging eyes when asked about eye pain, otherwise no."""
    pending = {record.concept for record in case.conversation_state.clue_concepts_asked
               if record.question_index == case.conversation_state.pending_clue_question_index}
    if pending & DRY_EYE:
        return turn(case, "我最近眼睛很乾", [HAIR, EYE_PART, symptom("眼睛乾澀", "眼睛很乾")],
                    assertion=lambda name: "present" if name in DRY_EYE else "absent")
    if pending & EYE_PAIN:
        return turn(case, "眼睛有刺痛", [HAIR, EYE_PART, symptom("眼睛乾澀", "眼睛很乾"), symptom("眼睛疼痛", "眼睛有刺痛")],
                    assertion=lambda name: "present" if name in EYE_PAIN else "absent")
    return turn(case, "沒有", [HAIR], assertion="absent")


def test_hair_loss_with_dry_eyes_never_ends_in_a_department_that_does_not_see_hair_loss():
    assert EYE not in {record["canonical_dept_id"] for record in lookup_concept("掉頭髮", RECORDS)}
    case, body = turn(new_case("chief-hair-eye"), "我掉了好多頭髮", [HAIR])
    for _ in range(8):
        if body["department_result"]:
            break
        case, body = answer_like_the_patient(case)
    asked = [(record.question_index, record.concept, record.answer) for record in case.conversation_state.clue_concepts_asked]
    assert body["department_result"], asked
    assert body["department_result"]["dept_id"] != EYE, asked
    # Dry eyes are also a rheumatology concept, so they point the hair loss there.
    assert body["department_result"]["dept_id"] == RHEUM, asked
    # Eye-only concepts are never asked: 眼科 cannot be recommended for hair loss.
    assert not any(concept in EYE_PAIN for _, concept, _ in asked), asked
