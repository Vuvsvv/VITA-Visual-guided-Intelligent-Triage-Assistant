"""STD, cancer and pregnancy concepts route from patient wording but are never asked as clues."""
from app.services import department_clues as clues
from app.services.department_knowledge import load_department_knowledge, lookup_concept

_, RECORDS = load_department_knowledge()
INFECTION, NOSE, GI, OBGYN = 1234, 1334, 1239, 1319


def askable(candidates):
    return {clue.concept for clue in clues.clue_concepts(RECORDS, candidates, set(), [])}


def test_cold_symptoms_never_ask_about_sexually_transmitted_diseases():
    # 「流鼻水咳嗽喉嚨痛」 reaches 感染科 and 鼻科; 2026-11-02 the clue asked 愛滋病 and 梅毒.
    concepts = askable({INFECTION, NOSE})
    assert "寒顫" in concepts
    assert not {"愛滋病", "梅毒"} & concepts


def test_cancer_and_pregnancy_are_never_asked():
    assert not {"肝癌", "胃癌", "大腸癌", "胰臟癌", "消化道腫瘤", "腹部腫瘤"} & askable({GI, NOSE})
    assert "懷孕" not in askable({OBGYN, NOSE})


def test_sensitive_concepts_still_route_when_the_patient_says_them():
    assert INFECTION in {record["canonical_dept_id"] for record in lookup_concept("梅毒", RECORDS)}
    assert OBGYN in {record["canonical_dept_id"] for record in lookup_concept("懷孕", RECORDS)}


def test_never_ask_markers():
    assert all(clues.never_ask_concept(concept) for concept in ("愛滋病", "淋巴癌", "白血病", "惡性肉瘤", "懷孕"))
    assert not any(clues.never_ask_concept(concept) for concept in ("寒顫", "皮下腫塊", "流鼻水", "小便疼痛"))
