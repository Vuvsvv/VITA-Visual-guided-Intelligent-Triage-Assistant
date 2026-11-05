"""A literal match never cuts into a word: an AI body part or a longer official concept.

The extractions below are what the live turn interpreter returned on 2026-11-04.
"""
from app.schemas import Message, SemanticExtraction, TriageCase
from app.services import department_reasoning_service as service
from app.services.department_knowledge import load_department_knowledge
from app.services.rule_engine import apply_semantic_extractions

_, RECORDS = load_department_knowledge()
RESOLUTIONS = [dict(status="resolved", canonical_dept_id=r["canonical_dept_id"], db_dept_id=r["canonical_dept_id"],
                    db_parent_dept=r["canonical_parent_dept"], db_child_dept=r["canonical_child_dept"])
               for r in RECORDS]


def item(field, value, source):
    return SemanticExtraction(field=field, normalized_value=value, source_text=source, semantic_status="available",
                              assertion=None if field == "body_part" else "present", confidence=.99,
                              extractor="ai")


def concepts(text, symptom, symptom_source, body_part=None):
    case = TriageCase(case_id="word-boundary")
    case.history_records.append(Message(role="user", content=text))
    extractions = [item("symptom", symptom, symptom_source)]
    if body_part:
        extractions.append(item("body_part", body_part, body_part))
    apply_semantic_extractions(case, extractions, allow_red_flag_completion=False)
    return {row["knowledge_concept"] for row in service.retrieve_official_evidence(case, RECORDS, RESOLUTIONS)}


def test_a_body_part_the_ai_split_off_does_not_hide_a_headache():
    assert "頭痛" not in concepts("我舌頭痛", "疼痛", "舌頭痛", "舌頭")
    assert "頭痛" not in concepts("我骨頭痛", "疼痛", "骨頭痛", "骨頭")
    assert "背痛" not in concepts("我手背痛", "疼痛", "手背痛", "手背")
    assert "囊腫" not in concepts("陰囊腫大", "腫大", "陰囊腫大", "陰囊")


def test_the_ai_reading_still_matches_when_it_names_the_concept():
    # 「額頭痛」: the AI's own symptom name is 頭痛, so the official 頭痛 still matches.
    assert "頭痛" in concepts("額頭痛", "頭痛", "額頭痛", "額頭")
    assert "胸痛" in concepts("左胸痛", "胸痛", "胸痛", "左胸")


def test_symptoms_without_a_body_part_are_unchanged():
    assert "經痛" in concepts("我經痛很嚴重", "經痛", "經痛")
    assert "頭痛" in concepts("我頭痛", "頭痛", "頭痛")


def test_a_longer_official_concept_keeps_its_inner_words():
    found = concepts("坐骨神經痛", "坐骨神經痛", "坐骨神經痛")
    assert "坐骨神經痛" in found and "經痛" not in found
    found = concepts("我有心肌炎", "心肌炎", "我有心肌炎")
    assert "心肌炎" in found and "肌炎" not in found


def test_literal_concepts():
    official = {"頭痛": "頭痛", "經痛": "經痛", "神經痛": "神經痛"}
    assert service.literal_concepts("舌頭痛又頭痛", official, {"舌頭"}) == {"頭痛"}
    assert service.literal_concepts("舌頭痛", official, {"舌頭"}) == set()
    assert service.literal_concepts("神經痛", official, set()) == {"神經痛"}
    assert service.literal_concepts("神經痛之外也有經痛", official, set()) == {"神經痛", "經痛"}
