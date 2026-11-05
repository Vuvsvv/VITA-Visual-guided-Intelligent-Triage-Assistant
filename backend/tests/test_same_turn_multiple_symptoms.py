from app.schemas import Message, SemanticExtraction, TriageCase
from app.services import department_reasoning_service as service
from app.services.department_knowledge import load_department_knowledge
from app.services.rag_triage_adapter import _build_symptom_collection_prompt
from app.services.rule_engine import apply_semantic_extractions


def symptom(value, source, **changes):
    fields = dict(field="symptom", normalized_value=value, source_text=source,
                  semantic_status="available", assertion="present", confidence=.99, extractor="ai")
    fields.update(changes)
    return SemanticExtraction(**fields)


def case_with(*texts):
    case = TriageCase(case_id="same-turn-symptoms")
    case.history_records = [Message(role="user", content=text) for text in texts]
    return case


def resolutions(records):
    return [dict(status="resolved", canonical_dept_id=r["canonical_dept_id"],
                 db_dept_id=r["canonical_dept_id"], db_parent_dept=r["canonical_parent_dept"],
                 db_child_dept=r["canonical_child_dept"]) for r in records]


def test_second_symptom_in_same_turn_reaches_department_retrieval():
    case = case_with("最近頭腦昏沉，吃不太下")
    apply_semantic_extractions(case, [symptom("頭腦昏沉", "頭腦昏沉"), symptom("食慾不振", "吃不太下")],
                               allow_red_flag_completion=False)

    assert case.patient_input.symptom == "頭腦昏沉"
    assert [item.normalized_value for item in case.semantic_extractions] == ["頭腦昏沉", "食慾不振"]
    assert {item["normalized_value"] for item in service.accepted_semantic_evidence(case)} == {"頭腦昏沉", "食慾不振"}
    _, records = load_department_knowledge()
    retrieved = service.retrieve_official_evidence(case, records, resolutions(records))
    assert {(item["patient_source_text"], item["knowledge_concept"]) for item in retrieved} == {("吃不太下", "食慾不振")}
    assert 1232 in {item["dept_id"] for item in retrieved}


def test_additional_symptom_in_later_turn_is_kept_without_overwriting():
    case = case_with("頭腦昏沉", "也吃不太下")
    apply_semantic_extractions(case, [symptom("頭腦昏沉", "頭腦昏沉")])
    apply_semantic_extractions(case, [symptom("食慾不振", "吃不太下")])

    assert case.patient_input.symptom == "頭腦昏沉"
    assert [item.normalized_value for item in case.semantic_extractions] == ["頭腦昏沉", "食慾不振"]


def test_repeated_symptom_concept_is_still_a_duplicate():
    case = case_with("頭腦昏沉", "還是頭腦昏沉")
    apply_semantic_extractions(case, [symptom("頭腦昏沉", "頭腦昏沉")])
    apply_semantic_extractions(case, [symptom("頭腦昏沉", "還是頭腦昏沉")])

    assert len(case.semantic_extractions) == 1
    assert case.patient_input.symptom == "頭腦昏沉"


def test_other_filled_fields_keep_duplicate_rejection():
    case = case_with("痛了三天", "其實五天")
    duration = dict(field="duration", semantic_status="available", confidence=.99, extractor="ai")
    apply_semantic_extractions(case, [SemanticExtraction(normalized_value="3天", source_text="三天", **duration)])
    apply_semantic_extractions(case, [SemanticExtraction(normalized_value="5天", source_text="五天", **duration)])

    assert case.patient_input.duration == "3天"
    assert [item.normalized_value for item in case.semantic_extractions] == ["3天"]


def test_low_confidence_additional_symptom_is_not_evidence():
    case = case_with("最近頭腦昏沉，吃不太下")
    apply_semantic_extractions(case, [symptom("頭腦昏沉", "頭腦昏沉"), symptom("食慾不振", "吃不太下", confidence=.3)])

    assert [item.normalized_value for item in case.semantic_extractions] == ["頭腦昏沉"]


def test_turn_prompt_routes_extra_symptoms_to_accompanying_symptoms():
    case = case_with("最近頭腦昏沉，吃不太下")
    prompt = _build_symptom_collection_prompt(case, ["最近頭腦昏沉，吃不太下"])

    assert "最主要的一個放 symptom，其餘各自以 accompanying_symptoms 輸出" in prompt
