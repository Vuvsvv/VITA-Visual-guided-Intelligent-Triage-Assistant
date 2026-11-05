import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.schemas import Message, SemanticExtraction, TriageCase
from app.services import department_reasoning_service as service
from app.services.conversation_service import (
    ClarificationSuggestion, DEPARTMENT_EVIDENCE_REPLY, advance_conversation,
)
from app.services.department_knowledge import load_department_knowledge


def ankle_case():
    case = TriageCase(case_id="semantic-retrieval-test")
    case.patient_input.symptom = "疼痛"
    case.patient_input.red_flags_checked = True
    case.history_records = [Message(role="user", content="腳踝疼痛一星期")]
    case.conversation_state.clarification_evidence = {"duration": "一星期"}
    case.semantic_extractions = [
        SemanticExtraction(field="symptom", normalized_value="疼痛", source_text="疼痛",
            semantic_status="available", assertion="present", confidence=.99),
        SemanticExtraction(field="body_part", normalized_value="腳踝", source_text="腳踝",
            semantic_status="available", confidence=.99),
    ]
    return case


def data():
    _, records = load_department_knowledge()
    return records, [dict(status="resolved", canonical_dept_id=r["canonical_dept_id"],
        db_dept_id=r["canonical_dept_id"], db_parent_dept=r["canonical_parent_dept"],
        db_child_dept=r["canonical_child_dept"]) for r in records]


def match(**changes):
    return dict(patient_source_text="疼痛", knowledge_concept="關節痛", confidence=.98, **changes)


def retrieve(case, matches):
    records, resolutions = data()
    provider = AsyncMock(side_effect=[json.dumps(dict(matches=matches), ensure_ascii=False),
        json.dumps(dict(judgments=[dict(patient_source_text=m["patient_source_text"],
            knowledge_concept=m["knowledge_concept"], supported=True, reason="語意相符") for m in matches]))])
    with patch.object(service, "complete_runtime_json", provider):
        result = asyncio.run(service.retrieve_semantic_official_evidence(case, records, resolutions))
    return result, provider


def test_split_anatomy_and_symptom_retrieve_only_existing_official_records():
    case = ankle_case()
    records, resolutions = data()
    assert service.retrieve_official_evidence(case, records, resolutions) == []
    result, provider = retrieve(case, [match()])
    assert result
    assert all(item["knowledge_concept"] == "關節痛" and item["patient_source_text"] == "疼痛" for item in result)
    allowed = {(r["canonical_dept_id"], r["source_id"], r["concept"]) for r in records}
    assert all((item["dept_id"], item["knowledge_source_id"], item["knowledge_concept"]) in allowed for item in result)
    prompt = provider.await_args.args[0]
    assert "腳踝" in prompt and "疼痛" in prompt


@pytest.mark.parametrize("field,value", [
    ("knowledge_concept", "不存在的概念"), ("patient_source_text", "膝蓋痛"),
    ("patient_source_text", "腳踝"), ("confidence", .1), ("confidence", True),
    ("confidence", float("nan")), ("confidence", 1.1),
])
def test_unsupported_semantic_match_is_not_routing_evidence(field, value):
    proposal = match()
    proposal[field] = value
    result, _ = retrieve(ankle_case(), [proposal])
    assert result == []


def test_absent_symptom_or_anatomy_alone_cannot_create_positive_candidates():
    case = ankle_case()
    case.semantic_extractions[0].assertion = "absent"
    result, provider = retrieve(case, [match()])
    assert result == []
    provider.assert_not_awaited()
    case.semantic_extractions = case.semantic_extractions[1:]
    result, provider = retrieve(case, [match()])
    assert result == []
    provider.assert_not_awaited()


def test_provider_failure_yields_no_invented_routing():
    records, resolutions = data()
    with patch.object(service, "complete_runtime_json", AsyncMock(side_effect=TimeoutError())):
        assert asyncio.run(service.retrieve_semantic_official_evidence(ankle_case(), records, resolutions)) == []


def test_entailment_audit_removes_added_symptom_qualities_and_cannot_expand():
    records, resolutions = data()
    provider = AsyncMock(side_effect=[
        json.dumps(dict(matches=[match(), dict(patient_source_text="疼痛", knowledge_concept="下肢酸痛", confidence=.99)])),
        json.dumps(dict(judgments=[dict(patient_source_text="疼痛", knowledge_concept="關節痛", supported=True, reason="相符"),
                                  dict(patient_source_text="疼痛", knowledge_concept="胸痛", supported=True, reason="新增不得採用")])),
    ])
    with patch.object(service, "complete_runtime_json", provider):
        result = asyncio.run(service.retrieve_semantic_official_evidence(ankle_case(), records, resolutions))
    assert result
    assert all(item["knowledge_concept"] == "關節痛" for item in result)


def test_failed_entailment_audit_cannot_be_used_as_routing_evidence():
    records, resolutions = data()
    provider = AsyncMock(side_effect=[json.dumps(dict(matches=[match()])), TimeoutError()])
    with patch.object(service, "complete_runtime_json", provider):
        assert asyncio.run(service.retrieve_semantic_official_evidence(ankle_case(), records, resolutions)) == []


def test_sufficient_symptoms_without_department_explain_evidence_gap():
    case = ankle_case()
    advance_conversation(case, ClarificationSuggestion("sufficient", None, None, "症狀資訊足夠"),
                         user_sources=["有腫脹"], current_extractions=[])
    assert case.triage.next_question == DEPARTMENT_EVIDENCE_REPLY
    assert case.conversation_state.reply_source == "DEPARTMENT_EVIDENCE_UNRESOLVED"
    assert not case.conversation_state.is_complete
    assert not case.conversation_state.confirmed
    assert case.department_result is None
