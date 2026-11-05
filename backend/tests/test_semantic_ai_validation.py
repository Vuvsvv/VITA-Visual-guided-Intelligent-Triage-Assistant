"""Strict structured extraction validation for the natural-language transport."""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.schemas import Message, SemanticExtraction, TriageCase
from app.services import rag_triage_adapter as adapter
from app.services.rule_engine import apply_semantic_extractions, apply_user_message


def extraction(field, value, source, **changes):
    return dict(field=field, normalized_value=value, source_text=source,
                semantic_status="available", confidence=0.98,
                assertion="present" if field in {"symptom", "accompanying_symptoms"} else None, **changes)


def payload(items, text):
    return dict(semantic_extractions=items, pending_answer=None, ttas_evidence=[], time_preferences=[],
        prior_answer_reference=None, department_preference=None,
        dialogue_act=dict(act="new_information", source_text=text, confidence=0.99))


def run_extraction(text, items):
    case = TriageCase(case_id="strict-ai")
    provider = AsyncMock(return_value=json.dumps(payload(items, text)))
    with patch.object(adapter, "_ai_available", return_value=True), patch.object(adapter, "complete_prompt", provider):
        apply_user_message(case, text)
        asyncio.run(adapter.refine_case_with_ai(case, user_sources=[text]))
    return case, provider


@pytest.mark.parametrize("field,text,value,span", [
    ("body_part", "鎖骨附近怪怪的", "鎖骨", "鎖骨附近"),
    ("body_part", "手腕那邊一直怪怪的", "手腕", "手腕那邊"),
    ("body_part", "右腳腳踝腫痛", "右腳腳踝", "右腳腳踝"),
    ("body_part", "胃部有點不舒服", "胃部", "胃部"),
    ("symptom", "吞嚥有異物感", "吞嚥異物感", "吞嚥有異物感"),
    ("duration", "大概已經半年了", "6個月", "半年"),
    ("severity", "還能正常上班", "mild", "還能正常上班"),
    ("severity", "痛到睡不著", "severe", "痛到睡不著"),
    ("onset", "突然開始", "突然", "突然"),
    ("preferred_days", "下週只有週二能來", ["週二"], "只有週二能來"),
    ("preferred_sessions", "晚上不行，下午可以", ["下午"], "晚上不行，下午可以"),
])
def test_grounded_ai_values_do_not_need_patient_keyword_dictionaries(field, text, value, span):
    case, provider = run_extraction(text, [extraction(field, value, span)])
    provider.assert_awaited_once()
    container = case.availability if field.startswith("preferred_") else case.patient_input
    expected = span if field == "symptom" else value
    assert getattr(container, field) == expected


@pytest.mark.parametrize("text,value,source", [
    ("我的屁股好痛", "臀部", "屁股"),
    ("右手腕，鎖骨，頭暈", "左手腕", "右手腕"),
    ("右手腕，鎖骨，頭暈", "右手腕", "手腕"),
    ("右手腕，鎖骨，頭暈", "膝蓋", "鎖骨"),
])
def test_body_part_renamed_by_ai_keeps_the_patients_words(text, value, source):
    case, _ = run_extraction(text, [extraction("body_part", value, source)])
    assert case.patient_input.body_part == source
    kept = [item for item in case.semantic_extractions if item.field == "body_part"]
    assert [(item.normalized_value, item.source_text) for item in kept] == [(source, source)]


@pytest.mark.parametrize("field,value,source,change", [
    ("body_part", "膝蓋", "鎖骨", {"confidence": 0.1}),
    ("body_part", "膝蓋", "不存在", {}),
    ("body_part", "膝蓋", "鎖骨", {"semantic_status": "invalid"}),
    ("body_part", ["手"], "手", {}),
    ("body_part", "手", "手", {"confidence": 0.1}),
    ("body_part", "手", "不存在", {}),
    ("body_part", "手", "手", {"confidence": float("nan")}),
    ("body_part", "手", "手", {"confidence": True}),
    ("body_part", "手", "手", {"confidence": 1.1}),
    ("body_part", "手", "手", {"semantic_status": "invalid"}),
    ("severity", "嚴重", "很痛", {}),
    ("severity", {"severity_level": "mild"}, "還好", {}),
    ("symptom", "頭暈", "頭暈", {"assertion": None}),
    ("symptom", "頭暈", "頭暈", {"assertion": "safe"}),
    ("accompanying_symptoms", ["耳鳴"], "耳鳴", {"assertion": None}),
    ("preferred_sessions", ["中午"], "下午", {}),
    ("preferred_days", ["下週二"], "週二", {}),
])
def test_invalid_shape_confidence_assertion_or_grounding_fails_closed(field, value, source, change):
    item = extraction(field, value, source)
    item.update(change)
    case, _ = run_extraction("右手腕，鎖骨，頭暈，耳鳴，很痛，還好，下午週二", [item])
    container = case.availability if field.startswith("preferred_") else case.patient_input
    assert not getattr(container, field)


def test_one_batch_keeps_all_grounded_extra_fields_without_rule_slot_guessing():
    text = "右腳腳踝痛三天，只能下午"
    items = [extraction("symptom", "腳踝痛", "腳踝痛"), extraction("body_part", "右腳腳踝", "右腳腳踝"),
             extraction("duration", "3天", "三天"), extraction("preferred_sessions", ["下午"], "只能下午")]
    case, provider = run_extraction(text, items)
    provider.assert_awaited_once()
    assert case.patient_input.body_part == "右腳腳踝"
    assert case.patient_input.duration == "3天"
    assert case.availability.preferred_sessions == ["下午"]
    assert {item.field for item in case.semantic_extractions} == {item["field"] for item in items}


@pytest.mark.parametrize("polarity", ["absent", "uncertain"])
def test_nonpositive_medical_assertions_are_retained_without_becoming_positive(polarity):
    item = extraction("symptom", "胸痛", "不確定胸痛")
    item["assertion"] = polarity
    case, _ = run_extraction("不確定胸痛", [item])
    assert case.patient_input.symptom == ""
    assert case.semantic_extractions[0].assertion == polarity


def test_ordinary_ai_cannot_overwrite_existing_grounded_value():
    case = TriageCase(case_id="duplicate")
    case.patient_input.body_part = "右手腕"
    case.history_records = [Message(role="user", content="右手腕，腳踝")]
    apply_semantic_extractions(case, [SemanticExtraction(**extraction("body_part", "腳踝", "腳踝"), extractor="ai")])
    assert case.patient_input.body_part == "右手腕"


@pytest.mark.parametrize("failure", ["{}", "not json", '{"semantic_extractions":[{"field":"confirmed","normalized_value":true}]}'])
def test_invalid_provider_response_cannot_fill_fields(failure):
    case = TriageCase(case_id="invalid-envelope")
    with patch.object(adapter, "_ai_available", return_value=True), patch.object(
        adapter, "complete_prompt", AsyncMock(return_value=failure),
    ):
        apply_user_message(case, "頭痛")
        asyncio.run(adapter.refine_case_with_ai(case, user_sources=["頭痛"]))
    assert not case.patient_input.symptom
    assert not case.semantic_extractions
    assert not case.confirmed
