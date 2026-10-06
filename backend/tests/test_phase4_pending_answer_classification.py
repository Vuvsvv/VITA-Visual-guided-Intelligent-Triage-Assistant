from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import (
    DepartmentResult,
    Message,
    PendingAnswerInterpretation,
    SemanticExtraction,
    TriageCase,
)
from app.services import conversation_service, rag_triage_adapter
from app.services.case_store import save_case
from app.services.conversation_service import (
    ClarificationSuggestion,
    PendingAnswerClassification,
    advance_conversation,
    capture_pending_answer,
    classify_pending_answer,
    request_clarification,
)
from app.services.rag_triage_adapter import RagTriageSuggestion
from app.services.department_knowledge import load_department_knowledge, resolve_department_names
from app.services.department_reasoning_service import retrieve_official_evidence


PENDING_QUESTION = "請問您是否有腰部或腹部的疼痛感？"


def pending_pain_case(case_id: str = "phase4-focused-pending") -> TriageCase:
    case = TriageCase(case_id=case_id)
    case.history_records = [
        Message(role="user", content="我最近有血尿，已經兩天了"),
        Message(role="assistant", content=PENDING_QUESTION),
    ]
    case.patient_input.symptom = "血尿"
    case.patient_input.red_flags_checked = True
    case.semantic_extractions = [SemanticExtraction(
        field="duration", normalized_value="2天", semantic_status="available",
        confidence=0.96, source_text="兩天", extractor="ai",
    )]
    case.conversation_state.free_text_mode = True
    case.conversation_state.pending_clarification_intent = "pain_presence"
    case.conversation_state.asked_clarification_intents = ["pain_presence"]
    case.conversation_state.department_status = "resolved"
    case.department_result = DepartmentResult(
        dept_id=1242, parentDept="內科部", childDept="腎臟科", confidence=0.94,
    )
    return case


def classifier_json(
    *,
    intent: object = "pain_presence",
    status: object = "answered",
    source: object,
    confidence: object = 0.95,
) -> str:
    return json.dumps({
        "answered_intent": intent,
        "answer_status": status,
        "answer_source_text": source,
        "answer_confidence": confidence,
    }, ensure_ascii=False)


def run_classifier(case: TriageCase, text: str, raw: object):
    provider = AsyncMock(
        side_effect=raw if isinstance(raw, Exception) else None,
        return_value=raw if isinstance(raw, str) else None,
    )
    with patch(
        "app.services.conversation_service.complete_runtime_json",
        new=provider,
    ):
        result = asyncio.run(classify_pending_answer(case, [text]))
    return result, provider


def test_explicit_negative_answer_is_grounded_and_clears_pending_without_semantic_extraction():
    case = pending_pain_case("phase4-negative-answer")
    text = "沒有腰痛，也沒有腹痛"
    result, provider = run_classifier(case, text, classifier_json(source=text))

    assert isinstance(result, PendingAnswerClassification)
    assert capture_pending_answer(case, result, [text]) is True
    assert case.conversation_state.pending_clarification_intent is None
    assert case.conversation_state.clarification_evidence["pain_presence"] == text
    assert len(case.semantic_extractions) == 1
    assert case.semantic_extractions[0].field == "duration"
    prompt = provider.await_args.args[0]
    assert "明確否定仍然是 answered" in prompt
    assert "沒有腰痛，也沒有腹痛" in prompt


def test_positive_answer_clears_pending():
    case = pending_pain_case("phase4-positive-answer")
    text = "右腰會痛"
    result, _ = run_classifier(case, text, classifier_json(source=text))

    assert capture_pending_answer(case, result, [text]) is True
    assert case.conversation_state.pending_clarification_intent is None
    assert case.conversation_state.clarification_evidence["pain_presence"] == text


@pytest.mark.parametrize("status", ["partial", "unclear"])
def test_partial_or_unclear_answer_keeps_pending(status):
    case = pending_pain_case(f"phase4-{status}-answer")
    text = "好像有一點，但不確定" if status == "partial" else "不知道"
    result, _ = run_classifier(
        case,
        text,
        classifier_json(status=status, source=text),
    )

    assert result is not None
    assert capture_pending_answer(case, result, [text]) is False
    assert case.conversation_state.pending_clarification_intent == "pain_presence"
    assert case.conversation_state.clarification_evidence == {}


@pytest.mark.parametrize(
    ("raw", "text", "expect_classification"),
    [
        (classifier_json(intent="other_intent", source="右腰會痛"), "右腰會痛", False),
        (classifier_json(source="先前說過腰痛"), "現在沒有了", False),
        (classifier_json(source="右腰會痛", confidence=0.2), "右腰會痛", True),
    ],
)
def test_wrong_intent_ungrounded_source_and_low_confidence_cannot_clear(
    raw,
    text,
    expect_classification,
):
    case = pending_pain_case(f"phase4-invalid-{len(text)}")
    result, _ = run_classifier(case, text, raw)

    assert (result is not None) is expect_classification
    assert capture_pending_answer(case, result, [text]) is False
    assert case.conversation_state.pending_clarification_intent == "pain_presence"
    assert case.conversation_state.clarification_evidence == {}


@pytest.mark.parametrize("raw", ["{bad json", TimeoutError()])
def test_malformed_or_provider_failure_fails_closed(raw):
    case = pending_pain_case(f"phase4-provider-failure-{type(raw).__name__}")
    result, _ = run_classifier(case, "沒有腰痛，也沒有腹痛", raw)

    assert result is None
    assert capture_pending_answer(case, result, ["沒有腰痛，也沒有腹痛"]) is False
    assert case.conversation_state.pending_clarification_intent == "pain_presence"


def test_planner_answer_fields_cannot_override_focused_classifier_failure():
    case = pending_pain_case("phase4-planner-cannot-override")
    text = "沒有腰痛，也沒有腹痛"
    planner_payload = {
        "status": "sufficient",
        "question": None,
        "intent": None,
        "reason": "planner attempted to classify the pending answer",
        "answered_intent": "pain_presence",
        "answer_source_text": text,
        "answer_status": "answered",
        "answer_confidence": 0.99,
    }
    with patch(
        "app.services.conversation_service.complete_runtime_json",
        new=AsyncMock(return_value=json.dumps(planner_payload, ensure_ascii=False)),
    ):
        suggestion = asyncio.run(request_clarification(
            case,
            [text],
            classify_answer_fields=False,
        ))

    assert suggestion is not None
    assert suggestion.answered_intent is None
    assert suggestion.answer_status is None
    advance_conversation(case, suggestion, user_sources=[text], current_extractions=[])
    assert case.conversation_state.pending_clarification_intent == "pain_presence"
    assert case.conversation_state.clarification_evidence == {}


def test_chat_orders_semantic_classifier_department_and_planner_and_avoids_neutral_retry():
    case = pending_pain_case("phase4-live-pain-presence")
    save_case(case)
    events: list[str] = []
    semantic = AsyncMock(side_effect=lambda *_args, **_kwargs: (
        events.append("semantic") or RagTriageSuggestion(
            semantic_extractions=[],
            pending_answer=PendingAnswerInterpretation(
                answered_intent="pain_presence",
                answer_status="answered",
                answer_source_text="沒有腰痛，也沒有腹痛",
                answer_confidence=0.95,
            ),
        )
    ))

    async def provider(_prompt: str, *, purpose: str) -> str:
        events.append(purpose)
        assert purpose == "conversation_clarification"
        return json.dumps({
            "status": "sufficient",
            "question": None,
            "intent": None,
            "reason": "上一題已由 focused classifier 確認",
            "answered_intent": None,
            "answer_source_text": None,
            "answer_status": None,
            "answer_confidence": None,
        }, ensure_ascii=False)

    async def keep_department_resolved(current: TriageCase) -> None:
        events.append("department")
        current.conversation_state.department_status = "resolved"
        current.department_result = DepartmentResult(
            dept_id=1242, parentDept="內科部", childDept="腎臟科", confidence=0.94,
        )

    settings = SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=False)
    legacy_classifier = AsyncMock(side_effect=AssertionError("primary flow must not classify twice"))
    with patch("app.routes.chat.get_settings", return_value=settings), patch(
        "app.routes.chat.refine_case_with_ai", new=semantic,
    ), patch(
        "app.services.conversation_service.classify_pending_answer", new=legacy_classifier,
    ), patch(
        "app.services.conversation_service.complete_runtime_json", new=provider,
    ), patch(
        "app.routes.chat.reason_about_departments", new=keep_department_resolved,
    ), patch(
        "app.routes.chat.generate_triage_reply",
        new=AsyncMock(side_effect=lambda **kwargs: kwargs["fallback_reply"]),
    ):
        response = TestClient(app).post("/chat", json={
            "case_id": case.case_id,
            "message": "沒有腰痛，也沒有腹痛",
        })

    assert response.status_code == 200
    data = response.json()
    assert events == [
        "semantic",
        "department",
        "conversation_clarification",
    ]
    legacy_classifier.assert_not_awaited()
    assert data["conversation_state"]["pending_clarification_intent"] is None
    assert data["conversation_state"]["clarification_evidence"]["pain_presence"] == "沒有腰痛，也沒有腹痛"
    assert len(data["triage_case"]["semantic_extractions"]) == 1
    assert data["next_question"] is None
    assert data["reply"] != "可以再補充和剛才問題相關的症狀細節嗎？"


def _turn_interpretation_case(case_id: str, intent: str, question: str, current_text: str) -> TriageCase:
    case = TriageCase(case_id=case_id)
    case.history_records = [
        Message(role="user", content="我最近會眩暈"),
        Message(role="assistant", content=question),
        Message(role="user", content=current_text),
    ]
    case.patient_input.symptom = "眩暈"
    case.patient_input.red_flags_checked = True
    case.conversation_state.pending_clarification_intent = intent
    case.conversation_state.asked_clarification_intents = [intent]
    return case


def _run_turn_interpreter(case: TriageCase, current_text: str, payload: dict | list[dict]):
    if isinstance(payload, list):
        provider = AsyncMock(side_effect=[json.dumps(item, ensure_ascii=False) for item in payload])
    else:
        provider = AsyncMock(return_value=json.dumps(payload, ensure_ascii=False))
    with patch.object(rag_triage_adapter, "_ai_available", return_value=True), patch.object(
        rag_triage_adapter, "complete_prompt", new=provider,
    ):
        result = asyncio.run(rag_triage_adapter.refine_case_with_ai(
            case,
            user_sources=[current_text],
        ))
    return result, provider


def test_single_turn_interpretation_captures_tinnitus_evidence_and_pending_answer():
    intent = "ask_hearing_loss_or_chest_pain"
    question = "請問您是否有聽力下降的情形？"
    text = "沒有聽力下降，但我右耳一直有耳鳴，眩暈的時候耳鳴會更明顯。"
    case = _turn_interpretation_case("phase4-tinnitus-turn", intent, question, text)
    payload = {
        "semantic_extractions": [
            {
                "field": "accompanying_symptoms", "normalized_value": ["聽力下降"],
                "semantic_status": "available", "assertion": "absent",
                "confidence": 0.98, "source_text": "沒有聽力下降",
            },
            {
                "field": "body_part", "normalized_value": "右耳",
                "semantic_status": "available", "assertion": None,
                "confidence": 0.96, "source_text": "右耳",
            },
            {
                "field": "accompanying_symptoms", "normalized_value": ["耳鳴"],
                "semantic_status": "available", "assertion": "present",
                "confidence": 0.97, "source_text": "右耳一直有耳鳴",
            },
            {
                "field": "accompanying_symptoms", "normalized_value": ["眩暈"],
                "semantic_status": "available", "assertion": "present",
                "confidence": 0.96, "source_text": "眩暈",
            },
        ],
        "pending_answer": {
            "answered_intent": intent,
            "answer_status": "answered",
            "answer_source_text": "沒有聽力下降",
            "answer_confidence": 0.97,
        },
    }

    result, provider = _run_turn_interpreter(case, text, payload)

    assert result is not None
    assert provider.await_count == 1
    assert any(
        item.field == "accompanying_symptoms"
        and item.normalized_value == ["耳鳴"]
        and item.assertion == "present"
        for item in result.semantic_extractions or []
    )
    assert any(
        item.field == "accompanying_symptoms"
        and item.normalized_value == ["聽力下降"]
        and item.assertion == "absent"
        for item in result.semantic_extractions or []
    )
    assert any(item.field == "body_part" and item.normalized_value == "右耳" for item in result.semantic_extractions or [])
    assert any(
        item.field == "accompanying_symptoms"
        and item.normalized_value == ["眩暈"]
        and item.assertion == "present"
        for item in result.semantic_extractions or []
    )
    assert capture_pending_answer(case, result.pending_answer, [text]) is True
    assert case.conversation_state.pending_clarification_intent is None
    assert case.conversation_state.clarification_evidence[intent] == "沒有聽力下降"
    prompt = provider.await_args.args[0]
    assert intent in prompt
    assert question in prompt
    assert text in prompt
    assert "opaque correlation key" in prompt


def test_answered_pending_without_clinical_evidence_retries_once_and_fails_closed():
    intent = "ask_hearing_loss_or_chest_pain"
    question = "請問您是否有聽力下降的情形？"
    text = "沒有聽力下降"
    case = _turn_interpretation_case("phase4-incomplete-turn", intent, question, text)
    incomplete = {
        "semantic_extractions": [],
        "pending_answer": {
            "answered_intent": intent,
            "answer_status": "answered",
            "answer_source_text": text,
            "answer_confidence": 0.97,
        },
    }

    result, provider = _run_turn_interpreter(case, text, [incomplete, incomplete])

    assert result is not None
    assert result.interpretation_complete is False
    assert result.pending_answer is None
    assert result.semantic_extractions == []
    assert provider.await_count == 2
    assert "修復要求" in provider.await_args_list[1].args[0]
    assert case.semantic_extractions == []
    assert case.conversation_state.pending_clarification_intent == intent
    assert case.conversation_state.department_status == "unresolved"


def test_answered_pending_repair_applies_evidence_before_pending_can_clear():
    intent = "ask_hearing_loss_or_chest_pain"
    question = "請問您是否有聽力下降的情形？"
    text = "沒有聽力下降，但我右耳一直有耳鳴"
    case = _turn_interpretation_case("phase4-repaired-turn", intent, question, text)
    incomplete = {
        "semantic_extractions": [],
        "pending_answer": {
            "answered_intent": intent,
            "answer_status": "answered",
            "answer_source_text": "沒有聽力下降",
            "answer_confidence": 0.97,
        },
    }
    repaired = {
        "semantic_extractions": [
            {
                "field": "accompanying_symptoms", "normalized_value": ["聽力下降"],
                "semantic_status": "available", "assertion": "absent",
                "confidence": 0.98, "source_text": "沒有聽力下降",
            },
            {
                "field": "body_part", "normalized_value": "右耳",
                "semantic_status": "available", "assertion": None,
                "confidence": 0.98, "source_text": "右耳",
            },
            {
                "field": "accompanying_symptoms", "normalized_value": ["耳鳴"],
                "semantic_status": "available", "assertion": "present",
                "confidence": 0.98, "source_text": "右耳一直有耳鳴",
            },
        ],
        "pending_answer": incomplete["pending_answer"],
    }

    result, provider = _run_turn_interpreter(case, text, [incomplete, repaired])

    assert result is not None
    assert result.interpretation_complete is True
    assert provider.await_count == 2
    assert {item.field for item in case.semantic_extractions} == {"accompanying_symptoms", "body_part"}
    assert any(item.normalized_value == ["耳鳴"] and item.assertion == "present" for item in case.semantic_extractions)
    assert capture_pending_answer(case, result.pending_answer, [text]) is True
    assert case.conversation_state.pending_clarification_intent is None


def test_route_repair_failure_keeps_pending_and_skips_department_resolution():
    intent = "ask_hearing_loss_or_chest_pain"
    question = "請問您是否有聽力下降的情形？"
    text = "沒有聽力下降"
    case = TriageCase(case_id="phase4-route-repair-failure")
    case.history_records = [
        Message(role="user", content="我最近會眩暈"),
        Message(role="assistant", content=question),
    ]
    case.patient_input.symptom = "眩暈"
    case.patient_input.red_flags_checked = True
    case.conversation_state.free_text_mode = True
    case.conversation_state.pending_clarification_intent = intent
    case.conversation_state.asked_clarification_intents = [intent]
    save_case(case)
    incomplete = json.dumps({
        "semantic_extractions": [],
        "pending_answer": {
            "answered_intent": intent,
            "answer_status": "answered",
            "answer_source_text": text,
            "answer_confidence": 0.97,
        },
    }, ensure_ascii=False)
    interpreter = AsyncMock(side_effect=[incomplete, incomplete])
    department = AsyncMock(side_effect=AssertionError("incomplete turn must not resolve department"))
    planner = AsyncMock(return_value=ClarificationSuggestion(
        "clarification_needed",
        "可以再具體說明剛才詢問的情況嗎？",
        intent,
        "本輪 interpretation 尚未留下可靠 evidence",
    ))
    legacy_classifier = AsyncMock(side_effect=AssertionError("legacy classifier must not run"))
    settings = SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=False)

    with patch("app.routes.chat.get_settings", return_value=settings), patch.object(
        rag_triage_adapter, "_ai_available", return_value=True,
    ), patch.object(
        rag_triage_adapter, "complete_prompt", new=interpreter,
    ), patch("app.routes.chat.reason_about_departments", new=department), patch(
        "app.routes.chat.request_clarification", new=planner,
    ), patch(
        "app.services.conversation_service.classify_pending_answer", new=legacy_classifier,
    ), patch(
        "app.routes.chat.generate_triage_reply",
        new=AsyncMock(side_effect=lambda **kwargs: kwargs["fallback_reply"]),
    ):
        response = TestClient(app).post("/chat", json={
            "case_id": case.case_id,
            "message": text,
        })

    assert response.status_code == 200
    result = response.json()
    assert interpreter.await_count == 2
    department.assert_not_awaited()
    legacy_classifier.assert_not_awaited()
    assert result["conversation_state"]["pending_clarification_intent"] == intent
    assert result["conversation_state"]["department_status"] == "unresolved"
    assert result["triage_case"]["semantic_extractions"] == []


def test_single_turn_interpretation_preserves_absent_cardiac_evidence_without_positive_retrieval():
    intent = "cardiac_symptom"
    question = "請問您近期有沒有感覺胸口悸動或胸痛？"
    text = "沒有胸痛，也沒有心悸。"
    case = _turn_interpretation_case("phase4-cardiac-negative-turn", intent, question, text)
    payload = {
        "semantic_extractions": [
            {
                "field": "accompanying_symptoms", "normalized_value": ["胸痛"],
                "semantic_status": "available", "assertion": "absent",
                "confidence": 0.97, "source_text": "沒有胸痛",
            },
            {
                "field": "accompanying_symptoms", "normalized_value": ["心悸"],
                "semantic_status": "available", "assertion": "absent",
                "confidence": 0.96, "source_text": "沒有心悸",
            },
        ],
        "pending_answer": {
            "answered_intent": intent,
            "answer_status": "answered",
            "answer_source_text": text,
            "answer_confidence": 0.97,
        },
    }

    result, _ = _run_turn_interpreter(case, text, payload)

    assert result is not None
    assert capture_pending_answer(case, result.pending_answer, [text]) is True
    absent = [item for item in case.semantic_extractions if item.assertion == "absent"]
    assert {tuple(item.normalized_value) for item in absent} == {("胸痛",), ("心悸",)}
    sources, records = load_department_knowledge()
    assert sources
    resolutions = resolve_department_names(records, [
        {"dept_id": 1232, "parentDept": "內科", "childDept": "一般內科"},
        {"dept_id": 1241, "parentDept": "內科", "childDept": "心臟內科"},
    ])
    assert retrieve_official_evidence(case, records, resolutions) == []


def test_present_tinnitus_reaches_production_official_kb_and_canonical_ear_department():
    case = TriageCase(case_id="phase4-production-tinnitus")
    text = "右耳一直有耳鳴"
    case.history_records = [Message(role="user", content=text)]
    case.semantic_extractions = [SemanticExtraction(
        field="accompanying_symptoms",
        normalized_value=["耳鳴"],
        semantic_status="available",
        assertion="present",
        confidence=0.97,
        source_text=text,
        extractor="ai",
    )]
    _, records = load_department_knowledge()
    resolutions = resolve_department_names(records, [{
        "dept_id": 1333, "parentDept": "五官科", "childDept": "耳科",
    }])

    retrieved = retrieve_official_evidence(case, records, resolutions)

    tinnitus = [item for item in retrieved if item["knowledge_concept"] == "耳鳴"]
    assert tinnitus
    assert tinnitus[0]["dept_id"] == 1333
    assert tinnitus[0]["childDept"] == "耳科"
    assert tinnitus[0]["patient_source_text"] == text


def test_turn_interpreter_rejects_ungrounded_tinnitus_source():
    intent = "ask_about_hearing_loss_or_tinnitus"
    question = "請問您有沒有聽力下降或耳鳴的情況？"
    text = "右耳有聲音"
    case = _turn_interpretation_case("phase4-ungrounded-tinnitus", intent, question, text)
    payload = {
        "semantic_extractions": [{
            "field": "accompanying_symptoms", "normalized_value": ["耳鳴"],
            "semantic_status": "available", "assertion": "present",
            "confidence": 0.97, "source_text": "耳朵嗡嗡叫",
        }],
        "pending_answer": None,
    }

    result, _ = _run_turn_interpreter(case, text, payload)

    assert result is not None
    assert result.semantic_extractions == []
    assert case.semantic_extractions == []


@pytest.mark.parametrize("failure", [TimeoutError(), "{bad json"])
def test_turn_interpreter_failure_keeps_pending_and_adds_no_evidence(failure):
    intent = "ask_about_hearing_loss_or_tinnitus"
    question = "請問您有沒有聽力下降或耳鳴的情況？"
    text = "有，我右耳一直有耳鳴"
    case = _turn_interpretation_case(f"phase4-turn-failure-{type(failure).__name__}", intent, question, text)
    provider = AsyncMock(
        side_effect=failure if isinstance(failure, Exception) else None,
        return_value=failure if isinstance(failure, str) else None,
    )
    with patch.object(rag_triage_adapter, "_ai_available", return_value=True), patch.object(
        rag_triage_adapter, "complete_prompt", new=provider,
    ):
        result = asyncio.run(rag_triage_adapter.refine_case_with_ai(case, user_sources=[text]))

    assert result is None
    assert case.semantic_extractions == []
    assert case.conversation_state.pending_clarification_intent == intent
    assert case.conversation_state.department_status == "unresolved"
