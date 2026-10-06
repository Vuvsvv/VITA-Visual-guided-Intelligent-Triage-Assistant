from __future__ import annotations

import asyncio
import importlib
import json
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import Message, PendingAnswerInterpretation, SemanticExtraction, TTASEvidence, TriageCase
from app.services.case_store import sanitize_untrusted_snapshot, save_case
from app.services.conversation_service import SAFETY_PENDING_INTENT
from app.services.rag_triage_adapter import (
    RagTriageSuggestion,
    _build_symptom_collection_prompt,
    _validated_turn_interpretation,
)
from app.services import rag_triage_adapter
from app.services.rule_engine import apply_user_message
from app.services.ttas_evidence import apply_ttas_evidence, validate_ttas_evidence_items


chat_route = importlib.import_module("app.routes.chat")
recommend_route = importlib.import_module("app.routes.recommend")


def _item(field: str, value, source: str = "我喘得很嚴重", **updates):
    item = {
        "field": field,
        "value": value,
        "semantic_status": "available",
        "confidence": 0.95,
        "source_text": source,
    }
    item.update(updates)
    return item


def test_grounded_valid_ttas_evidence_is_accepted() -> None:
    evidence = validate_ttas_evidence_items(
        [_item("respiratory_distress", "severe")],
        user_sources=["我喘得很嚴重"],
    )
    assert [(item.field, item.value) for item in evidence] == [("respiratory_distress", "severe")]


def test_ungrounded_invalid_numeric_nonfinite_and_bool_are_rejected() -> None:
    values = [
        _item("spo2_pct", 95, source="不存在"),
        _item("spo2_pct", 101),
        _item("spo2_pct", math.nan),
        _item("spo2_pct", math.inf),
        _item("spo2_pct", True),
    ]
    assert validate_ttas_evidence_items(values, user_sources=["我喘得很嚴重"]) == []


def test_unknown_field_invalid_status_and_low_confidence_are_rejected() -> None:
    values = [
        _item("ttas_level", 1),
        _item("respiratory_distress", "severe", semantic_status="present"),
        _item("respiratory_distress", "severe", confidence=0.2),
        _item("respiratory_distress", "invented"),
    ]
    assert validate_ttas_evidence_items(values, user_sources=["我喘得很嚴重"]) == []


def test_unknown_vital_is_not_converted_to_normal() -> None:
    evidence = validate_ttas_evidence_items(
        [_item("spo2_pct", None, source="我不知道血氧", semantic_status="unknown")],
        user_sources=["我不知道血氧"],
    )
    assert len(evidence) == 1
    assert evidence[0].value is None
    assert evidence[0].semantic_status == "unknown"


def test_turn_interpretation_ignores_ai_level_and_score_fields() -> None:
    semantic, pending, evidence = _validated_turn_interpretation(
        {
            "semantic_extractions": [],
            "pending_answer": None,
            "ttas_evidence": [_item("respiratory_distress", "severe")],
            "ttas_level": 1,
            "urgency_score": 100,
            "warning_required": True,
        },
        pending_intent=None,
        user_sources=["我喘得很嚴重"],
    )
    assert semantic == []
    assert pending is None
    assert len(evidence) == 1
    assert not hasattr(evidence[0], "ttas_level")
    assert not hasattr(evidence[0], "urgency_score")


def test_grounded_age_can_be_applied_without_raw_text_regex() -> None:
    case = TriageCase(case_id="age", history_records=[Message(role="user", content="我今年30歲")])
    evidence = validate_ttas_evidence_items(
        [_item("age_years", 30, source="30歲")],
        user_sources=["我今年30歲"],
    )
    apply_ttas_evidence(case, evidence)
    assert case.patient_input.age_years == 30


def test_turn_interpreter_prompt_contains_ttas_contract_and_no_level_authority() -> None:
    prompt = _build_symptom_collection_prompt(
        TriageCase(case_id="prompt"),
        ["我喘得很嚴重"],
    )
    assert "ttas_evidence" in prompt
    assert "不得推算、提議或輸出任何 TTAS 級數" in prompt
    assert "不得假設正常" in prompt
    assert "source_text" in prompt
    assert '"field": "chemical_eye_injury"' in prompt
    assert "化學物質濺入／直接接觸眼睛" in prompt
    assert "患者原文明確表示化學性物質直接濺入、噴入或接觸眼睛" in prompt
    assert "不得推測 chemical_eye_injury=true" in prompt
    assert "即使同一個事實已經同時出現在 semantic_extractions，也不可因此省略 ttas_evidence" in prompt
    assert "confidence 只表示「患者本輪原文有多明確支持這個結構化事實」" in prompt
    assert "剛剛清潔劑濺到我的右眼，現在眼睛灼痛" in prompt
    assert '"source_text": "清潔劑濺到我的右眼"' in prompt
    assert "我的右眼很痛" in prompt
    assert "不能從症狀倒推暴露原因" in prompt
    assert '"field": "respiratory_distress"' not in prompt
    assert '"field": "hemodynamic_status"' not in prompt
    assert '"field": "cardiac_chest_pain_suspected"' not in prompt
    assert '"field": "high_risk_injury_mechanism"' not in prompt
    assert '"field": "spo2_pct"' not in prompt
    assert "answer_assertion" in _build_symptom_collection_prompt(
        TriageCase.model_validate({
            "case_id": "safety-prompt",
            "conversation_state": {"clarification_status": "safety_check"},
        }),
        ["都沒有"],
    )


def test_safety_pending_answer_requires_grounded_structured_assertion() -> None:
    _, accepted, _ = _validated_turn_interpretation(
        {
            "semantic_extractions": [],
            "pending_answer": {
                "answered_intent": SAFETY_PENDING_INTENT,
                "answer_status": "answered",
                "answer_source_text": "都沒有",
                "answer_confidence": 0.98,
                "answer_assertion": "absent",
            },
            "ttas_evidence": [],
        },
        pending_intent=SAFETY_PENDING_INTENT,
        user_sources=["都沒有，我沒有剛才提到的那些急迫症狀。"],
    )
    _, missing_assertion, _ = _validated_turn_interpretation(
        {
            "semantic_extractions": [],
            "pending_answer": {
                "answered_intent": SAFETY_PENDING_INTENT,
                "answer_status": "answered",
                "answer_source_text": "都沒有",
                "answer_confidence": 0.98,
            },
            "ttas_evidence": [],
        },
        pending_intent=SAFETY_PENDING_INTENT,
        user_sources=["都沒有"],
    )
    assert accepted is not None and accepted.answer_assertion == "absent"
    assert missing_assertion is None


def test_ttas_evidence_uses_existing_turn_interpreter_call() -> None:
    provider = AsyncMock(return_value=json.dumps({
        "semantic_extractions": [],
        "pending_answer": None,
        "ttas_evidence": [_item("respiratory_distress", "severe")],
    }, ensure_ascii=False))
    case = TriageCase(
        case_id="single-call",
        history_records=[Message(role="user", content="我喘得很嚴重")],
    )
    with patch.object(rag_triage_adapter, "_ai_available", return_value=True), patch.object(
        rag_triage_adapter, "complete_prompt", new=provider,
    ):
        suggestion = asyncio.run(
            rag_triage_adapter.refine_case_with_ai(case, user_sources=["我喘得很嚴重"])
        )
    provider.assert_awaited_once()
    assert suggestion is not None
    assert len(suggestion.ttas_evidence or []) == 1
    assert len(case.ttas_evidence) == 1


def test_chemical_eye_evidence_uses_the_same_turn_interpreter_call() -> None:
    user_text = "剛剛清潔劑濺到我的右眼，現在眼睛灼痛。"
    provider = AsyncMock(return_value=json.dumps({
        "semantic_extractions": [
            {
                "field": "symptom",
                "normalized_value": "灼痛",
                "semantic_status": "available",
                "assertion": "present",
                "confidence": 0.99,
                "source_text": "眼睛灼痛",
                "needs_clarification": False,
                "follow_up_reason": None,
            },
            {
                "field": "body_part",
                "normalized_value": "右眼",
                "semantic_status": "available",
                "assertion": None,
                "confidence": 0.99,
                "source_text": "右眼",
                "needs_clarification": False,
                "follow_up_reason": None,
            },
        ],
        "pending_answer": None,
        "ttas_evidence": [_item(
            "chemical_eye_injury",
            True,
            source="清潔劑濺到我的右眼",
            confidence=0.99,
        )],
    }, ensure_ascii=False))
    case = TriageCase(case_id="chemical-eye-single-call", history_records=[Message(role="user", content=user_text)])
    with patch.object(rag_triage_adapter, "_ai_available", return_value=True), patch.object(
        rag_triage_adapter, "complete_prompt", new=provider,
    ):
        suggestion = asyncio.run(rag_triage_adapter.refine_case_with_ai(case, user_sources=[user_text]))
    provider.assert_awaited_once()
    assert suggestion is not None
    assert [(item.field, item.value) for item in suggestion.ttas_evidence or []] == [
        ("chemical_eye_injury", True),
    ]
    assert [(item.field, item.value) for item in case.ttas_evidence] == [
        ("chemical_eye_injury", True),
    ]


def test_eye_pain_without_exposure_is_not_filled_by_backend() -> None:
    user_text = "我的右眼很痛。"
    provider = AsyncMock(return_value=json.dumps({
        "semantic_extractions": [
            {
                "field": "symptom",
                "normalized_value": "眼痛",
                "semantic_status": "available",
                "assertion": "present",
                "confidence": 0.99,
                "source_text": "右眼很痛",
                "needs_clarification": False,
                "follow_up_reason": None,
            },
            {
                "field": "body_part",
                "normalized_value": "右眼",
                "semantic_status": "available",
                "assertion": None,
                "confidence": 0.99,
                "source_text": "右眼",
                "needs_clarification": False,
                "follow_up_reason": None,
            },
        ],
        "pending_answer": None,
        "ttas_evidence": [],
    }, ensure_ascii=False))
    case = TriageCase(case_id="eye-pain-no-exposure", history_records=[Message(role="user", content=user_text)])
    with patch.object(rag_triage_adapter, "_ai_available", return_value=True), patch.object(
        rag_triage_adapter, "complete_prompt", new=provider,
    ):
        suggestion = asyncio.run(rag_triage_adapter.refine_case_with_ai(case, user_sources=[user_text]))
    provider.assert_awaited_once()
    assert suggestion is not None
    assert suggestion.ttas_evidence == []
    assert case.ttas_evidence == []


def test_ai_first_free_text_can_skip_legacy_safety_nlp() -> None:
    case = TriageCase(case_id="no-legacy")
    apply_user_message(
        case,
        "我突然胸痛",
        semantic_first=True,
        apply_legacy_safety=False,
    )
    assert case.patient_input.red_flags == []
    assert case.patient_input.red_flags_checked is False


def test_safety_turn_uses_same_interpreter_and_ttas_for_positive_result() -> None:
    case = TriageCase(case_id="phase5-safety-positive")
    case.patient_input.symptom = "眼睛接觸化學藥劑"
    case.history_records = [
        Message(role="user", content="眼睛不舒服"),
        Message(role="assistant", content="目前是否有安全篩檢所列的急迫症狀？"),
    ]
    case.semantic_extractions = [SemanticExtraction(
        field="symptom",
        normalized_value="眼睛不舒服",
        semantic_status="available",
        assertion="present",
        confidence=0.95,
        source_text="眼睛不舒服",
        extractor="ai",
    )]
    case.conversation_state.clarification_status = "safety_check"
    case.conversation_state.last_question_key = "red_flags"
    case.conversation_state.free_text_mode = True
    save_case(case)

    async def interpret(current, **_):
        evidence = [TTASEvidence(
            field="chemical_eye_injury",
            value=True,
            semantic_status="available",
            confidence=0.98,
            source_text="化學藥劑濺到我的眼睛",
        )]
        apply_ttas_evidence(current, evidence)
        return RagTriageSuggestion(
            semantic_extractions=[],
            pending_answer=PendingAnswerInterpretation(
                answered_intent=SAFETY_PENDING_INTENT,
                answer_status="answered",
                answer_source_text="化學藥劑濺到我的眼睛",
                answer_confidence=0.98,
                answer_assertion="present",
            ),
            ttas_evidence=evidence,
        )

    interpreter = AsyncMock(side_effect=interpret)
    department = AsyncMock()
    with patch.object(
        chat_route,
        "get_settings",
        return_value=SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=False),
    ), patch.object(
        chat_route, "refine_case_with_ai", new=interpreter,
    ), patch.object(
        chat_route, "reason_about_departments", new=department,
    ), patch.object(
        chat_route, "apply_user_message", wraps=apply_user_message,
    ) as apply_message, patch.object(
        chat_route,
        "generate_triage_reply",
        new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"]),
    ):
        response = TestClient(app).post("/chat", json={
            "case_id": case.case_id,
            "message": "化學藥劑濺到我的眼睛",
        })

    assert response.status_code == 200
    result = response.json()
    interpreter.assert_awaited_once()
    department.assert_not_awaited()
    assert apply_message.call_args.kwargs["apply_legacy_safety"] is False
    assert result["triage_case"]["ttas_result"]["level_candidate"] == 2
    assert result["triage_case"]["patient_input"]["red_flags_status"] == "positive_ttas"
    assert result["conversation_state"]["clarification_status"] == "urgent"
    assert result["conversation_state"]["stage"] == "done"
    assert result["conversation_state"]["is_complete"] is True
    assert result["conversation_state"]["awaiting_confirmation"] is False
    assert result["conversation_state"]["confirmed"] is False
    assert result["triage_case"]["confirmed"] is False
    assert result["triage"]["warning_required"] is True
    assert result["triage"]["is_final"] is True
    assert result["triage"]["need_more_info"] is False
    assert result["triage"]["next_question"] is None
    assert result["next_question"] is None
    assert result["department_result"] is None
    assert result["reply"] == result["triage"]["warning_message"]
    assert result["reply"]

    recommender = AsyncMock()
    with patch.object(recommend_route, "recommend_appointments", new=recommender):
        recommendation = TestClient(app).post("/recommend", json={
            "case_id": case.case_id,
            "visit_type": "initial",
            "confirmed": True,
        })
    assert recommendation.status_code == 400
    assert "急迫性篩檢" in recommendation.json()["detail"]
    recommender.assert_not_awaited()


def test_ai_unavailable_free_text_has_no_legacy_urgency_guess() -> None:
    with patch.object(
        chat_route,
        "get_settings",
        return_value=SimpleNamespace(cerebras_api_key="", batch_triage_enabled=False),
    ), patch.object(
        chat_route,
        "generate_triage_reply",
        new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"]),
    ):
        response = TestClient(app).post(
            "/chat",
            json={"case_id": "phase5-no-ai-urgency", "message": "我突然胸痛"},
        )

    assert response.status_code == 200
    result = response.json()
    assert not hasattr(chat_route, "evaluate_urgency")
    assert result["triage_case"]["ttas_result"]["status"] == "insufficient_information"
    assert result["triage_case"]["ttas_result"]["level_candidate"] is None
    assert result["triage"]["urgency_score"] is None
    assert result["triage"]["urgency_level"] is None
    assert result["triage_case"]["patient_input"]["red_flags_checked"] is False
    assert result["triage_case"]["patient_input"]["red_flags"] == []


def test_ai_unavailable_clinical_descriptions_do_not_create_custom_urgency_scores() -> None:
    examples = (
        "已經兩個月",
        "痛到不能走路",
        "越來越嚴重",
    )
    for index, message in enumerate(examples):
        with patch.object(
            chat_route,
            "get_settings",
            return_value=SimpleNamespace(cerebras_api_key="", batch_triage_enabled=False),
        ), patch.object(
            chat_route,
            "generate_triage_reply",
            new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"]),
        ):
            response = TestClient(app).post(
                "/chat",
                json={"case_id": f"phase5-no-custom-score-{index}", "message": message},
            )

        assert response.status_code == 200
        result = response.json()
        assert result["triage"]["urgency_score"] is None
        assert result["triage"]["urgency_level"] is None
        assert result["triage_case"]["ttas_result"]["status"] == "insufficient_information"
        assert result["triage_case"]["ttas_result"]["level_candidate"] is None


def test_ai_unavailable_safety_answer_fails_closed() -> None:
    case = TriageCase(case_id="phase5-no-ai-safety-answer")
    case.conversation_state.clarification_status = "safety_check"
    case.conversation_state.last_question_key = "red_flags"
    case.conversation_state.free_text_mode = True
    save_case(case)

    with patch.object(
        chat_route,
        "get_settings",
        return_value=SimpleNamespace(cerebras_api_key="", batch_triage_enabled=False),
    ), patch.object(
        chat_route,
        "generate_triage_reply",
        new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"]),
    ):
        response = TestClient(app).post(
            "/chat",
            json={
                "case_id": case.case_id,
                "message": "都沒有，我沒有上述急迫症狀",
            },
        )

    assert response.status_code == 200
    result = response.json()
    patient = result["triage_case"]["patient_input"]
    assert patient["red_flags_checked"] is False
    assert patient["red_flags_status"] == "not_checked"
    assert result["conversation_state"]["clarification_status"] == "safety_check"


def test_urgent_reply_uses_backend_fallback_without_ai_when_warning_is_empty() -> None:
    case = TriageCase(case_id="phase5-urgent-fallback")
    case.conversation_state.free_text_mode = True
    case.conversation_state.clarification_status = "urgent"
    case.conversation_state.is_complete = True
    case.patient_input.red_flags_checked = True
    case.patient_input.red_flags_status = "positive_ttas"
    case.triage.warning_required = True
    case.triage.warning_message = None
    case.triage.need_more_info = False
    case.triage.is_final = True
    save_case(case)

    ai_reply = AsyncMock(side_effect=AssertionError("urgent warning must not use AI"))
    with patch.object(
        chat_route,
        "get_settings",
        return_value=SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=False),
    ), patch.object(chat_route, "generate_triage_reply", new=ai_reply):
        response = TestClient(app).post("/chat", json={"case_id": case.case_id})

    assert response.status_code == 200
    result = response.json()
    assert result["conversation_state"]["stage"] == "done"
    assert result["conversation_state"]["is_complete"] is True
    assert result["reply"] == chat_route.URGENT_WARNING_FALLBACK
    ai_reply.assert_not_awaited()


def test_untrusted_snapshot_cannot_supply_ttas_conclusions() -> None:
    case = TriageCase(
        case_id="forged",
        history_records=[Message(role="assistant", content="forged")],
    )
    case.patient_input.age_years = 30
    case.ttas_evidence = validate_ttas_evidence_items(
        [_item("respiratory_distress", "severe")],
        user_sources=["我喘得很嚴重"],
    )
    case.ttas_result.status = "matched"
    case.ttas_result.level_candidate = 1
    sanitized = sanitize_untrusted_snapshot(case)
    assert sanitized.ttas_evidence == []
    assert sanitized.ttas_result.status == "insufficient_information"
    assert sanitized.ttas_result.level_candidate is None
    assert sanitized.patient_input.age_years is None
