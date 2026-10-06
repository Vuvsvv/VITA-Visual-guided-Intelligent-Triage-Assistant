from __future__ import annotations

import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import DepartmentResult, Message, SemanticExtraction, TriageCase
from app.services import conversation_service, rag_triage_adapter
from app.services.case_store import save_case
from app.services.conversation_service import (
    ClarificationSuggestion, advance_conversation, capture_pending_answer, request_clarification,
)

chat_route = importlib.import_module("app.routes.chat")

SEVERITY_QUESTION = "目前血尿的程度大概如何，例如只有少量血絲，還是尿液明顯變紅？"
SECOND_USER_TEXT = (
    "就是尿裡面有血絲，還有尿尿時有灼熱感。另外，我沒有胸痛、呼吸困難、"
    "意識不清、大量出血、半邊無力或劇烈頭痛"
)


def extraction(field: str, value: object, source: str, assertion: str | None = None) -> dict:
    item = {
        "field": field, "normalized_value": value, "source_text": source,
        "confidence": 0.95, "semantic_status": "available",
    }
    if assertion is not None or field in {"symptom", "accompanying_symptoms"}:
        item["assertion"] = assertion or "present"
    return item


def pending_case() -> TriageCase:
    case = TriageCase(case_id="phase41-pending")
    case.history_records = [
        Message(role="user", content="我最近有血尿，已經兩天了"),
        Message(role="assistant", content=SEVERITY_QUESTION),
    ]
    case.patient_input.symptom = "血尿"
    case.patient_input.red_flags_checked = True
    case.semantic_extractions = [SemanticExtraction(
        field="duration", normalized_value="2天", source_text="兩天",
        confidence=0.95, semantic_status="available",
    )]
    state = case.conversation_state
    state.pending_clarification_intent = "severity"
    state.asked_clarification_intents = ["severity"]
    state.next_information_needed = ["血尿的程度仍不清楚"]
    state.department_status = "resolved"
    case.department_result = DepartmentResult(
        dept_id=1242, parentDept="內科系", childDept="腎臟科", confidence=0.9,
    )
    return case


class Phase41ClarificationTest(unittest.IsolatedAsyncioTestCase):
    def post(self, message: str, extractions: list[dict], plan: dict, *, triage_case: dict | None = None):
        settings = SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=False)
        semantic_payload = {"semantic_extractions": extractions, "pending_answer": None}
        if triage_case and plan.get("answered_intent"):
            semantic_payload["pending_answer"] = {
                "answered_intent": plan.get("answered_intent"),
                "answer_status": plan.get("answer_status"),
                "answer_source_text": plan.get("answer_source_text"),
                "answer_confidence": plan.get("answer_confidence"),
            }
        semantic = AsyncMock(return_value=json.dumps(semantic_payload, ensure_ascii=False))
        planner = AsyncMock(return_value=json.dumps(plan, ensure_ascii=False))
        detector = AsyncMock()

        async def keep_resolved(case):
            case.conversation_state.department_status = "resolved"
            case.department_result = DepartmentResult(
                dept_id=1242, parentDept="內科系", childDept="腎臟科", confidence=0.9,
            )

        with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
            rag_triage_adapter, "get_settings", return_value=settings,
        ), patch.object(rag_triage_adapter, "complete_prompt", new=semantic), patch.object(
            conversation_service, "complete_runtime_json", new=planner,
        ), patch.object(chat_route, "reason_about_departments", new=keep_resolved), patch.object(
            chat_route, "detect_department_result", new=detector,
        ), patch.object(chat_route, "generate_triage_reply", new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"])):
            response = TestClient(app).post("/chat", json={
                "message": message,
                **({"triage_case": triage_case} if triage_case else {}),
            })
        self.assertEqual(response.status_code, 200)
        detector.assert_not_awaited()
        return response.json(), planner, semantic

    def test_real_hematuria_two_turn_grounded_answer_clears_pending(self):
        initial = TriageCase(case_id="phase41-route-after-safety")
        initial.patient_input.red_flags_checked = True
        initial.patient_input.red_flags_status = "negative"
        save_case(initial)
        first, _, _ = self.post(
            "我最近有血尿，已經兩天了",
            [extraction("symptom", "血尿", "血尿"), extraction("duration", "2天", "兩天")],
            {"status": "clarification_needed", "question": SEVERITY_QUESTION,
             "intent": "severity", "reason": "血尿的程度仍不清楚"},
            triage_case=initial.model_dump(mode="json"),
        )
        self.assertEqual(first["conversation_state"]["pending_clarification_intent"], "severity")
        self.assertEqual(first["department_result"]["dept_id"], 1242)

        second, planner, semantic = self.post(
            SECOND_USER_TEXT,
            [extraction("accompanying_symptoms", ["灼熱感"], "灼熱感")],
            {"status": "sufficient", "question": None, "intent": None,
             "reason": "先前程度追問已由具體描述回答",
             "answered_intent": "severity", "answer_source_text": "有血絲",
             "answer_status": "answered", "answer_confidence": 0.95},
            triage_case=first["triage_case"],
        )
        interpretation_prompt = semantic.await_args.args[0]
        self.assertIn(SEVERITY_QUESTION, interpretation_prompt)
        self.assertIn('"pending_clarification_intent": "severity"', interpretation_prompt)
        self.assertEqual(planner.await_count, 1)
        self.assertIsNone(second["conversation_state"]["pending_clarification_intent"])
        self.assertEqual(second["conversation_state"]["clarification_evidence"]["severity"], "有血絲")
        self.assertTrue(second["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(second["triage_case"]["patient_input"]["red_flags"], [])
        self.assertIn("灼熱感", second["triage_case"]["patient_input"]["accompanying_symptoms"])
        self.assertEqual(second["department_result"]["dept_id"], 1242)
        self.assertEqual(second["department_result"]["childDept"], "腎臟科")
        self.assertNotIn("可以先補充上一個追問的答案", second["reply"])

    async def test_partial_answer_keeps_intent_and_accepts_targeted_follow_up(self):
        case = pending_case()
        text = "看起來不多"
        case.history_records.append(Message(role="user", content=text))
        follow_up = "你說的不多，是偶爾看到一點，還是每次都看得到？"
        suggestion = ClarificationSuggestion(
            "clarification_needed", follow_up, "severity", "程度仍待釐清",
            "severity", text, "partial", 0.9,
        )
        advance_conversation(case, suggestion, user_sources=[text], current_extractions=[])
        self.assertEqual(case.conversation_state.pending_clarification_intent, "severity")
        self.assertEqual(case.triage.next_question, follow_up)
        self.assertNotEqual(case.triage.next_question, SEVERITY_QUESTION)
        self.assertEqual(case.conversation_state.clarification_evidence, {})

    async def test_wrong_ungrounded_or_low_confidence_answer_keeps_pending(self):
        for intent, source, confidence in (
            ("duration", "只有一點血絲", 0.95),
            ("severity", "完全沒有血尿", 0.95),
            ("severity", "只有一點血絲", 0.54),
            ("severity", "只有一點血絲", True),
        ):
            with self.subTest(intent=intent, source=source, confidence=confidence):
                case = pending_case()
                text = "只有一點血絲"
                case.history_records.append(Message(role="user", content=text))
                suggestion = ClarificationSuggestion(
                    "sufficient", None, None, "症狀資訊足夠",
                    intent, source, "answered", confidence,
                )
                self.assertFalse(capture_pending_answer(case, suggestion, [text]))
                advance_conversation(case, suggestion, user_sources=[text], current_extractions=[])
                self.assertEqual(case.conversation_state.pending_clarification_intent, "severity")
                self.assertEqual(case.conversation_state.clarification_evidence, {})

    async def test_duplicate_or_multi_question_is_rejected(self):
        case = pending_case()
        text = "看起來不多"
        case.history_records.append(Message(role="user", content=text))
        for question in (
            SEVERITY_QUESTION,
            "血尿的量大概多少？是否有疼痛或其他不適感？",
        ):
            with self.subTest(question=question):
                proposal = {
                    "status": "clarification_needed", "question": question,
                    "intent": "severity", "reason": "程度仍待釐清",
                    "answered_intent": "severity", "answer_source_text": text,
                    "answer_status": "partial", "answer_confidence": 0.9,
                }
                with patch.object(conversation_service, "complete_runtime_json", new=AsyncMock(return_value=json.dumps(proposal, ensure_ascii=False))):
                    suggestion = await request_clarification(case, [text])
                if question != SEVERITY_QUESTION:
                    self.assertIsNone(suggestion.question)
                advance_conversation(case, suggestion, user_sources=[text], current_extractions=[])
                self.assertNotEqual(case.triage.next_question, question)
                self.assertEqual(case.conversation_state.pending_clarification_intent, "severity")

    async def test_invalid_follow_up_keeps_grounded_answer_fields_independent(self):
        case = pending_case()
        text = "只有一點血絲"
        proposal = {
            "status": "clarification_needed", "question": "程度如何？還有疼痛嗎？",
            "intent": "severity", "reason": "還需其他資訊",
            "answered_intent": "severity", "answer_source_text": text,
            "answer_status": "answered", "answer_confidence": 0.95,
        }
        with patch.object(conversation_service, "complete_runtime_json", new=AsyncMock(return_value=json.dumps(proposal, ensure_ascii=False))):
            suggestion = await request_clarification(case, [text])
        self.assertIsNone(suggestion.question)
        self.assertEqual(suggestion.answer_source_text, text)
        self.assertTrue(capture_pending_answer(case, suggestion, [text]))
        self.assertIsNone(case.conversation_state.pending_clarification_intent)

    async def test_new_grounded_detail_gets_nonduplicate_focused_retry(self):
        case = pending_case()
        text = "尿尿時有灼熱感"
        case.history_records.append(Message(role="user", content=text))
        detail = SemanticExtraction(
            field="accompanying_symptoms", normalized_value=["灼熱感"], source_text="灼熱感",
            confidence=0.9, semantic_status="available",
        )
        advance_conversation(case, None, user_sources=[text], current_extractions=[detail])
        first_retry = case.triage.next_question
        self.assertIn("血尿的程度", first_retry)
        self.assertNotIn("可以先補充上一個追問的答案", first_retry)
        self.assertEqual(case.conversation_state.pending_clarification_intent, "severity")
        case.history_records.append(Message(role="assistant", content=first_retry))
        case.history_records.append(Message(role="user", content="我已經回答了"))
        advance_conversation(case, None, user_sources=["我已經回答了"], current_extractions=[])
        self.assertNotEqual(case.triage.next_question, first_retry)
        self.assertNotEqual(case.triage.next_question, SEVERITY_QUESTION)
        self.assertEqual(case.conversation_state.pending_clarification_intent, "severity")
