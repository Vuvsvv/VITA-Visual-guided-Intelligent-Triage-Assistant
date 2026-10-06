from __future__ import annotations

import importlib
import inspect
import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import (
    DepartmentResult,
    Message,
    PendingAnswerInterpretation,
    SemanticExtraction,
    TriageCase,
    VisitType,
)
from app.services import department_reasoning_service as reasoning
from app.services import department_preference_service
from app.services.appointment_service import DepartmentResolutionError
from app.services.case_store import get_case, save_case
from app.services.conversation_service import (
    NEUTRAL_PENDING_RETRIES,
    SAFETY_PENDING_INTENT,
    ClarificationSuggestion,
    advance_conversation,
    request_clarification,
)
from app.services.department_knowledge import load_department_knowledge, resolve_department_names
from app.services.rag_triage_adapter import RagTriageSuggestion
from app.services.rule_engine import (
    QUESTION_TEXTS,
    RED_FLAG_QUESTION_KEY,
    apply_semantic_extractions,
    apply_user_message,
)

chat_route = importlib.import_module("app.routes.chat")
recommend_route = importlib.import_module("app.routes.recommend")


def fixture():
    sources = [{"source_id": "official_one"}, {"source_id": "official_two"}]
    def record(department, dept_id, concept, source_id, priority):
        return {
            "department_name": department,
            "canonical_dept_id": dept_id,
            "canonical_parent_dept": "測試系",
            "canonical_child_dept": department,
            "concept": concept,
            "evidence_type": "direct_routing_evidence",
            "source_id": source_id,
            "evidence_text": f"{department}：{concept}",
            "source_priority": priority,
        }
    records = [
        record("測試甲科", 101, "症狀甲", "official_one", 1),
        record("測試乙科", 102, "症狀甲", "official_two", 2),
        record("測試乙科", 102, "線索乙", "official_two", 2),
        record("近似測試科", 103, "症狀甲", "official_one", 1),
    ]
    active = [
        {"dept_id": "101", "parent_dept": "測試系", "child_dept": "測試甲科"},
        {"dept_id": 102, "parent_dept": "測試系", "child_dept": "測試乙科"},
    ]
    return sources, records, active


def case_with_symptom() -> TriageCase:
    case = TriageCase(case_id="phase4-synthetic")
    case.history_records = [Message(role="user", content="我有症狀甲")]
    case.patient_input.symptom = "症狀甲"
    case.semantic_extractions = [SemanticExtraction(
        field="symptom",
        normalized_value="症狀甲",
        semantic_status="available",
        assertion="present",
        confidence=0.95,
        source_text="症狀甲",
        extractor="ai",
    )]
    return case


def support(dept_id: int, concept: str = "症狀甲", source: str | None = None, text: str = "症狀甲") -> dict:
    return {
        "patient_source_text": text,
        "knowledge_source_id": source or ("official_one" if dept_id == 101 else "official_two"),
        "knowledge_concept": concept,
    }


def candidate(dept_id: int, confidence: object = 0.9, evidence: list[dict] | None = None) -> dict:
    return {"dept_id": dept_id, "confidence": confidence, "supporting_evidence": evidence or [support(dept_id)]}


class Phase4ValidationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sources, self.records, self.active = fixture()
        self.case = case_with_symptom()
        self.retrieved = reasoning.retrieve_official_evidence(
            self.case, self.records, resolve_department_names(self.records, self.active),
        )

    def validate(self, candidates, status="resolved"):
        return reasoning.validate_candidate_proposal(
            {"status": status, "candidates": candidates}, self.retrieved,
            self.sources, self.active, ["我有症狀甲"],
        )

    def test_invalid_dept_id_and_unresolved_near_name_are_rejected(self):
        for dept_id in (999, 103, 0, -1, True, "101"):
            with self.subTest(dept_id=dept_id):
                self.assertEqual(self.validate([candidate(dept_id)])[1], [])
        self.assertNotIn(103, {item["dept_id"] for item in self.retrieved})

    def test_source_id_must_exist_and_belong_to_retrieved_record(self):
        for source_id in ("invented", "official_two"):
            with self.subTest(source_id=source_id):
                self.assertEqual(self.validate([candidate(101, evidence=[support(101, source=source_id)])])[1], [])

    def test_patient_source_and_concept_must_match_grounded_retrieval(self):
        self.assertEqual(self.validate([candidate(101, evidence=[support(101, text="未曾說過")])])[1], [])
        self.assertEqual(self.validate([candidate(101, evidence=[support(101, concept="其他概念")])])[1], [])
        self.assertEqual(self.validate([candidate(101, evidence=[support(101, text="我有症狀甲")])])[1], [])

    def test_invalid_confidence_is_rejected(self):
        for confidence in (math.nan, math.inf, -math.inf, True, -0.1, 1.1, "0.9"):
            with self.subTest(confidence=confidence):
                self.assertEqual(self.validate([candidate(101, confidence)])[1], [])

    def test_multiple_departments_and_priorities_survive_retrieval(self):
        self.assertEqual({item["dept_id"] for item in self.retrieved}, {101, 102})
        self.assertEqual({item["source_priority"] for item in self.retrieved}, {1, 2})
        self.assertEqual(len(self.validate([candidate(101), candidate(102)])[1]), 2)

    def test_duplicate_departments_are_deduplicated_and_top_k_is_three(self):
        self.assertEqual(len(self.validate([candidate(101), candidate(101)])[1]), 1)
        retrieved = list(self.retrieved)
        active = list(self.active)
        candidates = [candidate(101), candidate(102)]
        for dept_id in (104, 105):
            active.append({"dept_id": dept_id, "parent_dept": "測試系", "child_dept": f"測試{dept_id}科"})
            retrieved.append({
                "dept_id": dept_id, "parentDept": "測試系", "childDept": f"測試{dept_id}科",
                "patient_source_text": "症狀甲", "knowledge_source_id": "official_two",
                "knowledge_concept": "症狀甲", "source_priority": 2,
            })
            candidates.append(candidate(dept_id))
        _, validated, _, _ = reasoning.validate_candidate_proposal(
            {"status": "ambiguous", "candidates": candidates}, retrieved,
            self.sources, active, ["我有症狀甲"],
        )
        self.assertEqual(len(validated), 3)

    async def test_backend_owns_ambiguous_and_resolved_gate(self):
        proposals = [
            {"status": "resolved", "candidates": [candidate(101), candidate(102)], "uncertainty_reason": "需要區分", "next_question_intent": "differentiate_signal"},
            {"status": "resolved", "candidates": [candidate(101, 0.54)]},
            {"status": "resolved", "candidates": [candidate(101, 0.9)], "stage": "confirmed", "is_complete": True},
        ]
        provider = AsyncMock(side_effect=[json.dumps(item, ensure_ascii=False) for item in proposals])
        with patch.object(reasoning, "fetch_active_departments", return_value=self.active), patch.object(
            reasoning, "load_department_knowledge", return_value=(self.sources, self.records),
        ), patch.object(reasoning, "complete_runtime_json", new=provider):
            await reasoning.reason_about_departments(self.case)
            self.assertEqual(self.case.conversation_state.department_status, "ambiguous")
            self.assertEqual(len(self.case.conversation_state.candidate_departments), 2)
            self.assertIsNone(self.case.department_result)
            self.assertEqual(self.case.conversation_state.department_next_question_intent, "differentiate_signal")
            await reasoning.reason_about_departments(self.case)
            self.assertEqual(self.case.conversation_state.department_status, "ambiguous")
            self.assertIsNone(self.case.department_result)
            await reasoning.reason_about_departments(self.case)
        self.assertEqual(self.case.conversation_state.department_status, "resolved")
        self.assertEqual(self.case.department_result.dept_id, 101)
        self.assertEqual(self.case.department_result.parentDept, "測試系")
        self.assertFalse(self.case.conversation_state.confirmed)
        self.assertEqual(self.case.conversation_state.stage.value, "collecting")

    async def test_single_valid_candidate_with_ambiguous_proposal_stays_ambiguous(self):
        provider = AsyncMock(return_value=json.dumps({"status": "ambiguous", "candidates": [candidate(101)]}, ensure_ascii=False))
        with patch.object(reasoning, "fetch_active_departments", return_value=self.active), patch.object(
            reasoning, "load_department_knowledge", return_value=(self.sources, self.records),
        ), patch.object(reasoning, "complete_runtime_json", new=provider):
            await reasoning.reason_about_departments(self.case)
        self.assertEqual(self.case.conversation_state.department_status, "ambiguous")
        self.assertIsNone(self.case.department_result)

    async def test_candidate_ai_payload_uses_structured_evidence_not_raw_clinical_text(self):
        self.case.conversation_state.clarification_evidence["severity"] = "raw clarification"
        provider = AsyncMock(return_value=json.dumps({
            "status": "ambiguous",
            "candidates": [candidate(101), candidate(102)],
        }, ensure_ascii=False))
        with patch.object(reasoning, "fetch_active_departments", return_value=self.active), patch.object(
            reasoning, "load_department_knowledge", return_value=(self.sources, self.records),
        ), patch.object(reasoning, "complete_runtime_json", new=provider):
            await reasoning.reason_about_departments(self.case)

        prompt = provider.await_args.args[0]
        payload = json.loads(prompt.split("\n資料：", 1)[1])
        self.assertEqual(
            set(payload),
            {"accepted_semantic_evidence", "retrieved_official_evidence", "previously_asked_intents"},
        )
        self.assertNotIn("conversation_history", payload)
        self.assertNotIn("grounded_patient_evidence", payload)
        self.assertNotIn("accepted_clarification_evidence", payload)
        self.assertEqual(payload["accepted_semantic_evidence"][0]["assertion"], "present")

    async def test_candidate_context_has_all_current_fields_but_kb_uses_only_present_concepts(self):
        case = case_with_symptom()
        case.history_records.extend([
            Message(role="user", content="一開始是左膝不舒服"),
            Message(role="user", content="現在是右下腹，兩天，中等程度，今天開始，還有伴隨線索"),
        ])
        case.semantic_extractions.extend([
            SemanticExtraction(
                field="body_part", normalized_value="左膝", semantic_status="available",
                confidence=0.9, source_text="左膝", extractor="ai",
            ),
            SemanticExtraction(
                field="body_part", normalized_value="右下腹", semantic_status="available",
                confidence=0.95, source_text="右下腹", extractor="ai",
            ),
            SemanticExtraction(
                field="duration", normalized_value="2天", semantic_status="available",
                confidence=0.97, source_text="兩天", extractor="ai",
            ),
            SemanticExtraction(
                field="severity", normalized_value="moderate", semantic_status="available",
                confidence=0.91, source_text="中等程度", extractor="ai",
            ),
            SemanticExtraction(
                field="onset", normalized_value="今天開始", semantic_status="available",
                confidence=0.93, source_text="今天開始", extractor="ai",
            ),
            SemanticExtraction(
                field="accompanying_symptoms", normalized_value=["伴隨線索"],
                semantic_status="available", assertion="present", confidence=0.9,
                source_text="伴隨線索", extractor="ai",
            ),
        ])
        nonconcept_records = [
            {"department_name": "測試甲科", "canonical_dept_id": 101,
             "canonical_parent_dept": "測試系", "canonical_child_dept": "測試甲科",
             "concept": concept, "evidence_type": "direct_routing_evidence", "source_id": "official_one",
             "evidence_text": f"測試甲科：{concept}", "source_priority": 1}
            for concept in ("右下腹", "2天", "moderate", "今天開始")
        ]
        records = [*self.records, *nonconcept_records]
        provider = AsyncMock(return_value=json.dumps({
            "status": "ambiguous",
            "candidates": [candidate(101), candidate(102)],
        }, ensure_ascii=False))
        with patch.object(reasoning, "fetch_active_departments", return_value=self.active), patch.object(
            reasoning, "load_department_knowledge", return_value=(self.sources, records),
        ), patch.object(reasoning, "complete_runtime_json", new=provider):
            await reasoning.reason_about_departments(case)

        payload = json.loads(provider.await_args.args[0].split("\n資料：", 1)[1])
        context = payload["accepted_semantic_evidence"]
        self.assertEqual(
            {item["field"] for item in context},
            {"symptom", "body_part", "duration", "severity", "onset", "accompanying_symptoms"},
        )
        self.assertEqual(
            [item["normalized_value"] for item in context if item["field"] == "body_part"],
            ["右下腹"],
        )
        self.assertEqual(
            {item["knowledge_concept"] for item in payload["retrieved_official_evidence"]},
            {"症狀甲"},
        )

    async def test_live_hematuria_english_normalization_resolves_canonical_kidney_candidate(self):
        case = TriageCase(case_id="live-hematuria-dual-surface")
        case.history_records = [Message(role="user", content="我最近有血尿，已經兩天了")]
        case.semantic_extractions = [
            SemanticExtraction(
                field="symptom", normalized_value="hematuria", semantic_status="available",
                assertion="present", confidence=0.96, source_text="血尿", extractor="ai",
            ),
            SemanticExtraction(
                field="duration", normalized_value="2天", semantic_status="available",
                confidence=0.96, source_text="兩天", extractor="ai",
            ),
        ]
        sources = [{"source_id": "kidney_official"}]
        records = [{
            "department_name": "腎臟科", "canonical_dept_id": 1242,
            "canonical_parent_dept": "內科部", "canonical_child_dept": "腎臟科",
            "concept": "血尿", "evidence_type": "direct_routing_evidence",
            "source_id": "kidney_official", "evidence_text": "腎臟科：血尿",
            "source_priority": 1,
        }]
        active = [{"dept_id": "1242", "parent_dept": "內科部", "child_dept": "腎臟科"}]
        proposal = {
            "status": "resolved",
            "candidates": [{
                "dept_id": 1242,
                "confidence": 0.94,
                "supporting_evidence": [{
                    "patient_source_text": "血尿",
                    "knowledge_source_id": "kidney_official",
                    "knowledge_concept": "血尿",
                }],
            }],
        }
        provider = AsyncMock(return_value=json.dumps(proposal, ensure_ascii=False))
        with patch.object(reasoning, "fetch_active_departments", return_value=active), patch.object(
            reasoning, "load_department_knowledge", return_value=(sources, records),
        ), patch.object(reasoning, "complete_runtime_json", new=provider):
            await reasoning.reason_about_departments(case)

        payload = json.loads(provider.await_args.args[0].split("\n資料：", 1)[1])
        self.assertEqual(len(payload["retrieved_official_evidence"]), 1)
        self.assertEqual(payload["retrieved_official_evidence"][0]["patient_source_text"], "血尿")
        self.assertEqual(case.conversation_state.department_status, "resolved")
        self.assertEqual(case.department_result.dept_id, 1242)
        self.assertEqual(case.department_result.childDept, "腎臟科")

    async def test_new_grounded_answer_recomputes_candidates_and_converges(self):
        first = {"status": "ambiguous", "candidates": [candidate(101, 0.8), candidate(102, 0.77)], "next_question_intent": "differentiate_signal"}
        second = {"status": "resolved", "candidates": [candidate(102, 0.91, [support(102, "線索乙", text="線索乙")])]}
        provider = AsyncMock(side_effect=[json.dumps(first, ensure_ascii=False), json.dumps(second, ensure_ascii=False)])
        with patch.object(reasoning, "fetch_active_departments", return_value=self.active), patch.object(
            reasoning, "load_department_knowledge", return_value=(self.sources, self.records),
        ), patch.object(reasoning, "complete_runtime_json", new=provider):
            await reasoning.reason_about_departments(self.case)
            self.case.history_records.append(Message(role="user", content="還有線索乙"))
            self.case.conversation_state.clarification_evidence["differentiate_signal"] = "線索乙"
            self.case.semantic_extractions.append(SemanticExtraction(
                field="accompanying_symptoms",
                normalized_value=["線索乙"],
                semantic_status="available",
                assertion="present",
                confidence=0.95,
                source_text="線索乙",
                extractor="ai",
            ))
            await reasoning.reason_about_departments(self.case)
        self.assertEqual(self.case.conversation_state.department_status, "resolved")
        self.assertEqual([item.dept_id for item in self.case.conversation_state.candidate_departments], [102])
        self.assertEqual(self.case.department_result.dept_id, 102)
        self.assertEqual(provider.await_count, 2)

    async def test_provider_failure_clears_stale_candidate_without_legacy_fallback(self):
        self.case.department_result = None
        for raw in (TimeoutError(), "{bad json"):
            with self.subTest(raw=type(raw).__name__):
                provider = AsyncMock(side_effect=raw if isinstance(raw, Exception) else None,
                                     return_value=raw if isinstance(raw, str) else None)
                with patch.object(reasoning, "fetch_active_departments", return_value=self.active), patch.object(
                    reasoning, "load_department_knowledge", return_value=(self.sources, self.records),
                ), patch.object(reasoning, "complete_runtime_json", new=provider):
                    await reasoning.reason_about_departments(self.case)
                self.assertEqual(self.case.conversation_state.department_status, "unresolved")
                self.assertEqual(self.case.conversation_state.candidate_departments, [])
                self.assertIsNone(self.case.department_result)

    async def test_explicit_department_preference_is_validated_but_not_kb_claim(self):
        self.case.history_records.append(Message(role="user", content="我想看測試甲科"))
        self.case.patient_input.requested_department_name = "測試甲科"
        self.case.patient_input.requested_department_id = 101
        provider = AsyncMock()
        with patch.object(reasoning, "fetch_active_departments", return_value=self.active), patch.object(
            reasoning, "complete_runtime_json", new=provider):
            await reasoning.reason_about_departments(self.case)
        provider.assert_not_awaited()
        self.assertEqual(self.case.conversation_state.department_status, "resolved")
        self.assertIn("使用者明確指定", self.case.department_result.reason[0])

    def test_production_service_does_not_use_audit_inventory(self):
        self.assertNotIn("PHASE3_310_DEPARTMENT_INVENTORY", inspect.getsource(reasoning))


class Phase4ChatGateTest(unittest.IsolatedAsyncioTestCase):
    def no_ai_chat(self, message: str, *, triage_case: dict | None = None, confirmed: bool = False):
        settings = SimpleNamespace(cerebras_api_key="", batch_triage_enabled=False)
        old_detector = AsyncMock()
        with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
            chat_route, "detect_department_result", new=old_detector,
        ), patch.object(chat_route, "generate_triage_reply", new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"])):
            response = TestClient(app).post("/chat", json={
                "message": message, "confirmed": confirmed,
                **({"triage_case": triage_case} if triage_case else {}),
            })
        self.assertEqual(response.status_code, 200)
        old_detector.assert_not_awaited()
        return response.json()

    def test_missing_ai_key_free_text_stays_unresolved(self):
        result = self.no_ai_chat("我最近一直頭暈", confirmed=True)
        self.assertTrue(result["conversation_state"]["free_text_mode"])
        self.assertEqual(result["conversation_state"]["department_status"], "unresolved")
        self.assertIsNone(result["department_result"])
        self.assertEqual(result["conversation_state"]["stage"], "collecting")
        self.assertFalse(result["conversation_state"]["awaiting_confirmation"])
        self.assertFalse(result["conversation_state"]["confirmed"])
        confirmed = self.no_ai_chat("", triage_case=result["triage_case"], confirmed=True)
        self.assertEqual(confirmed["conversation_state"]["stage"], "collecting")
        self.assertIsNone(confirmed["department_result"])

    def test_missing_ai_key_messages_user_text_is_also_free_text(self):
        settings = SimpleNamespace(cerebras_api_key="", batch_triage_enabled=False)
        old_detector = AsyncMock()
        with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
            chat_route, "detect_department_result", new=old_detector,
        ), patch.object(chat_route, "generate_triage_reply", new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"])):
            response = TestClient(app).post("/chat", json={"messages": [{"role": "user", "content": "我膝蓋很痛"}], "confirmed": True})
        self.assertEqual(response.status_code, 200)
        old_detector.assert_not_awaited()
        self.assertTrue(response.json()["conversation_state"]["free_text_mode"])
        self.assertIsNone(response.json()["department_result"])
        self.assertEqual(response.json()["conversation_state"]["stage"], "collecting")

    def test_missing_ai_key_knee_pain_cannot_use_legacy_mapping_even_later(self):
        first = self.no_ai_chat("我膝蓋很痛")
        self.assertIsNone(first["department_result"])
        second = self.no_ai_chat(
            "膝蓋痛兩週，爬樓梯很吃力，沒有胸痛呼吸困難意識不清大量出血，週一上午可以看診",
            triage_case=first["triage_case"], confirmed=True,
        )
        self.assertEqual(second["conversation_state"]["department_status"], "unresolved")
        self.assertIsNone(second["department_result"])
        self.assertEqual(second["conversation_state"]["stage"], "collecting")
        self.assertFalse(second["conversation_state"]["confirmed"])

    def test_missing_ai_key_explicit_preference_uses_exact_live_db(self):
        row = {"dept_id": 1234, "parent_dept": "內科系", "child_dept": "感染科"}
        with patch.object(department_preference_service, "fetch_active_departments", return_value=[row]) as live_db:
            result = self.no_ai_chat("我想看感染科")
        self.assertGreaterEqual(live_db.call_count, 1)
        self.assertEqual(result["department_result"]["dept_id"], 1234)
        self.assertEqual(result["department_result"]["childDept"], "感染科")
        self.assertEqual(result["conversation_state"]["department_status"], "resolved")
        self.assertIn("使用者明確指定", result["department_result"]["reason"][0])
        self.assertNotIn("官方 KB", result["department_result"]["reason"][0])

    def test_missing_ai_key_free_text_safety_fails_closed(self):
        result = self.no_ai_chat("我最近一直頭暈")
        self.assertFalse(result["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(result["triage_case"]["ttas_result"]["status"], "insufficient_information")
        self.assertIsNone(result["triage_case"]["ttas_result"]["level_candidate"])
        self.assertIsNone(result["triage"]["urgency_score"])
        self.assertIsNone(result["triage"]["urgency_level"])
        self.assertIsNone(result["department_result"])
        self.assertEqual(result["conversation_state"]["stage"], "collecting")

    def test_missing_ai_key_pending_safety_is_not_replaced_by_department(self):
        case = TriageCase(case_id="phase4-no-ai-pending-safety")
        case.history_records = [Message(role="user", content="我最近一直頭暈，上禮拜開始")]
        case.patient_input.symptom = "頭暈"
        case.semantic_extractions = [SemanticExtraction(
            field="duration", normalized_value="一週", source_text="上禮拜",
            confidence=0.9, semantic_status="available",
        )]
        case.conversation_state.clarification_status = "safety_check"
        case.conversation_state.last_question_key = RED_FLAG_QUESTION_KEY
        case.conversation_state.free_text_mode = True
        save_case(case)
        result = self.no_ai_chat("不太確定", triage_case=case.model_dump(mode="json"))
        self.assertEqual(result["conversation_state"]["clarification_status"], "safety_check")
        self.assertEqual(result["conversation_state"]["last_question_key"], RED_FLAG_QUESTION_KEY)
        self.assertIn("胸痛", result["next_question"])
        self.assertIsNone(result["department_result"])
        self.assertEqual(result["conversation_state"]["stage"], "collecting")

    def test_missing_ai_key_medical_phrase_does_not_trigger_legacy_urgency(self):
        result = self.no_ai_chat("我突然胸痛")
        self.assertEqual(result["triage_case"]["patient_input"]["red_flags"], [])
        self.assertFalse(result["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertFalse(result["triage"]["warning_required"])
        self.assertIsNone(result["triage"]["urgency_score"])
        self.assertIsNone(result["triage"]["urgency_level"])
        self.assertEqual(result["triage_case"]["ttas_result"]["status"], "insufficient_information")
        self.assertIsNone(result["department_result"])

    def test_live_dizziness_department_question_is_preempted_by_safety(self):
        settings = SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=False)
        symptom = SemanticExtraction(
            field="symptom", normalized_value="頭暈", semantic_status="available",
            assertion="present", confidence=0.96, source_text="頭暈", extractor="ai",
        )
        duration = SemanticExtraction(
            field="duration", normalized_value="3天", semantic_status="available",
            confidence=0.95, source_text="三天", extractor="ai",
        )
        accompanying = SemanticExtraction(
            field="accompanying_symptoms", normalized_value=["眩暈"],
            semantic_status="available", assertion="present", confidence=0.95,
            source_text="眩暈", extractor="ai",
        )

        async def semantic(case, **_):
            case.patient_input.symptom = "頭暈"
            case.semantic_extractions.extend([symptom, duration, accompanying])
            return SimpleNamespace(semantic_extractions=[symptom, duration, accompanying])

        async def converge(case):
            state = case.conversation_state
            state.department_status = "ambiguous"
            state.department_next_question_intent = "cardiac_symptom"

        planner = AsyncMock(return_value=ClarificationSuggestion(
            "clarification_needed",
            "請問您近期有沒有感覺胸口悸動或胸痛？",
            "cardiac_symptom",
            "需要區分候選科別",
        ))
        with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
            chat_route, "refine_case_with_ai", new=semantic,
        ), patch.object(chat_route, "reason_about_departments", new=converge), patch.object(
            chat_route, "request_clarification", new=planner,
        ), patch.object(
            chat_route, "generate_triage_reply",
            new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"]),
        ):
            response = TestClient(app).post("/chat", json={
                "message": "我這三天一直頭暈，會有眩暈、天旋地轉的感覺。",
            })

        self.assertEqual(response.status_code, 200)
        result = response.json()
        state = result["conversation_state"]
        planner.assert_awaited_once()
        self.assertEqual(state["clarification_status"], "safety_check")
        self.assertEqual(state["last_question_key"], RED_FLAG_QUESTION_KEY)
        self.assertEqual(result["next_question"], QUESTION_TEXTS[RED_FLAG_QUESTION_KEY])
        self.assertFalse(result["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertIsNone(state["pending_clarification_intent"])
        self.assertEqual(state["asked_clarification_intents"], [])
        self.assertEqual(state["department_next_question_intent"], "cardiac_symptom")
        self.assertNotEqual(result["next_question"], "請問您近期有沒有感覺胸口悸動或胸痛？")

    def test_completed_safety_turn_recomputes_department_and_plans_next_question(self):
        settings = SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=False)
        symptom = SemanticExtraction(
            field="symptom", normalized_value="頭暈", semantic_status="available",
            assertion="present", confidence=0.96, source_text="頭暈", extractor="ai",
        )
        duration = SemanticExtraction(
            field="duration", normalized_value="3天", semantic_status="available",
            confidence=0.95, source_text="三天", extractor="ai",
        )
        accompanying = SemanticExtraction(
            field="accompanying_symptoms", normalized_value=["眩暈"],
            semantic_status="available", assertion="present", confidence=0.95,
            source_text="眩暈", extractor="ai",
        )

        interpretation_calls = 0

        async def interpret(case, **_):
            nonlocal interpretation_calls
            interpretation_calls += 1
            if interpretation_calls == 1:
                case.patient_input.symptom = "頭暈"
                case.semantic_extractions.extend([symptom, duration, accompanying])
                return SimpleNamespace(
                    semantic_extractions=[symptom, duration, accompanying],
                    pending_answer=None,
                )
            return SimpleNamespace(
                semantic_extractions=[],
                pending_answer=PendingAnswerInterpretation(
                    answered_intent=SAFETY_PENDING_INTENT,
                    answer_status="answered",
                    answer_source_text="都沒有，我沒有剛才提到的那些急迫症狀。",
                    answer_confidence=0.98,
                    answer_assertion="absent",
                ),
            )

        semantic = AsyncMock(side_effect=interpret)
        intent_before_recompute: list[str | None] = []

        async def recompute(case):
            state = case.conversation_state
            intent_before_recompute.append(state.department_next_question_intent)
            state.department_status = "ambiguous"
            state.department_next_question_intent = "ask_hearing_related_symptoms"

        department = AsyncMock(side_effect=recompute)
        question = "請問你頭暈時，有沒有伴隨耳鳴或聽力變化？"
        planner_seen_intents: list[str | None] = []

        async def plan(case, *_args, **_kwargs):
            planner_seen_intents.append(case.conversation_state.department_next_question_intent)
            return ClarificationSuggestion(
                "clarification_needed",
                question,
                "ask_hearing_related_symptoms",
                "需要區分目前的官方候選科別",
            )

        planner = AsyncMock(side_effect=plan)
        with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
            chat_route, "refine_case_with_ai", new=semantic,
        ), patch.object(
            chat_route, "apply_user_message", wraps=apply_user_message,
        ) as apply_message, patch(
            "app.services.conversation_service.classify_pending_answer",
            new=AsyncMock(side_effect=AssertionError("standalone classifier must not run")),
        ), patch.object(chat_route, "reason_about_departments", new=department), patch.object(
            chat_route, "request_clarification", new=planner,
        ), patch.object(
            chat_route, "generate_triage_reply",
            new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"]),
        ):
            client = TestClient(app)
            first_response = client.post("/chat", json={
                "message": "我這三天一直頭暈，會有眩暈、天旋地轉的感覺。",
            })
            self.assertEqual(first_response.status_code, 200)
            first = first_response.json()
            self.assertEqual(first["conversation_state"]["clarification_status"], "safety_check")
            self.assertEqual(first["next_question"], QUESTION_TEXTS[RED_FLAG_QUESTION_KEY])

            stored = get_case(first["case_id"])
            self.assertIsNotNone(stored)
            stored.conversation_state.department_next_question_intent = "stale_before_safety"
            save_case(stored)

            second_response = client.post("/chat", json={
                "case_id": first["case_id"],
                "message": "都沒有，我沒有剛才提到的那些急迫症狀。",
            })

        self.assertEqual(second_response.status_code, 200)
        second = second_response.json()
        self.assertEqual(semantic.await_count, 2)
        self.assertFalse(apply_message.call_args_list[-1].kwargs["apply_legacy_safety"])
        self.assertEqual(department.await_count, 2)
        self.assertEqual(intent_before_recompute, [None, "stale_before_safety"])
        self.assertEqual(planner.await_count, 2)
        self.assertEqual(planner_seen_intents, [
            "ask_hearing_related_symptoms", "ask_hearing_related_symptoms",
        ])
        self.assertTrue(second["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(second["triage_case"]["patient_input"]["red_flags_status"], "negative")
        self.assertEqual(second["next_question"], question)
        self.assertNotIn(second["next_question"], NEUTRAL_PENDING_RETRIES)
        self.assertEqual(
            second["conversation_state"]["pending_clarification_intent"],
            "ask_hearing_related_symptoms",
        )

    def test_live_a1_a2_a3_keeps_new_evidence_and_uses_contextual_follow_up(self):
        settings = SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=False)
        a1_text = "我這三天一直頭暈，會有眩暈、天旋地轉的感覺。"
        a3_text = "沒有聽力下降，但我右耳一直有耳鳴，眩暈的時候耳鳴會更明顯。"
        hearing_key = "ask_hearing_loss_or_chest_pain"
        next_key = "ask_nausea_or_visual_disturbance"
        initial = [
            SemanticExtraction(
                field="symptom", normalized_value="頭暈", semantic_status="available",
                assertion="present", confidence=0.97, source_text="頭暈", extractor="ai",
            ),
            SemanticExtraction(
                field="duration", normalized_value="3天", semantic_status="available",
                confidence=0.96, source_text="三天", extractor="ai",
            ),
            SemanticExtraction(
                field="accompanying_symptoms", normalized_value=["眩暈"],
                semantic_status="available", assertion="present", confidence=0.96,
                source_text="眩暈", extractor="ai",
            ),
        ]
        a3_evidence = [
            SemanticExtraction(
                field="accompanying_symptoms", normalized_value=["聽力下降"],
                semantic_status="available", assertion="absent", confidence=0.98,
                source_text="沒有聽力下降", extractor="ai",
            ),
            SemanticExtraction(
                field="body_part", normalized_value="右耳", semantic_status="available",
                confidence=0.98, source_text="右耳", extractor="ai",
            ),
            SemanticExtraction(
                field="accompanying_symptoms", normalized_value=["耳鳴"],
                semantic_status="available", assertion="present", confidence=0.98,
                source_text="右耳一直有耳鳴", extractor="ai",
            ),
            SemanticExtraction(
                field="accompanying_symptoms", normalized_value=["眩暈"],
                semantic_status="available", assertion="present", confidence=0.98,
                source_text="眩暈", extractor="ai",
            ),
        ]
        interpretation_calls = 0

        async def interpret(case, **_):
            nonlocal interpretation_calls
            interpretation_calls += 1
            if interpretation_calls == 1:
                apply_semantic_extractions(case, initial, allow_red_flag_completion=False)
                return RagTriageSuggestion(semantic_extractions=initial)
            if interpretation_calls == 2:
                return RagTriageSuggestion(
                    semantic_extractions=[],
                    pending_answer=PendingAnswerInterpretation(
                        answered_intent=SAFETY_PENDING_INTENT,
                        answer_status="answered",
                        answer_source_text="都沒有，我沒有剛才提到的那些急迫症狀。",
                        answer_confidence=0.98,
                        answer_assertion="absent",
                    ),
                )
            apply_semantic_extractions(case, a3_evidence, allow_red_flag_completion=False)
            return RagTriageSuggestion(
                semantic_extractions=a3_evidence,
                pending_answer=PendingAnswerInterpretation(
                    answered_intent=hearing_key,
                    answer_status="answered",
                    answer_source_text="沒有聽力下降",
                    answer_confidence=0.98,
                ),
            )

        semantic = AsyncMock(side_effect=interpret)
        _, records = load_department_knowledge()
        resolutions = resolve_department_names(records, [{
            "dept_id": 1333, "parentDept": "五官科", "childDept": "耳科",
        }])
        retrieved_by_turn: list[list[dict]] = []

        async def recompute(case):
            retrieved_by_turn.append(
                reasoning.retrieve_official_evidence(case, records, resolutions)
            )
            state = case.conversation_state
            state.department_status = "ambiguous"
            state.department_next_question_intent = (
                next_key if len(retrieved_by_turn) == 3 else hearing_key
            )

        department = AsyncMock(side_effect=recompute)
        hearing_question = "請問您是否有聽力下降的情形？"
        nausea_question = "請問眩暈時會不會伴隨噁心或想吐？"

        async def plan(case, *_args, **_kwargs):
            key = case.conversation_state.department_next_question_intent
            return ClarificationSuggestion(
                "clarification_needed",
                nausea_question if key == next_key else hearing_question,
                key,
                "需要進一步區分目前候選",
            )

        planner = AsyncMock(side_effect=plan)
        legacy_classifier = AsyncMock(side_effect=AssertionError("legacy classifier must not run"))
        with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
            chat_route, "refine_case_with_ai", new=semantic,
        ), patch.object(chat_route, "reason_about_departments", new=department), patch.object(
            chat_route, "request_clarification", new=planner,
        ), patch(
            "app.services.conversation_service.classify_pending_answer", new=legacy_classifier,
        ), patch.object(
            chat_route, "generate_triage_reply",
            new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"]),
        ):
            client = TestClient(app)
            a1 = client.post("/chat", json={"message": a1_text}).json()
            self.assertEqual(a1["conversation_state"]["clarification_status"], "safety_check")

            a2 = client.post("/chat", json={
                "case_id": a1["case_id"],
                "message": "都沒有，我沒有剛才提到的那些急迫症狀。",
            }).json()
            self.assertEqual(a2["next_question"], hearing_question)
            self.assertEqual(a2["conversation_state"]["pending_clarification_intent"], hearing_key)

            a3 = client.post("/chat", json={
                "case_id": a1["case_id"],
                "message": a3_text,
            }).json()

        self.assertEqual(semantic.await_count, 3)
        legacy_classifier.assert_not_awaited()
        self.assertEqual(department.await_count, 3)
        self.assertEqual(planner.await_count, 3)
        semantic_history = a3["triage_case"]["semantic_extractions"]
        self.assertTrue(any(
            item["normalized_value"] == ["聽力下降"] and item["assertion"] == "absent"
            for item in semantic_history
        ))
        self.assertTrue(any(
            item["normalized_value"] == ["耳鳴"] and item["assertion"] == "present"
            for item in semantic_history
        ))
        self.assertTrue(any(item["field"] == "body_part" and item["normalized_value"] == "右耳" for item in semantic_history))
        self.assertTrue(any(
            item["normalized_value"] == ["眩暈"] and item["assertion"] == "present"
            for item in semantic_history
        ))
        self.assertEqual(
            a3["conversation_state"]["clarification_evidence"][hearing_key],
            "沒有聽力下降",
        )
        tinnitus_support = [
            item for item in retrieved_by_turn[-1]
            if item["dept_id"] == 1333 and item["knowledge_concept"] == "耳鳴"
        ]
        self.assertTrue(tinnitus_support)
        self.assertEqual(tinnitus_support[0]["patient_source_text"], "右耳一直有耳鳴")
        self.assertEqual(a3["next_question"], nausea_question)
        self.assertNotIn(a3["next_question"], NEUTRAL_PENDING_RETRIES)
        self.assertEqual(a3["conversation_state"]["pending_clarification_intent"], next_key)

    async def test_safety_question_precedes_candidate_question(self):
        case = case_with_symptom()
        case.conversation_state.department_status = "ambiguous"
        case.conversation_state.department_next_question_intent = "differentiate_signal"
        case.semantic_extractions = [SemanticExtraction(
            field="body_part", normalized_value="症狀甲", semantic_status="available",
            confidence=0.9, source_text="症狀甲",
        )]
        advance_conversation(
            case, ClarificationSuggestion("sufficient", None, None, "症狀足夠"),
            user_sources=["我有症狀甲"], current_extractions=[],
        )
        self.assertEqual(case.conversation_state.clarification_status, "safety_check")
        self.assertEqual(case.triage.next_question, QUESTION_TEXTS[RED_FLAG_QUESTION_KEY])
        self.assertEqual(case.conversation_state.asked_clarification_intents, [])

    async def test_ambiguous_department_clarification_cannot_skip_safety(self):
        case = case_with_symptom()
        state = case.conversation_state
        state.department_status = "ambiguous"
        state.department_next_question_intent = "differentiate_signal"
        case.semantic_extractions.append(SemanticExtraction(
            field="body_part", normalized_value="症狀甲", semantic_status="available",
            confidence=0.95, source_text="症狀甲", extractor="ai",
        ))
        proposal = ClarificationSuggestion(
            "clarification_needed",
            "請問是否還有其他可區分症狀的細節？",
            "differentiate_signal",
            "需要區分候選科別",
        )

        advance_conversation(
            case,
            proposal,
            user_sources=["我有症狀甲"],
            current_extractions=[],
        )

        self.assertEqual(state.clarification_status, "safety_check")
        self.assertEqual(case.triage.next_question, QUESTION_TEXTS[RED_FLAG_QUESTION_KEY])
        self.assertEqual(state.last_question_key, RED_FLAG_QUESTION_KEY)
        self.assertFalse(case.patient_input.red_flags_checked)
        self.assertIsNone(state.pending_clarification_intent)
        self.assertEqual(state.asked_clarification_intents, [])
        self.assertEqual(state.department_next_question_intent, "differentiate_signal")

    def test_partial_negative_outside_safety_question_does_not_complete_screen(self):
        case = TriageCase(case_id="phase4-partial-negative-outside-safety")

        apply_user_message(case, "沒有胸痛，也沒有心悸。", semantic_first=True)

        self.assertFalse(case.patient_input.red_flags_checked)
        self.assertEqual(case.patient_input.red_flags_status, "not_checked")
        self.assertEqual(case.patient_input.red_flags, [])

    def test_uninterpreted_negative_answer_to_safety_question_fails_closed(self):
        case = TriageCase(case_id="phase4-negative-safety-answer")
        case.conversation_state.last_question_key = RED_FLAG_QUESTION_KEY

        apply_user_message(case, "都沒有。", semantic_first=True)

        self.assertFalse(case.patient_input.red_flags_checked)
        self.assertEqual(case.patient_input.red_flags_status, "not_checked")
        self.assertEqual(case.patient_input.red_flags, [])

    def test_uninterpreted_positive_phrase_does_not_bypass_ttas(self):
        case = TriageCase(case_id="phase4-positive-safety-signal")

        apply_user_message(case, "我突然胸痛", semantic_first=True)

        self.assertFalse(case.patient_input.red_flags_checked)
        self.assertEqual(case.patient_input.red_flags_status, "not_checked")
        self.assertEqual(case.patient_input.red_flags, [])

    async def test_hard_cap_with_ambiguous_candidates_stays_unresolved(self):
        case = case_with_symptom()
        case.patient_input.red_flags_checked = True
        case.conversation_state.turn_count = 7
        case.conversation_state.department_status = "ambiguous"
        case.semantic_extractions = [SemanticExtraction(
            field="body_part", normalized_value="症狀甲", semantic_status="available",
            confidence=0.9, source_text="症狀甲",
        )]
        advance_conversation(
            case, ClarificationSuggestion("sufficient", None, None, "症狀足夠"),
            user_sources=["我有症狀甲"], current_extractions=[],
        )
        self.assertEqual(case.conversation_state.turn_count, 8)
        self.assertEqual(case.conversation_state.clarification_status, "unresolved")
        self.assertIsNone(case.department_result)

    async def test_required_intent_mismatch_is_rejected(self):
        case = case_with_symptom()
        case.conversation_state.department_status = "ambiguous"
        case.conversation_state.department_next_question_intent = "differentiate_signal"
        raw = {"status": "clarification_needed", "question": "可以補充症狀如何變化嗎？", "intent": "wrong_intent", "reason": "待釐清"}
        with patch("app.services.conversation_service.complete_runtime_json", new=AsyncMock(return_value=json.dumps(raw, ensure_ascii=False))):
            suggestion = await request_clarification(case, ["我有症狀甲"])
        self.assertIsNone(suggestion.question)
        self.assertIsNone(suggestion.intent)

    async def test_compound_required_key_accepts_single_focus_question_when_echoed(self):
        case = case_with_symptom()
        required = "ask_nausea_or_visual_disturbance"
        case.conversation_state.department_status = "ambiguous"
        case.conversation_state.department_next_question_intent = required
        raw = {
            "status": "clarification_needed",
            "question": "請問眩暈時會不會伴隨噁心或想吐？",
            "intent": required,
            "reason": "需要區分目前候選",
            "answered_intent": None,
            "answer_source_text": None,
            "answer_status": None,
            "answer_confidence": None,
        }
        provider = AsyncMock(return_value=json.dumps(raw, ensure_ascii=False))
        with patch("app.services.conversation_service.complete_runtime_json", new=provider):
            suggestion = await request_clarification(
                case, ["我有症狀甲"], classify_answer_fields=False,
            )

        self.assertEqual(suggestion.question, raw["question"])
        self.assertEqual(suggestion.intent, required)
        prompt = provider.await_args.args[0]
        self.assertIn("opaque correlation key", prompt)
        self.assertIn("原樣 echo", prompt)

    async def test_compound_required_key_still_rejects_changed_correlation_key(self):
        case = case_with_symptom()
        case.conversation_state.department_status = "ambiguous"
        case.conversation_state.department_next_question_intent = "ask_nausea_or_visual_disturbance"
        raw = {
            "status": "clarification_needed",
            "question": "請問眩暈時會不會伴隨噁心或想吐？",
            "intent": "ask_nausea",
            "reason": "需要區分目前候選",
            "answered_intent": None,
            "answer_source_text": None,
            "answer_status": None,
            "answer_confidence": None,
        }
        with patch(
            "app.services.conversation_service.complete_runtime_json",
            new=AsyncMock(return_value=json.dumps(raw, ensure_ascii=False)),
        ):
            suggestion = await request_clarification(
                case, ["我有症狀甲"], classify_answer_fields=False,
            )

        self.assertIsNone(suggestion.question)
        self.assertIsNone(suggestion.intent)

    async def test_free_text_never_calls_legacy_single_department_detector(self):
        settings = SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=False)
        async def semantic(case, **_):
            case.patient_input.symptom = "症狀甲"
            return SimpleNamespace(
                semantic_extractions=[], pending_answer=None, interpretation_complete=True,
            )
        async def converge(case):
            case.conversation_state.department_status = "ambiguous"
        planner = ClarificationSuggestion("clarification_needed", "可以補充症狀如何變化嗎？", "differentiate_signal", "待釐清")
        old_detector = AsyncMock()
        with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
            chat_route, "refine_case_with_ai", new=semantic,
        ), patch.object(chat_route, "reason_about_departments", new=converge), patch.object(
            chat_route, "request_clarification", new=AsyncMock(return_value=planner),
        ), patch.object(chat_route, "detect_department_result", new=old_detector), patch.object(
            chat_route, "generate_triage_reply", new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"]),
        ):
            response = TestClient(app).post("/chat", json={"message": "我有症狀甲"})
        self.assertEqual(response.status_code, 200)
        old_detector.assert_not_awaited()
        self.assertEqual(response.json()["conversation_state"]["department_status"], "ambiguous")
        self.assertEqual(response.json()["conversation_state"]["stage"], "collecting")

    async def test_recommend_rejects_missing_result_without_detector(self):
        case = case_with_symptom()
        case.visit_type = VisitType.INITIAL
        case.patient_input.red_flags_checked = True
        case.conversation_state.is_complete = True
        case.conversation_state.confirmed = True
        case.triage.need_more_info = False
        case.confirmed = True
        save_case(case)
        detector = AsyncMock()
        with patch.object(recommend_route, "detect_department_result", new=detector):
            response = TestClient(app).post("/recommend", json={"triage_case": case.model_dump(mode="json"), "visit_type": "initial"})
        self.assertEqual(response.status_code, 422)
        detector.assert_not_awaited()

    async def test_recommend_allows_only_live_validated_explicit_preference_exception(self):
        case = case_with_symptom()
        case.visit_type = VisitType.INITIAL
        case.patient_input.red_flags_checked = True
        case.patient_input.requested_department_id = 101
        case.patient_input.requested_department_name = "測試甲科"
        case.conversation_state.clarification_status = "sufficient"
        case.conversation_state.is_complete = True
        case.conversation_state.confirmed = True
        case.triage.need_more_info = False
        case.confirmed = True
        save_case(case)
        explicit = DepartmentResult(dept_id=101, parentDept="測試系", childDept="測試甲科", reason=["使用者明確指定"])
        recommender = AsyncMock(side_effect=DepartmentResolutionError("test stop"))
        with patch.object(recommend_route, "resolve_requested_department", return_value=explicit) as validator, patch.object(
            recommend_route, "recommend_appointments", new=recommender,
        ), patch.object(recommend_route, "detect_department_result", new=AsyncMock()) as detector:
            response = TestClient(app).post("/recommend", json={"triage_case": case.model_dump(mode="json"), "visit_type": "initial"})
        validator.assert_called_once()
        recommender.assert_awaited_once()
        detector.assert_not_awaited()
        self.assertEqual(response.status_code, 422)

    async def test_resolved_case_safety_then_confirmation_keeps_validated_result(self):
        settings = SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=False)
        interpretation_calls = 0

        async def semantic(case, **_):
            nonlocal interpretation_calls
            interpretation_calls += 1
            if interpretation_calls == 2:
                return SimpleNamespace(
                    semantic_extractions=[],
                    pending_answer=PendingAnswerInterpretation(
                        answered_intent=SAFETY_PENDING_INTENT,
                        answer_status="answered",
                        answer_source_text="沒有胸痛、呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛",
                        answer_confidence=0.98,
                        answer_assertion="absent",
                    ),
                    interpretation_complete=True,
                )
            case.patient_input.symptom = "頭暈"
            case.semantic_extractions.append(SemanticExtraction(
                field="duration", normalized_value="一週", source_text="上禮拜",
                confidence=0.9, semantic_status="available",
            ))
            return SimpleNamespace(
                semantic_extractions=[], pending_answer=None, interpretation_complete=True,
            )
        async def converge(case):
            case.conversation_state.department_status = "resolved"
            case.department_result = DepartmentResult(
                dept_id=101, parentDept="測試系", childDept="測試甲科", confidence=0.9,
            )
        old_detector = AsyncMock()
        with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
            chat_route, "refine_case_with_ai", new=semantic,
        ) as semantic_mock, patch.object(chat_route, "reason_about_departments", new=converge), patch.object(
            chat_route, "request_clarification", new=AsyncMock(return_value=ClarificationSuggestion("sufficient", None, None, "足夠")),
        ) as planner, patch.object(chat_route, "detect_department_result", new=old_detector), patch.object(
            chat_route, "generate_triage_reply", new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"]),
        ):
            client = TestClient(app)
            first = client.post("/chat", json={"message": "我頭暈，上禮拜開始"}).json()
            self.assertEqual(first["conversation_state"]["clarification_status"], "safety_check")
            second = client.post("/chat", json={
                "triage_case": first["triage_case"],
                "message": "沒有胸痛、呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛",
            }).json()
            self.assertEqual(second["conversation_state"]["stage"], "waiting_confirmation")
            self.assertEqual(second["department_result"]["dept_id"], 101)
            third = client.post("/chat", json={"triage_case": second["triage_case"], "confirmed": True}).json()
        self.assertEqual(third["conversation_state"]["stage"], "recommending")
        self.assertTrue(third["conversation_state"]["confirmed"])
        self.assertEqual(third["department_result"]["dept_id"], 101)
        self.assertEqual(planner.await_count, 2)
        old_detector.assert_not_awaited()
