from __future__ import annotations

import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import Message, TriageCase
from app.services import batch_extraction_service, rag_triage_adapter
from app.services.case_store import save_case


chat_route = importlib.import_module("app.routes.chat")
recommend_route = importlib.import_module("app.routes.recommend")


def extraction(field, value, source, confidence=0.95, assertion=None):
    item = {
        "field": field,
        "normalized_value": value,
        "semantic_status": "available",
        "confidence": confidence,
        "source_text": source,
    }
    if assertion is not None or field in {"symptom", "accompanying_symptoms"}:
        item["assertion"] = assertion or "present"
    return item


class Phase1AiFirstTest(unittest.TestCase):
    def test_semantic_prompt_requires_minimal_directly_supporting_source_span(self):
        case = TriageCase(case_id="phase1-source-span-prompt")
        case.history_records = [
            Message(role="user", content="我沒有頭暈，但是現在有胸口痛"),
        ]

        prompt = rag_triage_adapter._build_symptom_collection_prompt(case)

        self.assertIn("最短連續逐字原文", prompt)
        self.assertIn("直接支持該 extraction", prompt)
        self.assertIn("不得包含無關的前後症狀、否定句或其他子句", prompt)
        self.assertIn("present 表示患者陳述存在", prompt)
        self.assertIn("若同一句包含不同 polarity，必須拆成多筆 extraction", prompt)

    def post_text(self, message, provider_result, *, triage_case=None):
        settings = SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=False)
        provider = AsyncMock(
            side_effect=provider_result if isinstance(provider_result, Exception) else None,
            return_value=(
                json.dumps(provider_result, ensure_ascii=False)
                if isinstance(provider_result, dict) else provider_result
            ),
        )
        with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
            rag_triage_adapter, "get_settings", return_value=settings,
        ), patch.object(
            rag_triage_adapter, "complete_prompt", new=provider,
        ), patch.object(
            chat_route, "detect_department_result", new=AsyncMock(return_value=None),
        ), patch.object(
            chat_route, "generate_triage_reply",
            new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"]),
        ):
            response = TestClient(app).post(
                "/chat", json={"message": message, **({"triage_case": triage_case} if triage_case else {})},
            )
        self.assertEqual(response.status_code, 200)
        return response.json(), provider

    def test_clear_free_text_calls_ai_and_applies_grounded_extractions(self):
        result, provider = self.post_text(
            "我頭暈兩天",
            {"semantic_extractions": [
                extraction("symptom", "頭暈", "頭暈"),
                extraction("duration", "2天", "兩天"),
            ]},
        )
        provider.assert_awaited_once()
        patient = result["triage_case"]["patient_input"]
        self.assertEqual(patient["symptom"], "頭暈")
        self.assertEqual(patient["duration"], "2天")
        self.assertEqual(result["triage_case"]["semantic_extractions"][0]["extractor"], "ai")

    def test_natural_phrases_normalize_duration_severity_and_body_part(self):
        cases = [
            ("上禮拜開始", "duration", "1週", "上禮拜", "duration"),
            ("痛到晚上一直醒", "severity", "severe", "痛到晚上一直醒", "severity"),
            ("右腳腳背腫得很明顯", "body_part", "右腳腳背", "右腳腳背", "body_part"),
        ]
        for message, field, value, source, patient_field in cases:
            with self.subTest(field=field):
                result, provider = self.post_text(
                    message, {"semantic_extractions": [extraction(field, value, source)]},
                )
                provider.assert_awaited_once()
                self.assertEqual(result["triage_case"]["patient_input"][patient_field], value)

    def test_onset_and_accompanying_symptoms_require_literal_support(self):
        message = "上禮拜從樓梯踩空，右腳腳背腫，走路會痛"
        result, _ = self.post_text(message, {"semantic_extractions": [
            extraction("onset", "從樓梯踩空", "上禮拜從樓梯踩空"),
            extraction("accompanying_symptoms", ["右腳腳背腫", "走路會痛"], message),
        ]})
        patient = result["triage_case"]["patient_input"]
        self.assertEqual(patient["onset"], "從樓梯踩空")
        self.assertEqual(patient["accompanying_symptoms"], ["右腳腳背腫", "走路會痛"])

    def test_missing_or_invalid_medical_assertion_fails_closed(self):
        for assertion in (None, "yes"):
            with self.subTest(assertion=assertion):
                item = extraction("symptom", "胸痛", "有胸痛")
                if assertion is None:
                    item.pop("assertion")
                else:
                    item["assertion"] = assertion
                result, _ = self.post_text("我有胸痛", {"semantic_extractions": [item]})
                self.assertEqual(result["triage_case"]["patient_input"]["symptom"], "")
                self.assertEqual(result["triage_case"]["semantic_extractions"], [])

    def test_absent_and_uncertain_medical_assertions_do_not_fill_positive_patient_state(self):
        cases = (
            ("我沒有胸痛", "沒有胸痛", "absent", "available"),
            ("我不知道這算不算胸痛", "不知道這算不算胸痛", "uncertain", "ambiguous"),
        )
        for message, source, assertion, status in cases:
            with self.subTest(assertion=assertion):
                item = extraction(
                    "accompanying_symptoms", ["胸痛"], source, assertion=assertion,
                )
                item["semantic_status"] = status
                result, _ = self.post_text(message, {"semantic_extractions": [item]})
                case = result["triage_case"]
                self.assertEqual(case["patient_input"]["accompanying_symptoms"], [])
                self.assertEqual(case["semantic_extractions"][0]["assertion"], assertion)

    def test_malicious_medical_extraction_metadata_is_rejected(self):
        base = extraction("symptom", "胸痛", "有胸痛")
        invalid_items = []
        for confidence in (float("nan"), float("inf"), True, -0.1, 1.1):
            item = dict(base, confidence=confidence)
            invalid_items.append(item)
        invalid_items.extend([
            dict(base, field="confirmed"),
            dict(base, source_text="使用者沒說過"),
            dict(base, normalized_value=["胸痛"]),
        ])

        for item in invalid_items:
            with self.subTest(item=item):
                result, _ = self.post_text("我有胸痛", {"semantic_extractions": [item]})
                self.assertEqual(result["triage_case"]["patient_input"]["symptom"], "")
                self.assertEqual(result["triage_case"]["semantic_extractions"], [])

    def test_ungrounded_disallowed_and_invalid_values_are_rejected(self):
        result, _ = self.post_text(
            "我頭暈",
            {"semantic_extractions": [
                extraction("duration", "3個月", "三個月"),
                extraction("confirmed", True, "頭暈"),
                extraction("severity", "extreme", "頭暈"),
            ]},
        )
        patient = result["triage_case"]["patient_input"]
        self.assertIsNone(patient["duration"])
        self.assertIsNone(patient["severity"])
        self.assertFalse(result["conversation_state"]["confirmed"])
        self.assertEqual(result["triage_case"]["semantic_extractions"], [])

    def test_low_confidence_does_not_fill_or_complete(self):
        result, _ = self.post_text(
            "我頭暈兩天",
            {"semantic_extractions": [extraction("duration", "2天", "兩天", 0.2)]},
        )
        self.assertIsNone(result["triage_case"]["patient_input"]["duration"])
        self.assertTrue(result["needMoreInfo"])
        self.assertFalse(result["conversation_state"]["is_complete"])

    def test_provider_failure_timeout_and_malformed_json_do_not_fill_values(self):
        for provider_result in (RuntimeError("quota"), TimeoutError(), "{bad json"):
            with self.subTest(provider_result=repr(provider_result)):
                result, provider = self.post_text("我頭暈兩天", provider_result)
                provider.assert_awaited_once()
                patient = result["triage_case"]["patient_input"]
                self.assertEqual(patient["symptom"], "")
                self.assertIsNone(patient["duration"])
                self.assertTrue(result["needMoreInfo"])

    def test_malformed_ai_triage_is_ignored(self):
        result, _ = self.post_text(
            "我頭暈", {"semantic_extractions": [], "triage": {"urgency_score": "invalid"}},
        )
        self.assertTrue(result["needMoreInfo"])
        self.assertEqual(result["conversation_state"]["stage"], "collecting")

    def test_prior_turn_quote_does_not_ground_current_extraction(self):
        case = TriageCase(case_id="phase1-prior-source")
        case.history_records.append(Message(role="user", content="三個月"))
        result, _ = self.post_text(
            "我頭暈", {"semantic_extractions": [extraction("duration", "3個月", "三個月")]},
            triage_case=case.model_dump(mode="json"),
        )
        self.assertIsNone(result["triage_case"]["patient_input"]["duration"])

    def test_low_confidence_remains_missing_after_attempt_limit(self):
        case = TriageCase(case_id="phase1-low-confidence-attempts")
        case.conversation_state.question_attempts["duration"] = 2
        result, _ = self.post_text(
            "我頭暈兩天",
            {"semantic_extractions": [extraction("duration", "2天", "兩天", 0.2)]},
            triage_case=case.model_dump(mode="json"),
        )
        self.assertNotIn("duration", result["conversation_state"]["consumed_fields"])
        self.assertTrue(result["needMoreInfo"])

    def test_symptom_revision_clears_stale_dependent_values(self):
        case = TriageCase(case_id="phase1-symptom-revision")
        case.patient_input.symptom = "膝蓋痛"
        case.patient_input.body_part = "膝"
        case.patient_input.duration = "3天"
        case.conversation_state.awaiting_confirmation = True
        save_case(case)
        result, _ = self.post_text(
            "我想把症狀改成頭暈",
            {"semantic_extractions": [extraction("symptom", "頭暈", "頭暈")]},
            triage_case=case.model_dump(mode="json"),
        )
        patient = result["triage_case"]["patient_input"]
        self.assertEqual(patient["symptom"], "頭暈")
        self.assertIsNone(patient["body_part"])
        self.assertIsNone(patient["duration"])

    def test_ai_workflow_fields_and_red_flags_cannot_take_effect(self):
        result, _ = self.post_text(
            "我頭暈",
            {
                "semantic_extractions": [
                    extraction("symptom", "頭暈", "頭暈"),
                    extraction("red_flags", ["突發胸痛"], "頭暈"),
                    extraction("stage", "confirmed", "頭暈"),
                ],
                "stage": "confirmed", "is_complete": True, "confirmed": True,
                "awaiting_confirmation": True, "recommendation_generated": True,
                "script_generated": True, "red_flags_checked": True,
                "triage": {"is_final": True, "need_more_info": False},
            },
        )
        case = result["triage_case"]
        self.assertEqual(case["patient_input"]["symptom"], "頭暈")
        self.assertFalse(case["patient_input"]["red_flags_checked"])
        self.assertFalse(case["recommendation_generated"])
        self.assertFalse(case["script_generated"])
        self.assertFalse(case["confirmed"])
        self.assertEqual(result["conversation_state"]["stage"], "collecting")
        self.assertFalse(result["conversation_state"]["is_complete"])

    def test_ai_normalized_symptom_cannot_indirectly_complete_red_flag_screen(self):
        result, _ = self.post_text(
            "我頭暈", {"semantic_extractions": [extraction("symptom", "胸痛", "頭暈")]},
        )
        patient = result["triage_case"]["patient_input"]
        self.assertEqual(patient["symptom"], "頭暈")
        self.assertNotEqual(patient["symptom"], "胸痛")
        self.assertEqual(
            result["triage_case"]["semantic_extractions"][0]["normalized_value"], "胸痛",
        )
        self.assertFalse(patient["red_flags_checked"])
        self.assertEqual(patient["red_flags"], [])
        self.assertTrue(result["needMoreInfo"])

    def test_ai_symptom_concept_keeps_grounded_patient_wording(self):
        wording = "心臟有時候突然砰砰跳很快"
        result, provider = self.post_text(
            wording, {"semantic_extractions": [extraction("symptom", "心悸", wording)]},
        )
        provider.assert_awaited_once()
        case = result["triage_case"]
        self.assertEqual(case["patient_input"]["symptom"], wording)
        self.assertEqual(case["semantic_extractions"][0]["source_text"], wording)
        self.assertEqual(case["semantic_extractions"][0]["normalized_value"], "心悸")

    def test_keyed_free_text_calls_ai_but_explicit_choice_remains_structured(self):
        settings = SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=True)
        provider = AsyncMock(return_value=json.dumps({
            "extractions": [extraction("duration", "1週", "上禮拜")],
        }, ensure_ascii=False))
        with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
            batch_extraction_service, "get_settings", return_value=settings,
        ), patch.object(batch_extraction_service, "complete_prompt", new=provider), patch.object(
            chat_route, "detect_department_result", new=AsyncMock(return_value=None),
        ):
            response = TestClient(app).post("/chat", json={
                "visit_type": "initial",
                "answers": [
                    {"key": "duration", "answer": "上禮拜開始"},
                    {"key": "preferred_sessions", "answer": "上午"},
                ],
            })
        self.assertEqual(response.status_code, 200)
        provider.assert_awaited_once()
        case = response.json()["triage_case"]
        self.assertEqual(case["patient_input"]["duration"], "1週")
        self.assertEqual(case["availability"]["preferred_sessions"], ["上午"])

    def test_keyed_free_text_failure_does_not_fill_duration(self):
        settings = SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=True)
        provider = AsyncMock(side_effect=TimeoutError())
        with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
            batch_extraction_service, "get_settings", return_value=settings,
        ), patch.object(batch_extraction_service, "complete_prompt", new=provider), patch.object(
            chat_route, "detect_department_result", new=AsyncMock(return_value=None),
        ):
            response = TestClient(app).post("/chat", json={
                "visit_type": "initial", "answers": [{"key": "duration", "answer": "上禮拜開始"}],
            })
        self.assertEqual(response.status_code, 200)
        provider.assert_awaited_once()
        self.assertIsNone(response.json()["triage_case"]["patient_input"]["duration"])

    def test_keyed_free_text_with_existing_value_still_calls_ai_without_overwrite(self):
        settings = SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=True)
        case = TriageCase(case_id="phase1-keyed-existing")
        case.patient_input.duration = "2天"
        provider = AsyncMock(return_value=json.dumps({
            "extractions": [extraction("duration", "1週", "上禮拜")],
        }, ensure_ascii=False))
        with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
            batch_extraction_service, "get_settings", return_value=settings,
        ), patch.object(batch_extraction_service, "complete_prompt", new=provider), patch.object(
            chat_route, "detect_department_result", new=AsyncMock(return_value=None),
        ):
            response = TestClient(app).post("/chat", json={
                "visit_type": "initial", "triage_case": case.model_dump(mode="json"),
                "answers": [{"key": "duration", "answer": "上禮拜開始"}],
            })
        self.assertEqual(response.status_code, 200)
        provider.assert_awaited_once()
        self.assertEqual(response.json()["triage_case"]["patient_input"]["duration"], "2天")

    def test_recommend_user_query_uses_same_ai_first_extraction(self):
        settings = SimpleNamespace(cerebras_api_key="test-key")
        case = TriageCase(case_id="phase1-recommend-query")
        provider = AsyncMock(return_value='{"semantic_extractions":[]}')
        with patch.object(recommend_route, "get_settings", return_value=settings), patch.object(
            recommend_route, "create_case", return_value=case,
        ), patch.object(rag_triage_adapter, "get_settings", return_value=settings), patch.object(
            rag_triage_adapter, "complete_prompt", new=provider,
        ):
            response = TestClient(app).post("/recommend", json={
                "visit_type": "initial", "userQuery": "我頭暈兩天", "confirmed": True,
            })
        self.assertEqual(response.status_code, 400)
        provider.assert_awaited_once()
        self.assertEqual(case.patient_input.symptom, "")
        self.assertIsNone(case.patient_input.duration)


if __name__ == "__main__":
    unittest.main()
