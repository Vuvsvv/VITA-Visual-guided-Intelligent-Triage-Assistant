from __future__ import annotations

import json
import unittest
from unittest.mock import AsyncMock, patch

from app.schemas import BatchAnswer, SemanticExtraction, TriageCase, VisitType
from app.services import batch_extraction_service
from app.services.batch_extraction_service import extract_batch_answers
from app.services.field_acceptance import has_symptom_semantics
from app.services.rule_engine import apply_semantic_extractions, missing_checklist_fields


class SemanticAiValidationTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _provider(extraction: dict) -> AsyncMock:
        return AsyncMock(
            return_value=json.dumps({"extractions": [extraction]}, ensure_ascii=False)
        )

    async def _extract(self, field: str, answer: str, extraction: dict):
        case = TriageCase(case_id=f"semantic-ai-{field}", visit_type=VisitType.INITIAL)
        provider = self._provider(extraction)
        with patch.object(
            batch_extraction_service,
            "runtime_ai_available",
            return_value=True,
        ), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            outcome = await extract_batch_answers(
                case,
                [BatchAnswer(key=field, answer=answer)],
            )
        return case, outcome, provider

    async def test_ai_body_part_can_bypass_deterministic_dictionary_for_clavicle(self):
        answer = "鎖骨附近怪怪的"
        case, outcome, provider = await self._extract(
            "body_part",
            answer,
            {
                "field": "body_part",
                "normalized_value": "鎖骨",
                "semantic_status": "available",
                "confidence": 0.9,
                "source_text": "鎖骨附近",
            },
        )

        provider.assert_awaited_once()
        self.assertTrue(outcome.ai_attempted)
        self.assertEqual(case.patient_input.body_part, "鎖骨")
        self.assertNotIn("body_part", missing_checklist_fields(case))

    async def test_ai_body_part_can_replace_broad_rule_match_with_grounded_wrist(self):
        answer = "手腕那邊一直怪怪的"
        case, outcome, provider = await self._extract(
            "body_part",
            answer,
            {
                "field": "body_part",
                "normalized_value": "手腕",
                "semantic_status": "available",
                "confidence": 0.88,
                "source_text": "手腕那邊",
            },
        )

        provider.assert_awaited_once()
        self.assertTrue(outcome.ai_attempted)
        self.assertEqual(case.patient_input.body_part, "手腕")

    async def test_lossy_secondary_wrist_is_refined_in_the_same_batch(self):
        answer = "我最近左手手腕會酸痛還有灼熱感，尤其是晚上的時候"
        case, outcome, provider = await self._extract(
            "symptom",
            answer,
            {
                "field": "body_part",
                "normalized_value": "手腕",
                "semantic_status": "available",
                "confidence": 0.9,
                "source_text": "左手手腕",
            },
        )

        provider.assert_awaited_once()
        self.assertEqual(case.patient_input.symptom, answer)
        self.assertEqual(case.patient_input.body_part, "手腕")
        self.assertEqual(case.availability.preferred_sessions, [])
        self.assertNotEqual(case.patient_input.body_part, "手")
        self.assertEqual(outcome.ai_fields, ["body_part"])

    async def test_lossy_secondary_ankle_is_refined_in_the_same_batch(self):
        answer = "右腳腳踝腫痛"
        case, _, provider = await self._extract(
            "symptom",
            answer,
            {
                "field": "body_part",
                "normalized_value": "腳踝",
                "semantic_status": "available",
                "confidence": 0.9,
                "source_text": "右腳腳踝",
            },
        )

        provider.assert_awaited_once()
        self.assertEqual(case.patient_input.body_part, "腳踝")

    async def test_compound_session_answers_use_one_semantic_request(self):
        examples = (
            ("我不要上午和晚上", ["下午"]),
            ("上午跟晚上都不方便", ["下午"]),
            ("下午可以，但是晚上不要", ["下午"]),
        )
        for answer, expected in examples:
            with self.subTest(answer=answer):
                case, _, provider = await self._extract(
                    "preferred_sessions",
                    answer,
                    {
                        "field": "preferred_sessions",
                        "normalized_value": expected,
                        "semantic_status": "partial",
                        "confidence": 0.9,
                        "source_text": answer,
                    },
                )

                provider.assert_awaited_once()
                self.assertEqual(case.availability.preferred_sessions, expected)

    async def test_compound_day_answers_use_one_semantic_request(self):
        examples = (
            (
                "我下週三有事其他時間應該都可以",
                ["週一", "週二", "週四", "週五", "週六", "週日"],
            ),
            (
                "我下周三有事其他時間應該都可以",
                ["週一", "週二", "週四", "週五", "週六", "週日"],
            ),
            (
                "週三跟週五不行，其他都可以",
                ["週一", "週二", "週四", "週六", "週日"],
            ),
        )
        for answer, expected in examples:
            with self.subTest(answer=answer):
                with self.assertLogs(level="INFO") as logs:
                    case, _, provider = await self._extract(
                        "preferred_days",
                        answer,
                        {
                            "field": "preferred_days",
                            "normalized_value": expected,
                            "semantic_status": "partial",
                            "confidence": 0.9,
                            "source_text": answer,
                            "needs_clarification": False,
                        },
                    )

                provider.assert_awaited_once()
                self.assertEqual(case.availability.preferred_days, expected)
                self.assertNotIn("preferred_days", missing_checklist_fields(case))
                output = "\n".join(logs.output)
                self.assertIn("reason=compound_day_semantics", output)
                self.assertIn("field=preferred_days accepted=true", output)

    async def test_simple_day_values_keep_zero_ai_fast_path(self):
        for answer, expected in (
            ("週三", ["週三"]),
            ("周三", ["週三"]),
            ("週三週四", ["週三", "週四"]),
            ("周三周四", ["週三", "週四"]),
        ):
            with self.subTest(answer=answer):
                case = TriageCase(case_id=f"semantic-ai-simple-day-{answer}")
                provider = AsyncMock(side_effect=AssertionError("simple days must not call AI"))
                with patch.object(
                    batch_extraction_service,
                    "runtime_ai_available",
                    return_value=True,
                ), patch.object(
                    batch_extraction_service,
                    "complete_prompt",
                    new=provider,
                ):
                    outcome = await extract_batch_answers(
                        case,
                        [BatchAnswer(key="preferred_days", answer=answer)],
                    )

                provider.assert_not_awaited()
                self.assertFalse(outcome.ai_attempted)
                self.assertEqual(case.availability.preferred_days, expected)

    async def test_simple_session_values_keep_zero_ai_fast_path(self):
        for answer, expected in (("下午", ["下午"]), ("都可以", ["上午", "下午", "夜間"])):
            with self.subTest(answer=answer):
                case = TriageCase(case_id=f"semantic-ai-simple-session-{answer}")
                provider = AsyncMock(side_effect=AssertionError("simple session must not call AI"))
                with patch.object(
                    batch_extraction_service,
                    "runtime_ai_available",
                    return_value=True,
                ), patch.object(
                    batch_extraction_service,
                    "complete_prompt",
                    new=provider,
                ):
                    outcome = await extract_batch_answers(
                        case,
                        [BatchAnswer(key="preferred_sessions", answer=answer)],
                    )

                provider.assert_not_awaited()
                self.assertFalse(outcome.ai_attempted)
                self.assertEqual(case.availability.preferred_sessions, expected)

    async def test_ai_symptom_is_not_limited_to_symptom_terms(self):
        answer = "一直有燒灼的感覺"
        self.assertFalse(has_symptom_semantics(answer))
        case, _, provider = await self._extract(
            "symptom",
            answer,
            {
                "field": "symptom",
                "normalized_value": "灼熱感",
                "semantic_status": "available",
                "assertion": "present",
                "confidence": 0.87,
                "source_text": "燒灼的感覺",
            },
        )

        provider.assert_awaited_once()
        self.assertEqual(case.patient_input.symptom, "燒灼的感覺")
        self.assertEqual(case.semantic_extractions[0].normalized_value, "灼熱感")

    async def test_indirect_weak_severity_answers_use_ai_and_land_as_mild(self):
        for answer in ("不影響吧", "應該還好吧"):
            with self.subTest(answer=answer):
                case, _, provider = await self._extract(
                    "severity",
                    answer,
                    {
                        "field": "severity",
                        "normalized_value": "mild",
                        "semantic_status": "available",
                        "confidence": 0.82,
                        "source_text": answer,
                        "needs_clarification": False,
                    },
                )

                provider.assert_awaited_once()
                self.assertEqual(case.patient_input.severity, "mild")
                self.assertNotIn("severity", missing_checklist_fields(case))

    async def test_weak_duration_with_clear_core_keeps_the_fast_path(self):
        case = TriageCase(case_id="semantic-ai-duration", visit_type=VisitType.INITIAL)
        provider = AsyncMock(side_effect=AssertionError("clear duration must not call Cerebras"))
        with patch.object(
            batch_extraction_service,
            "runtime_ai_available",
            return_value=True,
        ), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            outcome = await extract_batch_answers(
                case,
                [BatchAnswer(key="duration", answer="應該半年左右吧")],
            )

        provider.assert_not_awaited()
        self.assertFalse(outcome.ai_attempted)
        self.assertEqual(case.patient_input.duration, "6個月")

    async def test_unknown_ai_result_does_not_fill_body_part(self):
        answer = "我真的不知道是哪裡"
        case, _, provider = await self._extract(
            "body_part",
            answer,
            {
                "field": "body_part",
                "normalized_value": None,
                "semantic_status": "unknown",
                "confidence": 0.2,
                "source_text": answer,
                "needs_clarification": True,
            },
        )

        provider.assert_awaited_once()
        self.assertIsNone(case.patient_input.body_part)
        self.assertIn("body_part", missing_checklist_fields(case, apply_attempt_fallback=False))

    async def test_ai_source_text_must_be_contiguous_in_the_same_keyed_answer(self):
        answer = "我不知道是哪裡"
        with self.assertLogs(batch_extraction_service.logger, level="INFO") as logs:
            case, outcome, _ = await self._extract(
                "body_part",
                answer,
                {
                    "field": "body_part",
                    "normalized_value": "頭",
                    "semantic_status": "available",
                    "confidence": 0.9,
                    "source_text": "頭",
                },
            )

        self.assertIsNone(case.patient_input.body_part)
        self.assertEqual(outcome.ai_fields, [])
        self.assertIn("reject_reason=source_not_grounded", "\n".join(logs.output))

    async def test_grounded_source_cannot_support_unrelated_body_part(self):
        answer = "鎖骨附近怪怪的"
        with self.assertLogs(batch_extraction_service.logger, level="INFO") as logs:
            case, outcome, _ = await self._extract(
                "body_part",
                answer,
                {
                    "field": "body_part",
                    "normalized_value": "膝蓋",
                    "semantic_status": "available",
                    "confidence": 0.9,
                    "source_text": "鎖骨附近",
                },
            )

        self.assertIsNone(case.patient_input.body_part)
        self.assertEqual(outcome.ai_fields, [])
        self.assertIn(
            "reject_reason=normalized_value_not_supported_by_source",
            "\n".join(logs.output),
        )

    async def test_semantic_ai_logs_request_parsed_value_and_final_decision(self):
        answer = "鎖骨附近怪怪的"
        with self.assertLogs(level="INFO") as logs:
            case, _, _ = await self._extract(
                "body_part",
                answer,
                {
                    "field": "body_part",
                    "normalized_value": "鎖骨",
                    "semantic_status": "available",
                    "confidence": 0.9,
                    "source_text": "鎖骨附近",
                },
            )

        output = "\n".join(logs.output)
        self.assertEqual(case.patient_input.body_part, "鎖骨")
        self.assertIn("[SEMANTIC_AI_REQUEST]", output)
        self.assertIn("[SEMANTIC_AI_PARSED]", output)
        self.assertIn("[SEMANTIC_AI_DECISION]", output)
        self.assertIn("accepted=true", output)

    async def test_fast_path_logs_rejected_secondary_and_clear_value(self):
        with self.assertLogs(batch_extraction_service.logger, level="INFO") as logs:
            await self._extract(
                "symptom",
                "左手手腕酸痛",
                {
                    "field": "body_part",
                    "normalized_value": "手腕",
                    "semantic_status": "available",
                    "confidence": 0.9,
                    "source_text": "左手手腕",
                },
            )

        output = "\n".join(logs.output)
        self.assertIn("[SEMANTIC_FAST_PATH]", output)
        self.assertIn("field=body_part", output)
        self.assertIn("accepted=false", output)
        self.assertIn("reason=coarse_secondary_extraction", output)

    def test_ai_body_part_must_be_lexically_grounded_and_laterality_safe(self):
        examples = (
            ("右下腹", "左下腹", False, "laterality_conflict"),
            ("左手腕", "手腕", True, None),
            ("左手腕", "右手腕", False, "laterality_conflict"),
            ("左手腕", "左手臂", False, "normalized_value_not_supported_by_source"),
            ("右腳踝", "右腳掌", False, "normalized_value_not_supported_by_source"),
            ("手腕", "左手腕", False, "laterality_not_grounded"),
            ("鎖骨附近", "鎖骨", True, None),
            ("腸胃", "腹", True, None),
            ("左邊手腕", "左手腕", True, None),
        )
        for source, normalized, accepted, reason in examples:
            with self.subTest(source=source, normalized=normalized):
                with self.assertLogs(batch_extraction_service.logger, level="INFO") as logs:
                    parsed = batch_extraction_service._parse_ai_extractions(
                        json.dumps(
                            {
                                "extractions": [
                                    {
                                        "field": "body_part",
                                        "normalized_value": normalized,
                                        "semantic_status": "available",
                                        "confidence": 0.9,
                                        "source_text": source,
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        ),
                        {"body_part": source},
                        ["body_part"],
                    )

                self.assertEqual(bool(parsed), accepted)
                if reason:
                    self.assertIn(f"reject_reason={reason}", "\n".join(logs.output))

    def test_ai_cannot_overwrite_an_existing_confirmed_body_part(self):
        case = TriageCase(case_id="semantic-ai-no-overwrite", visit_type=VisitType.INITIAL)
        case.patient_input.body_part = "右下腹"
        extraction = SemanticExtraction(
            field="body_part",
            normalized_value="右邊",
            semantic_status="available",
            confidence=0.8,
            source_text="右邊",
            extractor="ai_batch",
        )

        with self.assertLogs(level="INFO") as logs:
            apply_semantic_extractions(case, [extraction])

        self.assertEqual(case.patient_input.body_part, "右下腹")
        self.assertIn("reject_reason=duplicate_existing_value", "\n".join(logs.output))

    def test_prompt_contains_current_question_and_weak_uncertainty_contract(self):
        case = TriageCase(case_id="semantic-ai-prompt", visit_type=VisitType.INITIAL)
        prompt = batch_extraction_service._build_batch_prompt(
            case,
            {"severity": "不影響吧"},
            ["severity"],
        )

        self.assertIn("current_question_field", prompt)
        self.assertIn("請問症狀程度", prompt)
        self.assertIn("不影響吧", prompt)
        self.assertIn("弱語氣不等於無法回答", prompt)
        self.assertIn("ambiguous 只用於互相衝突", prompt)
        self.assertIn("preferred_days 與 preferred_sessions 是封閉集合", prompt)
        self.assertIn("我不喜歡早上跟晚上", prompt)
        self.assertIn("其他都可以／其餘都可以", prompt)
        self.assertIn("partial 是可直接寫入的有效答案", prompt)
        self.assertIn("assertion=present/absent/uncertain", prompt)
        self.assertIn("symptom 與 accompanying_symptoms", prompt)
        self.assertIn("Backend 不會替你解析否定 scope", prompt)

    def test_batch_ai_symptom_requires_valid_assertion(self):
        answer = "我有胸痛"
        for assertion in (None, "yes"):
            with self.subTest(assertion=assertion):
                item = {
                    "field": "symptom",
                    "normalized_value": "胸痛",
                    "semantic_status": "available",
                    "confidence": 0.95,
                    "source_text": "有胸痛",
                }
                if assertion is not None:
                    item["assertion"] = assertion
                try:
                    parsed = batch_extraction_service._parse_ai_extractions(
                        json.dumps({"extractions": [item]}, ensure_ascii=False),
                        {"symptom": answer},
                        ["symptom"],
                    )
                except ValueError:
                    parsed = []
                self.assertEqual(parsed, [])

    def test_batch_ai_accompanying_symptoms_requires_valid_assertion(self):
        answer = "還有尿尿很痛"
        for assertion in (None, "yes"):
            with self.subTest(assertion=assertion):
                item = {
                    "field": "accompanying_symptoms",
                    "normalized_value": ["小便疼痛"],
                    "semantic_status": "available",
                    "confidence": 0.95,
                    "source_text": "尿尿很痛",
                }
                if assertion is not None:
                    item["assertion"] = assertion
                try:
                    parsed = batch_extraction_service._parse_ai_extractions(
                        json.dumps({"extractions": [item]}, ensure_ascii=False),
                        {"accompanying_symptoms": answer},
                        ["accompanying_symptoms"],
                    )
                except ValueError:
                    parsed = []
                self.assertEqual(parsed, [])


if __name__ == "__main__":
    unittest.main()
