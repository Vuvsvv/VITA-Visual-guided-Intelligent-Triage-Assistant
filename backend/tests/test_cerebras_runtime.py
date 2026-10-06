from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.schemas import BatchAnswer, TriageCase, VisitType
from app.services import ai_service, batch_extraction_service, project_smart_department_adapter
from app.services.batch_extraction_service import extract_batch_answers
from app.services.project_smart_department_adapter import detect_department_with_project_smart_adapter


def _settings(*, key: str = "cerebras-test-key") -> SimpleNamespace:
    return SimpleNamespace(
        ai_provider="gemini",  # A stale environment value must not switch runtime.
        batch_extraction_provider="gemini",
        cerebras_api_key=key,
        cerebras_model="gpt-oss-120b",
        google_api_key="",
        llm_model="gemini-must-not-run",
        ai_timeout_seconds=1.0,
    )


class CerebrasRuntimeContractTest(unittest.IsolatedAsyncioTestCase):
    def test_cerebras_key_is_sufficient_without_google_key(self):
        self.assertTrue(ai_service.runtime_ai_available(_settings()))
        self.assertFalse(ai_service.runtime_ai_available(_settings(key="")))

    async def test_default_completion_cannot_be_switched_to_gemini_by_stale_config(self):
        cerebras = AsyncMock(return_value='{"ok":true}')
        gemini_client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=AsyncMock(
                        side_effect=AssertionError("Gemini runtime call is forbidden")
                    )
                )
            )
        )
        with patch.object(ai_service, "get_settings", return_value=_settings()), patch.object(
            ai_service, "initialize_ai", return_value=None
        ), patch.object(ai_service, "_complete_cerebras", new=cerebras), patch.object(
            ai_service, "_google_client", gemini_client
        ):
            result = await ai_service.complete_prompt("runtime")

        self.assertEqual(result, '{"ok":true}')
        cerebras.assert_awaited_once()
        gemini_client.aio.models.generate_content.assert_not_awaited()

    async def test_runtime_json_uses_cerebras_json_mode_and_fixed_model_contract(self):
        provider = AsyncMock(return_value='{"extractions":[]}')
        with patch.object(ai_service, "get_settings", return_value=_settings()), patch.object(
            ai_service, "complete_prompt", new=provider
        ):
            result = await ai_service.complete_runtime_json(
                "extract",
                purpose="semantic_extraction",
            )

        self.assertEqual(result, '{"extractions":[]}')
        provider.assert_awaited_once_with(
            "extract",
            provider="cerebras",
            json_mode=True,
        )
        self.assertEqual(ai_service.RUNTIME_AI_MODEL, "gpt-oss-120b")

    async def test_deterministic_semantic_answer_uses_zero_llm_calls(self):
        case = TriageCase(case_id="deterministic", visit_type=VisitType.INITIAL)
        provider = AsyncMock(side_effect=AssertionError("deterministic answer called LLM"))
        with patch.object(batch_extraction_service, "get_settings", return_value=_settings()), patch.object(
            batch_extraction_service, "complete_prompt", new=provider
        ):
            outcome = await extract_batch_answers(
                case,
                [BatchAnswer(key="preferred_days", answer="星期五上午")],
            )

        provider.assert_not_awaited()
        self.assertFalse(outcome.ai_attempted)
        self.assertEqual(case.availability.preferred_days, ["週五"])
        self.assertEqual(case.availability.preferred_sessions, ["上午"])

    async def test_unresolved_semantic_answer_uses_exactly_one_cerebras_call(self):
        case = TriageCase(case_id="semantic", visit_type=VisitType.INITIAL)
        provider = AsyncMock(
            return_value=json.dumps(
                {
                    "extractions": [
                        {
                            "field": "body_part",
                            "normalized_value": "肩膀",
                            "semantic_status": "available",
                            "confidence": 0.91,
                            "source_text": "不知道是不是肩膀附近",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        with patch.object(batch_extraction_service, "get_settings", return_value=_settings()), patch.object(
            batch_extraction_service, "complete_prompt", new=provider
        ):
            outcome = await extract_batch_answers(
                case,
                [BatchAnswer(key="body_part", answer="不知道是不是肩膀附近")],
            )

        provider.assert_awaited_once()
        self.assertTrue(outcome.ai_attempted)
        self.assertEqual(case.patient_input.body_part, "肩膀")

    async def test_invalid_semantic_field_is_rejected(self):
        case = TriageCase(case_id="invalid-field", visit_type=VisitType.INITIAL)
        provider = AsyncMock(
            return_value=json.dumps(
                {
                    "extractions": [
                        {
                            "field": "stage",
                            "normalized_value": "recommending",
                            "semantic_status": "available",
                            "confidence": 1.0,
                            "source_text": "不知道是不是肩膀附近",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        original_stage = case.conversation_state.stage
        with patch.object(batch_extraction_service, "get_settings", return_value=_settings()), patch.object(
            batch_extraction_service, "complete_prompt", new=provider
        ):
            outcome = await extract_batch_answers(
                case,
                [BatchAnswer(key="body_part", answer="不知道是不是肩膀附近")],
            )

        provider.assert_awaited_once()
        self.assertEqual(outcome.ai_fields, [])
        self.assertEqual(case.conversation_state.stage, original_stage)
        self.assertIsNone(case.patient_input.body_part)

    async def test_red_flags_never_call_cerebras(self):
        case = TriageCase(case_id="red-flags", visit_type=VisitType.INITIAL)
        provider = AsyncMock(side_effect=AssertionError("red flags called generative AI"))
        with patch.object(batch_extraction_service, "get_settings", return_value=_settings()), patch.object(
            batch_extraction_service, "complete_prompt", new=provider
        ):
            await extract_batch_answers(
                case,
                [BatchAnswer(key="red_flags", answer="有一點")],
            )

        provider.assert_not_awaited()
        self.assertFalse(case.patient_input.red_flags_checked)

    async def test_department_receives_candidates_and_accepts_exact_tuple(self):
        departments = [
            {"dept_id": 1333, "parent_dept": "五官科", "child_dept": "耳科"},
            {"dept_id": 2001, "parent_dept": "內科系", "child_dept": "一般內科"},
        ]
        provider = AsyncMock(
            return_value=json.dumps(
                {
                    "dept_id": 1333,
                    "confidence": 0.91,
                    "reason": ["耳部症狀"],
                },
                ensure_ascii=False,
            )
        )
        case = TriageCase(case_id="department")
        case.patient_input.symptom = "耳朵疼痛"
        with patch.object(
            project_smart_department_adapter, "get_settings", return_value=_settings()
        ), patch.object(project_smart_department_adapter, "complete_prompt", new=provider):
            result = await detect_department_with_project_smart_adapter(case, departments)

        provider.assert_awaited_once()
        prompt = provider.await_args.args[0]
        self.assertIn('"dept_id": 1333', prompt)
        self.assertIn('"dept_id": 2001', prompt)
        self.assertIn('"parent_dept": "五官科"', prompt)
        response_contract = prompt.split("請只輸出以下 JSON", 1)[1]
        self.assertNotIn("parent_dept", response_contract)
        self.assertNotIn("child_dept", response_contract)
        self.assertEqual(result.dept_id, 1333)
        self.assertEqual(result.parentDept, "五官科")
        self.assertEqual(result.childDept, "耳科")

    async def test_department_rejects_invented_id_without_family_fallback(self):
        departments = [
            {"dept_id": 1333, "parent_dept": "五官科", "child_dept": "耳科"},
        ]
        provider = AsyncMock(
            return_value='{"dept_id":9999,"confidence":1,"reason":"test"}'
        )
        case = TriageCase(case_id="invalid-department")
        case.patient_input.symptom = "耳朵疼痛"
        with patch.object(
            project_smart_department_adapter, "get_settings", return_value=_settings()
        ), patch.object(project_smart_department_adapter, "complete_prompt", new=provider):
            result = await detect_department_with_project_smart_adapter(case, departments)

        self.assertIsNone(result)

    async def test_complete_simple_triage_uses_zero_semantic_and_at_most_one_department_call(self):
        case = TriageCase(case_id="simple-triage", visit_type=VisitType.INITIAL)
        semantic = AsyncMock(side_effect=AssertionError("simple triage called semantic LLM"))
        department = AsyncMock(
            return_value=json.dumps(
                {
                    "dept_id": 2001,
                    "confidence": 0.9,
                    "reason": ["一般頭痛評估"],
                },
                ensure_ascii=False,
            )
        )
        answers = [
            BatchAnswer(key="symptom", answer="我頭痛"),
            BatchAnswer(key="red_flags", answer="都沒有"),
            BatchAnswer(key="body_part", answer="頭部"),
            BatchAnswer(key="duration", answer="三天"),
            BatchAnswer(key="severity", answer="中等"),
            BatchAnswer(key="preferred_days", answer="星期五上午"),
        ]
        with patch.object(batch_extraction_service, "get_settings", return_value=_settings()), patch.object(
            batch_extraction_service, "complete_prompt", new=semantic
        ):
            await extract_batch_answers(case, answers)
        with patch.object(
            project_smart_department_adapter, "get_settings", return_value=_settings()
        ), patch.object(project_smart_department_adapter, "complete_prompt", new=department):
            result = await detect_department_with_project_smart_adapter(
                case,
                [{"dept_id": 2001, "parent_dept": "內科系", "child_dept": "一般內科"}],
            )

        semantic.assert_not_awaited()
        department.assert_awaited_once()
        self.assertEqual(result.dept_id, 2001)

    async def test_semantic_source_grounding_failure_is_rejected(self):
        case = TriageCase(case_id="ungrounded", visit_type=VisitType.INITIAL)
        provider = AsyncMock(
            return_value=json.dumps(
                {
                    "extractions": [
                        {
                            "field": "body_part",
                            "normalized_value": "右肩",
                            "semantic_status": "available",
                            "confidence": 0.91,
                            "source_text": "使用者沒有說過的文字",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        with patch.object(batch_extraction_service, "get_settings", return_value=_settings()), patch.object(
            batch_extraction_service, "complete_prompt", new=provider
        ):
            outcome = await extract_batch_answers(
                case,
                [BatchAnswer(key="body_part", answer="不知道是不是肩膀附近")],
            )

        self.assertEqual(outcome.ai_fields, [])
        self.assertIsNone(case.patient_input.body_part)

    async def test_wrong_semantic_envelope_is_schema_error_not_silent_empty_success(self):
        case = TriageCase(case_id="wrong-envelope", visit_type=VisitType.INITIAL)
        provider = AsyncMock(return_value='{"semantic_extractions":[]}')
        with patch.object(batch_extraction_service, "get_settings", return_value=_settings()), patch.object(
            batch_extraction_service, "complete_prompt", new=provider
        ):
            outcome = await extract_batch_answers(
                case,
                [BatchAnswer(key="body_part", answer="不知道是不是肩膀附近")],
            )

        self.assertEqual(outcome.ai_fields, [])
        self.assertEqual(outcome.fallback_reason, "ValueError")


if __name__ == "__main__":
    unittest.main()
