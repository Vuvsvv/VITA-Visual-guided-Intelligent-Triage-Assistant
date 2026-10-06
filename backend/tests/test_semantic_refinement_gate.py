from __future__ import annotations

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import TriageCase
from app.services import ai_reply_generator, ai_service, rag_triage_adapter
from app.services.case_store import save_case


chat_route = importlib.import_module("app.routes.chat")


class SemanticRefinementGateTest(unittest.IsolatedAsyncioTestCase):
    def _post_with_mocked_ai(self, payload: dict, *, ai_available: bool = False):
        payload = dict(payload)
        if payload.get("triage_case"):
            stored = TriageCase.model_validate(payload["triage_case"])
            save_case(stored)
            payload["case_id"] = stored.case_id
        semantic = AsyncMock(return_value=None)

        async def deterministic_reply(**kwargs) -> str:
            return kwargs["fallback_reply"]

        contextual = AsyncMock(side_effect=deterministic_reply)
        with patch.object(chat_route, "refine_case_with_ai", new=semantic), patch.object(
            chat_route,
            "generate_triage_reply",
            new=contextual,
        ), patch.object(
            chat_route,
            "get_settings",
            return_value=SimpleNamespace(
                cerebras_api_key="test-key" if ai_available else "",
                batch_triage_enabled=False,
            ),
        ), patch.object(
            chat_route,
            "detect_department_result",
            new=AsyncMock(return_value=None),
        ):
            response = TestClient(app).post("/chat", json=payload)
        return response, semantic, contextual

    @staticmethod
    def _case_waiting_for(field: str) -> TriageCase:
        case = TriageCase(case_id=f"semantic-gate-{field}")
        case.patient_input.symptom = "頭暈"
        case.patient_input.body_part = "頭"
        case.patient_input.duration = "2天"
        if field != "red_flags":
            case.patient_input.red_flags_checked = True
        if field in {"preferred_days", "preferred_sessions"}:
            case.patient_input.severity = "中度"
        if field == "preferred_sessions":
            case.availability.preferred_days = ["週二"]
        case.conversation_state.last_question_key = field
        return case

    async def test_clear_symptom_and_duration_skip_semantic_provider(self):
        response, semantic, contextual = self._post_with_mocked_ai(
            {"message": "我頭暈兩天"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["triage_case"]["patient_input"]["duration"], "2天")
        semantic.assert_not_awaited()
        contextual.assert_awaited_once()

    async def test_red_flag_denial_fails_closed_without_ai(self):
        case = self._case_waiting_for("red_flags")
        response, semantic, _ = self._post_with_mocked_ai(
            {
                "triage_case": case.model_dump(mode="json"),
                "message": "沒有胸痛、呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛",
            },
        )

        self.assertEqual(response.status_code, 200)
        patient = response.json()["triage_case"]["patient_input"]
        self.assertFalse(patient["red_flags_checked"])
        self.assertEqual(patient["red_flags_status"], "not_checked")
        semantic.assert_not_awaited()

    async def test_ambiguous_red_flag_answer_fails_closed_without_ai(self):
        case = self._case_waiting_for("red_flags")
        response, semantic, _ = self._post_with_mocked_ai(
            {
                "triage_case": case.model_dump(mode="json"),
                "message": "好像有",
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(
            response.json()["triage_case"]["patient_input"]["red_flags_status"],
            "not_checked",
        )
        semantic.assert_not_awaited()

    async def test_clear_severity_skips_semantic_provider(self):
        case = self._case_waiting_for("severity")
        response, semantic, _ = self._post_with_mocked_ai(
            {
                "triage_case": case.model_dump(mode="json"),
                "message": "中等，會有點影響生活",
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["triage_case"]["patient_input"]["severity"], "中度")
        semantic.assert_not_awaited()

    async def test_clear_day_skips_semantic_provider(self):
        case = self._case_waiting_for("preferred_days")
        response, semantic, _ = self._post_with_mocked_ai(
            {
                "triage_case": case.model_dump(mode="json"),
                "message": "下周二",
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["triage_case"]["availability"]["preferred_days"], ["週二"])
        semantic.assert_not_awaited()

    async def test_clear_session_skips_semantic_provider(self):
        case = self._case_waiting_for("preferred_sessions")
        response, semantic, _ = self._post_with_mocked_ai(
            {
                "triage_case": case.model_dump(mode="json"),
                "message": "上午",
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["triage_case"]["availability"]["preferred_sessions"], ["上午"])
        semantic.assert_not_awaited()

    async def test_unresolved_requested_field_is_not_blocked_by_plausibility_rule(self):
        case = self._case_waiting_for("duration")
        case.patient_input.duration = None

        response, semantic, _ = self._post_with_mocked_ai(
            {
                "triage_case": case.model_dump(mode="json"),
                "message": "我想直接掛皮膚科",
            },
            ai_available=True,
        )

        self.assertEqual(response.status_code, 200)
        semantic.assert_awaited_once()
        self.assertIsNone(response.json()["triage_case"]["patient_input"]["duration"])

    async def test_compound_severity_calls_semantic_provider(self):
        case = self._case_waiting_for("severity")

        response, semantic, _ = self._post_with_mocked_ai(
            {
                "triage_case": case.model_dump(mode="json"),
                "message": "白天還好，可是晚上常常痛醒",
            },
            ai_available=True,
        )

        self.assertEqual(response.status_code, 200)
        semantic.assert_awaited_once()

    async def test_weak_tone_half_year_duration_skips_semantic_provider(self):
        case = self._case_waiting_for("duration")
        case.patient_input.duration = None

        response, semantic, _ = self._post_with_mocked_ai(
            {
                "triage_case": case.model_dump(mode="json"),
                "message": "大概半年了吧",
            }
        )

        self.assertEqual(response.status_code, 200)
        semantic.assert_not_awaited()
        self.assertEqual(response.json()["triage_case"]["patient_input"]["duration"], "6個月")

    async def test_ambiguous_natural_language_calls_semantic_provider(self):
        contextual = (
            "了解你有暈眩不適。請問是否有胸痛、呼吸困難、意識不清、大量出血、"
            "半邊無力或劇烈頭痛？"
        )

        class FakeCompletions:
            def __init__(self):
                self.call_count = 0

            async def create(self, **_kwargs):
                self.call_count += 1
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        content='{"patient_input":{},"semantic_extractions":[]}'
                    ))]
                )

        models = FakeCompletions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=models))
        summaries = []
        settings = SimpleNamespace(
            ai_reply_generation_enabled=True,
            ai_reply_timeout_seconds=20,
            ai_timeout_seconds=8,
            cerebras_api_key="test-key",
            cerebras_model="gpt-oss-120b",
            batch_triage_enabled=False,
        )
        original_finish = chat_route.finish_chat_perf

        def capture_finish(perf, token):
            summaries.append(perf.summary())
            original_finish(perf, token)

        with patch.object(
            rag_triage_adapter,
            "get_settings",
            return_value=settings,
        ), patch.object(
            chat_route,
            "get_settings",
            return_value=settings,
        ), patch.object(
            ai_reply_generator,
            "get_settings",
            return_value=settings,
        ), patch.object(
            ai_service,
            "get_settings",
            return_value=settings,
        ), patch.object(
            ai_service,
            "initialize_ai",
            return_value=None,
        ), patch.object(
            ai_service,
            "_cerebras_client",
            fake_client,
        ), patch.object(
            chat_route,
            "finish_chat_perf",
            side_effect=capture_finish,
        ):
            response = TestClient(app).post(
                "/chat",
                json={"message": "最近不知道為什麼老是暈暈的，差不多從前幾天開始吧"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], response.json()["next_question"])
        self.assertEqual(models.call_count, 2)
        self.assertEqual(len(summaries), 1)
        self.assertTrue(summaries[0]["semantic_ai_called"])
        self.assertFalse(summaries[0]["ai_reply_ai_called"])
        self.assertEqual(summaries[0]["ai_call_count"], 2)
        self.assertEqual(summaries[0]["gemini_call_count"], 0)

    async def test_semantic_failure_keeps_deterministic_chat_flow(self):
        provider = AsyncMock(side_effect=RuntimeError("simulated provider failure"))

        async def deterministic_reply(**kwargs) -> str:
            return kwargs["fallback_reply"]

        with patch.object(
            rag_triage_adapter,
            "get_settings",
            return_value=SimpleNamespace(cerebras_api_key="test-key"),
        ), patch.object(
            chat_route,
            "get_settings",
            return_value=SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=False),
        ), patch.object(
            rag_triage_adapter,
            "complete_prompt",
            new=provider,
        ), patch.object(
            chat_route,
            "generate_triage_reply",
            new=AsyncMock(side_effect=deterministic_reply),
        ):
            response = TestClient(app).post(
                "/chat",
                json={"message": "有時候會不舒服，但也說不上來嚴不嚴重"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["needMoreInfo"])
        self.assertTrue(response.json()["next_question"])
        provider.assert_awaited_once()

    async def test_controlled_question_variant_keeps_reply_ai_disabled(self):
        contextual = (
            "了解你頭暈兩天。請問是否有胸痛、呼吸困難、意識不清、大量出血、"
            "半邊無力或劇烈頭痛？"
        )
        class FakeModels:
            def __init__(self):
                self.call_count = 0

            async def generate_content(self, **_kwargs):
                self.call_count += 1
                return SimpleNamespace(text=f'{{"reply":"{contextual}"}}')

        models = FakeModels()
        fake_client = SimpleNamespace(aio=SimpleNamespace(models=models))
        semantic = AsyncMock(return_value=None)
        summaries = []
        settings = SimpleNamespace(
            ai_reply_generation_enabled=True,
            ai_reply_timeout_seconds=20,
            ai_timeout_seconds=8,
            cerebras_api_key="test-key",
            cerebras_model="gpt-oss-120b",
        )
        original_finish = chat_route.finish_chat_perf

        def capture_finish(perf, token):
            summaries.append(perf.summary())
            original_finish(perf, token)

        with patch.object(chat_route, "refine_case_with_ai", new=semantic), patch.object(
            ai_reply_generator,
            "get_settings",
            return_value=settings,
        ), patch.object(
            ai_service,
            "get_settings",
            return_value=settings,
        ), patch.object(
            ai_service,
            "initialize_ai",
            return_value=None,
        ), patch.object(
            ai_service,
            "_llama_settings",
            None,
        ), patch.object(
            ai_service,
            "_google_client",
            fake_client,
        ), patch.object(
            chat_route,
            "finish_chat_perf",
            side_effect=capture_finish,
        ):
            response = TestClient(app).post("/chat", json={"message": "我頭暈兩天"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["reply"], response.json()["next_question"])
        semantic.assert_awaited_once()
        self.assertEqual(models.call_count, 0)
        self.assertEqual(len(summaries), 1)
        self.assertFalse(summaries[0]["ai_reply_ai_called"])
        self.assertEqual(summaries[0]["gemini_call_count"], 0)


if __name__ == "__main__":
    unittest.main()
