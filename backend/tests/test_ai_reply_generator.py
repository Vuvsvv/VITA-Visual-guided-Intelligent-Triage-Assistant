from __future__ import annotations

import asyncio
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ConversationStage, DepartmentResult, TriageCase
from app.services import ai_reply_generator
from app.services.question_specs import QUESTION_SPEC_BY_ID
from app.services.rule_engine import QUESTION_TEXTS

chat_route = importlib.import_module("app.routes.chat")


def _settings(*, enabled: bool = True, key: str = "test-key", timeout: float = 0.05):
    return SimpleNamespace(
        ai_reply_generation_enabled=enabled,
        ai_reply_timeout_seconds=timeout,
        google_api_key=key,
        llm_model="gemini-test",
    )


class AiReplyGeneratorTest(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_uses_deterministic_fallback(self):
        case = TriageCase(case_id="disabled")
        with patch.object(ai_reply_generator, "get_settings", return_value=_settings(enabled=False)), patch.object(
            ai_reply_generator, "complete_prompt", new_callable=AsyncMock
        ) as provider:
            reply = await ai_reply_generator.generate_triage_reply(
                case=case,
                next_question=QUESTION_TEXTS["symptom"],
                fallback_reply=QUESTION_TEXTS["symptom"],
            )

        self.assertEqual(reply, QUESTION_TEXTS["symptom"])
        provider.assert_not_awaited()

    async def test_missing_key_does_not_call_provider(self):
        case = TriageCase(case_id="missing-key")
        with patch.object(ai_reply_generator, "get_settings", return_value=_settings(key="")), patch.object(
            ai_reply_generator, "complete_prompt", new_callable=AsyncMock
        ) as provider:
            reply = await ai_reply_generator.generate_triage_reply(
                case=case,
                next_question=QUESTION_TEXTS["symptom"],
                fallback_reply=QUESTION_TEXTS["symptom"],
            )

        self.assertEqual(reply, QUESTION_TEXTS["symptom"])
        provider.assert_not_awaited()

    async def test_provider_exception_keeps_chat_200_and_fallback(self):
        async def fail_provider(_: str) -> str:
            raise RuntimeError("quota")

        with patch.object(ai_reply_generator, "get_settings", return_value=_settings()), patch.object(
            ai_reply_generator, "complete_prompt", side_effect=fail_provider
        ), patch.object(chat_route, "refine_case_with_ai", new=AsyncMock(return_value=None)):
            response = TestClient(app).post("/chat", json={"message": "我頭暈兩天"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["reply"], data["next_question"])

    async def test_timeout_uses_fallback(self):
        async def slow_provider(_: str) -> str:
            await asyncio.sleep(0.1)
            return '{"reply":"不應抵達"}'

        case = TriageCase(case_id="timeout")
        with patch.object(
            ai_reply_generator,
            "get_settings",
            return_value=_settings(timeout=0.001),
        ), patch.object(ai_reply_generator, "complete_prompt", side_effect=slow_provider):
            reply = await ai_reply_generator.generate_triage_reply(
                case=case,
                next_question=QUESTION_TEXTS["symptom"],
                fallback_reply=QUESTION_TEXTS["symptom"],
            )

        self.assertEqual(reply, QUESTION_TEXTS["symptom"])

    async def test_unvalidated_turn_uses_controlled_fallback_without_reply_ai(self):
        contextual = (
            "了解你已經頭暈兩天。請問是否有胸痛、呼吸困難、意識不清、大量出血、"
            "半邊無力或劇烈頭痛？"
        )
        provider = AsyncMock(return_value=f'{{"reply":"{contextual}"}}')
        with patch.object(ai_reply_generator, "get_settings", return_value=_settings()), patch.object(
            ai_reply_generator, "complete_prompt", new=provider
        ), patch.object(chat_route, "refine_case_with_ai", new=AsyncMock(return_value=None)):
            response = TestClient(app).post("/chat", json={"message": "我頭暈兩天"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["reply"], data["next_question"])
        self.assertTrue(data["next_question"])
        self.assertEqual(data["conversation_state"]["stage"], ConversationStage.COLLECTING.value)
        self.assertFalse(data["conversation_state"]["confirmed"])
        self.assertFalse(data["triage_case"]["confirmed"])
        provider.assert_not_awaited()

    async def test_ai_reply_does_not_change_confirmation_gate(self):
        complete_message = (
            "我頭暈兩天，程度中等，沒有胸痛、呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛，"
            "週一上午可以看診"
        )
        department = DepartmentResult(
            parentDept="一般內科",
            childDept="一般內科",
            confidence=0.9,
            reason=["測試"],
        )
        provider = AsyncMock(return_value='{"reply":"資料已整理完成，請確認是否正確。"}')
        with patch.object(ai_reply_generator, "get_settings", return_value=_settings()), patch.object(
            ai_reply_generator, "complete_prompt", new=provider
        ), patch.object(chat_route, "refine_case_with_ai", new=AsyncMock(return_value=None)), patch.object(
            chat_route, "detect_department_result", new=AsyncMock(return_value=department)
        ) as detector:
            response = TestClient(app).post("/chat", json={"message": complete_message})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["conversation_state"]["stage"], ConversationStage.COLLECTING.value)
        self.assertFalse(data["conversation_state"]["awaiting_confirmation"])
        self.assertFalse(data["conversation_state"]["confirmed"])
        self.assertIsNone(data["department_result"])
        detector.assert_not_awaited()

    def test_department_confirmation_reason_uses_only_collected_case_fields(self):
        case = TriageCase(case_id="grounded-confirmation")
        case.patient_input.symptom = "頭部不舒服"
        case.patient_input.body_part = "頭部"
        case.department_result = DepartmentResult(
            parentDept="內科系",
            childDept="神經內科",
            confidence=0.9,
        )

        reply = ai_reply_generator.build_department_confirmation_reply(case)

        self.assertIn("頭部不舒服", reply)
        self.assertIn("頭部", reply)
        self.assertIn("神經內科", reply)
        self.assertNotIn("頭痛", reply)
        self.assertNotIn("暈眩", reply)
        self.assertNotIn("腦血管疾病", reply)

    def test_department_confirmation_separately_echoes_exact_date_preference(self):
        case = TriageCase(case_id="confirmation-date")
        case.patient_input.symptom = "腿痛"
        case.patient_input.body_part = "大腿"
        case.patient_input.duration = "3週"
        case.availability.preferred_dates = ["2026-09-20"]
        case.availability.preferred_days = ["週日"]
        case.availability.preferred_sessions = ["下午"]
        case.department_result = DepartmentResult(
            dept_id=1298,
            parentDept="外科系",
            childDept="一般骨科",
            confidence=0.9,
        )

        reply = ai_reply_generator.build_department_confirmation_reply(case)

        self.assertIn("一般骨科", reply)
        self.assertIn("就醫時間偏好：9 月 20 日下午。", reply)

    async def test_incomplete_ai_red_flag_screen_is_rejected(self):
        case = TriageCase(case_id="red-flag")
        case.conversation_state.last_question_key = "red_flags"
        fallback = QUESTION_TEXTS["red_flags"]
        with patch.object(ai_reply_generator, "get_settings", return_value=_settings()), patch.object(
            ai_reply_generator,
            "complete_prompt",
            new=AsyncMock(return_value='{"reply":"請問有胸痛或呼吸困難嗎？"}'),
        ):
            reply = await ai_reply_generator.generate_triage_reply(
                case=case,
                next_question=fallback,
                fallback_reply=fallback,
            )

        self.assertEqual(reply, fallback)
        for term in ("胸痛", "呼吸困難", "意識不清", "大量出血", "半邊無力", "劇烈頭痛"):
            self.assertIn(term, reply)

    async def test_question_targets_never_call_provider(self):
        examples = {
            "duration": "了解你有頭暈不適。這個症狀持續多久了？",
            "severity": "你提到已經不舒服一段時間，請問嚴重程度或對生活的影響如何？",
            "preferred_days": "症狀資料已整理，請問最近哪幾天方便就醫？",
            "preferred_sessions": "可看診日期已記下，偏好的時段是上午、下午還是夜間？",
        }
        for target, expected in examples.items():
            with self.subTest(target=target):
                case = TriageCase(case_id=f"example-{target}")
                case.conversation_state.last_question_key = target
                next_question = QUESTION_TEXTS.get(target)
                provider = AsyncMock(return_value=f'{{"reply":"{expected}"}}')
                with patch.object(
                    ai_reply_generator, "get_settings", return_value=_settings()
                ), patch.object(
                    ai_reply_generator,
                    "complete_prompt",
                    new=provider,
                ):
                    reply = await ai_reply_generator.generate_triage_reply(
                        case=case,
                        next_question=next_question,
                        fallback_reply=next_question or "請確認資料。",
                    )

                self.assertEqual(reply, next_question)
                provider.assert_not_awaited()

    async def test_status_reply_never_sends_identifiers_or_calls_provider(self):
        case = TriageCase(case_id="must-not-be-sent")
        case.patient_input.symptom = (
            "我是王小明，住在台北市測試路一號，學號：S123456，頭暈，"
            "電話：0912345678，Email：a@example.com"
        )
        case.conversation_state.last_question_key = "duration"
        case.conversation_state.awaiting_confirmation = True
        provider = AsyncMock(return_value='{"reply":"頭暈持續多久了？"}')
        with patch.object(ai_reply_generator, "get_settings", return_value=_settings()), patch.object(
            ai_reply_generator, "complete_prompt", new=provider
        ):
            await ai_reply_generator.generate_triage_reply(
                case=case,
                next_question=None,
                fallback_reply="請確認資料。",
            )

        provider.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
