from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services import ai_service


def _settings(*, key: str = "cerebras-test-key") -> SimpleNamespace:
    return SimpleNamespace(
        ai_provider="gemini",  # A stale environment value must not switch runtime.
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

    async def test_every_request_fixes_temperature_and_seed_by_default(self):
        reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])
        create = AsyncMock(return_value=reply)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        settings = _settings()
        settings.ai_temperature_zero = True
        with patch.object(ai_service, "_cerebras_client", client):
            await ai_service._complete_cerebras("p", settings, json_mode=True)

        request = create.await_args.kwargs
        self.assertEqual(request["temperature"], 0)
        self.assertEqual(request["seed"], ai_service.DETERMINISTIC_SEED)

    async def test_switch_off_keeps_temperature_zero_only_when_asked(self):
        reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))])
        create = AsyncMock(return_value=reply)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        settings = _settings()
        settings.ai_temperature_zero = False
        with patch.object(ai_service, "_cerebras_client", client):
            await ai_service._complete_cerebras("p", settings, json_mode=True, deterministic=True)
            await ai_service._complete_cerebras("p", settings, json_mode=True)

        deterministic, default = (call.kwargs for call in create.await_args_list)
        self.assertEqual(deterministic["temperature"], 0)
        self.assertEqual(deterministic["seed"], ai_service.DETERMINISTIC_SEED)
        self.assertNotIn("temperature", default)
        self.assertNotIn("seed", default)

    def test_temperature_zero_is_the_configured_default(self):
        from app.config import Settings

        self.assertTrue(Settings.model_fields["ai_temperature_zero"].default)
