from __future__ import annotations

import asyncio
from datetime import date, timedelta
from io import BytesIO
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from app.main import app
from app.routes import chat as chat_route
from app.routes import voice as voice_route
from app.schemas import (
    ConversationStage,
    DepartmentResult,
    Message,
    SemanticExtraction,
    TriageCase,
    TriageResult,
    VisitType,
)
from app.services import (
    ai_service,
    appointment_service,
    department_preference_service,
    rag_triage_adapter,
)
from app.services.conversation_service import ClarificationSuggestion
from app.services.case_store import save_case


def _case_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


def _department() -> DepartmentResult:
    return DepartmentResult(dept_id=7, parentDept="外科系", childDept="一般骨科", confidence=1.0)


class VisitTypeRecommendationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_fetch = appointment_service.fetch_available_slots
        self.original_fetch_departments = appointment_service.fetch_active_departments
        self.original_score = appointment_service.score_doctor_specialties

        async def neutral_score(*_args, **_kwargs):
            return {}

        appointment_service.score_doctor_specialties = neutral_score
        appointment_service.fetch_active_departments = lambda: [
            {"dept_id": 7, "parent_dept": "外科系", "child_dept": "一般骨科"}
        ]

    def tearDown(self):
        appointment_service.fetch_available_slots = self.original_fetch
        appointment_service.fetch_active_departments = self.original_fetch_departments
        appointment_service.score_doctor_specialties = self.original_score

    async def test_initial_and_followup_are_exact_and_case_is_source_of_truth(self):
        rows = [_slot("初診醫師", "初診"), _slot("複診醫師", "複診")]
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: rows

        for visit_type, expected in [(VisitType.INITIAL, "初診醫師"), (VisitType.FOLLOWUP, "複診醫師")]:
            with self.subTest(visit_type=visit_type):
                case = _confirmed_case(visit_type)
                save_case(case)
                response = TestClient(app).post("/recommend", json={"case_id": case.case_id})
                self.assertEqual(response.status_code, 200)
                doctors = [item["doctor"] for item in response.json()["recommendations"]["specialty_first"]]
                self.assertEqual(doctors, [expected])

    async def test_return_visit_maps_to_followup_schedule(self):
        self.assertEqual(appointment_service.schedule_visit_type_for(VisitType.RETURN_VISIT), "複診")
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: [
            _slot("初診醫師", "初診"),
            _slot("回診可掛醫師", "複診"),
        ]
        case = _confirmed_case(VisitType.RETURN_VISIT)
        save_case(case)

        response = TestClient(app).post("/recommend", json={"case_id": case.case_id})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["doctor"] for item in response.json()["recommendations"]["specialty_first"]],
            ["回診可掛醫師"],
        )

    async def test_no_compatible_or_blank_visit_type_never_falls_back(self):
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: [
            _slot("只有複診", "複診"),
            _slot("空白類型", ""),
        ]
        case = _confirmed_case(VisitType.INITIAL)
        save_case(case)

        response = TestClient(app).post("/recommend", json={"case_id": case.case_id})

        self.assertEqual(response.status_code, 503)

    async def test_no_matching_availability_returns_503_instead_of_open_rows(self):
        row = _slot("初診醫師", "初診")
        row.update({"date": "2026-07-13", "session": "上午"})
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: [row]
        case = _confirmed_case(VisitType.INITIAL)
        case.availability.preferred_days = ["週二"]
        case.availability.preferred_sessions = ["晚上"]
        case.availability.can_take_leave = False
        save_case(case)

        response = TestClient(app).post("/recommend", json={"case_id": case.case_id})

        self.assertEqual(response.status_code, 503)

    async def test_recommend_request_conflict_is_rejected(self):
        case = _confirmed_case(VisitType.INITIAL)
        save_case(case)
        response = TestClient(app).post(
            "/recommend",
            json={"case_id": case.case_id, "visit_type": "followup"},
        )
        self.assertEqual(response.status_code, 409)


class VoiceCompatibilityTest(unittest.IsolatedAsyncioTestCase):
    async def test_voice_gateway_connection_failure_is_observable_and_user_readable(self):
        settings = SimpleNamespace(
            voice_enabled=True,
            voice_gateway_url="http://localhost:8000",
            voice_gateway_key="test-key",
            voice_default_lang="taiwanese",
            voice_use_fixed_cache=False,
        )
        gateway_error = voice_route.VoiceGatewayError(
            "語音服務目前無法連線，請確認 Voice Gateway 已啟動。",
            upstream_status="connection_error",
            exception_type="ConnectionError",
            exception_message="connection refused",
        )

        def fail(*_args, **_kwargs):
            raise gateway_error

        client = SimpleNamespace(transcribe=fail, synthesize_to_base64=fail)
        with patch.object(voice_route, "get_settings", return_value=settings), patch.object(
            voice_route,
            "_get_client",
            return_value=client,
        ), self.assertLogs("app.routes.voice", level="INFO") as logs:
            asr = await voice_route.voice_asr(
                UploadFile(filename="voice.wav", file=BytesIO(b"RIFF")),
                "taiwanese",
            )
            tts = await voice_route.voice_tts({"text": "測試語音", "lang": "taiwanese"})

        self.assertTrue(asr["asr_failed"])
        self.assertIn("Voice Gateway", asr["error"])
        self.assertTrue(tts["tts_failed"])
        self.assertIn("Voice Gateway", tts["error"])
        combined = "\n".join(logs.output)
        self.assertIn("[VOICE] operation=asr backend_route=/voice/asr", combined)
        self.assertIn("[VOICE] operation=tts backend_route=/voice/tts", combined)
        self.assertIn("upstream_provider=taiwanese_asr", combined)
        self.assertIn("upstream_provider=taiwanese_tts", combined)
        self.assertIn("upstream_status=connection_error", combined)
        self.assertIn("exception_type=ConnectionError", combined)

    async def test_voice_chat_passes_optional_visit_type_to_shared_chat(self):
        case = TriageCase(case_id=_case_id("voice"), visit_type=VisitType.INITIAL)
        triage_result = TriageResult(
            case_id=case.case_id,
            triage_case=case,
            conversation_state=case.conversation_state,
            triage=case.triage,
            reply="請繼續回答",
        )
        chat_handler = AsyncMock(return_value=triage_result)
        client = SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: "我膝蓋痛",
            synthesize_to_base64=lambda *_args, **_kwargs: SimpleNamespace(
                audio_base64="dGVzdA==",
                audio_format="wav",
            ),
        )
        settings = SimpleNamespace(
            voice_enabled=True,
            voice_gateway_url="https://voice.invalid",
            voice_gateway_key="test-key",
            voice_default_lang="chinese",
            voice_use_fixed_cache=False,
        )
        upload = UploadFile(filename="voice.wav", file=BytesIO(b"RIFF"))
        with patch.object(voice_route, "get_settings", return_value=settings), patch.object(
            voice_route,
            "_get_client",
            return_value=client,
        ), patch.object(voice_route, "chat_handler", new=chat_handler):
            response = await voice_route.voice_chat(
                file=upload,
                case_id=None,
                visit_type=VisitType.INITIAL,
                lang="chinese",
                confirmed=False,
            )

        self.assertEqual(response.case_id, case.case_id)
        forwarded = chat_handler.await_args.args[0]
        self.assertEqual(forwarded.visit_type, VisitType.INITIAL)


class _ProviderStatusError(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__(f"status={status_code}")
        self.status_code = status_code


def _slot(doctor: str, visit_type: str) -> dict:
    today = date.today()
    next_monday = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
    return {
        "parent_dept": "外科系",
        "child_dept": "一般骨科",
        "doctor_id": doctor,
        "doctor": doctor,
        "schedule_id": f"s-{doctor}",
        "date": next_monday.isoformat(),
        "session": "上午",
        "slot": "3201診",
        "room": "3201診",
        "specialty_tags": "",
        "status": "open",
        "source": "db",
        "visit_type": visit_type,
        "dept_id": 7,
    }


def _confirmed_case(visit_type: VisitType) -> TriageCase:
    case = TriageCase(case_id=_case_id("recommend"), visit_type=visit_type)
    case.patient_input.symptom = "膝痛"
    case.patient_input.body_part = "膝蓋"
    case.patient_input.duration = "2週"
    case.patient_input.severity = "moderate"
    case.patient_input.red_flags_checked = True
    case.availability.preferred_days = ["週一"]
    case.availability.preferred_sessions = ["上午"]
    case.triage.need_more_info = False
    case.triage.is_final = True
    case.conversation_state.is_complete = True
    case.conversation_state.stage = ConversationStage.RECOMMENDING
    case.conversation_state.confirmed = True
    case.confirmed = True
    case.department_result = _department()
    return case


if __name__ == "__main__":
    unittest.main()
