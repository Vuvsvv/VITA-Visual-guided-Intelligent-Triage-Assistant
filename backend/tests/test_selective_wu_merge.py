from __future__ import annotations

import asyncio
import unittest
import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import DepartmentResult, TriageCase
from app.services.conversation_service import ClarificationSuggestion
from app.services import appointment_service, department_preference_service, rag_triage_adapter, specialty_scoring
from app.services.rule_engine import apply_user_message
from app.services.specialty_scoring import score_doctor_specialties


class SelectiveWuMergeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_fetch_departments = appointment_service.fetch_active_departments
        self.original_preference_departments = department_preference_service.fetch_active_departments
        self.original_fetch_slots = appointment_service.fetch_available_slots
        self.original_rag_settings = rag_triage_adapter.get_settings
        self.original_rag_complete = rag_triage_adapter.complete_prompt
        self.original_specialty_settings = specialty_scoring.get_settings
        self.chat_route = importlib.import_module("app.routes.chat")
        self.original_create_case = self.chat_route.create_case
        self.original_runtime_ai_available = self.chat_route.runtime_ai_available

        rag_triage_adapter.get_settings = lambda: _NoAiSettings()
        specialty_scoring.get_settings = lambda: _NoAiSettings()

        def create_prescreened_case(case_id=None):
            case = self.original_create_case(case_id)
            case.patient_input.red_flags_checked = True
            case.patient_input.red_flags_status = "negative"
            return case

        self.chat_route.create_case = create_prescreened_case
        self.chat_route.runtime_ai_available = lambda _settings: False
        appointment_service.fetch_active_departments = _active_departments
        department_preference_service.fetch_active_departments = _active_departments
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: _available_slots()
        self.generate_script_route = importlib.import_module("app.routes.generate_script")
        self.original_revalidate = self.generate_script_route.revalidate_schedule
        self.generate_script_route.revalidate_schedule = lambda recommendation, **_kwargs: recommendation

        async def fail_complete(_: str) -> str:
            raise RuntimeError("AI disabled in selective merge tests")

        rag_triage_adapter.complete_prompt = fail_complete

    def tearDown(self):
        appointment_service.fetch_active_departments = self.original_fetch_departments
        department_preference_service.fetch_active_departments = self.original_preference_departments
        appointment_service.fetch_available_slots = self.original_fetch_slots
        rag_triage_adapter.get_settings = self.original_rag_settings
        rag_triage_adapter.complete_prompt = self.original_rag_complete
        specialty_scoring.get_settings = self.original_specialty_settings
        self.chat_route.create_case = self.original_create_case
        self.chat_route.runtime_ai_available = self.original_runtime_ai_available
        self.generate_script_route.revalidate_schedule = self.original_revalidate

    def test_deterministic_specialty_fallback_is_neutral_without_keyword_inference(self):
        case = TriageCase(case_id="case_selective_reason")
        case.patient_input.symptom = "腹痛兩天"
        department = DepartmentResult(parentDept="內科系", childDept="胃腸肝膽科", confidence=0.8)
        row = {
            "doctor_id": "gi-1",
            "doctor": "腸胃專長醫師",
            "child_dept": "胃腸肝膽科",
            "specialty_tags": "腸胃疾病、消化系統、腹痛",
        }

        with patch.object(specialty_scoring, "get_settings", return_value=SimpleNamespace(ai_doctor_scoring_enabled=False)):
            score = asyncio.run(score_doctor_specialties(case, department, [row]))["gi-1"]

        self.assertEqual(score.score, 0.5)
        self.assertEqual(score.source, "neutral")
        self.assertIn("中性值 0.50", score.reason)
        self.assertNotIn("腹痛", score.reason)
        self.assertNotIn("腸胃", score.reason)

    def test_main_chat_confirm_recommend_generate_script_flow_still_runs(self):
        client = TestClient(app)

        async def converge(case):
            case.department_result = DepartmentResult(
                dept_id=203, parentDept="外科系", childDept="一般骨科", confidence=0.9,
            )
            case.conversation_state.department_status = "resolved"

        proposal = dict(semantic_extractions=[dict(field=field, normalized_value=value,
            assertion="present" if field == "symptom" else None, source_text=source,
            semantic_status="available", confidence=0.98) for field, value, source in (
                ("symptom", "膝痛", "左膝痛"), ("duration", "2週", "2週"),
                ("preferred_days", ["週一"], "週一"), ("preferred_sessions", ["上午"], "上午"),
            )], pending_answer=None, ttas_evidence=[], time_preferences=[],
            prior_answer_reference=None, department_preference=None,
            dialogue_act=dict(act="new_information", source_text="左膝痛", confidence=0.99))
        with patch.object(self.chat_route, "runtime_ai_available", return_value=True), patch.object(
            rag_triage_adapter, "_ai_available", return_value=True,
        ), patch.object(rag_triage_adapter, "complete_prompt", AsyncMock(return_value=json.dumps(proposal))), patch.object(
            self.chat_route, "reason_about_departments", AsyncMock(side_effect=converge),
        ), patch.object(self.chat_route, "request_clarification", AsyncMock(return_value=ClarificationSuggestion(
            status="sufficient", question=None, intent=None, reason="grounded fixture",
        ))):
            chat_response = client.post(
                "/chat",
                json={"message": "左膝痛2週，爬樓梯很吃力，想看一般骨科，週一上午可以看診", "visit_type": "initial"},
            )
        self.assertEqual(chat_response.status_code, 200)
        chat_data = chat_response.json()
        self.assertEqual(chat_data["conversation_state"]["stage"], "waiting_confirmation")

        confirm_response = client.post("/chat", json={"case_id": chat_data["case_id"], "confirmed": True})
        self.assertEqual(confirm_response.status_code, 200)

        recommend_response = client.post("/recommend", json={"case_id": chat_data["case_id"]})
        self.assertEqual(recommend_response.status_code, 200)
        recommendation = recommend_response.json()["recommendations"]["specialty_first"][0]

        script_response = client.post(
            "/generate_script",
            json={
                "case_id": chat_data["case_id"],
                "recommendation_id": recommendation["recommendation_id"],
            },
        )
        self.assertEqual(script_response.status_code, 200)
        self.assertGreaterEqual(script_response.json()["step_count"], 1)


class _NoAiSettings:
    google_api_key = ""


def _case_from_message(message: str) -> TriageCase:
    case = TriageCase(case_id="case_selective_wu")
    apply_user_message(case, message)
    return case


def _active_departments() -> list[dict]:
    return [
        {"dept_id": 201, "parent_dept": "婦幼醫學部", "child_dept": "兒童腸胃科"},
        {"dept_id": 202, "parent_dept": "內科系", "child_dept": "胃腸肝膽科"},
        {"dept_id": 203, "parent_dept": "外科系", "child_dept": "一般骨科"},
        {"dept_id": 204, "parent_dept": "一般內科", "child_dept": "一般內科"},
    ]


def _available_slots() -> list[dict]:
    return [
        {
            "parent_dept": "外科系",
            "child_dept": "一般骨科",
            "dept_id": 203,
            "doctor_id": "ortho-1",
            "doctor": "骨科專長醫師",
            "schedule_id": "selective-slot-1",
            "date": _next_weekday(0),
            "session": "上午",
            "slot": "3201診",
            "room": "3201診",
            "specialty_tags": "膝關節、運動傷害、骨科",
            "status": "open",
            "visit_type": "初診",
        }
    ]


def _next_weekday(weekday: int) -> date:
    today = date.today()
    offset = (weekday - today.weekday()) % 7
    if offset == 0:
        offset = 7
    return today + timedelta(days=offset)


if __name__ == "__main__":
    unittest.main()
