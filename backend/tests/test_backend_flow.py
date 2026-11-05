from __future__ import annotations

import unittest
import importlib
import json as json_module
from unittest.mock import AsyncMock, patch
from datetime import date, timedelta
from fastapi.testclient import TestClient

from app import db
from app.main import app
from app.schemas import DepartmentResult, RecommendationItem, TriageCase, VisitType
from app.routes import chat as chat_route
from app.services import appointment_service, department_preference_service, rag_triage_adapter
from app.services.appointment_service import DepartmentResolutionError, recommend_appointments
from app.services.case_store import get_case
from app.services.rule_engine import apply_user_message
from app.services.script_service import build_navigation_script
from app.services.ttas_evaluator import apply_ttas_evaluation
from app.services.conversation_service import ClarificationSuggestion


class BackendFlowTest(unittest.IsolatedAsyncioTestCase):
    def post_accepted_description(self, client, path, *, json):
        proposal = {"semantic_extractions": [
            {"field": "symptom", "normalized_value": "膝痛", "source_text": "左膝痛", "assertion": "present", "semantic_status": "available", "confidence": 0.98},
            {"field": "duration", "normalized_value": "2週", "source_text": "2週", "semantic_status": "available", "confidence": 0.98},
            {"field": "preferred_days", "normalized_value": ["週一"], "source_text": "週一", "semantic_status": "available", "confidence": 0.98},
            {"field": "preferred_sessions", "normalized_value": ["上午"], "source_text": "上午", "semantic_status": "available", "confidence": 0.98},
        ], "pending_answer": None, "ttas_evidence": [], "time_preferences": [],
            "prior_answer_reference": None, "department_preference": None,
            "dialogue_act": {"act": "new_information", "source_text": "左膝痛", "confidence": 0.99}}

        async def converge(case):
            case.department_result = DepartmentResult(dept_id=7, parentDept="外科系", childDept="一般骨科")
            case.conversation_state.department_status = "resolved"

        with patch.object(chat_route, "runtime_ai_available", return_value=True), patch.object(
            rag_triage_adapter, "_ai_available", return_value=True,
        ), patch.object(rag_triage_adapter, "complete_prompt", AsyncMock(return_value=json_module.dumps(proposal))), patch.object(
            chat_route, "reason_about_departments", AsyncMock(side_effect=converge),
        ), patch.object(chat_route, "request_clarification", AsyncMock(return_value=ClarificationSuggestion("sufficient", None, None, "已有可靠資料"))):
            return client.post(path, json=json)

    def setUp(self):
        self.original_create_case = chat_route.create_case
        self.original_departments = appointment_service.fetch_active_departments
        self.original_preference_departments = department_preference_service.fetch_active_departments
        self.original_rag_complete = rag_triage_adapter.complete_prompt
        self.original_rag_settings = rag_triage_adapter.get_settings
        self.original_generate_triage_reply = chat_route.generate_triage_reply
        self.original_runtime_ai_available = chat_route.runtime_ai_available
        appointment_service.fetch_active_departments = lambda: [
            {"dept_id": 7, "parent_dept": "外科系", "child_dept": "一般骨科"},
            {"dept_id": 8, "parent_dept": "一般內科", "child_dept": "一般內科"},
            {"dept_id": 9, "parent_dept": "一般內科", "child_dept": "心臟內科"},
            {"dept_id": 1333, "parent_dept": "五官科", "child_dept": "耳科"},
        ]
        department_preference_service.fetch_active_departments = appointment_service.fetch_active_departments

        async def fail_complete(_: str) -> str:
            raise RuntimeError("llm disabled in tests")

        rag_triage_adapter.complete_prompt = fail_complete

        async def deterministic_reply(**kwargs) -> str:
            return kwargs["fallback_reply"]

        chat_route.generate_triage_reply = deterministic_reply
        chat_route.runtime_ai_available = lambda _settings: False
        def create_pre_screened_case(case_id=None):
            case = self.original_create_case(case_id)
            case.patient_input.red_flags_checked = True
            case.patient_input.red_flags_status = "negative"
            return case

        chat_route.create_case = create_pre_screened_case
        self.generate_script_route = importlib.import_module("app.routes.generate_script")
        self.original_revalidate = self.generate_script_route.revalidate_schedule
        self.generate_script_route.revalidate_schedule = lambda recommendation, **_kwargs: recommendation

    def tearDown(self):
        appointment_service.fetch_active_departments = self.original_departments
        department_preference_service.fetch_active_departments = self.original_preference_departments
        rag_triage_adapter.complete_prompt = self.original_rag_complete
        rag_triage_adapter.get_settings = self.original_rag_settings
        chat_route.generate_triage_reply = self.original_generate_triage_reply
        chat_route.runtime_ai_available = self.original_runtime_ai_available
        chat_route.create_case = self.original_create_case
        self.generate_script_route.revalidate_schedule = self.original_revalidate

    async def test_high_urgency_red_flag(self):
        case = TriageCase(case_id="case_high")
        apply_user_message(case, "突發胸痛而且呼吸困難")
        case.triage = apply_ttas_evaluation(case)

        self.assertIsNone(case.triage.urgency_score)
        self.assertIsNone(case.triage.urgency_level)
        self.assertFalse(case.triage.warning_required)
        self.assertEqual(case.ttas_result.status, "insufficient_information")
        self.assertTrue(case.triage.reasons)

    async def test_db_failure_returns_no_schedule_rows(self):
        original_create = db.create_db_connection
        db.create_db_connection = lambda: (_ for _ in ()).throw(RuntimeError("db down"))
        try:
            with self.assertRaises(db.DatabaseUnavailableError):
                db.fetch_available_slots("一般骨科", max_slots=2)
        finally:
            db.create_db_connection = original_create


    async def test_db_empty_slots_return_no_schedule_rows(self):
        original_fetch_db = db._fetch_available_slots_from_db
        db._fetch_available_slots_from_db = lambda *_, **__: []
        try:
            slots = db.fetch_available_slots("一般骨科", max_slots=2)
        finally:
            db._fetch_available_slots_from_db = original_fetch_db

        self.assertEqual(slots, [])

    async def test_rag_adapter_can_refine_patient_input_without_changing_schema(self):
        class FakeSettings:
            cerebras_api_key = "test-key"

        async def fake_complete(_: str) -> str:
            return """
            {
              "semantic_extractions": [
                {
                  "field": "body_part",
                  "normalized_value": "膝",
                  "semantic_status": "available",
                  "confidence": 0.9,
                  "source_text": "膝蓋"
                }
              ],
              "pending_answer": null,
              "ttas_evidence": [],
              "time_preferences": [],
              "prior_answer_reference": null,
              "department_preference": null,
              "dialogue_act": {"act": "new_information", "source_text": "膝蓋", "confidence": 0.99}
            }
            """

        rag_triage_adapter.get_settings = lambda: FakeSettings()
        rag_triage_adapter.complete_prompt = fake_complete

        case = TriageCase(case_id="case_rag_refine")
        apply_user_message(case, "膝蓋不舒服")
        suggestion = await rag_triage_adapter.refine_case_with_ai(case)

        self.assertIsNotNone(suggestion)
        self.assertEqual(case.patient_input.body_part, "膝")
        self.assertIsNone(case.patient_input.duration)
        self.assertNotIn("爬樓梯吃力", case.patient_input.accompanying_symptoms)
        self.assertTrue(case.semantic_extractions)
        self.assertIsNone(suggestion.triage)
        self.assertIsNone(case.triage.urgency_score)

    async def test_ai_semantic_extraction_cannot_replace_explicit_value_or_complete_red_flags(self):
        class FakeSettings:
            cerebras_api_key = "test-key"

        async def fake_complete(_: str) -> str:
            return """
            {
              "semantic_extractions": [
                {
                  "field": "body_part",
                  "normalized_value": "胸口",
                  "semantic_status": "available",
                  "confidence": 0.99,
                  "source_text": "AI 猜測"
                },
                {
                  "field": "red_flags",
                  "normalized_value": ["突發胸痛"],
                  "semantic_status": "available",
                  "confidence": 1.0,
                  "source_text": "AI 猜測"
                }
              ]
            }
            """

        rag_triage_adapter.get_settings = lambda: FakeSettings()
        rag_triage_adapter.complete_prompt = fake_complete
        case = TriageCase(case_id="case_ai_no_overwrite")
        case.patient_input.body_part = "頭部"

        await rag_triage_adapter.refine_case_with_ai(case)

        self.assertEqual(case.patient_input.body_part, "頭部")
        self.assertFalse(case.patient_input.red_flags_checked)
        self.assertEqual(case.patient_input.red_flags, [])

    async def test_recommendations_are_capped_at_five_per_column(self):
        original_fetch = appointment_service.fetch_available_slots
        today = date.today()

        def fake_slots(*_, **__):
            return [
                {
                    "dept_id": 7,
                    "parent_dept": "外科系",
                    "child_dept": "一般骨科",
                    "doctor_id": f"fixture-doctor-{i}",
                    "doctor": f"測試醫師{i}",
                    "schedule_id": f"fixture-schedule-{i}",
                    "date": today + timedelta(days=i + 1),
                    "session": "上午",
                    "slot": f"32{i:02d}診",
                    "status": "open",
                    "source": "test_fixture",
                    "visit_type": "初診",
                }
                for i in range(8)
            ]

        appointment_service.fetch_available_slots = fake_slots
        try:
            case = TriageCase(case_id="case_recommend", visit_type=VisitType.INITIAL)
            apply_user_message(case, "左膝痛2週，爬樓梯很吃力，週一上午")
            case.department_result = DepartmentResult(dept_id=7, parentDept="外科系", childDept="一般骨科")
            result = await recommend_appointments(case)
        finally:
            appointment_service.fetch_available_slots = original_fetch

        self.assertGreaterEqual(len(result.recommendations.specialty_first), 1)
        self.assertLessEqual(len(result.recommendations.specialty_first), 5)
        self.assertGreaterEqual(len(result.recommendations.time_first), 1)
        self.assertLessEqual(len(result.recommendations.time_first), 5)
        self.assertEqual(result.recommendations.specialty_first[0].rank, 1)
        self.assertTrue(result.recommendations.specialty_first[0].is_best_match)
        self.assertLessEqual(result.total_count, 10)

    async def test_no_matching_department_does_not_cross_recommend(self):
        original_fetch = appointment_service.fetch_available_slots
        requested_departments = []

        def wrong_department_rows(department, *_args, **_kwargs):
            requested_departments.append(department)
            return [
                {
                    "parent_dept": "外科系",
                    "child_dept": "一般骨科",
                    "doctor": "骨科醫師",
                    "date": date.today() + timedelta(days=1),
                    "session": "上午",
                    "visit_type": "初診",
                }
            ]

        appointment_service.fetch_available_slots = wrong_department_rows
        try:
            case = TriageCase(case_id="case_no_cross_department", visit_type=VisitType.INITIAL)
            case.department_result = DepartmentResult(
                dept_id=9,
                parentDept="一般內科",
                childDept="心臟內科",
                confidence=1.0,
            )
            result = await recommend_appointments(case)
        finally:
            appointment_service.fetch_available_slots = original_fetch

        self.assertEqual(requested_departments, ["心臟內科"])
        self.assertEqual(result.recommendations.specialty_first, [])
        self.assertEqual(result.recommendations.time_first, [])
        self.assertEqual(result.fallback_departments, [])

    async def test_recommendation_routes_by_department_id(self):
        original_fetch = appointment_service.fetch_available_slots
        captured: dict = {}

        def exact_department_rows(department, *_args, **kwargs):
            captured["name"] = department
            captured["dept_id"] = kwargs.get("department_id")
            return [
                {
                    "dept_id": 1333,
                    "parent_dept": "五官科",
                    "child_dept": "耳科",
                    "doctor_id": "fixture-ear-doctor",
                    "doctor": "耳科醫師",
                    "schedule_id": "fixture-ear-schedule",
                    "date": date.today() + timedelta(days=1),
                    "session": "上午",
                    "status": "open",
                    "source": "test_fixture",
                    "visit_type": "初診",
                }
            ]

        appointment_service.fetch_available_slots = exact_department_rows
        try:
            case = TriageCase(case_id="case_exact_dept", visit_type=VisitType.INITIAL)
            case.department_result = DepartmentResult(
                dept_id=1333,
                parentDept="五官科",
                childDept="耳科",
                confidence=1.0,
            )
            result = await recommend_appointments(case)
        finally:
            appointment_service.fetch_available_slots = original_fetch

        self.assertEqual(captured, {"name": "耳科", "dept_id": 1333})
        self.assertEqual(result.recommendations.specialty_first[0].dept_id, 1333)

    async def test_invalid_department_id_does_not_fallback(self):
        original_fetch = appointment_service.fetch_available_slots
        captured: list[int | None] = []

        def no_rows(_department, *_args, **kwargs):
            captured.append(kwargs.get("department_id"))
            return []

        appointment_service.fetch_available_slots = no_rows
        try:
            case = TriageCase(case_id="case_invalid_dept", visit_type=VisitType.INITIAL)
            case.department_result = DepartmentResult(
                dept_id=999999,
                parentDept="不存在父科",
                childDept="不存在科",
                confidence=1.0,
            )
            with self.assertRaises(DepartmentResolutionError):
                await recommend_appointments(case)
        finally:
            appointment_service.fetch_available_slots = original_fetch

        self.assertEqual(captured, [])

    async def test_script_template_applies_recommendation(self):
        item = RecommendationItem(
            recommendation_id="rec_test",
            parentDept="外科系",
            childDept="一般骨科",
            doctor="邱方遙",
            date="2026-05-18",
            session="上午",
            slot="3209診",
            score=95,
            reasons=["專長最符合"],
        )

        steps = build_navigation_script(item)

        self.assertEqual(steps[0].action, "open_app")
        self.assertEqual(steps[4].target, "外科系")
        self.assertEqual(steps[5].target, "一般骨科")
        self.assertEqual(steps[8].target, "3209診邱方遙")
        self.assertEqual(steps[8].text, "3209診邱方遙")
        self.assertGreaterEqual(steps[8].retry, 1)

    async def test_incomplete_triage_case_is_rejected_by_recommend(self):
        client = TestClient(app)
        response = client.post(
            "/recommend",
            json={
                "triage_case": {
                    "case_id": "case_incomplete",
                    "visit_type": "initial",
                    "patient_input": {"symptom": "膝蓋痛"},
                    "triage": {"need_more_info": True},
                    "conversation_state": {"stage": "collecting", "is_complete": False},
                }
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("請先完成 /chat", response.json()["detail"])

    async def test_completed_triage_waits_for_confirmation_before_recommend(self):
        client = TestClient(app)
        chat_response = self.post_accepted_description(client,
            "/chat",
            json={"message": "左膝痛2週，爬樓梯很吃力，想看一般骨科，沒有胸痛呼吸困難意識不清大量出血，週一上午可以看診", "visit_type": "initial"},
        )
        self.assertEqual(chat_response.status_code, 200)
        chat_data = chat_response.json()
        self.assertEqual(chat_data["conversation_state"]["stage"], "waiting_confirmation")
        self.assertTrue(chat_data["conversation_state"]["awaiting_confirmation"])
        self.assertFalse(chat_data["conversation_state"]["confirmed"])
        self.assertTrue(chat_data["triage_reasons"])

        recommend_response = client.post("/recommend", json={"case_id": chat_data["case_id"]})
        self.assertEqual(recommend_response.status_code, 400)
        self.assertIn("尚未確認", recommend_response.json()["detail"])

        confirm_response = client.post(
            "/chat",
            json={"case_id": chat_data["case_id"], "confirmed": True},
        )
        self.assertEqual(confirm_response.status_code, 200)
        confirm_data = confirm_response.json()
        self.assertEqual(confirm_data["conversation_state"]["stage"], "recommending")
        self.assertTrue(confirm_data["conversation_state"]["confirmed"])
        self.assertFalse(confirm_data["conversation_state"]["awaiting_confirmation"])

    async def test_short_red_flag_denial_does_not_repeat_red_flag_question(self):
        client = TestClient(app)
        first_response = client.post(
            "/chat",
            json={"message": "左膝痛2週，爬樓梯很吃力"},
        )
        self.assertEqual(first_response.status_code, 200)
        first_data = first_response.json()
        self.assertNotEqual(first_data["conversation_state"]["last_question_key"], "red_flags")

        second_response = client.post(
            "/chat",
            json={"case_id": first_data["case_id"], "message": "沒有"},
        )
        self.assertEqual(second_response.status_code, 200)
        second_data = second_response.json()

        patient_input = second_data["triage_case"]["patient_input"]
        self.assertTrue(patient_input["red_flags_checked"])
        self.assertEqual(patient_input["red_flags"], [])
        self.assertNotEqual(second_data["conversation_state"]["last_question_key"], "red_flags")
        self.assertNotIn("突發胸痛", second_data.get("next_question") or "")

    async def test_unavailable_ai_does_not_guess_all_negative_red_flags_consumes_field_and_does_not_repeat(self):
        # Former checklist-route expectation; primary free text must now fail closed.
        chat_route.create_case = self.original_create_case
        response = TestClient(app).post("/chat", json={"message": "膝蓋痛一週，爬樓梯很吃力"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["triage_case"]["semantic_extractions"], [])
        self.assertIsNone(data["triage_case"]["ttas_result"]["level_candidate"])
        self.assertIsNone(data["triage"]["urgency_score"])
        self.assertFalse(data["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(data["conversation_state"]["stage"], "collecting")
        self.assertFalse(data["conversation_state"]["confirmed"])

    async def test_unavailable_ai_does_not_guess_detailed_red_flag_denial_reaches_waiting_confirmation(self):
        # Former checklist-route expectation; primary free text must now fail closed.
        chat_route.create_case = self.original_create_case
        response = TestClient(app).post("/chat", json={"message": "左膝痛2週，爬樓梯很吃力，沒有突發胸痛，沒有呼吸困難，沒有意識不清，沒有大量出血，沒有半邊無力或劇烈頭痛，週一上午可以看診"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["triage_case"]["semantic_extractions"], [])
        self.assertIsNone(data["triage_case"]["ttas_result"]["level_candidate"])
        self.assertIsNone(data["triage"]["urgency_score"])
        self.assertFalse(data["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(data["conversation_state"]["stage"], "collecting")
        self.assertFalse(data["conversation_state"]["confirmed"])

    async def test_ai_question_does_not_override_deterministic_red_flag_state(self):
        class FakeSettings:
            cerebras_api_key = "test-key"

        async def fake_complete(_: str) -> str:
            return """
            {
              "patient_input": {
                "red_flags": []
              },
              "triage": {
                "urgency_score": 20,
                "urgency_level": "low",
                "warning_required": false,
                "warning_message": null,
                "need_more_info": true,
                "next_question": "請問是否有突發胸痛或呼吸困難？",
                "reasons": ["AI 建議再問紅旗"],
                "is_final": false
              },
              "reply": "請問是否有突發胸痛或呼吸困難？"
            }
            """

        rag_triage_adapter.get_settings = lambda: FakeSettings()
        rag_triage_adapter.complete_prompt = fake_complete

        client = TestClient(app)
        first_response = client.post(
            "/chat",
            json={"message": "左膝痛2週，爬樓梯很吃力"},
        )
        first_data = first_response.json()

        second_response = client.post(
            "/chat",
            json={"case_id": first_data["case_id"], "message": "沒有"},
        )
        self.assertEqual(second_response.status_code, 200)
        second_data = second_response.json()

        self.assertTrue(second_data["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertNotEqual(second_data["conversation_state"]["last_question_key"], "red_flags")
        self.assertNotIn("突發胸痛", second_data.get("next_question") or "")

    async def test_unavailable_ai_does_not_guess_all_available_answer_moves_from_days_to_sessions(self):
        # Former checklist-route expectation; primary free text must now fail closed.
        chat_route.create_case = self.original_create_case
        response = TestClient(app).post("/chat", json={"message": "左膝痛2週，爬樓梯很吃力，沒有胸痛呼吸困難意識不清大量出血"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["triage_case"]["semantic_extractions"], [])
        self.assertIsNone(data["triage_case"]["ttas_result"]["level_candidate"])
        self.assertIsNone(data["triage"]["urgency_score"])
        self.assertFalse(data["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(data["conversation_state"]["stage"], "collecting")
        self.assertFalse(data["conversation_state"]["confirmed"])

    async def test_unavailable_ai_does_not_guess_weekday_range_and_morning_session_reach_waiting_confirmation(self):
        # Former checklist-route expectation; primary free text must now fail closed.
        chat_route.create_case = self.original_create_case
        response = TestClient(app).post("/chat", json={"message": "左膝痛2週，爬樓梯很吃力，沒有胸痛呼吸困難意識不清大量出血"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["triage_case"]["semantic_extractions"], [])
        self.assertIsNone(data["triage_case"]["ttas_result"]["level_candidate"])
        self.assertIsNone(data["triage"]["urgency_score"])
        self.assertFalse(data["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(data["conversation_state"]["stage"], "collecting")
        self.assertFalse(data["conversation_state"]["confirmed"])

    async def test_unavailable_ai_does_not_guess_ai_question_does_not_override_deterministic_availability_state(self):
        # Former checklist-route expectation; primary free text must now fail closed.
        chat_route.create_case = self.original_create_case
        response = TestClient(app).post("/chat", json={"message": "左膝痛2週，爬樓梯很吃力，沒有胸痛呼吸困難意識不清大量出血"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["triage_case"]["semantic_extractions"], [])
        self.assertIsNone(data["triage_case"]["ttas_result"]["level_candidate"])
        self.assertIsNone(data["triage"]["urgency_score"])
        self.assertFalse(data["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(data["conversation_state"]["stage"], "collecting")
        self.assertFalse(data["conversation_state"]["confirmed"])

    async def test_unavailable_ai_does_not_guess_uncertain_all_available_answer_does_not_loop(self):
        # Former checklist-route expectation; primary free text must now fail closed.
        chat_route.create_case = self.original_create_case
        response = TestClient(app).post("/chat", json={"message": "左膝痛2週，爬樓梯很吃力，沒有胸痛呼吸困難意識不清大量出血"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["triage_case"]["semantic_extractions"], [])
        self.assertIsNone(data["triage_case"]["ttas_result"]["level_candidate"])
        self.assertIsNone(data["triage"]["urgency_score"])
        self.assertFalse(data["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(data["conversation_state"]["stage"], "collecting")
        self.assertFalse(data["conversation_state"]["confirmed"])

    async def test_unavailable_ai_does_not_guess_mixed_availability_answer_consumes_days_and_does_not_loop(self):
        # Former checklist-route expectation; primary free text must now fail closed.
        chat_route.create_case = self.original_create_case
        response = TestClient(app).post("/chat", json={"message": "膝蓋痛1週，爬樓梯很吃力，沒有胸痛呼吸困難意識不清大量出血"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["triage_case"]["semantic_extractions"], [])
        self.assertIsNone(data["triage_case"]["ttas_result"]["level_candidate"])
        self.assertIsNone(data["triage"]["urgency_score"])
        self.assertFalse(data["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(data["conversation_state"]["stage"], "collecting")
        self.assertFalse(data["conversation_state"]["confirmed"])

    async def test_unavailable_ai_does_not_guess_unknown_answer_falls_back_after_two_attempts(self):
        # Former checklist-route expectation; primary free text must now fail closed.
        chat_route.create_case = self.original_create_case
        response = TestClient(app).post("/chat", json={"message": "左膝痛2週，沒有胸痛呼吸困難意識不清大量出血"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["triage_case"]["semantic_extractions"], [])
        self.assertIsNone(data["triage_case"]["ttas_result"]["level_candidate"])
        self.assertIsNone(data["triage"]["urgency_score"])
        self.assertFalse(data["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(data["conversation_state"]["stage"], "collecting")
        self.assertFalse(data["conversation_state"]["confirmed"])

    async def test_unavailable_ai_does_not_guess_negated_red_flags_are_structured_and_do_not_loop(self):
        # Former checklist-route expectation; primary free text must now fail closed.
        chat_route.create_case = self.original_create_case
        response = TestClient(app).post("/chat", json={"message": "左膝痛2週"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["triage_case"]["semantic_extractions"], [])
        self.assertIsNone(data["triage_case"]["ttas_result"]["level_candidate"])
        self.assertIsNone(data["triage"]["urgency_score"])
        self.assertFalse(data["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(data["conversation_state"]["stage"], "collecting")
        self.assertFalse(data["conversation_state"]["confirmed"])

    async def test_unavailable_ai_does_not_guess_ai_semantic_suggestion_does_not_override_deterministic_next_question(self):
        # Former checklist-route expectation; primary free text must now fail closed.
        chat_route.create_case = self.original_create_case
        response = TestClient(app).post("/chat", json={"message": "左膝痛2週，爬樓梯很吃力，沒有胸痛呼吸困難意識不清大量出血"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["triage_case"]["semantic_extractions"], [])
        self.assertIsNone(data["triage_case"]["ttas_result"]["level_candidate"])
        self.assertIsNone(data["triage"]["urgency_score"])
        self.assertFalse(data["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(data["conversation_state"]["stage"], "collecting")
        self.assertFalse(data["conversation_state"]["confirmed"])

    async def test_full_chat_recommend_generate_script_flow(self):
        original_fetch_slots = appointment_service.fetch_available_slots
        self.addCleanup(setattr, appointment_service, "fetch_available_slots", original_fetch_slots)
        days_until_monday = (7 - date.today().weekday()) % 7 or 7
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: [
            {
                "dept_id": 7,
                "parent_dept": "外科系",
                "child_dept": "一般骨科",
                "doctor_id": "fixture-doctor",
                "doctor": "測試醫師",
                "schedule_id": "fixture-schedule",
                "date": date.today() + timedelta(days=days_until_monday),
                "session": "上午",
                "slot": "fixture-room",
                "room": "fixture-room",
                "specialty_tags": "膝關節",
                "status": "open",
                "source": "test_fixture",
                "visit_type": "初診",
            }
        ]
        client = TestClient(app)

        chat_response = self.post_accepted_description(client,
            "/chat",
            json={"message": "左膝痛2週，爬樓梯很吃力，想看一般骨科，沒有胸痛呼吸困難意識不清大量出血，週一上午可以看診", "visit_type": "initial"},
        )
        self.assertEqual(chat_response.status_code, 200)
        chat_data = chat_response.json()
        self.assertFalse(chat_data["triage"]["need_more_info"])
        self.assertTrue(chat_data["conversation_state"]["is_complete"])
        self.assertEqual(chat_data["conversation_state"]["stage"], "waiting_confirmation")

        confirm_response = client.post(
            "/chat",
            json={"case_id": chat_data["case_id"], "confirmed": True},
        )
        self.assertEqual(confirm_response.status_code, 200)

        recommend_response = client.post(
            "/recommend",
            json={"case_id": chat_data["case_id"]},
        )
        self.assertEqual(recommend_response.status_code, 200)
        recommend_data = recommend_response.json()
        specialty_first = recommend_data["recommendations"]["specialty_first"]
        time_first = recommend_data["recommendations"]["time_first"]
        self.assertGreaterEqual(len(specialty_first), 1)
        self.assertGreaterEqual(len(time_first), 1)

        script_response = client.post(
            "/generate_script",
            json={
                "case_id": chat_data["case_id"],
                "recommendation_id": specialty_first[0]["recommendation_id"],
            },
        )
        self.assertEqual(script_response.status_code, 200)
        script_data = script_response.json()
        self.assertEqual(script_data["script_id"], "vgh_booking_001")
        self.assertEqual(script_data["recommendation"]["recommendation_id"], specialty_first[0]["recommendation_id"])
        self.assertEqual(script_data["recommendation"]["doctor"], specialty_first[0]["doctor"])
        self.assertGreaterEqual(len(script_data["steps"]), 1)
        self.assertEqual(script_data["step_count"], len(script_data["steps"]))

        saved_case = get_case(chat_data["case_id"])
        self.assertIsNotNone(saved_case)
        self.assertEqual(saved_case.selected_recommendation_id, specialty_first[0]["recommendation_id"])
        self.assertEqual(saved_case.conversation_state.stage, "script_ready")


if __name__ == "__main__":
    unittest.main()
