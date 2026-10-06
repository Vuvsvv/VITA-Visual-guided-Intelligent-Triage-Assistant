from __future__ import annotations

import unittest
import importlib
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ConversationStage, DepartmentResult, TriageCase, VisitType
from app.services import appointment_service, project_smart_department_adapter, specialty_scoring
from app.services.appointment_service import recommend_appointments
from app.services.appointment_service import (
    _build_recommendations,
    _compact_recommendation_reason,
    weighted_score,
)
from app.services.case_store import get_case, save_case, save_recommendation_result
from app.services.rule_engine import apply_user_message, evaluate_urgency
from app.services.specialty_scoring import SpecialtyScore, score_doctor_deterministically


class AppointmentRankingTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_fetch_slots = appointment_service.fetch_available_slots
        self.original_fetch_departments = appointment_service.fetch_active_departments
        self.original_score = appointment_service.score_doctor_specialties
        self.original_adapter_settings = project_smart_department_adapter.get_settings
        self.original_specialty_settings = specialty_scoring.get_settings
        project_smart_department_adapter.get_settings = lambda: _NoAiSettings()
        specialty_scoring.get_settings = lambda: _NoAiSettings()
        appointment_service.fetch_active_departments = lambda: [
            {"dept_id": 1298, "parent_dept": "外科系", "child_dept": "一般骨科"},
            {"dept_id": 1302, "parent_dept": "外科系", "child_dept": "骨科"},
        ]
        self.generate_script_route = importlib.import_module("app.routes.generate_script")
        self.original_revalidate = self.generate_script_route.revalidate_schedule
        self.generate_script_route.revalidate_schedule = lambda recommendation, **_kwargs: recommendation

    def tearDown(self):
        appointment_service.fetch_available_slots = self.original_fetch_slots
        appointment_service.fetch_active_departments = self.original_fetch_departments
        appointment_service.score_doctor_specialties = self.original_score
        project_smart_department_adapter.get_settings = self.original_adapter_settings
        specialty_scoring.get_settings = self.original_specialty_settings
        self.generate_script_route.revalidate_schedule = self.original_revalidate

    async def test_specialty_first_sorting(self):
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: _ranking_rows()
        case = _complete_case()

        result = await recommend_appointments(case)

        self.assertEqual(result.recommendations.specialty_first[0].doctor, "專長高醫師")
        self.assertGreaterEqual(result.recommendations.specialty_first[0].specialty_score, 0.7)

    async def test_time_first_uses_weighted_total_before_earlier_date(self):
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: _ranking_rows()
        case = _complete_case()

        result = await recommend_appointments(case)

        self.assertEqual(result.recommendations.time_first[0].doctor, "專長高醫師")
        self.assertEqual(
            result.recommendations.specialty_first[0].recommendation_id,
            result.recommendations.time_first[0].recommendation_id.replace("rec_t_", "rec_s_"),
        )

    async def test_time_first_prefers_monday_morning(self):
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: _ranking_rows()
        case = _complete_case()

        result = await recommend_appointments(case)
        first = result.recommendations.time_first[0]

        self.assertEqual(first.doctor, "專長高醫師")
        self.assertEqual(first.date, "2026-07-27")
        self.assertEqual(first.session, "上午")
        self.assertEqual(first.time_score, 1.0)
        self.assertIn("time_score*70 + specialty_score*30", "\n".join(first.reasons))

    async def test_time_first_prefers_tomorrow_or_after_tomorrow_afternoon(self):
        tomorrow = date.today() + timedelta(days=1)
        after_tomorrow = date.today() + timedelta(days=2)
        rows = [
            _row("doc-specialty", "專長高但時間不符", (after_tomorrow + timedelta(days=3)).isoformat(), "上午", "膝關節、運動傷害"),
            _row("doc-tomorrow", "明天下午醫師", tomorrow.isoformat(), "下午", ""),
            _row("doc-after", "後天下午醫師", after_tomorrow.isoformat(), "下午", ""),
            _row("doc-session-only", "只有下午醫師", (after_tomorrow + timedelta(days=4)).isoformat(), "下午", ""),
        ]
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: rows
        case = _case_for_message("膝蓋痛兩週，明天或後天下午可以看診")

        result = await recommend_appointments(case)
        first = result.recommendations.time_first[0]

        self.assertEqual(first.doctor, "明天下午醫師")
        self.assertEqual(first.date, tomorrow.isoformat())
        self.assertEqual(first.session, "下午")
        self.assertEqual(first.time_score, 1.0)

    async def test_time_first_with_any_time_orders_by_earliest_date_and_session(self):
        rows = [
            _row("doc-late", "較晚醫師", "2026-07-22", "上午", ""),
            _row("doc-earliest-pm", "最早下午醫師", "2026-07-20", "下午", ""),
            _row("doc-earliest-am", "最早上午醫師", "2026-07-20", "上午", ""),
        ]
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: rows
        case = _complete_case()
        case.availability.preferred_days = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]
        case.availability.preferred_sessions = ["上午", "下午", "夜間"]

        result = await recommend_appointments(case)
        first = result.recommendations.time_first[0]

        self.assertEqual(first.doctor, "最早上午醫師")
        self.assertEqual(first.date, "2026-07-20")
        self.assertEqual(first.session, "上午")
        self.assertEqual(first.time_score, 1.0)

    async def test_injected_schedule_fixture_is_ranked(self):
        fixture_row = _row("fixture-doc", "測試醫師", "2026-07-13", "上午")
        fixture_row["source"] = "test_fixture"
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: [fixture_row]
        case = _complete_case()

        result = await recommend_appointments(case)

        self.assertGreaterEqual(result.total_count, 1)
        self.assertEqual(result.recommendations.specialty_first[0].doctor, "測試醫師")

    async def test_deduplicates_repeated_doctor_slot(self):
        duplicate = _row("doc-1", "專長高醫師", "2026-07-20", "上午", "膝關節")
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: [duplicate, duplicate]
        case = _complete_case()

        result = await recommend_appointments(case)

        self.assertEqual(len(result.recommendations.specialty_first), 1)

    async def test_availability_filter_sees_matching_slot_after_first_thirty_rows(self):
        today = date.today()

        def next_weekday(index: int) -> date:
            return today + timedelta(days=(index - today.weekday()) % 7 or 7)

        non_matching_date = next_weekday(0).isoformat()
        matching_date = next_weekday(4).isoformat()
        rows = [
            _row(f"doc-{index}", f"不符醫師{index}", non_matching_date, "上午")
            for index in range(30)
        ]
        rows.append(_row("doc-match", "週五下午醫師", matching_date, "下午"))
        calls = []

        def slots(*_args, **kwargs):
            calls.append(kwargs)
            max_slots = kwargs.get("max_slots")
            return rows if max_slots is None else rows[:max_slots]

        appointment_service.fetch_available_slots = slots
        case = _complete_case()
        case.availability.preferred_days = ["週五"]
        case.availability.preferred_sessions = ["下午"]

        result = await recommend_appointments(case)

        self.assertIsNone(calls[0]["max_slots"])
        self.assertEqual(calls[0]["search_days"], 21)
        self.assertEqual(
            [item.doctor for item in result.recommendations.specialty_first],
            ["週五下午醫師"],
        )

    async def test_recommendation_reasons_explain_score_bases(self):
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: _ranking_rows()
        case = _complete_case()

        result = await recommend_appointments(case)
        item = result.recommendations.specialty_first[0]
        reasons = "\n".join(item.reasons)

        self.assertIn("科別依據", reasons)
        self.assertIn("時間依據", reasons)
        self.assertIn("狀態依據", reasons)
        self.assertIn("專長依據", reasons)
        self.assertIn("排序依據", reasons)
        self.assertIn("Specialty score", reasons)
        self.assertIn("Time score", reasons)
        self.assertIn("總分", reasons)

    async def test_missing_specialty_tags_are_not_primary_basis(self):
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: [
            _row("doc-no-tags", "資料不足醫師", "2026-07-20", "上午", "")
        ]
        case = _complete_case()

        result = await recommend_appointments(case)
        item = result.recommendations.specialty_first[0]
        reasons = "\n".join(item.reasons)

        self.assertEqual(item.specialty_score, 0.5)
        self.assertIn("專長資料不足，不列為主要依據", reasons)

    async def test_compact_reason_generation_adds_no_cerebras_call(self):
        provider = AsyncMock(side_effect=AssertionError("compact reasons must not call AI"))
        original_complete = specialty_scoring.complete_prompt
        self.addCleanup(setattr, specialty_scoring, "complete_prompt", original_complete)
        specialty_scoring.complete_prompt = provider
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: _ranking_rows()

        result = await recommend_appointments(_complete_case())

        provider.assert_not_awaited()
        self.assertTrue(
            result.recommendations.specialty_first[0].reasons[0].startswith("推薦理由：")
        )

    def test_synthetic_scores_produce_distinct_specialty_and_time_orders(self):
        case = _complete_case()
        rows = [
            _row("doctor-a", "Doctor A", "2026-07-21", "上午", "膝關節、運動傷害"),
            _row("doctor-b", "Doctor B", "2026-07-20", "上午", "骨科"),
            _row("doctor-c", "Doctor C", "2026-07-21", "下午", "一般外科"),
        ]
        scores = {
            "doctor-a": SpecialtyScore("doctor-a", "Doctor A", "一般骨科", 0.95, "專長高"),
            "doctor-b": SpecialtyScore("doctor-b", "Doctor B", "一般骨科", 0.72, "專長中"),
            "doctor-c": SpecialtyScore("doctor-c", "Doctor C", "一般骨科", 0.40, "專長低"),
        }

        specialty_order = _build_recommendations(
            case, case.case_id, rows, "specialty", True, scores
        )
        time_order = _build_recommendations(
            case, case.case_id, rows, "time", False, scores
        )

        self.assertEqual([item.doctor for item in specialty_order], ["Doctor A", "Doctor B", "Doctor C"])
        self.assertEqual([item.doctor for item in time_order], ["Doctor B", "Doctor A", "Doctor C"])

    def test_specialty_first_ranking_uses_weighted_total_as_primary_key(self):
        case = _complete_case()
        rows = [
            _row("doctor-a", "Doctor A", "2026-07-20", "上午"),
            _row("doctor-b", "Doctor B", "2026-07-21", "下午"),
        ]
        scores = {
            "doctor-a": SpecialtyScore("doctor-a", "Doctor A", "一般骨科", 0.78, "A"),
            "doctor-b": SpecialtyScore("doctor-b", "Doctor B", "一般骨科", 0.79, "B"),
        }

        ranked = _build_recommendations(case, case.case_id, rows, "specialty", True, scores)

        self.assertEqual([item.doctor for item in ranked], ["Doctor A", "Doctor B"])
        self.assertEqual(ranked[0].score, 84.6)
        self.assertEqual(ranked[1].score, 65.8)
        self.assertEqual(weighted_score(0.78, 1.0, True), 84.6)

    def test_time_first_ranking_uses_weighted_total_as_primary_key(self):
        case = _complete_case()
        rows = [
            _row("doctor-a", "Doctor A", "2026-07-20", "上午"),
            _row("doctor-b", "Doctor B", "2026-07-20", "下午"),
        ]
        scores = {
            "doctor-a": SpecialtyScore("doctor-a", "Doctor A", "一般骨科", 0.40, "A"),
            "doctor-b": SpecialtyScore("doctor-b", "Doctor B", "一般骨科", 0.95, "B"),
        }

        ranked = _build_recommendations(case, case.case_id, rows, "time", False, scores)

        self.assertEqual([item.doctor for item in ranked], ["Doctor A", "Doctor B"])
        self.assertEqual(ranked[0].score, 82.0)
        self.assertEqual(ranked[1].score, 78.9)
        self.assertEqual(weighted_score(0.40, 1.0, False), 82.0)

    def test_ai_specialty_reason_is_exposed_as_match_reason(self):
        case = _complete_case()
        row = _row("doctor-ai", "AI 醫師", "2026-07-20", "上午", "膝關節")
        ai_reason = "膝部不適與此醫師的膝關節診療專長直接相關。"
        scores = {
            "doctor-ai": SpecialtyScore(
                "doctor-ai",
                "AI 醫師",
                "一般骨科",
                0.88,
                ai_reason,
                source="ai",
            )
        }

        item = _build_recommendations(case, case.case_id, [row], "specialty", True, scores)[0]

        self.assertEqual(item.match_reason, ai_reason)
        self.assertEqual(item.specialty_score, 0.88)

    def test_deterministic_specialty_reason_is_not_exposed_as_match_reason(self):
        case = _complete_case()
        row = _row("doctor-rule", "規則醫師", "2026-07-20", "上午", "膝關節")
        technical_reason = "未找到明確專長關鍵字，不列為主要依據；專長分數使用中性值 0.50"
        scores = {
            "doctor-rule": SpecialtyScore(
                "doctor-rule",
                "規則醫師",
                "一般骨科",
                0.50,
                technical_reason,
                source="deterministic",
            )
        }

        item = _build_recommendations(case, case.case_id, [row], "specialty", True, scores)[0]

        self.assertIsNone(item.match_reason)
        self.assertEqual(item.specialty_score, 0.50)
        self.assertIn(technical_reason, "\n".join(item.reasons))

    async def test_ai_timeout_keeps_deterministic_score_without_match_reason(self):
        class AiSettings:
            cerebras_api_key = "test-key"
            ai_doctor_scoring_enabled = True

        provider = AsyncMock(side_effect=TimeoutError("doctor scoring timeout"))
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: _ranking_rows()

        with patch.object(specialty_scoring, "get_settings", return_value=AiSettings()), patch.object(
            specialty_scoring,
            "complete_prompt",
            new=provider,
        ):
            result = await recommend_appointments(_complete_case())

        provider.assert_awaited_once()
        item = result.recommendations.specialty_first[0]
        self.assertGreater(item.specialty_score, 0.0)
        self.assertIsNone(item.match_reason)
        self.assertTrue(any(reason.startswith("專長依據：") for reason in item.reasons))

    def test_head_symptom_can_distinguish_neurology_specialty(self):
        case = _case_for_message("頭部不舒服")
        case.department_result = DepartmentResult(parentDept="內科系", childDept="神經內科")
        row = _row("doctor-neuro", "神經專長醫師", "2026-07-20", "上午", "腦血管疾病、腦血管剝離")
        row["parent_dept"] = "內科系"
        row["child_dept"] = "神經內科"

        result = score_doctor_deterministically(case, case.department_result, row)

        self.assertGreater(result.score, 0.5)
        self.assertIn("相關", result.reason)

    def test_compact_reason_prefers_relevant_specialty_data(self):
        case = _case_for_message("鼻塞、過敏性鼻炎反覆發作")
        slot = _row("doc-nose", "鼻科醫師", "2026-07-20", "上午")
        slot["child_dept"] = "鼻科"
        slot["specialty_tags"] = "過敏性鼻炎之診斷與治療、鼻塞、鼻竇炎之診斷與治療"

        reason = _compact_recommendation_reason(case, slot)

        self.assertEqual(reason, "醫師專長相符：過敏性鼻炎診療、鼻塞")

    def test_compact_reason_falls_back_to_preferred_time(self):
        case = _complete_case()
        slot = _row("doc-time", "時段醫師", "2026-07-20", "上午", "")

        self.assertEqual(
            _compact_recommendation_reason(case, slot),
            "符合您的方便看診時段",
        )

    def test_compact_reason_falls_back_to_department(self):
        case = _complete_case()
        case.availability.preferred_days = []
        case.availability.preferred_sessions = []
        slot = _row("doc-dept", "科別醫師", "2026-07-20", "上午", "")

        self.assertEqual(_compact_recommendation_reason(case, slot), "符合目前推薦科別")

    async def test_unbookable_selected_department_is_not_silently_remapped(self):
        calls = []

        def no_slots(*_args, **kwargs):
            calls.append(kwargs)
            return []

        appointment_service.fetch_available_slots = no_slots
        case = _complete_case()
        case.department_result = DepartmentResult(
            dept_id=1302,
            parentDept="外科系",
            childDept="骨科",
            confidence=0.9,
        )

        result = await recommend_appointments(case)

        self.assertEqual(result.total_count, 0)
        self.assertEqual(case.department_result.dept_id, 1302)
        self.assertEqual(calls[0]["department_id"], 1302)

    async def test_explicit_requested_department_overrides_old_triage_for_schedule_query(self):
        row = _row("doc-internal", "一般內科醫師", "2026-07-20", "上午", "一般內科")
        row["dept_id"] = 1232
        row["parent_dept"] = "內科系"
        row["child_dept"] = "一般內科"
        calls = []

        def slots(*_args, **kwargs):
            calls.append(kwargs)
            return [row]

        appointment_service.fetch_available_slots = slots
        case = _complete_case()
        case.department_result = DepartmentResult(
            dept_id=1501,
            parentDept="其他科",
            childDept="睡眠醫學中心",
            confidence=0.9,
        )
        case.patient_input.requested_department_id = 1232
        case.patient_input.requested_department_name = "一般內科"
        departments = [
            {"dept_id": 1232, "parent_dept": "內科系", "child_dept": "一般內科"},
            {"dept_id": 1501, "parent_dept": "其他科", "child_dept": "睡眠醫學中心"},
        ]

        with patch.object(appointment_service, "fetch_active_departments", return_value=departments):
            result = await recommend_appointments(case)

        self.assertEqual(calls[0]["department_id"], 1232)
        self.assertEqual(calls[0]["schedule_visit_type"], "初診")
        self.assertEqual(case.department_result.childDept, "一般內科")
        self.assertEqual(result.recommendations.specialty_first[0].doctor, "一般內科醫師")

    async def test_general_orthopedics_exact_id_with_missing_session_returns_allowed_schedule(self):
        today = date.today()

        def next_weekday(index: int) -> date:
            days_ahead = (index - today.weekday()) % 7 or 7
            return today + timedelta(days=days_ahead)

        allowed = _row(
            "doc-allowed",
            "可掛號骨科醫師",
            next_weekday(0).isoformat(),
            "上午",
            "膝關節、運動傷害",
        )
        excluded = _row(
            "doc-wednesday",
            "週三骨科醫師",
            next_weekday(2).isoformat(),
            "下午",
            "膝關節",
        )
        for row in (allowed, excluded):
            row["dept_id"] = 1298
        calls = []

        def slots(*_args, **kwargs):
            calls.append(kwargs)
            return [excluded, allowed]

        appointment_service.fetch_available_slots = slots
        case = _complete_case()
        case.department_result = DepartmentResult(
            dept_id=1298,
            parentDept="外科系",
            childDept="一般骨科",
            confidence=0.9,
        )
        case.availability.preferred_days = ["週一", "週二", "週四", "週五", "週六", "週日"]
        case.availability.preferred_sessions = []

        result = await recommend_appointments(case)

        self.assertEqual(calls[0]["department_id"], 1298)
        self.assertEqual(calls[0]["schedule_visit_type"], "初診")
        self.assertEqual(
            [item.doctor for item in result.recommendations.specialty_first],
            ["可掛號骨科醫師"],
        )
        self.assertNotIn(
            "週三骨科醫師",
            [item.doctor for item in result.recommendations.time_first],
        )
        self.assertTrue(
            all(item.dept_id == 1298 for item in result.recommendations.specialty_first)
        )

    async def test_recommendation_id_can_generate_script(self):
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: _ranking_rows()
        case = _complete_case()
        save_case(case)
        result = await recommend_appointments(case)
        save_recommendation_result(result)
        selected = result.recommendations.specialty_first[0]

        response = TestClient(app).post(
            "/generate_script",
            json={"case_id": case.case_id, "recommendation_id": selected.recommendation_id},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recommendation_id"], selected.recommendation_id)

    async def test_recommend_route_returns_both_priority_lists(self):
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: _ranking_rows()
        case = _complete_case()
        save_case(case)

        response = TestClient(app).post("/recommend", json={"case_id": case.case_id})

        self.assertEqual(response.status_code, 200)
        data = response.json()["recommendations"]
        self.assertIn("specialty_first", data)
        self.assertIn("time_first", data)
        self.assertEqual(data["specialty_first"][0]["doctor"], "專長高醫師")
        self.assertEqual(data["time_first"][0]["doctor"], "專長高醫師")
        self.assertEqual(
            data["specialty_first"][0]["schedule_id"],
            data["time_first"][0]["schedule_id"],
        )

    async def test_recommend_route_filters_compatible_visit_type(self):
        initial = _row("doc-initial", "初診醫師", "2026-07-20", "上午")
        initial["visit_type"] = "初診"
        followup = _row("doc-followup", "複診醫師", "2026-07-20", "上午")
        followup["visit_type"] = "複診"
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: [initial, followup]
        case = _complete_case()
        case.visit_type = VisitType.FOLLOWUP
        save_case(case)

        response = TestClient(app).post(
            "/recommend",
            json={"case_id": case.case_id, "visit_type": "followup"},
        )

        self.assertEqual(response.status_code, 200)
        items = response.json()["recommendations"]["specialty_first"]
        self.assertEqual([item["doctor"] for item in items], ["複診醫師"])
        self.assertTrue(all("複診" in item["visit_type"] for item in items))

    async def test_recommend_route_preserves_and_uses_exact_date(self):
        exact = _row("doc-exact", "指定日期醫師", "2026-09-21", "下午")
        exact["visit_type"] = "初診"
        other = _row("doc-other", "其他週一醫師", "2026-09-28", "下午")
        other["visit_type"] = "初診"
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: [other, exact]
        case = _complete_case()
        case.department_result = DepartmentResult(
            dept_id=1298,
            parentDept="外科系",
            childDept="一般骨科",
            confidence=0.9,
        )
        case.availability.preferred_dates = ["2026-09-21"]
        case.availability.preferred_days = ["週一"]
        case.availability.preferred_sessions = ["下午"]
        save_case(case)

        response = TestClient(app).post("/recommend", json={"case_id": case.case_id, "visit_type": "initial"})

        self.assertEqual(response.status_code, 200)
        items = response.json()["recommendations"]["specialty_first"]
        self.assertEqual([item["doctor"] for item in items], ["指定日期醫師"])
        self.assertEqual(get_case(case.case_id).availability.preferred_dates, ["2026-09-21"])

    async def test_recommend_route_empty_exact_date_returns_grounded_detail(self):
        only_other_date = _row("doc-other", "其他日期醫師", "2026-09-21", "下午")
        only_other_date["visit_type"] = "初診"
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: [only_other_date]
        case = _complete_case()
        case.department_result = DepartmentResult(
            dept_id=1298,
            parentDept="外科系",
            childDept="一般骨科",
            confidence=0.9,
        )
        case.availability.preferred_dates = ["2026-09-20"]
        case.availability.preferred_days = ["週日"]
        case.availability.preferred_sessions = ["下午"]
        save_case(case)

        response = TestClient(app).post("/recommend", json={"case_id": case.case_id, "visit_type": "initial"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.json()["detail"],
            "目前找不到 9 月 20 日下午符合條件的一般骨科班表。",
        )

    async def test_unknown_visit_type_is_rejected(self):
        case = _complete_case()
        save_case(case)

        response = TestClient(app).post(
            "/recommend",
            json={"case_id": case.case_id, "visit_type": "unknown"},
        )

        self.assertEqual(response.status_code, 422)

    async def test_missing_case_visit_type_is_rejected(self):
        case = _complete_case()
        case.visit_type = None
        save_case(case)

        response = TestClient(app).post("/recommend", json={"case_id": case.case_id})

        self.assertEqual(response.status_code, 422)
        self.assertIn("visit_type", response.json()["detail"])

    async def test_unconfirmed_case_is_rejected_by_route(self):
        case = _complete_case()
        case.confirmed = False
        case.conversation_state.confirmed = False
        save_case(case)

        response = TestClient(app).post("/recommend", json={"case_id": case.case_id})

        self.assertEqual(response.status_code, 400)
        self.assertIn("尚未確認", response.json()["detail"])


class _NoAiSettings:
    google_api_key = ""
    ai_doctor_scoring_enabled = False


def _complete_case() -> TriageCase:
    case = TriageCase(case_id="case_ranking", visit_type=VisitType.INITIAL)
    apply_user_message(case, "膝蓋走路疼痛2週，中等程度，慢慢變嚴重，沒有胸痛呼吸困難意識不清大量出血，週一上午可以看診")
    case.patient_input.red_flags_checked = True
    case.patient_input.red_flags_status = "negative"
    case.triage = evaluate_urgency(case)
    case.conversation_state.is_complete = True
    case.conversation_state.stage = ConversationStage.RECOMMENDING
    case.confirmed = True
    case.conversation_state.confirmed = True
    case.availability.preferred_days = ["週一"]
    case.availability.preferred_sessions = ["上午"]
    case.department_result = DepartmentResult(
        dept_id=1298,
        parentDept="外科系",
        childDept="一般骨科",
        confidence=0.8,
    )
    return case


def _case_for_message(message: str) -> TriageCase:
    case = TriageCase(case_id="case_dynamic_ranking", visit_type=VisitType.INITIAL)
    apply_user_message(case, message)
    case.triage = evaluate_urgency(case)
    case.conversation_state.is_complete = True
    case.conversation_state.stage = ConversationStage.RECOMMENDING
    case.confirmed = True
    case.conversation_state.confirmed = True
    case.department_result = DepartmentResult(
        dept_id=1298,
        parentDept="外科系",
        childDept="一般骨科",
        confidence=0.8,
    )
    return case


def _ranking_rows():
    return [
        _row("doc-1", "專長高醫師", "2026-07-27", "上午", "膝關節、運動傷害"),
        _row("doc-2", "時間優先醫師", "2026-07-20", "上午", ""),
        _row("doc-3", "一般醫師", "2026-08-03", "上午", ""),
    ]


def _row(doctor_id: str, doctor: str, date: str, session: str, specialty_tags: str = ""):
    return {
        "parent_dept": "外科系",
        "child_dept": "一般骨科",
        "doctor_id": doctor_id,
        "doctor": doctor,
        "schedule_id": f"s-{doctor_id}",
        "date": date,
        "session": session,
        "slot": "3201診",
        "room": "3201診",
        "specialty_tags": specialty_tags,
        "status": "open",
        "visit_type": "初診",
    }


if __name__ == "__main__":
    unittest.main()
