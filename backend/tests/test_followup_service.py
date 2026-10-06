from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.db import DatabaseUnavailableError
from app.main import app
from app.schemas import DepartmentResult, TriageCase
from app.services import followup_service
from app.services import quick_search_service
from app.services.case_store import save_case


class FollowupServiceTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_fetch_slots = followup_service.fetch_return_visit_slots
        self.original_fetch_departments = followup_service.fetch_active_departments
        self.original_fetch_doctors = followup_service.fetch_reference_doctors
        followup_service.fetch_active_departments = lambda: [
            {"dept_id": 7, "parent_dept": "外科系", "child_dept": "一般骨科"},
            {"dept_id": 8, "parent_dept": "Surgery", "child_dept": "Orthopedics"},
        ]
        names = ["原醫師", "回診醫師", "指定日期醫師", "複診醫師", "醫師A", "Original Doctor"]
        followup_service.fetch_reference_doctors = lambda _department, **_kwargs: [
            {"doctor_id": "101", "name": name} for name in names
        ]

    def tearDown(self):
        followup_service.fetch_return_visit_slots = self.original_fetch_slots
        followup_service.fetch_active_departments = self.original_fetch_departments
        followup_service.fetch_reference_doctors = self.original_fetch_doctors

    def test_original_doctor_available(self):
        followup_service.fetch_return_visit_slots = lambda *_args, **_kwargs: [
            _row("原醫師", "2026-07-20", "上午"),
            _row("替代醫師", "2026-07-21", "下午"),
        ]

        response = TestClient(app).post(
            "/followup/recommend",
            json={"childDept": "一般骨科", "parentDept": "外科系", "original_doctor": "原醫師"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recommendations"][0]["doctor"], "原醫師")

    def test_original_doctor_unavailable_does_not_use_same_department_alternative(self):
        followup_service.fetch_return_visit_slots = lambda *_args, **_kwargs: [_row("替代醫師", "2026-07-21", "下午")]

        response = TestClient(app).post(
            "/followup/recommend",
            json={"childDept": "一般骨科", "parentDept": "外科系", "original_doctor": "原醫師"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recommendations"], [])
        self.assertEqual(response.json()["fallback_departments"], [])

    def test_missing_case(self):
        response = TestClient(app).post(
            "/followup/recommend",
            json={"case_id": "case_missing", "original_doctor": "原醫師"},
        )

        self.assertEqual(response.status_code, 404)

    def test_missing_required_fields(self):
        response = TestClient(app).post("/followup/recommend", json={})

        self.assertEqual(response.status_code, 422)

    def test_missing_original_doctor_is_rejected(self):
        client = TestClient(app)

        missing = client.post(
            "/followup/recommend",
            json={"childDept": "一般骨科", "parentDept": "外科系"},
        )
        blank = client.post(
            "/followup/recommend",
            json={
                "childDept": "一般骨科",
                "parentDept": "外科系",
                "original_doctor": "   ",
            },
        )

        self.assertEqual(missing.status_code, 422)
        self.assertEqual(blank.status_code, 422)

    def test_db_unavailable_returns_503_without_mock_doctor(self):
        followup_service.fetch_return_visit_slots = lambda *_args, **_kwargs: (_ for _ in ()).throw(DatabaseUnavailableError("db down"))

        response = TestClient(app).post(
            "/followup/recommend",
            json={"childDept": "一般骨科", "parentDept": "外科系", "original_doctor": "原醫師"},
        )

        self.assertEqual(response.status_code, 503)

    def test_empty_result(self):
        followup_service.fetch_return_visit_slots = lambda *_args, **_kwargs: []

        response = TestClient(app).post(
            "/followup/recommend",
            json={"childDept": "一般骨科", "parentDept": "外科系", "original_doctor": "原醫師"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recommendations"], [])
        self.assertEqual(response.json()["fallback_departments"], [])

    def test_no_matching_availability_returns_empty_instead_of_open_rows(self):
        followup_service.fetch_return_visit_slots = lambda *_args, **_kwargs: [
            _row("複診醫師", "2026-07-13", "上午")
        ]

        response = TestClient(app).post(
            "/followup/recommend",
            json={
                "childDept": "一般骨科",
                "parentDept": "外科系",
                "original_doctor": "複診醫師",
                "availability": {
                    "preferred_days": ["週二"],
                    "preferred_sessions": ["晚上"],
                    "can_take_leave": False,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recommendations"], [])
        self.assertEqual(response.json()["fallback_departments"], [])

    def test_return_visit_does_not_filter_by_schedule_visit_type(self):
        initial = _row("原醫師", "2026-07-20", "上午")
        initial["visit_type"] = "初診"
        followup = _row("複診醫師", "2026-07-21", "下午")
        blank = _row("空白類型醫師", "2026-07-22", "上午")
        blank["visit_type"] = ""
        followup_service.fetch_return_visit_slots = lambda *_args, **_kwargs: [initial, followup, blank]

        response = TestClient(app).post(
            "/followup/recommend",
            json={
                "childDept": "一般骨科",
                "parentDept": "外科系",
                "original_doctor": "原醫師",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["doctor"] for item in response.json()["recommendations"]],
            ["原醫師"],
        )
        self.assertEqual(response.json()["recommendations"][0]["visit_type"], "初診")

    def test_preferred_date_does_not_match_another_same_weekday(self):
        followup_service.fetch_return_visit_slots = lambda *_args, **_kwargs: [
            _row("指定日期醫師", "2026-08-24", "下午"),
            _row("其他週一醫師", "2026-08-31", "下午"),
        ]

        response = TestClient(app).post(
            "/followup/recommend",
            json={
                "childDept": "一般骨科",
                "parentDept": "外科系",
                "original_doctor": "指定日期醫師",
                "availability": {
                    "preferred_dates": ["2026-08-24"],
                    "preferred_sessions": ["下午"],
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["date"] for item in response.json()["recommendations"]],
            ["2026-08-24"],
        )

    def test_original_doctor_without_requested_date_slot_returns_empty(self):
        followup_service.fetch_return_visit_slots = lambda *_args, **_kwargs: [
            _row("醫師A", "2026-08-31", "下午"),
            _row("醫師B", "2026-08-24", "下午"),
        ]

        response = TestClient(app).post(
            "/followup/recommend",
            json={
                "childDept": "一般骨科",
                "parentDept": "外科系",
                "original_doctor": "醫師A",
                "availability": {
                    "preferred_dates": ["2026-08-24"],
                    "preferred_sessions": ["下午"],
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recommendations"], [])

    def test_return_visit_requires_exact_doctor_department_date_and_session(self):
        exact = _row("原醫師", "2026-09-07", "下午")
        exact.update(
            doctor_id="101",
            dept_id="7",
            schedule_id="exact-schedule",
            visit_type="初診",
        )
        wrong_doctor_id = {**exact, "doctor_id": "102", "schedule_id": "wrong-doctor"}
        wrong_dept_id = {**exact, "dept_id": "8", "schedule_id": "wrong-department"}
        wrong_date = {**exact, "date": "2026-09-14", "schedule_id": "wrong-date"}
        wrong_session = {**exact, "session": "上午", "schedule_id": "wrong-session"}
        followup_service.fetch_return_visit_slots = lambda *_args, **_kwargs: [
            wrong_doctor_id,
            wrong_dept_id,
            wrong_date,
            wrong_session,
            exact,
        ]

        response = TestClient(app).post(
            "/followup/recommend",
            json={
                "childDept": "一般骨科",
                "dept_id": 7,
                "original_doctor": "原醫師",
                "original_doctor_id": 101,
                "availability": {
                    "preferred_dates": ["2026-09-07"],
                    "preferred_sessions": ["下午"],
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["schedule_id"] for item in response.json()["recommendations"]],
            ["exact-schedule"],
        )
        self.assertEqual(response.json()["recommendations"][0]["doctor_id"], "101")
        self.assertEqual(response.json()["recommendations"][0]["visit_type"], "初診")

    def test_original_doctor_after_first_thirty_rows_is_still_found(self):
        rows = [
            _row(f"其他醫師{index}", "2026-08-24", "下午")
            for index in range(30)
        ] + [_row("原醫師", "2026-08-24", "下午")]

        def fetch_without_early_limit(*_args, **kwargs):
            self.assertNotIn("schedule_visit_type", kwargs)
            self.assertEqual(kwargs["preferred_dates"], ["2026-08-24"])
            return rows

        followup_service.fetch_return_visit_slots = fetch_without_early_limit

        response = TestClient(app).post(
            "/followup/recommend",
            json={
                "childDept": "一般骨科",
                "parentDept": "外科系",
                "original_doctor": "原醫師",
                "availability": {
                    "preferred_dates": ["2026-08-24"],
                    "preferred_sessions": ["下午"],
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["doctor"] for item in response.json()["recommendations"]],
            ["原醫師"],
        )

    def test_followup_recommendation_can_generate_script(self):
        followup_service.fetch_return_visit_slots = lambda *_args, **_kwargs: [
            _row("回診醫師", "2026-08-24", "下午")
        ]
        client = TestClient(app)

        recommend_response = client.post(
            "/followup/recommend",
            json={
                "childDept": "一般骨科",
                "parentDept": "外科系",
                "original_doctor": "回診醫師",
                "availability": {"preferred_dates": ["2026-08-24"]},
            },
        )
        self.assertEqual(recommend_response.status_code, 200)
        recommend_data = recommend_response.json()
        recommendation = recommend_data["recommendations"][0]
        self.assertEqual(recommendation["doctor"], "回診醫師")

        original_fetch = quick_search_service.fetch_quick_search_slot_by_id
        quick_search_service.fetch_quick_search_slot_by_id = lambda _schedule_id: {
            "parent_dept": "外科系", "child_dept": "一般骨科", "doctor": "回診醫師",
            "doctor_id": "101", "schedule_id": recommendation["schedule_id"], "dept_id": "7",
            "date": recommendation["date"], "session": recommendation["session"],
            "room": "3201診", "slot": "3201診", "status": "open", "is_placeholder": False,
            "supports_followup": True, "visit_type": "複診", "specialty_tags": "膝關節",
        }
        try:
            script_response = client.post(
                "/generate_script",
                json={
                    "case_id": recommend_data["case_id"],
                    "recommendation_id": recommendation["recommendation_id"],
                },
            )
        finally:
            quick_search_service.fetch_quick_search_slot_by_id = original_fetch

        self.assertEqual(script_response.status_code, 200)
        self.assertTrue(script_response.json()["isSuccess"])
        self.assertEqual(
            script_response.json()["recommendation_id"],
            recommendation["recommendation_id"],
        )

    def test_case_department_can_seed_followup(self):
        case = TriageCase(case_id="case_followup")
        case.department_result = DepartmentResult(parentDept="外科系", childDept="一般骨科")
        save_case(case)
        followup_service.fetch_return_visit_slots = lambda *_args, **_kwargs: [_row("原醫師", "2026-07-20", "上午")]

        response = TestClient(app).post(
            "/followup/recommend",
            json={"case_id": "case_followup", "original_doctor": "原醫師"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["department"]["childDept"], "一般骨科")


    def test_end_to_end_followup_recommendation_route(self):
        case = TriageCase(case_id="case_followup_e2e")
        case.department_result = DepartmentResult(parentDept="Surgery", childDept="Orthopedics")
        save_case(case)
        followup_service.fetch_return_visit_slots = lambda *_args, **_kwargs: [
            {
                "parent_dept": "Surgery",
                "child_dept": "Orthopedics",
                "doctor_id": "101",
                "doctor": "Original Doctor",
                "schedule_id": "followup-1",
                "date": "2026-07-20",
                "session": "morning",
                "slot": "Room 3201",
                "room": "Room 3201",
                "specialty_tags": "knee follow-up",
                "status": "open",
                "source": "db",
                "visit_type": "複診",
                "dept_id": "8",
            },
            {
                "parent_dept": "Surgery",
                "child_dept": "Orthopedics",
                "doctor_id": "102",
                "doctor": "Same Dept Doctor",
                "schedule_id": "followup-2",
                "date": "2026-07-21",
                "session": "afternoon",
                "slot": "Room 3202",
                "room": "Room 3202",
                "specialty_tags": "knee follow-up",
                "status": "open",
                "source": "db",
                "visit_type": "複診",
                "dept_id": "8",
            },
        ]

        response = TestClient(app).post(
            "/followup/recommend",
            json={
                "case_id": "case_followup_e2e",
                "original_doctor": "Original Doctor",
                "followup_reason": "post-visit follow-up",
            },
        )

        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["case_id"], "case_followup_e2e")
        self.assertEqual(data["department"]["childDept"], "Orthopedics")
        self.assertGreaterEqual(len(data["recommendations"]), 1)
        self.assertEqual(data["recommendations"][0]["doctor"], "Original Doctor")


def _row(doctor: str, date: str, session: str, source: str = "db"):
    return {
        "parent_dept": "外科系",
        "child_dept": "一般骨科",
        "doctor_id": "101",
        "doctor": doctor,
        "schedule_id": f"s-{doctor}",
        "date": date,
        "session": session,
        "slot": "3201診",
        "room": "3201診",
        "specialty_tags": "膝關節",
        "status": "open",
        "source": source,
        "visit_type": "複診",
        "dept_id": "7",
    }


if __name__ == "__main__":
    unittest.main()
