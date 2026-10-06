from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import db
from app.main import app


client = TestClient(app)
DEPARTMENT = {"dept_id": "7", "parent_dept": "外科系", "child_dept": "一般骨科"}
TARGET_DATE = "2099-09-20"


class QuickSearchRouteTest(unittest.TestCase):
    def _search(self, period: str, db_session: str):
        with patch("app.routes.schedules.fetch_reference_departments", return_value=[DEPARTMENT]), patch(
            "app.services.quick_search_service.fetch_quick_search_slots",
            return_value=[_slot(session=db_session)],
        ) as fetch_slots:
            response = client.get(
                "/schedules/search",
                params={"dept_id": 7, "date": TARGET_DATE, "period": period},
            )
        self.assertEqual(fetch_slots.call_args.kwargs["department_id"], 7)
        self.assertEqual(fetch_slots.call_args.kwargs["target_date"], date(2099, 9, 20))
        self.assertEqual(fetch_slots.call_args.kwargs["period"], period)
        return response

    def test_morning_search_returns_real_schedule(self):
        response = self._search("morning", "上午")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json()["dept_id"], int)
        self.assertEqual(response.json()["dept_id"], 7)
        self.assertEqual(response.json()["results"][0]["session"], "上午")

    def test_afternoon_search_returns_real_schedule(self):
        response = self._search("afternoon", "午診")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"][0]["session"], "下午")

    def test_evening_search_returns_real_schedule(self):
        for db_session in ("夜診", "夜間"):
            with self.subTest(db_session=db_session):
                response = self._search("evening", db_session)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["results"][0]["session"], "晚上")

    def test_no_schedule_returns_empty_results(self):
        with patch("app.routes.schedules.fetch_reference_departments", return_value=[DEPARTMENT]), patch(
            "app.services.quick_search_service.fetch_quick_search_slots",
            return_value=[],
        ):
            response = client.get(
                "/schedules/search",
                params={"dept_id": 7, "date": TARGET_DATE, "period": "morning"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"], [])
        self.assertEqual(response.json()["total_count"], 0)

    def test_visit_type_rows_for_same_logical_slot_return_one_card(self):
        with patch("app.routes.schedules.fetch_reference_departments", return_value=[DEPARTMENT]), patch(
            "app.services.quick_search_service.fetch_quick_search_slots",
            return_value=[
                _slot(schedule_id="51215", visit_type="初診"),
                _slot(schedule_id="47511", visit_type="複診"),
            ],
        ):
            response = client.get(
                "/schedules/search",
                params={"dept_id": 7, "date": TARGET_DATE, "period": "morning"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total_count"], 1)
        self.assertEqual(len(response.json()["results"]), 1)
        self.assertEqual(response.json()["results"][0]["schedule_id"], "47511")
        self.assertEqual(response.json()["results"][0]["visit_type"], "複診")

    def test_initial_only_rows_do_not_fallback_into_quick_search(self):
        with patch("app.routes.schedules.fetch_reference_departments", return_value=[DEPARTMENT]), patch(
            "app.services.quick_search_service.fetch_quick_search_slots",
            return_value=[_slot(schedule_id="51215", visit_type="初診", supports_followup=False)],
        ):
            response = client.get(
                "/schedules/search",
                params={"dept_id": 7, "date": TARGET_DATE, "period": "morning"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total_count"], 0)
        self.assertEqual(response.json()["results"], [])

    def test_available_shared_slot_is_not_removed_by_visit_type_label(self):
        with patch("app.routes.schedules.fetch_reference_departments", return_value=[DEPARTMENT]), patch(
            "app.services.quick_search_service.fetch_quick_search_slots",
            return_value=[
                _slot(schedule_id="51356", visit_type="初診", supports_followup=True),
            ],
        ):
            response = client.get(
                "/schedules/search",
                params={"dept_id": 7, "date": TARGET_DATE, "period": "morning"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total_count"], 1)
        self.assertEqual(response.json()["results"][0]["schedule_id"], "51356")
        self.assertEqual(response.json()["results"][0]["visit_type"], "初診")

    def test_invalid_department_is_rejected_before_schedule_query(self):
        with patch("app.routes.schedules.fetch_reference_departments", return_value=[DEPARTMENT]), patch(
            "app.services.quick_search_service.fetch_quick_search_slots"
        ) as fetch_slots:
            response = client.get(
                "/schedules/search",
                params={"dept_id": 999, "date": TARGET_DATE, "period": "morning"},
            )

        self.assertEqual(response.status_code, 404)
        fetch_slots.assert_not_called()

    def test_invalid_date_is_rejected(self):
        response = client.get(
            "/schedules/search",
            params={"dept_id": 7, "date": "09/20/2099", "period": "morning"},
        )
        self.assertEqual(response.status_code, 422)

    def test_invalid_period_is_rejected(self):
        response = client.get(
            "/schedules/search",
            params={"dept_id": 7, "date": TARGET_DATE, "period": "night"},
        )
        self.assertEqual(response.status_code, 422)

    def test_quick_search_does_not_call_ai_or_return_visit_recommendation(self):
        with patch("app.routes.schedules.fetch_reference_departments", return_value=[DEPARTMENT]), patch(
            "app.services.quick_search_service.fetch_quick_search_slots",
            return_value=[_slot()],
        ), patch("app.services.ai_service.complete_prompt") as ai_call, patch(
            "app.services.followup_service.recommend_followup"
        ) as followup_call:
            response = client.get(
                "/schedules/search",
                params={"dept_id": 7, "date": TARGET_DATE, "period": "morning"},
            )

        self.assertEqual(response.status_code, 200)
        ai_call.assert_not_called()
        followup_call.assert_not_called()

    def test_quick_search_does_not_create_or_persist_case_or_recommendations(self):
        with patch("app.routes.schedules.fetch_reference_departments", return_value=[DEPARTMENT]), patch(
            "app.services.quick_search_service.fetch_quick_search_slots",
            return_value=[_slot()],
        ), patch("app.services.case_store.create_case") as create_case, patch(
            "app.services.case_store.save_case"
        ) as save_case, patch("app.services.case_store.save_recommendations") as save_recommendations:
            response = client.get(
                "/schedules/search",
                params={"dept_id": 7, "date": TARGET_DATE, "period": "morning"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["case_id"].startswith("quick_"))
        create_case.assert_not_called()
        save_case.assert_not_called()
        save_recommendations.assert_not_called()

    def test_selected_quick_schedule_uses_existing_generate_script_flow(self):
        with patch("app.routes.schedules.fetch_reference_departments", return_value=[DEPARTMENT]), patch(
            "app.services.quick_search_service.fetch_quick_search_slots",
            return_value=[_slot()],
        ), patch(
            "app.services.quick_search_service.fetch_quick_search_slot_by_id",
            return_value=_slot(),
        ), patch("app.services.ai_service.complete_prompt") as ai_call, patch(
            "app.services.followup_service.recommend_followup"
        ) as followup_call:
            search_response = client.get(
                "/schedules/search",
                params={"dept_id": 7, "date": TARGET_DATE, "period": "morning"},
            )
            selected = search_response.json()["results"][0]
            selected["doctor"] = "Android 傳入的錯誤醫師名稱"
            selected["room"] = "Android 傳入的錯誤診間"
            selected["slot"] = "Android 傳入的錯誤診間"
            script_response = client.post(
                "/generate_script",
                json={
                    "case_id": search_response.json()["case_id"],
                    "recommendation_id": selected["recommendation_id"],
                    "recommendation": selected,
                },
            )

        self.assertEqual(search_response.status_code, 200)
        self.assertEqual(script_response.status_code, 200)
        self.assertTrue(script_response.json()["isSuccess"])
        self.assertGreater(len(script_response.json()["steps"]), 0)
        returned = script_response.json()["recommendation"]
        self.assertEqual(returned["schedule_id"], "9001")
        self.assertEqual(returned["doctor_id"], "101")
        self.assertEqual(returned["dept_id"], 7)
        self.assertEqual(returned["date"], TARGET_DATE)
        self.assertEqual(returned["session"], "上午")
        self.assertEqual(returned["doctor"], "測試醫師")
        self.assertEqual(returned["room"], "3201診")
        ai_call.assert_not_called()
        followup_call.assert_not_called()

    def test_shared_physical_slot_can_pass_generate_script_revalidation(self):
        selected = _stateless_recommendation()
        selected.update(
            recommendation_id="qs_quick_abcd1234_51356",
            schedule_id="51356",
            visit_type="初診",
        )
        with patch(
            "app.services.quick_search_service.fetch_quick_search_slot_by_id",
            return_value=_slot(
                schedule_id="51356",
                visit_type="初診",
                supports_followup=True,
            ),
        ):
            response = client.post(
                "/generate_script",
                json={
                    "case_id": "quick_abcd1234",
                    "recommendation_id": "qs_quick_abcd1234_51356",
                    "recommendation": selected,
                },
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["isSuccess"])
        self.assertEqual(response.json()["recommendation"]["schedule_id"], "51356")
        self.assertEqual(response.json()["recommendation"]["visit_type"], "初診")

    def test_stateless_script_rejects_schedule_identity_mismatches(self):
        mutations = {
            "missing schedule": {"schedule_id": "9999"},
            "doctor mismatch": {"doctor_id": "999"},
            "department mismatch": {"dept_id": 8},
            "date mismatch": {"date": "2099-09-21"},
            "session mismatch": {"session": "下午"},
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), patch(
                "app.services.quick_search_service.fetch_quick_search_slot_by_id",
                return_value=None if name == "missing schedule" else _slot(),
            ):
                selected = _stateless_recommendation()
                selected.update(mutation)
                response = client.post(
                    "/generate_script",
                    json={
                        "case_id": "quick_abcd1234",
                        "recommendation_id": "qs_quick_abcd1234_9001",
                        "recommendation": selected,
                    },
                )
            self.assertEqual(response.status_code, 409, response.text)

    def test_stateless_script_rechecks_availability_and_db_errors(self):
        for name, row in {
            "closed status": _slot(status="額滿"),
            "placeholder doctor": _slot(is_placeholder=True),
            "initial-only schedule": _slot(visit_type="初診", supports_followup=False),
        }.items():
            with self.subTest(name=name), patch(
                "app.services.quick_search_service.fetch_quick_search_slot_by_id",
                return_value=row,
            ):
                response = client.post(
                    "/generate_script",
                    json={
                        "case_id": "quick_abcd1234",
                        "recommendation_id": "qs_quick_abcd1234_9001",
                        "recommendation": _stateless_recommendation(),
                    },
                )
            self.assertEqual(response.status_code, 409, response.text)

        with patch(
            "app.services.quick_search_service.fetch_quick_search_slot_by_id",
            side_effect=RuntimeError("db unavailable"),
        ):
            response = client.post(
                "/generate_script",
                json={
                    "case_id": "quick_abcd1234",
                    "recommendation_id": "qs_quick_abcd1234_9001",
                    "recommendation": _stateless_recommendation(),
                },
            )
        self.assertEqual(response.status_code, 503)

    def test_db_exception_is_503_without_mock_results(self):
        with patch("app.routes.schedules.fetch_reference_departments", return_value=[DEPARTMENT]), patch(
            "app.services.quick_search_service.fetch_quick_search_slots",
            side_effect=RuntimeError("db unavailable"),
        ):
            response = client.get(
                "/schedules/search",
                params={"dept_id": 7, "date": TARGET_DATE, "period": "morning"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("results", response.json())
        self.assertNotIn("mock", response.text.lower())

    def test_quick_search_is_rejected_by_chat_and_recommend_pipelines(self):
        with patch("app.services.ai_service.complete_prompt") as ai_call, patch(
            "app.services.followup_service.recommend_followup"
        ) as followup_call:
            chat_response = client.post(
                "/chat",
                json={"case_id": "case_quick_chat_guard", "visit_type": "quick_search"},
            )
            recommend_response = client.post(
                "/recommend",
                json={
                    "visit_type": "quick_search",
                    "confirmed": True,
                    "triage_case": {
                        "case_id": "case_quick_recommend_guard",
                        "visit_type": "quick_search",
                        "triage": {"need_more_info": False},
                        "conversation_state": {"is_complete": True, "confirmed": True},
                        "confirmed": True,
                    },
                },
            )

        self.assertEqual(chat_response.status_code, 400)
        self.assertEqual(recommend_response.status_code, 400)
        ai_call.assert_not_called()
        followup_call.assert_not_called()


class QuickSearchDbAdapterTest(unittest.TestCase):
    def test_query_uses_exact_department_date_period_and_existing_availability_rules(self):
        connection = _FakeConnection([_quick_search_row()])
        with patch("app.db.create_db_connection", return_value=connection):
            slots = db.fetch_quick_search_slots(
                department_id=7,
                target_date=date(2099, 9, 20),
                period="evening",
            )

        self.assertEqual([slot["schedule_id"] for slot in slots], ["9001"])
        self.assertEqual(slots[0]["dept_id"], "7")
        self.assertIn("dep.dept_id = ?", connection.cursor_obj.query)
        self.assertIn("s.date = ?", connection.cursor_obj.query)
        self.assertIn("doc.is_active = 1", connection.cursor_obj.query)
        self.assertIn("doc.is_placeholder = 0", connection.cursor_obj.query)
        self.assertIn("s.status IS NULL", connection.cursor_obj.query)
        self.assertIn("LTRIM(RTRIM(s.session)) IN", connection.cursor_obj.query)
        self.assertIn("AS supports_followup", connection.cursor_obj.query)
        self.assertIn("FROM Schedule followup", connection.cursor_obj.query)
        self.assertEqual(connection.cursor_obj.params[:3], ("複診", 7, date(2099, 9, 20)))
        self.assertIn("夜診", connection.cursor_obj.params)
        self.assertTrue(connection.closed)

    def test_revalidation_query_uses_schedule_id_and_availability_rules(self):
        connection = _FakeConnection([_quick_search_row()])
        with patch("app.db.create_db_connection", return_value=connection):
            slot = db.fetch_quick_search_slot_by_id("9001")

        self.assertIsNotNone(slot)
        self.assertEqual(slot["schedule_id"], "9001")
        self.assertIn("s.schedule_id = ?", connection.cursor_obj.query)
        self.assertIn("doc.is_active = 1", connection.cursor_obj.query)
        self.assertIn("doc.is_placeholder = 0", connection.cursor_obj.query)
        self.assertIn("s.status IS NULL", connection.cursor_obj.query)
        self.assertIn("AS supports_followup", connection.cursor_obj.query)
        self.assertEqual(connection.cursor_obj.params, ("複診", "9001"))
        self.assertTrue(connection.closed)


def _slot(
    session: str = "上午",
    *,
    status: str = "open",
    is_placeholder: bool = False,
    schedule_id: str = "9001",
    visit_type: str = "複診",
    supports_followup: bool = True,
) -> dict:
    return {
        "parent_dept": "外科系",
        "child_dept": "一般骨科",
        "dept_id": "7",
        "doctor_id": "101",
        "doctor": "測試醫師",
        "schedule_id": schedule_id,
        "date": TARGET_DATE,
        "session": session,
        "slot": "3201診",
        "room": "3201診",
        "specialty_tags": "膝關節",
        "status": status,
        "is_placeholder": is_placeholder,
        "source": "db",
        "visit_type": visit_type,
        "supports_followup": supports_followup,
    }


def _stateless_recommendation() -> dict:
    return {
        "recommendation_id": "qs_quick_abcd1234_9001",
        "parentDept": "外科系",
        "childDept": "一般骨科",
        "doctor": "測試醫師",
        "doctor_id": "101",
        "schedule_id": "9001",
        "dept_id": 7,
        "date": TARGET_DATE,
        "session": "上午",
        "session_time": "08:30-12:00",
        "room": "3201診",
        "slot": "3201診",
        "visit_type": "複診",
        "score": 0.0,
        "reasons": ["符合指定條件"],
    }


def _quick_search_row():
    return (
        "外科系",
        "一般骨科",
        "測試醫師",
        "夜診",
        101,
        9001,
        "3201診",
        "膝關節",
        "可掛號",
        date(2099, 9, 20),
        "複診",
        False,
        7,
        True,
    )


class _FakeConnection:
    def __init__(self, rows):
        self.cursor_obj = _FakeCursor(rows)
        self.closed = False

    def cursor(self):
        return self.cursor_obj

    def close(self):
        self.closed = True


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = ""
        self.params = ()

    def execute(self, query, *params):
        self.query = query
        self.params = params

    def fetchall(self):
        return self.rows


if __name__ == "__main__":
    unittest.main()
