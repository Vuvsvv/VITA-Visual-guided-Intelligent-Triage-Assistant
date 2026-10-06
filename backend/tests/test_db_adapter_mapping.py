from __future__ import annotations

import unittest
from datetime import date, timedelta

from fastapi.testclient import TestClient

from app import db
from app.main import app
from app.schemas import ConversationStage, DepartmentResult, TriageCase, VisitType
from app.services import appointment_service
from app.services.case_store import save_case


class DbAdapterMappingTest(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "original_create_db_connection"):
            db.create_db_connection = self.original_create_db_connection
        if hasattr(self, "original_fetch_slots"):
            appointment_service.fetch_available_slots = self.original_fetch_slots
        if hasattr(self, "original_score"):
            appointment_service.score_doctor_specialties = self.original_score
        if hasattr(self, "original_fetch_active"):
            appointment_service.fetch_active_departments = self.original_fetch_active

    def test_fetch_active_departments_uses_category_schema(self):
        self.original_create_db_connection = db.create_db_connection
        rows = [(5, "一般內科", "心臟內科"), (7, "外科系", "一般外科")]
        conn = _FakeConnection(rows)
        db.create_db_connection = lambda: conn

        departments = db.fetch_active_departments()

        self.assertEqual(
            departments,
            [
                {"dept_id": 5, "parent_dept": "一般內科", "child_dept": "心臟內科"},
                {"dept_id": 7, "parent_dept": "外科系", "child_dept": "一般外科"},
            ],
        )
        self.assertIn("DepartmentCategory", conn.cursor_obj.query)
        self.assertIn("dc.name AS parent_dept", conn.cursor_obj.query)
        self.assertIn("d.name AS child_dept", conn.cursor_obj.query)
        self.assertIn("d.dept_id", conn.cursor_obj.query)

    def test_reference_departments_come_from_db_without_mock_fallback(self):
        self.original_create_db_connection = db.create_db_connection
        rows = [(5, "內科系", "一般內科"), (7, "外科系", "一般骨科")]
        conn = _FakeConnection(rows)
        db.create_db_connection = lambda: conn

        departments = db.fetch_reference_departments()

        self.assertEqual(
            departments,
            [
                {"dept_id": "5", "parent_dept": "內科系", "child_dept": "一般內科"},
                {"dept_id": "7", "parent_dept": "外科系", "child_dept": "一般骨科"},
            ],
        )
        self.assertIn("FROM Department d", conn.cursor_obj.query)
        self.assertIn("DepartmentCategory", conn.cursor_obj.query)
        self.assertTrue(conn.closed)

    def test_return_visit_query_uses_exact_ids_date_session_without_visit_type_filter(self):
        self.original_create_db_connection = db.create_db_connection
        conn = _FakeConnection([_return_visit_row()])
        db.create_db_connection = lambda: conn

        slots = db.fetch_return_visit_slots(
            department_name="一般骨科",
            doctor_name="原醫師",
            preferred_dates=["2026-09-07"],
            preferred_sessions=["下午"],
            department_id=7,
            doctor_id=101,
        )

        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["schedule_id"], "9001")
        self.assertEqual(slots[0]["doctor_id"], "101")
        self.assertEqual(slots[0]["dept_id"], "7")
        self.assertEqual(slots[0]["visit_type"], "初診")
        self.assertIn("s.doctor_id = selected.doctor_id", conn.cursor_obj.query)
        self.assertIn("s.dept_id = selected.dept_id", conn.cursor_obj.query)
        self.assertIn("s.date IN", conn.cursor_obj.query)
        self.assertIn("LTRIM(RTRIM(s.session)) IN", conn.cursor_obj.query)
        self.assertNotIn("s.visit_type =", conn.cursor_obj.query)
        self.assertEqual(conn.cursor_obj.params[:4], ("一般骨科", "原醫師", 7, 101))
        self.assertTrue(conn.closed)

    def test_return_visit_query_without_selected_date_uses_bounded_future_window(self):
        self.original_create_db_connection = db.create_db_connection
        future = db.datetime.now(db.TAIPEI_ZONE).date() + timedelta(days=10)
        row = list(_return_visit_row())
        row[9] = future
        conn = _FakeConnection([tuple(row)])
        db.create_db_connection = lambda: conn

        slots = db.fetch_return_visit_slots(
            department_name="一般骨科",
            doctor_name="原醫師",
            preferred_dates=[],
            preferred_sessions=["下午"],
            department_id=7,
            doctor_id=101,
        )

        self.assertEqual(len(slots), 1)
        self.assertIn("s.date >= ?", conn.cursor_obj.query)
        self.assertIn("DATEADD(day, 20, ?)", conn.cursor_obj.query)
        self.assertNotIn("s.date IN", conn.cursor_obj.query)

    def test_reference_doctors_use_exact_department_without_date_window(self):
        self.original_create_db_connection = db.create_db_connection
        conn = _FakeConnection([(12, "骨科醫師甲"), (13, "骨科醫師乙")])
        db.create_db_connection = lambda: conn

        doctors = db.fetch_reference_doctors("一般骨科")

        self.assertEqual(
            doctors,
            [
                {"doctor_id": "12", "name": "骨科醫師甲"},
                {"doctor_id": "13", "name": "骨科醫師乙"},
            ],
        )
        self.assertIn("dep.name = ?", conn.cursor_obj.query)
        self.assertNotIn("s.date", conn.cursor_obj.query)
        self.assertEqual(conn.cursor_obj.params, ("一般骨科",))
        self.assertTrue(conn.closed)

    def test_reference_doctors_can_be_constrained_by_department_id(self):
        self.original_create_db_connection = db.create_db_connection
        conn = _FakeConnection([(101, "測試醫師")])
        db.create_db_connection = lambda: conn

        doctors = db.fetch_reference_doctors("同名科", department_id=7)

        self.assertEqual(doctors, [{"doctor_id": "101", "name": "測試醫師"}])
        self.assertIn("dep.dept_id = ?", conn.cursor_obj.query)
        self.assertNotIn("dep.name = ?", conn.cursor_obj.query)
        self.assertEqual(conn.cursor_obj.params, (7,))
        self.assertTrue(conn.closed)

    def test_fetch_available_slots_maps_date_schedule_rows(self):
        self.original_create_db_connection = db.create_db_connection
        conn = _FakeConnection(
            [
                _schedule_row(schedule_id=101, visit_type="初診"),
                _schedule_row(schedule_id=102, visit_type="複診"),
                _schedule_row(schedule_id=103, status="額滿", visit_type="初診"),
            ]
        )
        db.create_db_connection = lambda: conn

        slots = db._fetch_available_slots_from_db("一般內科", search_days=21, max_slots=10)

        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["parent_dept"], "一般內科")
        self.assertEqual(slots[0]["child_dept"], "一般內科")
        self.assertEqual(slots[0]["doctor"], "測試醫師")
        self.assertEqual(slots[0]["schedule_id"], "101")
        self.assertEqual(slots[0]["date"], date(2026, 7, 25))
        self.assertEqual(slots[0]["room"], "3411")
        self.assertEqual(slots[0]["status"], "open")
        self.assertEqual(slots[0]["source"], "db")
        self.assertEqual(slots[0]["visit_type"], "初診/複診")
        self.assertEqual(conn.cursor_obj.params[0], "%一般內科%")
        self.assertEqual(conn.cursor_obj.params[2], 21)

    def test_status_null_is_recommendable(self):
        self.original_create_db_connection = db.create_db_connection
        conn = _FakeConnection([_schedule_row(status=None)])
        db.create_db_connection = lambda: conn

        slots = db._fetch_available_slots_from_db("一般內科", search_days=21, max_slots=10)

        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["status"], "open")

    def test_fetch_available_slots_uses_exact_department_id(self):
        self.original_create_db_connection = db.create_db_connection
        conn = _FakeConnection([_schedule_row()])
        db.create_db_connection = lambda: conn

        slots = db._fetch_available_slots_from_db(
            None,
            search_days=21,
            max_slots=10,
            department_id=123,
        )

        self.assertEqual(slots[0]["dept_id"], 123)
        self.assertIn("dep.dept_id = ?", conn.cursor_obj.query)
        self.assertNotIn("dep.name LIKE ?", conn.cursor_obj.query)
        self.assertIn("s.date >= ?", conn.cursor_obj.query)
        self.assertNotIn("s.date > ?", conn.cursor_obj.query)
        self.assertEqual(conn.cursor_obj.params[0], 123)

    def test_visit_type_is_filtered_in_sql_and_before_max_slots(self):
        self.original_create_db_connection = db.create_db_connection
        conn = _FakeConnection(
            [
                _schedule_row(schedule_id=301, visit_type="複診"),
                _schedule_row(schedule_id=302, visit_type="初診"),
                _schedule_row(schedule_id=303, visit_type="初診"),
            ]
        )
        db.create_db_connection = lambda: conn

        slots = db._fetch_available_slots_from_db(
            "一般內科",
            search_days=21,
            max_slots=1,
            schedule_visit_type="初診",
        )

        self.assertEqual(len(slots), 1)
        self.assertEqual(slots[0]["schedule_id"], "302")
        self.assertIn("s.visit_type = ?", conn.cursor_obj.query)
        self.assertEqual(conn.cursor_obj.params[-1], "初診")

    def test_none_max_slots_returns_all_usable_rows(self):
        self.original_create_db_connection = db.create_db_connection
        rows = [
            _schedule_row(schedule_id=400 + index, doctor_id=index, visit_type="複診")
            for index in range(35)
        ]
        conn = _FakeConnection(rows)
        db.create_db_connection = lambda: conn

        slots = db._fetch_available_slots_from_db(
            "一般內科",
            search_days=21,
            max_slots=None,
            schedule_visit_type="複診",
        )

        self.assertEqual(len(slots), 35)
        self.assertEqual(slots[-1]["schedule_id"], "434")

    def test_closed_statuses_are_not_recommendable(self):
        self.original_create_db_connection = db.create_db_connection
        conn = _FakeConnection(
            [
                _schedule_row(schedule_id=201, status="請假休診"),
                _schedule_row(schedule_id=202, status="額滿"),
                _schedule_row(schedule_id=203, status="額滿關診"),
            ]
        )
        db.create_db_connection = lambda: conn

        slots = db._fetch_available_slots_from_db("一般內科", search_days=21, max_slots=10)

        self.assertEqual(slots, [])

    def test_db_failure_raises_typed_error_without_mock_schedule(self):
        original_fetch_db = db._fetch_available_slots_from_db
        db._fetch_available_slots_from_db = lambda *_, **__: (_ for _ in ()).throw(RuntimeError("db down"))
        try:
            with self.assertRaises(db.DatabaseUnavailableError):
                db.fetch_available_slots("一般內科", max_slots=1)
        finally:
            db._fetch_available_slots_from_db = original_fetch_db


    def test_recommend_route_accepts_db_slots_without_mock_fallback(self):
        self.original_fetch_slots = appointment_service.fetch_available_slots
        self.original_fetch_active = appointment_service.fetch_active_departments
        self.original_score = appointment_service.score_doctor_specialties
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: [
            {
                "parent_dept": "一般內科",
                "child_dept": "一般內科",
                "doctor_id": "1",
                "doctor": "測試醫師",
                "schedule_id": "310",
                "date": date(2026, 7, 25),
                "session": "早上",
                "slot": "3411",
                "room": "3411",
                "specialty_tags": "",
                "status": "open",
                "source": "db",
                "visit_type": "初診/複診",
            }
        ]
        appointment_service.fetch_active_departments = lambda: [
            {"dept_id": 123, "parent_dept": "一般內科", "child_dept": "一般內科"}
        ]

        async def neutral_score(*_args, **_kwargs):
            return {}

        appointment_service.score_doctor_specialties = neutral_score
        case = _complete_case()
        save_case(case)

        response = TestClient(app).post("/recommend", json={"case_id": case.case_id, "confirmed": True})

        self.assertEqual(response.status_code, 200)
        item = response.json()["recommendations"]["specialty_first"][0]
        self.assertEqual(item["schedule_id"], "310")
        self.assertFalse(item["schedule_id"].startswith("mock_"))
        self.assertTrue(any("掛號別" in reason for reason in item["reasons"]))
        self.assertFalse(any("mock" in reason.lower() for reason in item["reasons"]))


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


def _schedule_row(
    *,
    schedule_id: int = 101,
    status: str | None = None,
    visit_type: str = "初診",
    is_placeholder: bool = False,
    doctor_id: int = 1,
):
    return (
        "一般內科",
        "一般內科",
        "測試醫師",
        "早上",
        doctor_id,
        schedule_id,
        "3411",
        "頭痛",
        status,
        date(2026, 7, 25),
        visit_type,
        is_placeholder,
        123,
    )


def _return_visit_row():
    return (
        "外科系",
        "一般骨科",
        "原醫師",
        "下午",
        101,
        9001,
        "3201診",
        "膝關節",
        "可掛號",
        date(2026, 9, 7),
        "初診",
        False,
        7,
    )


def _complete_case() -> TriageCase:
    case = TriageCase(case_id="case_db_adapter_route", visit_type=VisitType.INITIAL)
    case.patient_input.symptom = "頭暈"
    case.patient_input.red_flags_checked = True
    case.triage.need_more_info = False
    case.triage.is_final = True
    case.conversation_state.is_complete = True
    case.conversation_state.stage = ConversationStage.RECOMMENDING
    case.conversation_state.confirmed = True
    case.confirmed = True
    case.department_result = DepartmentResult(dept_id=123, parentDept="一般內科", childDept="一般內科", confidence=1.0)
    return case


if __name__ == "__main__":
    unittest.main()
