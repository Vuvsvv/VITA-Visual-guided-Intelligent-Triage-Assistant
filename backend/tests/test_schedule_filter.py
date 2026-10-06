from __future__ import annotations

import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from app.schemas import Availability
from app.services.schedule_filter import (
    availability_from_project_smart_slots,
    filter_rows_by_availability,
    normalize_session,
    row_matches_availability,
    select_feasible_rows,
    session_is_open,
    session_range,
    weekday_label,
)


class ScheduleFilterTest(unittest.TestCase):
    def setUp(self):
        self.before_rows = datetime(2026, 7, 12, 9, 0)

    def test_weekday_and_session_helpers(self):
        self.assertEqual(weekday_label("2026-07-13"), "週一")
        self.assertEqual(normalize_session("早上"), "上午")
        self.assertEqual(normalize_session("夜診"), "晚上")
        self.assertEqual(session_range("下午"), "13:30-17:00")

    def test_project_smart_slots_adapter(self):
        availability = availability_from_project_smart_slots(
            [{"day": "週一", "session": "早上"}, {"day": "週三", "session": "夜診"}],
            can_take_leave=True,
        )

        self.assertEqual(availability.preferred_days, ["週一", "週三"])
        self.assertEqual(availability.preferred_sessions, ["上午", "晚上"])
        self.assertTrue(availability.can_take_leave)

    def test_specified_weekday_and_morning_match(self):
        availability = Availability(preferred_days=["週一"], preferred_sessions=["上午"])
        self.assertTrue(row_matches_availability(_row("2026-07-13", "上午"), availability))
        self.assertFalse(row_matches_availability(_row("2026-07-14", "上午"), availability))
        self.assertFalse(row_matches_availability(_row("2026-07-13", "下午"), availability))

    def test_explicit_date_takes_priority_over_same_weekday(self):
        availability = Availability(
            preferred_dates=["2026-09-20"],
            preferred_days=["週日"],
            preferred_sessions=["下午"],
        )

        self.assertTrue(row_matches_availability(_row("2026-09-20", "下午"), availability))
        self.assertFalse(row_matches_availability(_row("2026-09-27", "下午"), availability))

    def test_today_expired_session_is_filtered(self):
        now = datetime(2026, 7, 13, 16, 45)
        self.assertFalse(session_is_open(_row("2026-07-13", "上午"), now=now))
        self.assertTrue(session_is_open(_row("2026-07-13", "晚上"), now=now))
        self.assertTrue(session_is_open(_row("2026-07-14", "上午"), now=now))

    def test_today_fixed_cutoff_boundaries(self):
        day = "2026-07-13"
        cases = [
            ("上午", datetime(2026, 7, 13, 9, 59), True),
            ("上午", datetime(2026, 7, 13, 10, 0), False),
            ("下午", datetime(2026, 7, 13, 14, 59), True),
            ("下午", datetime(2026, 7, 13, 15, 0), False),
            ("晚上", datetime(2026, 7, 13, 18, 59), True),
            ("晚上", datetime(2026, 7, 13, 19, 0), False),
        ]
        for session, now, expected in cases:
            with self.subTest(session=session, now=now):
                self.assertEqual(session_is_open(_row(day, session), now=now), expected)

    def test_future_sessions_ignore_today_cutoffs(self):
        now = datetime(2026, 7, 13, 20, 0, tzinfo=ZoneInfo("Asia/Taipei"))

        for session in ("上午", "下午", "晚上"):
            with self.subTest(session=session):
                self.assertTrue(session_is_open(_row("2026-07-14", session), now=now))

    def test_taipei_date_boundary_converts_aware_clock(self):
        taipei_midnight = datetime(2026, 7, 12, 16, 0, tzinfo=timezone.utc)
        taipei_morning_cutoff = datetime(2026, 7, 13, 2, 0, tzinfo=timezone.utc)

        self.assertTrue(session_is_open(_row("2026-07-13", "上午"), now=taipei_midnight))
        self.assertFalse(session_is_open(_row("2026-07-13", "上午"), now=taipei_morning_cutoff))

    def test_no_matching_slot_without_leave_returns_empty(self):
        availability = Availability(preferred_days=["週二"], preferred_sessions=["晚上"], can_take_leave=False)
        rows = [_row("2026-07-13", "上午"), _row("2026-07-15", "下午")]

        self.assertEqual(filter_rows_by_availability(rows, availability, now=self.before_rows), [])

    def test_can_take_leave_relaxes_when_no_preference_match(self):
        availability = Availability(preferred_days=["週二"], preferred_sessions=["晚上"], can_take_leave=True)
        rows = [_row("2026-07-13", "上午"), _row("2026-07-15", "下午")]

        self.assertEqual(filter_rows_by_availability(rows, availability, now=self.before_rows), rows)

    def test_select_feasible_prefers_requested_doctor_then_relaxes(self):
        availability = Availability(preferred_days=["週五"], preferred_sessions=["晚上"], can_take_leave=True)
        rows = [_row("2026-07-13", "上午", doctor="王醫師"), _row("2026-07-14", "下午", doctor="李醫師")]

        feasible, relaxed = select_feasible_rows(rows, availability, doctor_preference="王醫師", now=self.before_rows)

        self.assertEqual([row["doctor"] for row in feasible], ["王醫師"])
        self.assertTrue(relaxed)

    def test_empty_and_cross_day_data(self):
        availability = Availability(preferred_days=["週日"], preferred_sessions=["上午"])
        self.assertEqual(filter_rows_by_availability([], availability, now=self.before_rows), [])
        rows = [_row("2026-07-19", "上午")]
        self.assertEqual(filter_rows_by_availability(rows, availability, now=self.before_rows), rows)

    def test_leave_without_substitute_is_unavailable(self):
        availability = Availability()
        rows = [
            _row("2026-07-13", "上午", is_leave=True),
            _row("2026-07-13", "下午", is_leave=True, substitute_doctor="代理醫師"),
        ]

        result = filter_rows_by_availability(rows, availability, now=self.before_rows)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["substitute_doctor"], "代理醫師")


def _row(date: str, session: str, doctor: str = "測試醫師", **extra):
    return {
        "parent_dept": "外科系",
        "child_dept": "一般骨科",
        "doctor": doctor,
        "date": date,
        "session": session,
        "slot": "3201診",
        **extra,
    }


if __name__ == "__main__":
    unittest.main()
