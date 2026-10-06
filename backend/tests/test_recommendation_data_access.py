from __future__ import annotations

import unittest

from app import db
from app.db import map_schedule_record


class RecommendationDataAccessTest(unittest.TestCase):
    def test_schema_record_mapping(self):
        mapped = map_schedule_record(
            {
                "parentDept": "外科系",
                "childDept": "一般骨科",
                "doctorId": "D1",
                "doctorName": "王醫師",
                "scheduleId": "S1",
                "date": "2026-07-13",
                "session": "上午",
                "room": "3201診",
                "specialtyTags": "膝關節",
            }
        )

        self.assertEqual(mapped["parent_dept"], "外科系")
        self.assertEqual(mapped["child_dept"], "一般骨科")
        self.assertEqual(mapped["doctor_id"], "D1")
        self.assertEqual(mapped["schedule_id"], "S1")
        self.assertEqual(mapped["specialty_tags"], "膝關節")

    def test_mock_repository(self):
        class FakeRepository:
            def fetch_available_slots(self, *_args, **_kwargs):
                return [map_schedule_record({"childDept": "一般骨科", "doctor": "測試醫師"})]

        repo = FakeRepository()

        self.assertEqual(repo.fetch_available_slots("一般骨科")[0]["doctor"], "測試醫師")

    def test_db_connection_failure_raises_typed_error(self):
        original_fetch_db = db._fetch_available_slots_from_db
        db._fetch_available_slots_from_db = lambda *_, **__: (_ for _ in ()).throw(RuntimeError("db down"))
        try:
            with self.assertRaises(db.DatabaseUnavailableError):
                db.fetch_available_slots("一般骨科", max_slots=2)
        finally:
            db._fetch_available_slots_from_db = original_fetch_db


    def test_empty_query_result_returns_no_slots(self):
        original_fetch_db = db._fetch_available_slots_from_db
        db._fetch_available_slots_from_db = lambda *_, **__: []
        try:
            slots = db.fetch_available_slots("一般骨科", max_slots=2)
        finally:
            db._fetch_available_slots_from_db = original_fetch_db

        self.assertEqual(slots, [])

    def test_leave_substitute_mapping(self):
        mapped = map_schedule_record(
            {
                "childDept": "一般骨科",
                "doctor": "代理醫師",
                "is_leave": True,
                "substituteDoctor": "代理醫師",
            }
        )

        self.assertTrue(mapped["is_leave"])
        self.assertEqual(mapped["substitute_doctor"], "代理醫師")

    def test_injected_schedule_fixture_maps_required_ids(self):
        slot = map_schedule_record(
            {
                "childDept": "Test Department",
                "doctorId": "fixture-doctor",
                "scheduleId": "fixture-schedule",
            }
        )

        self.assertEqual(slot["doctor_id"], "fixture-doctor")
        self.assertEqual(slot["schedule_id"], "fixture-schedule")


if __name__ == "__main__":
    unittest.main()
