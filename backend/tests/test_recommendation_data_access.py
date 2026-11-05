from __future__ import annotations

import unittest

from app import db


class RecommendationDataAccessTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
