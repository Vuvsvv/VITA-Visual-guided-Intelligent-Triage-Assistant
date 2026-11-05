"""Legacy keys are correlation metadata, not authority to parse free text."""
from unittest.mock import patch

from app.services import department_preference_service as preference


def test_duplicate_names_do_not_require_alias_guessing_when_ai_selects_exact_id():
    rows = [
        {"dept_id": 204, "parent_dept": "其他科", "child_dept": "皮膚科"},
        {"dept_id": 304, "parent_dept": "特別門診", "child_dept": "皮膚科"},
    ]
    with patch.object(preference, "fetch_active_departments", return_value=rows):
        accepted = preference.validate_department_preference({
            "dept_id": 304, "department_name": "皮膚科", "source_text": "我要看皮膚科", "confidence": 0.99,
        }, ["我要看皮膚科"])
    assert accepted is not None
    assert accepted.dept_id == 304
    assert accepted.department_name == "皮膚科"
