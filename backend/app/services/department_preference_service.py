from __future__ import annotations

import logging
from typing import Any

from app.db import fetch_active_departments
from pydantic import ValidationError

from app.schemas import DepartmentPreferenceInterpretation, DepartmentResult, TriageCase
from app.services.confidence_scoring import ACCEPT_THRESHOLD


logger = logging.getLogger(__name__)


def active_department_candidates() -> list[dict[str, Any]]:
    """Read current official identities; provider failure yields no candidates."""
    try:
        return _canonical_departments(fetch_active_departments())
    except Exception as exc:
        logger.warning("department candidates unavailable error_type=%s", type(exc).__name__)
        return []


def validate_department_preference(
    value: Any, user_sources: list[str],
) -> DepartmentPreferenceInterpretation | None:
    """Validate against current SQL identities without mutating patient state."""
    if value is None:
        return None
    try:
        preference = DepartmentPreferenceInterpretation.model_validate(value)
    except (ValidationError, TypeError, ValueError):
        return None
    matches = [
        item for item in active_department_candidates()
        if item["dept_id"] == preference.dept_id
    ]
    valid = (
        preference.confidence >= ACCEPT_THRESHOLD
        and preference.source_text.strip()
        and any(preference.source_text in source for source in user_sources)
        and len(matches) == 1
        and matches[0]["child_dept"] == preference.department_name
    )
    return preference if valid else None


def resolve_requested_department(
    case: TriageCase,
    departments: list[dict[str, Any]] | None = None,
) -> DepartmentResult | None:
    """Resolve the stored user preference against current canonical department data."""
    if case.patient_input.red_flags:
        return None
    candidates = _canonical_departments(
        departments if departments is not None else fetch_active_departments()
    )
    requested_id = case.patient_input.requested_department_id
    requested_name = case.patient_input.requested_department_name
    matches = [
        item
        for item in candidates
        if (
            (requested_id is not None and item["dept_id"] == requested_id)
            or (requested_id is None and requested_name and item["child_dept"] == requested_name)
        )
        and (not requested_name or item["child_dept"] == requested_name)
    ]
    if len(matches) != 1:
        return None
    item = matches[0]
    return DepartmentResult(
        dept_id=item["dept_id"],
        parentDept=item["parent_dept"],
        childDept=item["child_dept"],
        confidence=0.9,
        reason=[f"使用者明確指定正式科別 {item['child_dept']}，後續推薦依此偏好查詢"],
    )


def _canonical_departments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for row in rows:
        if row.get("is_active") is False or row.get("is_active") == 0:
            continue
        try:
            dept_id = int(row.get("dept_id"))
        except (TypeError, ValueError):
            continue
        parent = str(row.get("parent_dept") or row.get("parentDept") or "").strip()
        child = str(row.get("child_dept") or row.get("childDept") or "").strip()
        if child:
            canonical.append({"dept_id": dept_id, "parent_dept": parent, "child_dept": child})
    return canonical
