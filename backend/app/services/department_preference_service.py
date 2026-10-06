from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.db import fetch_active_departments
from app.schemas import DepartmentResult, TriageCase
from app.services.field_acceptance import looks_like_department_request


@dataclass(frozen=True)
class DepartmentPreference:
    name: str
    dept_id: int | None
    resolved: bool


_DEPARTMENT_ALIASES = {
    "骨科": "一般骨科",
    "腸胃科": "胃腸肝膽科",
}


def capture_department_preference(case: TriageCase, text: str) -> DepartmentPreference | None:
    """Store an explicit preference without turning it into a triage result."""
    if not looks_like_department_request(text):
        return None

    departments = _canonical_departments(fetch_active_departments())
    mentioned_names = sorted(
        {item["child_dept"] for item in departments if item["child_dept"] in text},
        key=len,
        reverse=True,
    )
    name = mentioned_names[0] if mentioned_names else _department_name_from_text(text)
    if not name:
        return None

    alias_target = _DEPARTMENT_ALIASES.get(name)
    if alias_target and any(item["child_dept"] == alias_target for item in departments):
        name = alias_target

    matches = [item for item in departments if item["child_dept"] == name]
    dept_id = matches[0]["dept_id"] if len(matches) == 1 else None
    case.patient_input.requested_department_name = name
    case.patient_input.requested_department_id = dept_id
    return DepartmentPreference(name=name, dept_id=dept_id, resolved=dept_id is not None)


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


def _department_name_from_text(text: str) -> str | None:
    visit_match = re.search(
        r"(?:我)?(?:想要|想|要|希望|直接|改成|改為|改)?\s*"
        r"(?:看|掛號|掛診|掛)\s*([\u4e00-\u9fffA-Za-z0-9()（）/、-]{1,20}?科)",
        text,
    )
    if visit_match:
        return visit_match.group(1).strip()
    change_match = re.search(
        r"(?:我)?(?:想)?(?:改成|改為)\s*"
        r"([\u4e00-\u9fffA-Za-z0-9()（）/、-]{1,20}?科)",
        text,
    )
    return change_match.group(1).strip() if change_match else None


def _canonical_departments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for row in rows:
        try:
            dept_id = int(row.get("dept_id"))
        except (TypeError, ValueError):
            continue
        parent = str(row.get("parent_dept") or row.get("parentDept") or "").strip()
        child = str(row.get("child_dept") or row.get("childDept") or "").strip()
        if child:
            canonical.append({"dept_id": dept_id, "parent_dept": parent, "child_dept": child})
    return canonical
