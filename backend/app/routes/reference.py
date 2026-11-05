import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query

from app.db import fetch_reference_departments, fetch_reference_doctors
from app.schemas import (
    ReferenceDepartment,
    ReferenceDepartmentResponse,
    ReferenceDoctor,
    ReferenceDoctorResponse,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reference", tags=["reference"])


@router.get("/departments", response_model=ReferenceDepartmentResponse)
async def reference_departments() -> ReferenceDepartmentResponse:
    try:
        rows = await asyncio.to_thread(fetch_reference_departments)
    except Exception as exc:
        logger.error("Reference department query failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="正式科別資料目前無法載入，請稍後重試。") from exc

    return ReferenceDepartmentResponse(
        departments=[
            ReferenceDepartment(
                dept_id=row["dept_id"],
                parentDept=row["parent_dept"],
                childDept=row["child_dept"],
            )
            for row in rows
        ]
    )


@router.get("/doctors", response_model=ReferenceDoctorResponse)
async def reference_doctors(
    department: str = Query(min_length=1),
) -> ReferenceDoctorResponse:
    exact_department = department.strip()
    if not exact_department:
        raise HTTPException(status_code=422, detail="請提供正式科別名稱。")
    try:
        rows = await asyncio.to_thread(fetch_reference_doctors, exact_department)
    except Exception as exc:
        logger.error(
            "Reference doctor query failed error_type=%s",
            type(exc).__name__,
        )
        raise HTTPException(status_code=503, detail="正式醫師資料目前無法載入，請稍後重試。") from exc

    return ReferenceDoctorResponse(
        department=exact_department,
        doctors=[ReferenceDoctor(doctor_id=row["doctor_id"], name=row["name"]) for row in rows],
    )
