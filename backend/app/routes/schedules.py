import asyncio
import logging
from datetime import date as Date
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query

from app.db import fetch_reference_departments
from app.schemas import (
    QuickSearchPeriod,
    QuickSearchResponse,
)
from app.services.quick_search_service import search_quick_schedules


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/schedules", tags=["schedules"])


@router.get("/search", response_model=QuickSearchResponse)
async def quick_search_schedules(
    dept_id: int = Query(gt=0),
    date_: Date = Query(alias="date"),
    period: QuickSearchPeriod = Query(),
) -> QuickSearchResponse:
    try:
        departments = await asyncio.to_thread(fetch_reference_departments)
    except Exception as exc:
        logger.error("Quick search department lookup failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="正式科別資料目前無法載入，請稍後重試。") from exc

    department = next(
        (item for item in departments if str(item.get("dept_id") or "").strip() == str(dept_id)),
        None,
    )
    if department is None:
        raise HTTPException(status_code=404, detail="找不到指定 dept_id 的正式科別。")

    # Correlation-only identifier for this response and its item IDs. It is not persisted.
    case_id = f"quick_{uuid4().hex[:8]}"
    try:
        results = await search_quick_schedules(
            case_id=case_id,
            department=department,
            target_date=date_,
            period=period,
        )
    except Exception as exc:
        logger.error(
            "Quick search schedule query failed dept_id=%s error_type=%s",
            dept_id,
            type(exc).__name__,
        )
        raise HTTPException(status_code=503, detail="正式班表目前無法查詢，請稍後重試。") from exc

    return QuickSearchResponse(
        case_id=case_id,
        dept_id=dept_id,
        parentDept=str(department.get("parent_dept") or "").strip(),
        childDept=str(department.get("child_dept") or "").strip(),
        date=date_,
        period=period,
        results=results,
        total_count=len(results),
    )
