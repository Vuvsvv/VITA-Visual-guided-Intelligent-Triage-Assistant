from fastapi import APIRouter, HTTPException

from app.db import DatabaseUnavailableError
from app.schemas import FollowupRecommendRequest, FollowupRecommendResponse, VisitType
from app.services.case_store import create_case, get_case, save_case, save_recommendations
from app.services.followup_service import ReturnVisitResolutionError, recommend_followup

router = APIRouter(prefix="/followup", tags=["followup"])


@router.post("/recommend", response_model=FollowupRecommendResponse)
async def followup_recommend(req: FollowupRecommendRequest) -> FollowupRecommendResponse:
    if req.case_id:
        case = get_case(req.case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="找不到指定 case_id，請重新建立回診查詢。")
        if case.visit_type not in {None, VisitType.RETURN_VISIT}:
            raise HTTPException(status_code=409, detail="此 case_id 不是回診流程，不能查詢回診班表。")
    else:
        case = create_case()
    working_case = case.model_copy(deep=True)

    child = (req.childDept or "").strip()
    parent = (req.parentDept or "").strip()
    if not child and working_case.department_result:
        child = working_case.department_result.childDept
        parent = parent or working_case.department_result.parentDept
    dept_id = req.dept_id or (working_case.department_result.dept_id if working_case.department_result else None)

    request = req.model_copy(
        update={
            "case_id": working_case.case_id,
            "childDept": child,
            "parentDept": parent or None,
            "dept_id": dept_id,
        }
    )

    if not request.childDept:
        raise HTTPException(status_code=422, detail="請提供 childDept 或有效 case_id。")

    try:
        result = await recommend_followup(request)
    except ReturnVisitResolutionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DatabaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail="正式回診資料目前無法查詢，請稍後重試。") from exc
    if result.recommendations:
        working_case.visit_type = VisitType.RETURN_VISIT
        working_case.department_result = result.department
        working_case.recommendation_generated = True
        save_case(working_case)
        save_recommendations(result.case_id, result.recommendations)
    return result
