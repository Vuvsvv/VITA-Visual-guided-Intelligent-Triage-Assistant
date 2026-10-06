from fastapi import APIRouter, HTTPException

from app.db import DatabaseUnavailableError
from app.schemas import ConversationStage, ScriptRequest, ScriptResponse
from app.services.case_store import find_recommendation, get_case, get_recommendation, save_case
from app.services.quick_search_service import revalidate_schedule
from app.services.script_service import SCRIPT_ID, build_navigation_script

router = APIRouter(prefix="/generate_script", tags=["generate_script"])


@router.post("", response_model=ScriptResponse)
def generate_script(req: ScriptRequest) -> ScriptResponse:
    stored_recommendation = (
        get_recommendation(req.case_id, req.recommendation_id)
        if req.case_id
        else find_recommendation(req.recommendation_id)
    )
    recommendation = stored_recommendation or req.recommendation
    if recommendation is None:
        raise HTTPException(
            status_code=404,
            detail="找不到 recommendation，請先取得並選擇一筆班表。",
        )
    if recommendation.recommendation_id != req.recommendation_id:
        raise HTTPException(status_code=422, detail="recommendation_id 與 recommendation 內容不一致。")
    if (
        not recommendation.schedule_id
        or not recommendation.doctor_id
        or recommendation.dept_id is None
        or not recommendation.date
        or not recommendation.session
    ):
        raise HTTPException(status_code=422, detail="班表缺少 schedule_id、doctor_id、dept_id、date 或 session。")
    is_quick_search = stored_recommendation is None
    if is_quick_search:
        if not req.case_id or not req.case_id.startswith("quick_"):
            raise HTTPException(status_code=404, detail="找不到已儲存的 recommendation。")
        if not req.recommendation_id.startswith(f"qs_{req.case_id}_"):
            raise HTTPException(status_code=422, detail="Quick Search recommendation_id 與 case_id 不一致。")
    try:
        recommendation = revalidate_schedule(
            recommendation,
            require_followup_evidence=is_quick_search,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DatabaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail="正式班表目前無法重新確認，請稍後重試。") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="正式班表目前無法重新確認，請稍後重試。") from exc

    steps = build_navigation_script(recommendation)
    if req.case_id:
        case = get_case(req.case_id)
        if case:
            case.selected_recommendation_id = recommendation.recommendation_id
            case.script_generated = True
            case.conversation_state.stage = ConversationStage.SCRIPT_READY
            save_case(case)

    return ScriptResponse(
        isSuccess=True,
        script_id=SCRIPT_ID,
        recommendation_id=recommendation.recommendation_id,
        recommendation=recommendation,
        steps=steps,
        message=f"已產生導引劇本：{recommendation.childDept} / {recommendation.doctor} / {recommendation.date}",
        step_count=len(steps),
    )
