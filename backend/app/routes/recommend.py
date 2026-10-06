from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.db import DatabaseUnavailableError
from app.schemas import ConversationStage, RecommendRequest, RecommendationResult, VisitType
from app.services.appointment_service import (
    DepartmentResolutionError,
    detect_department_result,  # Legacy patch seam; /recommend does not invoke it.
    normalize_visit_type,
    recommend_appointments,
)
from app.services.ai_reply_generator import build_no_schedule_message
from app.services.case_store import create_case, get_case, sanitize_untrusted_snapshot, save_case, save_recommendation_result
from app.services.conversation_service import safety_screen_resolved
from app.services.department_preference_service import resolve_requested_department
from app.services.ai_service import runtime_ai_available
from app.services.rag_triage_adapter import refine_case_with_ai
from app.services.rule_engine import apply_user_message
from app.services.ttas_evaluator import apply_ttas_evaluation

router = APIRouter(prefix="/recommend", tags=["recommend"])


@router.post("", response_model=RecommendationResult)
async def recommend(req: RecommendRequest) -> RecommendationResult:
    supplied_case = req.triage_case
    if req.case_id and supplied_case is not None and supplied_case.case_id != req.case_id:
        raise HTTPException(status_code=422, detail="triage_case.case_id 與 request case_id 不一致。")
    lookup_id = req.case_id or (supplied_case.case_id if supplied_case else None)
    stored_case = get_case(lookup_id) if lookup_id else None
    if supplied_case is not None and stored_case is not None:
        if (
            supplied_case.visit_type is not None
            and stored_case.visit_type is not None
            and supplied_case.visit_type != stored_case.visit_type
        ):
            raise HTTPException(status_code=409, detail="triage_case.visit_type 與已保存案件不一致。")
    case = stored_case or (sanitize_untrusted_snapshot(supplied_case) if supplied_case else None)
    if case is None and req.userQuery and req.userQuery.strip():
        case = create_case(req.case_id)
        semantic_first = runtime_ai_available(get_settings())
        apply_user_message(
            case,
            req.userQuery,
            semantic_first=True,
            apply_legacy_safety=False,
        )
        if semantic_first:
            await refine_case_with_ai(case, user_sources=[req.userQuery])
        case.triage = apply_ttas_evaluation(case)
        case.conversation_state.is_complete = not case.triage.need_more_info

    if case is None:
        raise HTTPException(status_code=400, detail="請提供 triage_case、case_id 或 userQuery。")

    # Recommendation is transactional with respect to the in-memory workflow store.
    # Validation and DB work happen on a detached copy; only success is committed.
    case = case.model_copy(deep=True)

    if (
        case.conversation_state.clarification_status == "urgent"
        or (
            case.ttas_result.status == "matched"
            and case.ttas_result.level_candidate in {1, 2}
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="目前急迫性篩檢結果不適合繼續一般掛號推薦，請優先尋求醫療評估。",
        )

    request_visit_type = normalize_visit_type(req.visit_type) if req.visit_type is not None else None
    if case.visit_type is not None and request_visit_type is not None and case.visit_type != request_visit_type:
        raise HTTPException(
            status_code=409,
            detail="visit_type 與問診案件建立時的選擇不一致。",
        )
    effective_visit_type = case.visit_type or request_visit_type
    if effective_visit_type is None:
        raise HTTPException(
            status_code=422,
            detail="缺少合法 visit_type，請重新選擇 initial、followup、quick_search 或 return_visit。",
        )
    if effective_visit_type == VisitType.QUICK_SEARCH:
        raise HTTPException(
            status_code=400,
            detail="quick_search 不使用推薦流程，請改呼叫 /schedules/search。",
        )
    if case.visit_type is None:
        case.visit_type = effective_visit_type

    if case.conversation_state.is_complete is not True or case.triage.need_more_info is True:
        raise HTTPException(status_code=400, detail="triage_case 尚未完成，請先完成 /chat 多輪問答。")
    if not safety_screen_resolved(case):
        raise HTTPException(status_code=400, detail="急迫症狀篩檢尚未完成，請先返回 /chat 確認安全狀況。")

    if req.confirmed:
        case.confirmed = True
        case.conversation_state.confirmed = True
        case.conversation_state.awaiting_confirmation = False

    if not (case.confirmed or case.conversation_state.confirmed):
        raise HTTPException(status_code=400, detail="triage_case 尚未確認，請先以 /chat 傳入 confirmed=true。")

    explicit_department = None
    if case.patient_input.requested_department_id is not None or case.patient_input.requested_department_name:
        explicit_department = resolve_requested_department(case)
    if case.department_result is None:
        case.department_result = explicit_department
    if case.department_result is None:
        raise HTTPException(status_code=422, detail="尚無已驗證的正式科別結果，請先返回 /chat 釐清。")
    if (
        case.conversation_state.clarification_status == "sufficient"
        and case.conversation_state.department_status != "resolved"
        and explicit_department is None
    ):
        raise HTTPException(status_code=422, detail="科別候選尚未收斂為已驗證結果。")

    try:
        result = await recommend_appointments(case, visit_type=effective_visit_type)
    except DepartmentResolutionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DatabaseUnavailableError as exc:
        raise HTTPException(status_code=503, detail="正式資料目前無法查詢，請稍後重試。") from exc

    if not result.recommendations.specialty_first or not result.recommendations.time_first:
        raise HTTPException(status_code=503, detail=build_no_schedule_message(case))

    case.confirmed = True
    case.conversation_state.stage = ConversationStage.RECOMMENDING
    case.conversation_state.confirmed = True
    case.conversation_state.awaiting_confirmation = False
    case.recommendation_generated = True
    save_case(case)
    save_recommendation_result(result)
    return result
