from __future__ import annotations

import logging
import time
from uuid import uuid4

from app.schemas import (
    ConversationState,
    RecommendationItem,
    RecommendationResult,
    SeverityNormalization,
    TriageCase,
    UrgencyNormalization,
    UrgencyResult,
)

logger = logging.getLogger(__name__)


# Temporary in-memory workflow store.
#
# Responsibilities:
# - /chat creates and updates TriageCase by case_id.
# - /recommend stores the latest RecommendationResult items by case_id.
# - /generate_script resolves case_id + recommendation_id back to the selected item.
#
# Deployment note:
# This module is intentionally lightweight for the current prototype. Data is lost
# on process restart and is not shared across workers. Replace it with SQLite,
# Redis, or the primary DB before production or multi-worker deployment.
_CASES: dict[str, TriageCase] = {}
_RECOMMENDATIONS_BY_CASE: dict[str, dict[str, RecommendationItem]] = {}
_LAST_TOUCHED: dict[str, float] = {}
CASE_TTL_SECONDS = 2 * 60 * 60
_clock = time.monotonic


def _prune_expired(now: float | None = None) -> None:
    current = _clock() if now is None else now
    expired = [
        case_id for case_id, touched in _LAST_TOUCHED.items()
        if current - touched >= CASE_TTL_SECONDS
    ]
    for case_id in expired:
        _CASES.pop(case_id, None)
        _RECOMMENDATIONS_BY_CASE.pop(case_id, None)
        _LAST_TOUCHED.pop(case_id, None)


def new_case_id() -> str:
    return f"case_{uuid4().hex[:8]}"


def sanitize_untrusted_snapshot(snapshot: TriageCase) -> TriageCase:
    """Keep compatibility data while removing every client-asserted server-owned conclusion."""
    case = snapshot.model_copy(deep=True)
    case.history_records = [
        message.model_copy(deep=True)
        for message in case.history_records
        if message.role == "user" and message.content.strip()
    ]
    case.confirmed = False
    case.recommendation_generated = False
    case.script_generated = False
    case.selected_recommendation_id = None
    case.department_result = None
    case.semantic_extractions = []
    case.ttas_evidence = []
    case.ttas_result = type(case.ttas_result)()
    case.patient_input.red_flags = []
    case.patient_input.red_flags_checked = False
    case.patient_input.red_flags_status = "not_checked"
    case.patient_input.severity_normalized = SeverityNormalization()
    case.patient_input.urgency_normalized = UrgencyNormalization()
    case.patient_input.collected_fields = []
    case.patient_input.age_years = None
    case.patient_input.age_months = None
    case.triage = UrgencyResult()
    case.conversation_state = ConversationState()
    return case


def create_case(case_id: str | None = None) -> TriageCase:
    _prune_expired()
    case = TriageCase(case_id=case_id or new_case_id())
    _CASES[case.case_id] = case
    _LAST_TOUCHED[case.case_id] = _clock()
    logger.info("case_store create_case case_id=%s", case.case_id)
    return case


def get_case(case_id: str) -> TriageCase | None:
    _prune_expired()
    case = _CASES.get(case_id)
    if case is not None:
        _LAST_TOUCHED[case_id] = _clock()
    logger.debug("case_store get_case case_id=%s found=%s", case_id, case is not None)
    return case


def save_case(case: TriageCase) -> TriageCase:
    _prune_expired()
    _CASES[case.case_id] = case
    _LAST_TOUCHED[case.case_id] = _clock()
    logger.debug(
        "case_store save_case case_id=%s history_len=%s stage=%s",
        case.case_id,
        len(case.history_records),
        case.conversation_state.stage,
    )
    return case


def get_recommendations_for_case(case_id: str) -> dict[str, RecommendationItem]:
    _prune_expired()
    if case_id in _CASES:
        _LAST_TOUCHED[case_id] = _clock()
    return _RECOMMENDATIONS_BY_CASE.get(case_id, {})


def save_recommendation_result(result: RecommendationResult) -> RecommendationResult:
    items = [
        *result.recommendations.specialty_first,
        *result.recommendations.time_first,
    ]
    save_recommendations(result.case_id, items)
    return result


def save_recommendations(
    case_id: str,
    items: list[RecommendationItem],
) -> list[RecommendationItem]:
    _prune_expired()
    _RECOMMENDATIONS_BY_CASE[case_id] = {
        item.recommendation_id: item for item in items
    }
    if case_id in _CASES:
        _LAST_TOUCHED[case_id] = _clock()
    return items


def get_recommendation(case_id: str, recommendation_id: str) -> RecommendationItem | None:
    return get_recommendations_for_case(case_id).get(recommendation_id)


def find_recommendation(recommendation_id: str) -> RecommendationItem | None:
    _prune_expired()
    for recommendations in _RECOMMENDATIONS_BY_CASE.values():
        if recommendation_id in recommendations:
            return recommendations[recommendation_id]
    return None
