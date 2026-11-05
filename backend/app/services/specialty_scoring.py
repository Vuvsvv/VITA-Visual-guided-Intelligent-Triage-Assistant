from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from time import monotonic
from typing import Any, Iterable

from app.config import get_settings
from app.schemas import DepartmentResult, TriageCase
from app.services.ai_service import complete_runtime_json as _complete_runtime_json, runtime_ai_available
from app.services.department_reasoning_service import effective_semantic_evidence

logger = logging.getLogger(__name__)

TAG_NEUTRAL_SCORE = 0.5
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_CANDIDATES = 40
DEFAULT_MAX_BATCHES = 4
DEFAULT_TOTAL_TIMEOUT_SECONDS = 12.0
# The AI picks one of five levels instead of a free decimal, so small wording drift
# between calls cannot reorder doctors; the API keeps its 0..1 score scale.
RELEVANCE_LEVELS = {
    "very_high": 1.0,
    "high": 0.75,
    "medium": 0.5,
    "low": 0.25,
    "none": 0.0,
}


async def complete_prompt(prompt: str) -> str:
    """Compatibility seam for tests; production sends one small doctor batch per call."""
    return await _complete_runtime_json(prompt, purpose="doctor_scoring", deterministic=True)


@dataclass(frozen=True)
class SpecialtyScore:
    doctor_id: str
    doctor: str
    childDept: str
    score: float
    reason: str
    source: str = "neutral"


async def score_doctor_specialties(
    case: TriageCase,
    department: DepartmentResult,
    rows: Iterable[dict[str, Any]],
    max_ai_candidates: int | None = None,
) -> dict[str, SpecialtyScore]:
    """Score each real doctor once; any incomplete batch makes the whole set neutral."""
    doctor_rows = _unique_doctor_rows(
        row
        for row in rows
        if _row_department(row) == department.childDept and _row_doctor_id(row)
    )
    neutral = {_row_doctor_id(row): _neutral_score(row, department) for row in doctor_rows}
    if not doctor_rows:
        return neutral

    settings = get_settings()
    if not bool(getattr(settings, "ai_doctor_scoring_enabled", False)) or not _ai_available():
        logger.info("specialty_scoring ai skipped case_id=%s reason=unavailable", case.case_id)
        return neutral

    # Fixed doctor order keeps the prompt identical for the same candidates.
    scorable = sorted(
        (row for row in doctor_rows if _has_specialty(row.get("specialty_tags") or row.get("specialty"))),
        key=_row_doctor_id,
    )
    if not scorable:
        return neutral

    batch_size = _positive_setting(settings, "doctor_scoring_batch_size", DEFAULT_BATCH_SIZE)
    max_batches = _positive_setting(settings, "doctor_scoring_max_batches", DEFAULT_MAX_BATCHES)
    configured_capacity = _positive_setting(
        settings, "doctor_scoring_max_candidates", DEFAULT_MAX_CANDIDATES
    )
    capacity = min(configured_capacity, batch_size * max_batches)
    if max_ai_candidates is not None:
        capacity = min(capacity, max(0, int(max_ai_candidates)))
    if len(scorable) > capacity:
        logger.info(
            "specialty_scoring ai skipped case_id=%s reason=capacity candidate_count=%s capacity=%s",
            case.case_id,
            len(scorable),
            capacity,
        )
        return neutral

    accepted: dict[str, float] = {}
    total_budget = _positive_float_setting(
        settings,
        "doctor_scoring_total_timeout_seconds",
        DEFAULT_TOTAL_TIMEOUT_SECONDS,
    )
    deadline = monotonic() + total_budget
    try:
        for start in range(0, len(scorable), batch_size):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("doctor-scoring total time budget exhausted")
            batch = scorable[start:start + batch_size]
            raw = await asyncio.wait_for(
                complete_prompt(_build_scoring_prompt(case, department, batch)),
                timeout=remaining,
            )
            batch_scores = _validate_complete_batch(_parse_json_object(raw), batch)
            if batch_scores is None:
                raise ValueError("incomplete or invalid doctor-scoring batch")
            accepted.update(batch_scores)
    except Exception as exc:
        logger.warning(
            "specialty_scoring ai failed case_id=%s error_type=%s; all candidates neutral",
            case.case_id,
            type(exc).__name__,
        )
        return neutral

    results = dict(neutral)
    for row in scorable:
        doctor_id = _row_doctor_id(row)
        results[doctor_id] = SpecialtyScore(
            doctor_id=doctor_id,
            doctor=str(row.get("doctor") or row.get("doctor_name") or ""),
            childDept=department.childDept,
            score=accepted[doctor_id],
            reason="AI 僅比較患者已接受症狀證據與 SQL specialty_tags 的語意相關程度",
            source="ai",
        )
    return results


def _neutral_score(row: dict[str, Any], department: DepartmentResult) -> SpecialtyScore:
    has_tags = _has_specialty(row.get("specialty_tags") or row.get("specialty"))
    reason = (
        "AI 專長語意評分未完整可用，本次所有候選統一使用中性值 0.50"
        if has_tags
        else "SQL specialty_tags 無資料，專長分數使用中性值 0.50"
    )
    return SpecialtyScore(
        doctor_id=_row_doctor_id(row),
        doctor=str(row.get("doctor") or row.get("doctor_name") or ""),
        childDept=_row_department(row) or department.childDept,
        score=TAG_NEUTRAL_SCORE,
        reason=reason,
        source="neutral",
    )


def _positive_setting(settings: object, name: str, default: int) -> int:
    value = getattr(settings, name, default)
    return int(value) if type(value) is int and value > 0 else default


def _positive_float_setting(settings: object, name: str, default: float) -> float:
    value = getattr(settings, name, default)
    if type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) <= 0:
        return default
    return min(float(value), 60.0)


def _ai_available() -> bool:
    return runtime_ai_available(get_settings())


def _has_specialty(value: Any) -> bool:
    return bool(str(value or "").strip())


def _unique_doctor_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        doctor_id = _row_doctor_id(row)
        if doctor_id and doctor_id not in unique:
            unique[doctor_id] = row
    return list(unique.values())


def _build_scoring_prompt(
    case: TriageCase,
    department: DepartmentResult,
    rows: list[dict[str, Any]],
) -> str:
    candidates = [
        {
            "doctor_id": _row_doctor_id(row),
            "specialty_tags": str(row.get("specialty_tags") or row.get("specialty") or ""),
        }
        for row in rows
    ]
    patient_evidence = [
        evidence
        for evidence in effective_semantic_evidence(case)
        if evidence.get("field") in {
            "symptom", "accompanying_symptoms", "body_part", "duration", "severity", "onset"
        }
    ]
    return f"""你只比較患者已接受的結構化症狀證據與 SQL specialty_tags 的語意相關程度。

已確認科別：{department.childDept}
患者結構化證據：
{json.dumps(patient_evidence, ensure_ascii=False, indent=2)}

候選醫師：
{json.dumps(candidates, ensure_ascii=False, indent=2)}

規則：
1. 每個 doctor_id 恰好輸出一次，且必須完全照抄。
2. level 是語意相關程度，只能是下列其一：very_high（非常相關）、high（相關）、medium（普通）、low（不太相關）、none（無關）。這不是診斷機率、醫師能力、品質或成功率。
3. 只能使用提供的患者證據與 specialty_tags，不得診斷、補專長或參考名氣年資。
4. 不要輸出自然語言理由。

只輸出 JSON：
{{"scores":[{{"doctor_id":"候選 doctor_id","level":"medium"}}]}}"""


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw).strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("AI response is not a JSON object")
    return data


def _validate_complete_batch(
    data: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, float] | None:
    raw_scores = data.get("scores")
    if not isinstance(raw_scores, list):
        return None
    expected = {_row_doctor_id(row) for row in rows}
    accepted: dict[str, float] = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            return None
        doctor_id = item.get("doctor_id")
        level = item.get("level")
        if (
            not isinstance(doctor_id, str)
            or doctor_id not in expected
            or doctor_id in accepted
            or not isinstance(level, str)
            or level not in RELEVANCE_LEVELS
        ):
            return None
        accepted[doctor_id] = RELEVANCE_LEVELS[level]
    return accepted if set(accepted) == expected else None


def _row_department(row: dict[str, Any]) -> str:
    return str(row.get("child_dept") or row.get("childDept") or "").strip()


def _row_doctor_id(row: dict[str, Any]) -> str:
    value = row.get("doctor_id")
    return str(value).strip() if value is not None else ""

