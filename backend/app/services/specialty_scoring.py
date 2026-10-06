from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.config import get_settings
from app.schemas import DepartmentResult, TriageCase
from app.services.ai_service import complete_runtime_json as _complete_runtime_json, runtime_ai_available
from app.services.negation_utils import strip_negated_red_flags

logger = logging.getLogger(__name__)


async def complete_prompt(prompt: str) -> str:
    """Compatibility seam for tests; production always uses one Cerebras batch call."""
    return await _complete_runtime_json(prompt, purpose="doctor_scoring")

TAG_NEUTRAL_SCORE = 0.5
MAX_AI_SCORE_CANDIDATES = 10
MAX_AI_REASON_LENGTH = 64
PROHIBITED_AI_REASON_TERMS = (
    "確診",
    "診斷為",
    "患有",
    "罹患",
    "保證",
    "一定",
    "最適合",
    "最佳選擇",
    "肯定是",
    "必須看這位",
    "一定要看",
)

SPECIALTY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "頭": ("神經", "腦", "腦血管", "頭痛", "眩暈", "暈眩"),
    "頭暈": ("神經", "腦", "腦血管", "眩暈", "平衡"),
    "眩暈": ("神經", "腦", "眩暈", "平衡"),
    "膝": ("膝", "關節", "骨科", "運動傷害", "復健", "疼痛"),
    "關節": ("關節", "骨科", "復健", "風濕", "疼痛"),
    "皮膚": ("皮膚", "紅疹", "濕疹", "蕁麻疹", "青春痘", "過敏"),
    "疹": ("皮膚", "紅疹", "濕疹", "蕁麻疹", "過敏"),
    "癢": ("皮膚", "濕疹", "過敏", "蕁麻疹"),
    "胸痛": ("心臟", "胸腔", "心律", "冠心", "肺"),
    "呼吸困難": ("胸腔", "肺", "心臟", "氣喘"),
    "心悸": ("心臟", "心律", "電生理"),
    "咳": ("胸腔", "肺", "感染", "氣喘"),
}


@dataclass(frozen=True)
class SpecialtyScore:
    doctor_id: str
    doctor: str
    childDept: str
    score: float
    reason: str
    source: str = "deterministic"


async def score_doctor_specialties(
    case: TriageCase,
    department: DepartmentResult,
    rows: Iterable[dict[str, Any]],
    max_ai_candidates: int = MAX_AI_SCORE_CANDIDATES,
) -> dict[str, SpecialtyScore]:
    normalized_rows = _unique_doctor_rows(
        row for row in rows if _row_department(row) == department.childDept
    )
    deterministic = {
        _row_key(row): score_doctor_deterministically(case, department, row)
        for row in normalized_rows
    }

    settings = get_settings()
    if not bool(getattr(settings, "ai_doctor_scoring_enabled", False)):
        logger.info(
            "specialty_scoring ai skipped case_id=%s reason=feature_disabled",
            case.case_id,
        )
        return deterministic

    ai_rows = [
        row
        for row in normalized_rows
        if _has_specialty(row.get("specialty_tags") or row.get("specialty"))
    ][:max_ai_candidates]
    if not ai_rows:
        return deterministic
    if not _ai_available():
        logger.info(
            "specialty_scoring ai skipped case_id=%s reason=cerebras_api_key_not_configured",
            case.case_id,
        )
        return deterministic

    prompt = _build_scoring_prompt(case, department, ai_rows)
    try:
        raw = await complete_prompt(prompt)
        data = _parse_json_object(raw)
    except Exception as exc:
        logger.warning(
            "specialty_scoring ai failed case_id=%s error_type=%s",
            case.case_id,
            type(exc).__name__,
        )
        return deterministic

    valid_keys = {_row_key(row): row for row in ai_rows}
    results = dict(deterministic)
    for item in _iter_ai_scores(data):
        key = str(item.get("doctor_id") or item.get("doctor") or "").strip()
        if key not in valid_keys:
            logger.warning("specialty_scoring rejected unknown doctor key=%s", key)
            continue
        row = valid_keys[key]
        if _row_department(row) != department.childDept:
            logger.warning("specialty_scoring rejected wrong department doctor=%s", key)
            continue
        score = _valid_ai_score(item.get("score"))
        reason = _validate_ai_reason(item.get("reason"))
        if score is None or reason is None:
            logger.warning("specialty_scoring rejected incomplete result doctor=%s", key)
            continue
        results[key] = SpecialtyScore(
            doctor_id=_row_doctor_id(row),
            doctor=str(row.get("doctor") or row.get("doctor_name") or ""),
            childDept=department.childDept,
            score=score,
            reason=reason,
            source="ai",
        )

    return results


def score_doctor_deterministically(
    case: TriageCase,
    department: DepartmentResult,
    row: dict[str, Any],
) -> SpecialtyScore:
    if _row_department(row) != department.childDept:
        return SpecialtyScore(
            doctor_id=_row_doctor_id(row),
            doctor=str(row.get("doctor") or row.get("doctor_name") or ""),
            childDept=_row_department(row),
            score=0.0,
            reason="醫師科別與推薦科別不符，已排除",
        )

    tags = str(row.get("specialty_tags") or row.get("specialty") or "").strip()
    if not tags:
        return SpecialtyScore(
            doctor_id=_row_doctor_id(row),
            doctor=str(row.get("doctor") or row.get("doctor_name") or ""),
            childDept=department.childDept,
            score=TAG_NEUTRAL_SCORE,
            reason="醫師專長資料不足，不列為主要依據；專長分數使用中性值 0.50",
        )

    patient_text = _case_text(case)
    matched = []
    for symptom_keyword, specialty_keywords in SPECIALTY_KEYWORDS.items():
        if symptom_keyword in patient_text and any(keyword in tags for keyword in specialty_keywords):
            matched.append(symptom_keyword)
    concept_matches = _explainable_specialty_matches(patient_text, tags)

    if matched or concept_matches:
        basis = sorted({*matched, *(match[0] for match in concept_matches)})
        specialties = sorted({match[1] for match in concept_matches})
        score = min(0.95, 0.68 + (len(set(matched)) + len(concept_matches)) * 0.09)
        if specialties:
            reason = f"症狀與{'、'.join(basis)}相關；醫師專長包含{'、'.join(specialties)}"
        else:
            reason = "專長與症狀相關：" + "、".join(basis)
    else:
        score = TAG_NEUTRAL_SCORE
        reason = "未找到明確專長關鍵字，不列為主要依據；專長分數使用中性值 0.50"

    return SpecialtyScore(
        doctor_id=_row_doctor_id(row),
        doctor=str(row.get("doctor") or row.get("doctor_name") or ""),
        childDept=department.childDept,
        score=round(score, 2),
        reason=reason,
    )


def clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return TAG_NEUTRAL_SCORE
    return max(0.0, min(score, 1.0))


def _valid_ai_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if 0.0 <= score <= 1.0 else None


def _normalize_ai_reason(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _truncate_ai_reason(value: Any) -> str:
    return _normalize_ai_reason(value)[:MAX_AI_REASON_LENGTH]


def _validate_ai_reason(value: Any) -> str | None:
    reason = _normalize_ai_reason(value)
    if not reason:
        return None
    if any(term in reason for term in PROHIBITED_AI_REASON_TERMS):
        return None
    return _truncate_ai_reason(reason)


def _ai_available() -> bool:
    return runtime_ai_available(get_settings())


def _has_specialty(value: Any) -> bool:
    return bool(str(value or "").strip())


def _unique_doctor_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one representative schedule row per doctor for specialty scoring."""
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        if key and key not in unique:
            unique[key] = row
    return list(unique.values())


def _build_scoring_prompt(case: TriageCase, department: DepartmentResult, rows: list[dict[str, Any]]) -> str:
    candidates = [
        {
            "doctor_id": _row_key(row),
            "doctor": row.get("doctor") or row.get("doctor_name") or "",
            "childDept": _row_department(row),
            "specialty_tags": row.get("specialty_tags") or row.get("specialty") or "",
        }
        for row in rows
    ]
    return f"""你是醫療分診助理。請用一次批次回應，針對指定科別內每位候選醫師的實際專長做評分與簡短推薦理由。

推薦科別：{department.childDept}
病患症狀資料：
{json.dumps(_sanitized_patient_data(case), ensure_ascii=False, indent=2)}

候選醫師：
{json.dumps(candidates, ensure_ascii=False, indent=2)}

評分與理由規則：
1. 每位候選醫師都輸出一筆，score 必須是 0.0 到 1.0。
2. 只能評估目前症狀方向與該醫師 specialty_tags 的相關程度；同科醫師仍須依實際專長拉開差異，不可因同科就全部給高分。
3. 不參考醫師名氣、學歷、職稱或年資，不可新增 specialty_tags 未包含的專長。
4. 不可自行診斷疾病，不可把普通症狀描述成癌症或其他特定重大疾病。
5. 專長只有廣泛相關時分數要保守，專長與目前症狀方向非常直接相關時才給高分。
6. reason 使用自然繁體中文，說明使用者症狀方向、醫師實際專長，以及兩者為何相關或關聯有限。
7. reason 建議 30 到 55 個中文字，絕對不可超過 64 個中文字；不要輸出分數公式。
8. reason 不得出現「最適合」、「保證」、「一定」、「確診」。
9. doctor_id 必須完全照抄候選資料，不可虛構醫師。

請只輸出 JSON，不要輸出其他文字：
{{
  "scores": [
    {{"doctor_id": "必須完全照抄候選 doctor_id", "score": 0.0, "reason": "相符原因"}}
  ]
}}"""


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw).strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("AI response is not a JSON object")
    return data


def _iter_ai_scores(data: dict[str, Any]) -> list[dict[str, Any]]:
    scores = data.get("scores", data)
    if isinstance(scores, dict):
        return [
            {"doctor_id": key, **value}
            for key, value in scores.items()
            if isinstance(value, dict)
        ]
    if isinstance(scores, list):
        return [item for item in scores if isinstance(item, dict)]
    return []


def _case_text(case: TriageCase) -> str:
    patient = case.patient_input
    parts = [
        patient.symptom,
        patient.body_part or "",
        patient.duration or "",
        patient.severity or "",
        patient.onset or "",
        " ".join(patient.accompanying_symptoms),
        " ".join(patient.red_flags),
    ]
    return strip_negated_red_flags("；".join(part for part in parts if part))


def _sanitized_patient_data(case: TriageCase) -> dict[str, Any]:
    data = case.patient_input.model_dump()
    data["symptom"] = strip_negated_red_flags(str(data.get("symptom") or ""))
    data["body_part"] = strip_negated_red_flags(str(data.get("body_part") or "")) or None
    data["accompanying_symptoms"] = [
        cleaned
        for item in data.get("accompanying_symptoms", [])
        if (cleaned := strip_negated_red_flags(str(item)).strip())
    ]
    data["red_flags"] = list(case.patient_input.red_flags)
    data["red_flags_checked"] = case.patient_input.red_flags_checked
    return data


def _explainable_specialty_matches(patient_text: str, tags: str) -> list[tuple[str, str]]:
    concepts = [
        (
            ("頭", "頭痛", "頭暈", "眩暈", "暈眩", "麻木", "手腳麻"),
            ("神經", "腦", "腦血管", "頭痛", "眩暈", "平衡"),
            "頭部/神經相關症狀",
            "神經與腦血管相關專長",
        ),
        (
            ("腹", "肚", "胃", "腸", "嘔吐", "腹瀉", "拉肚子"),
            ("腸胃", "胃腸", "消化", "腹", "肝膽"),
            "腹部疼痛/腸胃疾病",
            "腸胃疾病",
        ),
        (
            ("膝", "關節", "走路", "爬樓梯", "骨", "扭傷", "腰"),
            ("膝", "關節", "骨科", "復健", "運動傷害"),
            "膝關節/骨科或復健",
            "骨科復健",
        ),
        (
            ("皮膚", "紅疹", "發癢", "癢", "濕疹", "過敏"),
            ("皮膚", "過敏", "濕疹", "蕁麻疹"),
            "皮膚紅疹/過敏",
            "皮膚過敏",
        ),
        (
            ("胸痛", "呼吸困難", "喘", "心悸"),
            ("心臟", "胸腔", "呼吸", "肺", "心律"),
            "胸痛/呼吸困難",
            "心肺相關專長",
        ),
    ]
    matches: list[tuple[str, str]] = []
    for symptom_terms, specialty_terms, direction, specialty in concepts:
        if any(term in patient_text for term in symptom_terms) and any(term in tags for term in specialty_terms):
            matches.append((direction, specialty))
    return matches


def _row_department(row: dict[str, Any]) -> str:
    return str(row.get("child_dept") or row.get("childDept") or "").strip()


def _row_doctor_id(row: dict[str, Any]) -> str:
    return str(row.get("doctor_id") or row.get("doctor") or row.get("doctor_name") or "").strip()


def _row_key(row: dict[str, Any]) -> str:
    return _row_doctor_id(row)
