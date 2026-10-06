from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.schemas import DepartmentResult, PatientInput, TriageCase
from app.services.ai_service import complete_runtime_json as _complete_runtime_json, runtime_ai_available
from app.services.negation_utils import strip_negated_red_flags

logger = logging.getLogger(__name__)


async def complete_prompt(prompt: str) -> str:
    """Compatibility seam for tests; production always uses Cerebras JSON mode."""
    return await _complete_runtime_json(prompt, purpose="department_detection")


@dataclass(frozen=True)
class DepartmentKeywordHint:
    childDept: str
    matched_keywords: tuple[str, ...]
    coverage: float


# Minimal UTF-8 subset extracted from project-smart dept_keywords.py, with a
# few local synonyms so natural phrases map to the same project-smart concepts.
PROJECT_SMART_KEYWORD_TO_DEPT: dict[str, tuple[str, ...]] = {
    "皮膚科": (
        "疹子",
        "紅疹",
        "皮膚",
        "皮膚紅腫痛",
        "青春痘",
        "掉頭髮",
        "皮下腫塊",
        "癢",
        "發癢",
    ),
    "骨科": (
        "關節腫",
        "關節痛",
        "膝關節疼痛",
        "膝蓋",
        "膝",
        "走路",
        "爬樓梯",
        "腰酸背痛",
        "骨",
    ),
    "復健醫學": (
        "關節腫",
        "關節痛",
        "膝關節疼痛",
        "膝蓋",
        "走路不穩",
        "走路",
        "肌肉萎縮",
        "肌肉抽筋",
        "腰酸背痛",
        "手腳會麻",
    ),
    "心臟內科": (
        "胸痛",
        "胸悶",
        "心悸",
        "呼吸急促困難",
        "呼吸困難",
        "冒冷汗",
    ),
    "胸腔內科": (
        "胸痛",
        "呼吸急促困難",
        "呼吸困難",
        "咳嗽",
        "咳血",
        "嘯鳴",
    ),
    "一般內科": (
        "頭痛",
        "頭暈",
        "咳嗽",
        "發燒",
        "感冒",
    ),
    "家庭醫學科(一般門診/戒菸)": (
        "發燒",
        "頭痛",
        "頭暈",
        "癢",
        "疹子",
        "紅疹",
        "咳嗽",
    ),
}


async def detect_department_with_project_smart_adapter(
    case: TriageCase,
    departments: list[dict[str, Any]],
) -> DepartmentResult | None:
    """Use project-smart department ideas without importing its app stack.

    The adapter is intentionally optional: it returns None whenever AI is not
    configured, times out, fails, or proposes a department outside the backend's
    active department list. The caller must keep deterministic fallback logic.
    """
    hints = keyword_department_hints(case)
    logger.info(
        "project_smart_department_adapter keyword hints case_id=%s hint_count=%s",
        case.case_id,
        len(hints),
    )

    if not departments:
        logger.info(
            "project_smart_department_adapter skipped case_id=%s reason=no_active_departments",
            case.case_id,
        )
        return None

    requested_result = _requested_department_result(case, departments)
    if requested_result is not None:
        logger.info(
            "project_smart_department_adapter requested department selected case_id=%s dept_id=%s fallback=false",
            case.case_id,
            requested_result.dept_id,
        )
        return requested_result

    pediatric_result = _pediatric_department_result(case, departments)
    if pediatric_result is not None:
        logger.info(
            "project_smart_department_adapter pediatric rule selected case_id=%s dept_id=%s fallback=false",
            case.case_id,
            pediatric_result.dept_id,
        )
        return pediatric_result

    if not _ai_available():
        logger.info(
            "project_smart_department_adapter ai skipped case_id=%s reason=cerebras_api_key_not_configured",
            case.case_id,
        )
        return None

    prompt = _build_department_prompt(case, departments, hints)
    try:
        raw = await complete_prompt(prompt)
        logger.debug(
            "project_smart_department_adapter response received case_id=%s chars=%s",
            case.case_id,
            len(raw),
        )
        data = _parse_json_object(raw)
    except Exception as exc:
        status = _exception_status(exc)
        logger.warning(
            "[DEPARTMENT] detection_called=true candidate_count=%s selected_dept_id=null "
            "selected_parent=null selected_child=null validation_result=provider_failure "
            "failure_reason=%s:%s",
            len(departments),
            type(exc).__name__,
            status,
        )
        return None

    dept_id = _safe_int(data.get("dept_id"))
    selected = _candidate_by_id(departments, dept_id)
    if selected is None:
        logger.warning(
            "[DEPARTMENT] detection_called=true candidate_count=%s selected_dept_id=%s "
            "selected_parent=null selected_child=null validation_result=rejected "
            "failure_reason=dept_id_not_in_candidates",
            len(departments),
            dept_id,
        )
        return None

    child = str(selected["child_dept"])
    hint_coverage = next((hint.coverage for hint in hints if hint.childDept == child), 0.0)
    confidence_ai = _safe_float(data.get("confidence"), 0.0)
    confidence = round(max(confidence_ai * 0.7 + hint_coverage * 0.3, confidence_ai), 2)
    reasons = _reason_list(data.get("reason"))
    if hint_coverage:
        reasons.append(f"project-smart keyword hint matched coverage={hint_coverage:.2f}")
    reasons.append("來源：project-smart department adapter，AI 結果已限制於後端 active department list")

    result = DepartmentResult(
        dept_id=int(selected["dept_id"]),
        parentDept=str(selected["parent_dept"]),
        childDept=str(selected["child_dept"]),
        confidence=confidence,
        reason=reasons,
    )
    logger.info(
        "[DEPARTMENT] detection_called=true candidate_count=%s selected_dept_id=%s "
        "validation_result=adapter_valid failure_reason=null",
        len(departments),
        result.dept_id,
    )
    return result


def keyword_department_hints(case_or_patient: TriageCase | PatientInput) -> list[DepartmentKeywordHint]:
    patient = case_or_patient.patient_input if isinstance(case_or_patient, TriageCase) else case_or_patient
    fragments = _patient_fragments(patient)
    total = max(len(fragments), 1)
    hints: list[DepartmentKeywordHint] = []

    for child, keywords in PROJECT_SMART_KEYWORD_TO_DEPT.items():
        matched = {
            keyword
            for keyword in keywords
            if any(keyword in fragment or fragment in keyword for fragment in fragments)
        }
        if matched:
            hints.append(
                DepartmentKeywordHint(
                    childDept=child,
                    matched_keywords=tuple(sorted(matched)),
                    coverage=round(min(len(matched) / total, 1.0), 2),
                )
            )

    return sorted(hints, key=lambda hint: (-hint.coverage, hint.childDept))


def _ai_available() -> bool:
    return runtime_ai_available(get_settings())


def _exception_status(exc: Exception) -> str:
    direct = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if direct is not None:
        return str(direct)
    match = re.search(r"\b([45]\d{2})\b", str(exc))
    return match.group(1) if match else "unknown"


def _build_department_prompt(
    case: TriageCase,
    departments: list[dict[str, Any]],
    hints: list[DepartmentKeywordHint],
) -> str:
    candidate_payload = [
        {
            "dept_id": item.get("dept_id"),
            "parent_dept": item.get("parent_dept"),
            "child_dept": item.get("child_dept"),
        }
        for item in departments
        if item.get("dept_id") is not None and item.get("child_dept")
    ]
    if hints:
        hint_lines = "\n".join(
            f"- {hint.childDept}: 命中 {', '.join(hint.matched_keywords)}"
            for hint in hints
        )
    else:
        hint_lines = "（關鍵字未命中明顯科別，請依症狀自行判斷）"

    orthopedic_guidance = ""
    candidate_names = {str(item.get("child_dept") or "").strip() for item in candidate_payload}
    if {"一般骨科", "骨科"}.issubset(candidate_names):
        orthopedic_guidance = """

【骨科候選選擇規則】
一般肌肉骨骼、膝蓋、關節、走路或運動傷害症狀，若資料沒有支持兒童骨科或其他更細分方向，
優先選正式候選中的「一般骨科」。只有病患資料明確支持其他候選時才選其他骨科相關 dept_id。
不得只因文字出現「骨」或「骨科」泛稱，就忽略「一般骨科」候選。"""

    return f"""你是台北榮民總醫院的分診助理。以下是病患症狀資料：
{json.dumps(_sanitized_patient_data(case), ensure_ascii=False, indent=2)}

【project-smart 關鍵字初步分析】
{hint_lines}
{orthopedic_guidance}

可選擇的科別清單（只能選一個既有 dept_id；正式科別名稱由 Backend 從同一筆資料補入）：
{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}

請判斷最適合的正式科別。若資訊不足以可靠選擇，請輸出 null，不得自行套用預設科別。
請只輸出以下 JSON，不要輸出任何其他文字：
{{
  "dept_id": 1234,
  "confidence": 0.0,
  "reason": ["依病患症狀說明原因", "說明關鍵字命中或未命中情況"]
}}"""


def _patient_fragments(patient: PatientInput) -> list[str]:
    values = [
        patient.symptom,
        patient.body_part or "",
        patient.duration or "",
        patient.severity or "",
        patient.onset or "",
        patient.department_context or "",
        *patient.accompanying_symptoms,
        *patient.red_flags,
    ]
    return [strip_negated_red_flags(str(value)).strip() for value in values if strip_negated_red_flags(str(value)).strip()]


def _requested_department_result(case: TriageCase, departments: list[dict[str, Any]]) -> DepartmentResult | None:
    if case.patient_input.red_flags:
        return None
    requested_id = case.patient_input.requested_department_id
    requested_name = case.patient_input.requested_department_name
    if requested_id is not None:
        matches = [
            item
            for item in departments
            if _safe_int(item.get("dept_id")) == requested_id
            and (not requested_name or str(item.get("child_dept") or "").strip() == requested_name)
        ]
        if len(matches) == 1:
            item = matches[0]
            child = str(item.get("child_dept") or "").strip()
            return DepartmentResult(
                dept_id=requested_id,
                parentDept=str(item.get("parent_dept") or ""),
                childDept=child,
                confidence=0.9,
                reason=[f"使用者明確指定正式科別 {child}，且目前無陽性急迫症狀"],
            )
    text = _case_text(case)
    for item in sorted(departments, key=lambda row: len(str(row.get("child_dept") or "")), reverse=True):
        child = str(item.get("child_dept") or "").strip()
        if not child:
            continue
        if any(marker in text for marker in (f"想看{child}", f"想看 {child}", f"要看{child}", f"要看 {child}", f"看{child}", f"看 {child}")):
            return DepartmentResult(
                dept_id=_safe_int(item.get("dept_id")),
                parentDept=str(item.get("parent_dept") or ""),
                childDept=child,
                confidence=0.84,
                reason=[f"使用者明確表示想看 {child}，且目前無陽性急迫症狀，優先尊重科別偏好"],
            )
    return None


def _pediatric_department_result(
    case: TriageCase,
    departments: list[dict[str, Any]],
) -> DepartmentResult | None:
    age = _patient_age(case)
    if age is None or age >= 18:
        return None

    pediatric_candidates = [
        item
        for item in departments
        if item.get("child_dept") and _is_pediatric_department(item)
    ]
    if not pediatric_candidates:
        logger.info(
            "project_smart_department_adapter pediatric skipped case_id=%s reason=no_active_pediatric_department",
            case.case_id,
        )
        return None

    text = _case_text(case)
    ranked: list[tuple[int, dict[str, Any]]] = []
    for item in pediatric_candidates:
        score = _pediatric_match_score(text, item)
        if score > 0:
            ranked.append((score, item))
    if not ranked:
        logger.info(
            "project_smart_department_adapter pediatric skipped case_id=%s reason=no_equivalent_pediatric_department",
            case.case_id,
        )
        return None

    selected = sorted(
        ranked,
        key=lambda pair: (-pair[0], str(pair[1].get("child_dept") or "")),
    )[0][1]
    child = str(selected.get("child_dept") or "").strip()
    parent = str(selected.get("parent_dept") or "").strip()
    return DepartmentResult(
        dept_id=_safe_int(selected.get("dept_id")),
        parentDept=parent,
        childDept=child,
        confidence=0.82,
        reason=[
            f"病患年齡 {age} 歲，符合未滿 18 歲兒科優先規則",
            f"症狀與 {child} 可處理方向相符",
            "科別已由後端 active department list 驗證",
        ],
    )


def _patient_age(case: TriageCase) -> int | None:
    direct_age = getattr(case.patient_input, "age", None)
    if direct_age is not None:
        try:
            return int(direct_age)
        except (TypeError, ValueError):
            return None

    text = _case_text(case)
    match = re.search(r"(?<!\d)([1-9]\d?)\s*(?:歲|岁|y/o|yo|years?\s*old)(?!\d)", text, re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _case_text(case: TriageCase) -> str:
    patient = case.patient_input
    parts = [
        patient.symptom,
        patient.body_part or "",
        patient.duration or "",
        patient.severity or "",
        patient.onset or "",
        patient.department_context or "",
        patient.requested_department_name or "",
        " ".join(patient.accompanying_symptoms),
        " ".join(patient.red_flags),
        " ".join(message.content for message in case.history_records if message.role == "user"),
    ]
    return strip_negated_red_flags(" ".join(part for part in parts if part))


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


def _is_pediatric_department(item: dict[str, Any]) -> bool:
    child = str(item.get("child_dept") or "")
    parent = str(item.get("parent_dept") or "")
    text = f"{parent} {child}"
    return any(marker in text for marker in ("兒科", "兒童", "小兒", "孩童", "兒"))


def _pediatric_match_score(text: str, item: dict[str, Any]) -> int:
    child = str(item.get("child_dept") or "")
    dept_text = f"{item.get('parent_dept') or ''} {child}"

    rules = [
        (
            ("腹", "肚", "胃", "腸", "嘔吐", "吐", "腹瀉", "拉肚子"),
            ("腸胃", "胃腸", "消化", "腹", "肝膽", "內科"),
        ),
        (
            ("咳", "喘", "呼吸", "發燒", "喉嚨", "流鼻水", "感冒"),
            ("胸腔", "呼吸", "感染", "內科"),
        ),
        (
            ("皮膚", "紅疹", "發癢", "癢", "濕疹", "過敏"),
            ("皮膚", "過敏"),
        ),
    ]
    for symptom_terms, department_terms in rules:
        if any(term in text for term in symptom_terms) and any(term in dept_text for term in department_terms):
            return 100 if any(term in child for term in department_terms) else 70
    return 0


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw).strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("AI response is not a JSON object")
    return data


def _reason_list(value: Any) -> list[str]:
    if isinstance(value, list):
        reasons = [str(item).strip() for item in value if str(item).strip()]
    else:
        reasons = []
    return reasons or ["AI 依據症狀與 project-smart keyword hint 產生科別判斷"]


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _candidate_by_id(
    departments: list[dict[str, Any]],
    dept_id: int | None,
) -> dict[str, Any] | None:
    matches = [
        item
        for item in departments
        if _safe_int(item.get("dept_id")) == dept_id
    ]
    return matches[0] if len(matches) == 1 else None
