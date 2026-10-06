from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.schemas import (
    DepartmentResult,
    PendingAnswerInterpretation,
    SemanticExtraction,
    TTASEvidence,
    TriageCase,
    UrgencyResult,
)
from app.services.ai_service import complete_runtime_json as _complete_runtime_json, runtime_ai_available
from app.services.confidence_scoring import ACCEPT_THRESHOLD, accepted
from app.services.conversation_service import SAFETY_PENDING_INTENT
from app.services.field_acceptance import ai_normalized_value_valid
from app.services.negation_utils import strip_negated_red_flags
from app.services.ttas_evidence import apply_ttas_evidence, validate_ttas_evidence_items
from app.services.ttas_rule_loader import TTASRuleLoadError, load_ttas_rules

logger = logging.getLogger(__name__)
_EVIDENCE_ASSERTIONS = {"present", "absent", "uncertain"}
_MEDICAL_ASSERTION_FIELDS = {"symptom", "accompanying_symptoms"}
_CLINICAL_EVIDENCE_FIELDS = {
    "symptom", "accompanying_symptoms", "body_part", "duration", "severity", "onset",
}


async def complete_prompt(prompt: str) -> str:
    """Compatibility seam for the legacy adapter; runtime provider is Cerebras."""
    return await _complete_runtime_json(prompt, purpose="semantic_extraction")


@dataclass
class RagTriageSuggestion:
    triage: UrgencyResult | None = None
    reply: str | None = None
    semantic_extractions: list[SemanticExtraction] | None = None
    pending_answer: PendingAnswerInterpretation | None = None
    ttas_evidence: list[TTASEvidence] | None = None
    interpretation_complete: bool = True


async def refine_case_with_ai(
    case: TriageCase, *, user_sources: list[str] | None = None,
) -> RagTriageSuggestion | None:
    """Extract grounded evidence from the current free-text turn."""
    if not _ai_available():
        logger.info("rag_triage_adapter refine skipped: AI key is not configured")
        return None

    prompt = _build_symptom_collection_prompt(case, user_sources or [])
    try:
        raw = await complete_prompt(prompt)
        logger.debug("rag_triage_adapter response received case_id=%s chars=%s", case.case_id, len(raw))
        data = _parse_json_object(raw)
    except Exception as exc:
        logger.warning(
            "rag_triage_adapter refine failed case_id=%s error_type=%s",
            case.case_id,
            type(exc).__name__,
        )
        return None

    if user_sources is None:
        user_sources = [message.content for message in case.history_records if message.role == "user"]
    pending_intent = _active_pending_intent(case)
    semantic_extractions, pending_answer, ttas_evidence = _validated_turn_interpretation(
        data,
        pending_intent=pending_intent,
        user_sources=user_sources,
    )
    if _answered_without_clinical_evidence(
        pending_answer, semantic_extractions, safety_context=pending_intent == SAFETY_PENDING_INTENT,
    ):
        repair_prompt = _build_turn_interpretation_repair_prompt(prompt)
        try:
            repair_raw = await complete_prompt(repair_prompt)
            logger.debug(
                "rag_triage_adapter repair response received case_id=%s chars=%s",
                case.case_id,
                len(repair_raw),
            )
            repair_data = _parse_json_object(repair_raw)
            semantic_extractions, pending_answer, ttas_evidence = _validated_turn_interpretation(
                repair_data,
                pending_intent=pending_intent,
                user_sources=user_sources,
            )
        except Exception as exc:
            logger.warning(
                "rag_triage_adapter repair failed case_id=%s error_type=%s",
                case.case_id,
                type(exc).__name__,
            )
            return RagTriageSuggestion(
                semantic_extractions=[], pending_answer=None, ttas_evidence=[], interpretation_complete=False,
            )
        if _answered_without_clinical_evidence(
            pending_answer, semantic_extractions, safety_context=pending_intent == SAFETY_PENDING_INTENT,
        ):
            logger.warning("rag_triage_adapter repair incomplete case_id=%s", case.case_id)
            return RagTriageSuggestion(
                semantic_extractions=[], pending_answer=None, ttas_evidence=[], interpretation_complete=False,
            )
    if semantic_extractions:
        # Import locally to keep the adapter independent at module load time.
        # The rule engine performs field allow-listing, confidence checks and
        # merge protection. Generative AI may never complete red-flag safety.
        from app.services.rule_engine import apply_semantic_extractions

        apply_semantic_extractions(
            case,
            semantic_extractions,
            allow_red_flag_completion=False,
        )
    if ttas_evidence:
        apply_ttas_evidence(case, ttas_evidence)

    reply = data.get("reply")
    return RagTriageSuggestion(
        triage=None,
        reply=str(reply).strip() if isinstance(reply, str) and reply.strip() else None,
        semantic_extractions=semantic_extractions,
        pending_answer=pending_answer,
        ttas_evidence=ttas_evidence,
    )


def _validated_turn_interpretation(
    data: dict[str, Any],
    *,
    pending_intent: str | None,
    user_sources: list[str],
) -> tuple[list[SemanticExtraction], PendingAnswerInterpretation | None, list[TTASEvidence]]:
    return (
        _semantic_extractions_from_ai(data.get("semantic_extractions"), user_sources=user_sources),
        _pending_answer_from_ai(
            data.get("pending_answer"),
            pending_intent=pending_intent,
            user_sources=user_sources,
        ),
        validate_ttas_evidence_items(data.get("ttas_evidence"), user_sources=user_sources),
    )


def _answered_without_clinical_evidence(
    pending_answer: PendingAnswerInterpretation | None,
    extractions: list[SemanticExtraction],
    *,
    safety_context: bool = False,
) -> bool:
    return bool(
        pending_answer is not None
        and pending_answer.answer_status == "answered"
        and pending_answer.answer_confidence >= ACCEPT_THRESHOLD
        and not (safety_context and pending_answer.answer_assertion in _EVIDENCE_ASSERTIONS)
        and not any(_is_accepted_clinical_evidence(item) for item in extractions)
    )


def _active_pending_intent(case: TriageCase) -> str | None:
    if case.conversation_state.clarification_status == "safety_check":
        return SAFETY_PENDING_INTENT
    return case.conversation_state.pending_clarification_intent


def _is_accepted_clinical_evidence(extraction: SemanticExtraction) -> bool:
    if extraction.field not in _CLINICAL_EVIDENCE_FIELDS:
        return False
    if (
        extraction.field in _MEDICAL_ASSERTION_FIELDS
        and extraction.assertion in {"absent", "uncertain"}
    ):
        return extraction.confidence >= ACCEPT_THRESHOLD
    return accepted(extraction.confidence, extraction.semantic_status)


async def detect_department_with_ai(
    case: TriageCase,
    departments: list[dict[str, Any]],
) -> DepartmentResult | None:
    """Pick a department from the DB-backed department list.

    This ports the useful rag_demo idea into backend service form: the AI sees
    the active DB department names and must return one exact child department.
    Invalid or invented departments are rejected so the caller can fall back to
    deterministic rules.
    """
    if not departments or not _ai_available():
        logger.info(
            "rag_triage_adapter department skipped case_id=%s has_departments=%s ai_available=%s",
            case.case_id,
            bool(departments),
            _ai_available(),
        )
        return None

    candidates = [item for item in departments if item.get("dept_id") is not None and item.get("child_dept")]
    if not candidates:
        return None

    prompt = _build_department_prompt(case, candidates)
    try:
        raw = await complete_prompt(prompt)
        logger.debug(
            "rag_triage_adapter department response received case_id=%s chars=%s",
            case.case_id,
            len(raw),
        )
        data = _parse_json_object(raw)
    except Exception as exc:
        logger.warning(
            "rag_triage_adapter department parse failed case_id=%s error_type=%s",
            case.case_id,
            type(exc).__name__,
        )
        return None

    dept_id = _optional_int(data.get("dept_id"))
    matches = [
        item for item in candidates
        if _optional_int(item.get("dept_id")) == dept_id
    ]
    if len(matches) != 1:
        logger.warning(
            "rag_triage_adapter rejected department case_id=%s dept_id=%s reason=dept_id_not_in_candidates",
            case.case_id,
            dept_id,
        )
        return None

    selected = matches[0]
    reasons = [str(item) for item in data.get("reason", []) if str(item).strip()]
    if not reasons:
        reasons = ["AI 依據問診內容從資料庫科別清單中選出此科別"]
    reasons.append("來源：rag_demo adapter，已限制於後端 DB 科別清單")

    return DepartmentResult(
        dept_id=int(selected["dept_id"]),
        parentDept=str(selected["parent_dept"]),
        childDept=str(selected["child_dept"]),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        reason=reasons,
    )


def merge_ai_next_question(case: TriageCase, suggestion: RagTriageSuggestion | None) -> bool:
    if suggestion is None or suggestion.triage is None:
        return False

    ai_question = suggestion.triage.next_question or suggestion.reply
    if not ai_question:
        return False

    attempted_override = bool(case.triage.next_question)
    if case.triage.next_question:
        logger.info(
            "rag_triage_adapter kept deterministic next_question case_id=%s ai_attempted_override=%s last_question_key=%s consumed_fields=%s question_attempts=%s field_statuses=%s red_flags_checked=%s",
            case.case_id,
            attempted_override,
            case.conversation_state.last_question_key,
            case.conversation_state.consumed_fields,
            case.conversation_state.question_attempts,
            case.conversation_state.field_statuses,
            case.patient_input.red_flags_checked,
        )
    else:
        logger.info(
            "rag_triage_adapter ignored AI next_question case_id=%s ai_attempted_override=%s last_question_key=%s consumed_fields=%s question_attempts=%s field_statuses=%s red_flags_checked=%s",
            case.case_id,
            attempted_override,
            case.conversation_state.last_question_key,
            case.conversation_state.consumed_fields,
            case.conversation_state.question_attempts,
            case.conversation_state.field_statuses,
            case.patient_input.red_flags_checked,
        )

    case.triage.reasons.append("AI 追問建議已記錄，但 next_question 由後端 deterministic state machine 決定。")
    return attempted_override


def _ai_available() -> bool:
    return runtime_ai_available(get_settings())


def _build_department_prompt(case: TriageCase, departments: list[dict[str, Any]]) -> str:
    dept_list = json.dumps(
        [
            {
                "dept_id": item.get("dept_id"),
                "parent_dept": item.get("parent_dept"),
                "child_dept": item.get("child_dept"),
            }
            for item in departments
        ],
        ensure_ascii=False,
        indent=2,
    )
    symptom_text = _case_text(case)
    candidate_names = {str(item.get("child_dept") or "").strip() for item in departments}
    orthopedic_guidance = ""
    if {"一般骨科", "骨科"}.issubset(candidate_names):
        orthopedic_guidance = """

骨科候選規則：一般肌肉骨骼、膝蓋、關節、走路或運動傷害症狀，若沒有兒童或其他細分依據，
優先選正式候選中的「一般骨科」；只有資料明確支持其他候選時才選其他骨科相關 dept_id。"""
    return f"""你是台灣醫院掛號分診助理。你只能從下列資料庫科別清單中選一個科別，不可以自行創造科別。

科別清單：
{dept_list}
{orthopedic_guidance}

病患資料：
{symptom_text}

請只輸出 JSON，不要輸出其他文字：
{{
  "dept_id": 1234,
  "confidence": 0.0,
  "reason": ["理由1", "理由2"]
}}"""


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw).strip().replace("```json", "").replace("```", "").strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("AI response is not a JSON object")
    return data


def _semantic_extractions_from_ai(
    value: Any,
    *,
    user_sources: list[str] | None = None,
) -> list[SemanticExtraction]:
    if not isinstance(value, list):
        return []

    extractions: list[SemanticExtraction] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").strip()
        status = str(item.get("semantic_status") or "unknown")
        source_text = str(item.get("source_text") or "").strip()
        raw_confidence = item.get("confidence", 0.0)
        if type(raw_confidence) not in {int, float}:
            continue
        confidence = float(raw_confidence)
        raw_assertion = item.get("assertion")
        assertion = raw_assertion if isinstance(raw_assertion, str) and raw_assertion in _EVIDENCE_ASSERTIONS else None
        medical_concept = field in _MEDICAL_ASSERTION_FIELDS and _has_nonempty_medical_concept(
            field,
            item.get("normalized_value"),
        )
        if (
            field not in {
                "symptom", "body_part", "duration", "severity", "onset",
                "accompanying_symptoms", "preferred_days", "preferred_sessions",
            }
            or status not in {"available", "unavailable", "unknown", "partial", "ambiguous"}
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or (medical_concept and assertion is None)
            or not _semantic_value_valid(field, item.get("normalized_value"), status, source_text)
            or not _source_text_grounded(source_text, user_sources or [])
        ):
            continue
        try:
            extractions.append(
                SemanticExtraction(
                    field=field,
                    normalized_value=item.get("normalized_value"),
                    semantic_status=status,
                    assertion=assertion,
                    confidence=confidence,
                    source_text=source_text,
                    needs_clarification=bool(item.get("needs_clarification", False)),
                    follow_up_reason=item.get("follow_up_reason"),
                    extractor="ai",
                )
            )
        except (TypeError, ValueError):
            logger.warning("rag_triage_adapter skipped invalid semantic extraction")
    return extractions


def _pending_answer_from_ai(
    value: Any,
    *,
    pending_intent: str | None,
    user_sources: list[str],
) -> PendingAnswerInterpretation | None:
    if pending_intent is None or not isinstance(value, dict):
        return None
    base_keys = {"answered_intent", "answer_status", "answer_source_text", "answer_confidence"}
    if set(value) not in {frozenset(base_keys), frozenset(base_keys | {"answer_assertion"})}:
        return None
    intent = value.get("answered_intent")
    status = value.get("answer_status")
    source = value.get("answer_source_text")
    confidence = value.get("answer_confidence")
    assertion = value.get("answer_assertion")
    if (
        intent != pending_intent
        or status not in {"answered", "partial", "unclear"}
        or not isinstance(source, str)
        or not source.strip()
        or not any(source.strip() in user_text for user_text in user_sources)
        or type(confidence) not in {int, float}
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
        or (assertion is not None and assertion not in _EVIDENCE_ASSERTIONS)
        or (pending_intent == SAFETY_PENDING_INTENT and assertion not in _EVIDENCE_ASSERTIONS)
    ):
        return None
    return PendingAnswerInterpretation(
        answered_intent=intent,
        answer_status=status,
        answer_source_text=source.strip(),
        answer_confidence=float(confidence),
        answer_assertion=assertion,
    )


def _has_nonempty_medical_concept(field: str, value: Any) -> bool:
    if field == "symptom":
        return isinstance(value, str) and bool(value.strip())
    if field == "accompanying_symptoms":
        return isinstance(value, list) and any(
            isinstance(item, str) and item.strip() for item in value
        )
    return False


def _source_text_grounded(source_text: str, user_sources: list[str]) -> bool:
    return bool(source_text) and any(source_text in source for source in user_sources)


def _semantic_value_valid(field: str, value: Any, status: str, source_text: str) -> bool:
    return ai_normalized_value_valid(field, value, status, source_text)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
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
        " ".join(patient.accompanying_symptoms),
        " ".join(patient.red_flags),
    ]
    return strip_negated_red_flags("；".join(part for part in parts if part))


def _build_turn_interpretation_repair_prompt(original_prompt: str) -> str:
    return (
        original_prompt
        + "\n\n修復要求：上一份輸出表示本輪已回答 clinical pending question，"
        "但沒有留下任何可接受的本輪 clinical semantic evidence，因此整份 Turn Interpretation 不完整。"
        "請重新解析同一份 current_user_text，仍輸出完整的 semantic_extractions、pending_answer 與 ttas_evidence。"
        "保留所有有逐字 source_text 支持的 present、absent、uncertain medical evidence，以及可驗證的"
        " body_part、duration、severity、onset；不得猜測、不得只補 pending_answer。"
    )


def _build_symptom_collection_prompt(
    case: TriageCase,
    current_user_text: list[str] | None = None,
) -> str:
    history = "\n".join(
        f"{'使用者' if message.role == 'user' else '助理'}: {message.content}"
        for message in case.history_records
    )
    current_input = case.patient_input.model_dump()
    pending_intent = _active_pending_intent(case)
    pending_question = next(
        (message.content for message in reversed(case.history_records) if message.role == "assistant"),
        None,
    ) if pending_intent else None
    pending_context = {
        "pending_clarification_intent": pending_intent,
        "pending_question": pending_question,
        "current_user_text": current_user_text or [],
    }
    try:
        prompt_evidence_catalog = load_ttas_rules().prompt_evidence_catalog
    except TTASRuleLoadError:
        prompt_evidence_catalog = {}

    return f"""你是醫療問診的語意抽取器。你只能做 extraction、normalization、confidence estimation。
不要在這個 extraction 回應中決定 next_question、stage、waiting_confirmation 或流程轉移；追問由獨立 clarification call 提議，狀態由 Backend 控制。

目前對話紀錄：
{history}

目前已收集到的症狀資料：{json.dumps(current_input, ensure_ascii=False, indent=2)}

你的任務：
1. 將使用者自然語言整理成 semantic_extractions。
2. semantic_status 只能使用 available、unavailable、unknown、partial、ambiguous。
3. confidence 使用 0 到 1。
4. 若信心不足，設定 needs_clarification=true 與 follow_up_reason。
5. 只輸出使用者本輪實際表達的欄位；若本輪明確修正舊值，可輸出新值，不得猜測。
  6. 不得輸出 red_flags。急迫性只能抽成 ttas_evidence，由 Backend 依正式規則判斷。
  7. 不要輸出 patient_input、triage、TTAS level、urgency_score、warning_required、red_flags_checked、next_question、stage、科別、確認狀態或掛號資訊。
8. duration 必須正規化為「數字+天／週／個月／年」，例如 3天、2週、6個月、1年；半年轉為 6個月，一年半轉為 18個月。
9. source_text 必須是直接支持該 extraction 的最短連續逐字原文（患者原話），不可改寫；不得包含無關的前後症狀、否定句或其他子句。
10. symptom 與 accompanying_symptoms 只要 normalized_value 含有 medical concept，就必須輸出 assertion：present 表示患者陳述存在、absent 表示患者否認、uncertain 表示患者不確定；其他欄位可省略 assertion 或輸出 null。
11. 同一 extraction 中的所有 normalized concept 必須具有相同 assertion；若同一句包含不同 polarity，必須拆成多筆 extraction，且每筆各自引用最短的 grounded source_text。不得把 mixed-polarity 整句共用為單一 medical extraction。
12. severity 的 normalized_value 只能輸出字串 "mild"、"moderate" 或 "severe"：輕微／還好 → mild，普通／中等／中度 → moderate，嚴重／很嚴重／痛到無法睡覺 → severe；不得輸出「輕微」「中等」「嚴重程度低」等其他字串。
13. onset 為簡短的發作描述字串；accompanying_symptoms 為伴隨症狀字串陣列，可使用 grounded source_text 的語意正規化概念，不可添加原文未表達的症狀。
14. 同一次 interpretation 也要判斷本輪是否回答 pending clarification。pending context 如下：
{json.dumps(pending_context, ensure_ascii=False, indent=2)}
15. 若 pending_clarification_intent 為 null，pending_answer 必須為 null。否則 pending_answer 包含 answered_intent、answer_status、answer_source_text、answer_confidence，並可包含 answer_assertion；answered_intent 只是 Backend/AI 間的 opaque correlation key，必須原樣 echo pending intent。是否回答應只比較實際 pending_question 與 current_user_text，不得解析 key 名稱或要求患者回答 key 中看似列出的全部概念。answer_status 只能是 answered、partial、unclear，source 必須是 current_user_text 的最短連續逐字片段，confidence 為 0 到 1。明確否定仍可構成 answered。
  15a. pending intent 為 safety_screen 時，answer_assertion 必填：present 表示患者明確回報 safety 問題中的一項或多項狀況，absent 表示患者明確否認整份 safety 問題，uncertain 表示無法確定。不得輸出 red_flags_checked；Backend 只會把 grounded、高信心的 answered+absent 視為 safety negative。
  16. pending_answer 只表示本輪是否回答實際 pending_question，不能取代 semantic_extractions。即使 pending question 已回答，仍必須完整抽取本輪所有額外、明確、grounded 的 medical evidence；不得只輸出 pending_answer。
  17. 同一次 Turn Interpretation 必須完成 TTAS evidence 掃描。請逐一對照下方「目前 production 可執行的 TTAS evidence catalog」與 current_user_text。凡本輪患者原文直接支持的 evidence，都必須輸出；即使同一個事實已經同時出現在 semantic_extractions，也不可因此省略 ttas_evidence。
  18. catalog 的 label_zh / meaning_zh 是語意概念，不是固定關鍵字清單。患者可以使用不同自然說法；只要語意直接等價，而且 source_text 能逐字在 current_user_text 找到，即可抽取。不得只因出現相似字詞就硬套欄位。
  19. 每筆 ttas_evidence 只能包含 field、value、semantic_status、confidence、source_text。field 只能從下方 catalog 選；value 必須符合該 field 的 type。semantic_status 只能是 available、unknown、ambiguous。
  20. source_text 必須是本輪 current_user_text 中直接支持該 evidence 的最短連續逐字片段。不得從 prior history 製造本輪 TTAS evidence。若同一事實也出現在 semantic_extractions，兩邊可以引用相同或各自最精準的 source_text。
  21. confidence 只表示「患者本輪原文有多明確支持這個結構化事實」，不是病情嚴重度、疾病機率、TTAS level 機率，也不是你對醫療判斷的信心。原文沒有直接支持時，不得用較低 confidence 猜一筆 evidence。
  22. 缺少資料時不要補正常值。使用者未提供血壓、心率、SpO2、GCS、體溫、血糖、年齡、懷孕週數等資料時，不得假設正常，也不得自行換算。使用者明確表示不知道時，才可在有 grounded source_text 的前提下輸出 unknown / ambiguous。
  23. 不得把患者的主觀形容或一般症狀升格成臨床 modifier、病因或診斷。特別是：不得從「喘得很嚴重」自行產生 respiratory_distress=severe；不得自行判定 shock、ill_appearing、cardiac_chest_pain_suspected、high_risk_injury_mechanism 或 pain_location_class；不得由「昏迷／叫不醒」自行估算 GCS；不得由「很燒／很冷」自行估算 temperature_c；不得由眼痛／灼熱反推化學暴露；不得由紅疹／腫脹反推昆蟲螫傷；不得由疼痛／腫脹反推 open fracture 或骨折／脫臼變形。只有 current_user_text 直接支持 catalog 所描述的結構化事實時才可輸出。
  24. age_years / age_months 只在本輪有逐字年齡證據時抽取；若 Backend 另有 trusted profile，該 trusted fact 由 Backend 合併，不需要你猜。
  25. 你只負責抽 TTAS evidence。不得推算、提議或輸出任何 TTAS 級數；也不得輸出 urgency_score、warning_required、red_flags_checked、stage、is_complete、科別、醫師、掛號決策。

目前 production 可執行的 TTAS evidence catalog：
{json.dumps(prompt_evidence_catalog, ensure_ascii=False, indent=2)}

以下是 clinical pending-answer contract 範例。實際 answered_intent 必須換成目前 pending_clarification_intent 的原值，所有 source_text 必須來自實際 current_user_text。請只輸出 JSON，不要輸出其他文字：
{{
  "semantic_extractions": [
    {{
      "field": "accompanying_symptoms",
      "normalized_value": ["聽力下降"],
      "semantic_status": "available",
      "assertion": "absent",
      "confidence": 0.98,
      "source_text": "沒有聽力下降",
      "needs_clarification": false,
      "follow_up_reason": null
    }},
    {{
      "field": "body_part",
      "normalized_value": "右耳",
      "semantic_status": "available",
      "assertion": null,
      "confidence": 0.98,
      "source_text": "右耳",
      "needs_clarification": false,
      "follow_up_reason": null
    }},
    {{
      "field": "accompanying_symptoms",
      "normalized_value": ["耳鳴"],
      "semantic_status": "available",
      "assertion": "present",
      "confidence": 0.98,
      "source_text": "右耳一直有耳鳴",
      "needs_clarification": false,
      "follow_up_reason": null
    }},
    {{
      "field": "accompanying_symptoms",
      "normalized_value": ["眩暈"],
      "semantic_status": "available",
      "assertion": "present",
      "confidence": 0.98,
      "source_text": "眩暈",
      "needs_clarification": false,
      "follow_up_reason": null
    }}
  ],
  "pending_answer": {{
    "answered_intent": "<current pending intent>",
    "answer_status": "answered",
    "answer_source_text": "沒有聽力下降",
    "answer_confidence": 0.98,
    "answer_assertion": "absent"
  }},
  "ttas_evidence": []
}}

TTAS 正例（同一句可同時產生 semantic_extractions 與 ttas_evidence）：
current_user_text:「剛剛清潔劑濺到我的右眼，現在眼睛灼痛。」
{{
  "semantic_extractions": [
    {{
      "field": "symptom",
      "normalized_value": "灼痛",
      "semantic_status": "available",
      "assertion": "present",
      "confidence": 0.99,
      "source_text": "眼睛灼痛",
      "needs_clarification": false,
      "follow_up_reason": null
    }},
    {{
      "field": "body_part",
      "normalized_value": "右眼",
      "semantic_status": "available",
      "assertion": null,
      "confidence": 0.99,
      "source_text": "右眼",
      "needs_clarification": false,
      "follow_up_reason": null
    }}
  ],
  "pending_answer": null,
  "ttas_evidence": [
    {{
      "field": "chemical_eye_injury",
      "value": true,
      "semantic_status": "available",
      "confidence": 0.99,
      "source_text": "清潔劑濺到我的右眼"
    }}
  ]
}}

TTAS 反例（不能從症狀倒推暴露原因）：
current_user_text:「我的右眼很痛。」
{{
  "semantic_extractions": [
    {{
      "field": "symptom",
      "normalized_value": "眼痛",
      "semantic_status": "available",
      "assertion": "present",
      "confidence": 0.99,
      "source_text": "右眼很痛",
      "needs_clarification": false,
      "follow_up_reason": null
    }},
    {{
      "field": "body_part",
      "normalized_value": "右眼",
      "semantic_status": "available",
      "assertion": null,
      "confidence": 0.99,
      "source_text": "右眼",
      "needs_clarification": false,
      "follow_up_reason": null
    }}
  ],
  "pending_answer": null,
  "ttas_evidence": []
}}"""
