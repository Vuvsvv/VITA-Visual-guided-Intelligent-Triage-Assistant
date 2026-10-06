"""Official-KB-grounded department candidates with live-DB and source validation."""

from __future__ import annotations

import json
import logging
import math
import re

from app.db import fetch_active_departments
from app.schemas import CandidateDepartment, CandidateDepartmentEvidence, DepartmentResult, TriageCase
from app.services.ai_service import complete_runtime_json
from app.services.confidence_scoring import ACCEPT_THRESHOLD
from app.services.department_knowledge import (
    canonical_department_id, load_department_knowledge, lookup_concept, normalize_concept, resolve_department_names,
)
from app.services.department_preference_service import resolve_requested_department
from app.services.field_acceptance import ai_normalized_value_valid

logger = logging.getLogger(__name__)
_INTENT = re.compile(r"[a-z][a-z0-9_]{2,63}\Z")
_ASSERTED_MEDICAL_FIELDS = {"symptom", "accompanying_symptoms"}
_CANDIDATE_CONTEXT_FIELDS = {
    "symptom", "accompanying_symptoms", "body_part", "duration", "severity", "onset",
}
_EVIDENCE_ASSERTIONS = {"present", "absent", "uncertain"}
_SEMANTIC_STATUSES = {"available", "unavailable", "unknown", "partial", "ambiguous"}


def _user_texts(case: TriageCase) -> list[str]:
    return [message.content for message in case.history_records if message.role == "user"]


def validated_semantic_evidence_history(case: TriageCase) -> list[dict]:
    """Return validated evidence in append order without interpreting raw text."""
    history = _user_texts(case)
    evidence: list[dict] = []
    for item in case.semantic_extractions:
        source = item.source_text.strip()
        if (
            item.field not in _CANDIDATE_CONTEXT_FIELDS
            or item.semantic_status not in _SEMANTIC_STATUSES
            or type(item.confidence) not in {int, float}
            or not math.isfinite(item.confidence)
            or item.confidence < ACCEPT_THRESHOLD
            or not source
            or not any(source in text for text in history)
            or not ai_normalized_value_valid(
                item.field,
                item.normalized_value,
                item.semantic_status,
                source,
            )
        ):
            continue
        if item.field in _ASSERTED_MEDICAL_FIELDS:
            if item.assertion not in _EVIDENCE_ASSERTIONS:
                continue
            concepts = (
                [item.normalized_value]
                if isinstance(item.normalized_value, str)
                else item.normalized_value
            )
            for concept in concepts if isinstance(concepts, list) else []:
                if not isinstance(concept, str) or not concept.strip():
                    continue
                evidence.append({
                    "field": item.field,
                    "normalized_value": concept.strip(),
                    "source_text": source,
                    "assertion": item.assertion,
                    "semantic_status": item.semantic_status,
                    "confidence": float(item.confidence),
                })
        elif item.semantic_status in {"available", "partial"}:
            evidence.append({
                "field": item.field,
                "normalized_value": item.normalized_value,
                "source_text": source,
                "assertion": None,
                "semantic_status": item.semantic_status,
                "confidence": float(item.confidence),
            })
    return evidence


def effective_semantic_evidence(case: TriageCase) -> list[dict]:
    """Collapse validated evidence history into the latest current fact per key."""
    current: dict[tuple[str, ...], tuple[int, dict]] = {}
    for position, evidence in enumerate(validated_semantic_evidence_history(case)):
        field = evidence["field"]
        if field in _ASSERTED_MEDICAL_FIELDS:
            key = ("concept", field, normalize_concept(str(evidence["normalized_value"])))
        else:
            key = ("field", field)
        current[key] = (position, evidence)
    return [evidence for _, evidence in sorted(current.values(), key=lambda item: item[0])]


def accepted_semantic_evidence(case: TriageCase) -> list[dict]:
    """Compatibility name for current effective structured semantic evidence."""
    return effective_semantic_evidence(case)


def retrieve_official_evidence(case: TriageCase, records: list[dict], resolutions: list[dict]) -> list[dict]:
    """Match official concepts only against validated present semantic evidence."""
    resolved = {
        entry["canonical_dept_id"]: entry
        for entry in resolutions if entry["status"] == "resolved"
    }
    concepts = {record["concept"] for record in records}
    found: dict[tuple[int, str, str, str], dict] = {}
    for evidence in accepted_semantic_evidence(case):
        if (
            evidence["assertion"] != "present"
            or evidence["semantic_status"] not in {"available", "partial"}
        ):
            continue
        source = evidence["source_text"]
        retrieval_surfaces = {
            normalize_concept(str(value))
            for value in (evidence["normalized_value"], source)
            if isinstance(value, str) and value.strip()
        }
        for concept in concepts:
            normalized_concept = normalize_concept(concept)
            if not any(normalized_concept in surface for surface in retrieval_surfaces):
                continue
            for record in lookup_concept(concept, records):
                resolution = resolved.get(record["canonical_dept_id"])
                if resolution is None:
                    continue
                key = (resolution["db_dept_id"], source, record["source_id"], record["concept"])
                found[key] = {
                    "dept_id": resolution["db_dept_id"],
                    "parentDept": resolution["db_parent_dept"],
                    "childDept": resolution["db_child_dept"],
                    "patient_source_text": source,
                    "knowledge_source_id": record["source_id"],
                    "knowledge_concept": record["concept"],
                    "knowledge_evidence_text": record["evidence_text"],
                    "source_priority": record["source_priority"],
                }
    return list(found.values())


def validate_candidate_proposal(
    proposal: object, retrieved: list[dict], sources: list[dict], active_departments: list[dict],
    user_texts: list[str],
) -> tuple[str, list[CandidateDepartment], str | None, str | None]:
    if not isinstance(proposal, dict):
        return "unresolved", [], None, None
    status = proposal.get("status")
    if status not in {"resolved", "ambiguous", "unresolved"}:
        status = "unresolved"
    allowed_sources = {source["source_id"] for source in sources}
    allowed = {
        (item["dept_id"], item["patient_source_text"], item["knowledge_source_id"], item["knowledge_concept"]): item
        for item in retrieved
    }
    active_ids: dict[int, int] = {}
    for row in active_departments:
        number = canonical_department_id(row.get("dept_id"))
        if number is not None:
            active_ids[number] = active_ids.get(number, 0) + 1
    validated: list[CandidateDepartment] = []
    seen_ids: set[int] = set()
    raw_candidates = proposal.get("candidates")
    for raw in raw_candidates if isinstance(raw_candidates, list) else []:
        if not isinstance(raw, dict):
            continue
        dept_id, confidence = raw.get("dept_id"), raw.get("confidence")
        supports = raw.get("supporting_evidence")
        if (
            type(dept_id) is not int or dept_id <= 0 or dept_id in seen_ids
            or active_ids.get(dept_id) != 1
            or type(confidence) not in {int, float} or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or not isinstance(supports, list) or not supports
        ):
            continue
        valid_supports: list[CandidateDepartmentEvidence] = []
        live_item = None
        for support in supports:
            if not isinstance(support, dict):
                valid_supports = []
                break
            source_text = support.get("patient_source_text")
            source_id = support.get("knowledge_source_id")
            concept = support.get("knowledge_concept")
            if (
                not isinstance(source_text, str) or not source_text.strip()
                or source_text != source_text.strip()
                or not any(source_text in text for text in user_texts)
                or not isinstance(source_id, str) or source_id not in allowed_sources
                or not isinstance(concept, str)
            ):
                valid_supports = []
                break
            item = allowed.get((dept_id, source_text, source_id, concept))
            if item is None or (live_item is not None and (
                live_item["parentDept"], live_item["childDept"]
            ) != (item["parentDept"], item["childDept"])):
                valid_supports = []
                break
            live_item = item
            valid_supports.append(CandidateDepartmentEvidence(
                patient_source_text=source_text,
                knowledge_source_id=source_id,
                knowledge_concept=concept,
            ))
        if not valid_supports or live_item is None:
            continue
        seen_ids.add(dept_id)
        validated.append(CandidateDepartment(
            dept_id=dept_id, parentDept=live_item["parentDept"], childDept=live_item["childDept"],
            confidence=float(confidence), supporting_evidence=valid_supports,
        ))
    validated.sort(key=lambda item: item.confidence, reverse=True)
    validated = validated[:3]
    reason = proposal.get("uncertainty_reason")
    reason = reason.strip()[:240] if isinstance(reason, str) and reason.strip() else None
    intent = proposal.get("next_question_intent")
    intent = intent if isinstance(intent, str) and _INTENT.fullmatch(intent) else None
    return status, validated, reason, intent


async def reason_about_departments(case: TriageCase) -> None:
    """Recompute candidates from current evidence; AI output never owns workflow state."""
    state = case.conversation_state
    state.department_status = "unresolved"
    state.candidate_departments = []
    state.department_uncertainty_reason = None
    state.department_next_information_needed = None
    state.department_next_question_intent = None
    case.department_result = None
    try:
        active = fetch_active_departments()
        if not active:
            state.department_uncertainty_reason = "正式科別主資料目前無法取得"
            return
        preference = resolve_requested_department(case, active)
        if preference is not None:
            case.department_result = preference
            state.department_status = "resolved"
            return
        sources, records = load_department_knowledge()
        retrieved = retrieve_official_evidence(case, records, resolve_department_names(records, active))
        if not retrieved:
            state.department_uncertainty_reason = "目前沒有可精確對應正式科別的官方症狀證據"
            return
        payload = {
            "accepted_semantic_evidence": accepted_semantic_evidence(case),
            "retrieved_official_evidence": retrieved,
            "previously_asked_intents": state.asked_clarification_intents,
        }
        prompt = (
            "僅比較 retrieved_official_evidence 中的正式 DB 科別，不可新增科別、猜 dept_id、診斷或控制流程。"
            "accepted_semantic_evidence 是已驗證的 present/absent/uncertain 語意證據；"
            "retrieved_official_evidence 只會由 present 證據建立，候選 supporting_evidence 只能引用其中項目。"
            "輸出 JSON：status (resolved/ambiguous/unresolved), candidates (最多 3 個，每項 dept_id, "
            "confidence 0..1, supporting_evidence 陣列；每筆有逐字 patient_source_text, knowledge_source_id, "
            "knowledge_concept), uncertainty_reason, next_question_intent。"
            "如目前仍不確定，提出可區分候選的英文 snake_case next_question_intent。它是 Backend/AI 間的"
            " opaque correlation key，只代表下一個 patient-answerable clinical dimension，不是 workflow 指令；"
            "優先一個聚焦面向，避免產生含 _or_ 或 _and_ 的 compound key。不要直接向患者顯示科別。"
            f"\n資料：{json.dumps(payload, ensure_ascii=False)}"
        )
        proposal = json.loads((await complete_runtime_json(prompt, purpose="department_convergence")).strip())
        proposed_status, candidates, reason, intent = validate_candidate_proposal(
            proposal, retrieved, sources, active, _user_texts(case),
        )
        state.candidate_departments = candidates
        state.department_uncertainty_reason = reason
        state.department_next_question_intent = intent if (
            state.pending_clarification_intent is None and (proposed_status != "resolved" or len(candidates) != 1)
        ) else None
        # The proposal contract currently provides a diagnostic uncertainty reason and
        # a machine intent, not a separately validated patient-answerable information need.
        state.department_next_information_needed = None
        if not candidates:
            state.department_status = "unresolved"
        elif proposed_status == "resolved" and len(candidates) == 1 and candidates[0].confidence >= ACCEPT_THRESHOLD:
            winner = candidates[0]
            state.department_status = "resolved"
            case.department_result = DepartmentResult(
                dept_id=winner.dept_id, parentDept=winner.parentDept, childDept=winner.childDept,
                confidence=winner.confidence,
                reason=["依已驗證患者原文、官方科別證據與正式 Department 主資料收斂"],
            )
        else:
            state.department_status = "ambiguous"
    except Exception as exc:
        logger.warning("department convergence unavailable case_id=%s error=%s", case.case_id, type(exc).__name__)
        state.department_status = "unresolved"
        state.candidate_departments = []
        state.department_uncertainty_reason = "科別證據比對暫時無法完成"
