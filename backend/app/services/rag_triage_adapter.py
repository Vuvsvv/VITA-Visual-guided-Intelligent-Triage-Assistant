from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.config import get_settings
from app.schemas import (
    ConceptAnswer,
    DepartmentPreferenceInterpretation,
    DialogueActInterpretation,
    PendingAnswerInterpretation,
    PriorAnswerReference,
    SemanticExtraction,
    TTASEvidence,
    TimePreference,
    TriageCase,
    UrgencyResult,
)
from app.services.ai_service import complete_runtime_json as _complete_runtime_json, runtime_ai_available
from app.services.confidence_scoring import ACCEPT_THRESHOLD, accepted
from app.services.conversation_service import (
    AVAILABILITY_PENDING_INTENT, SAFETY_NET_PENDING_INTENT, SAFETY_PENDING_INTENT, accepted_prior_answer_evidence,
    invalidate_corrected_department_context,
    validate_prior_answer_reference,
)
from app.services.rule_engine import QUESTION_TEXTS, RED_FLAG_QUESTION_KEY
from app.services.department_clues import pending_clue_concepts
from app.services.field_acceptance import ai_normalized_value_rejection_reason, ai_normalized_value_valid
from app.services.department_preference_service import validate_department_preference, active_department_candidates
from app.services.pediatric_routing import apply_patient_profile, validate_patient_profile
from app.services.ttas_evidence import apply_ttas_evidence, validate_ttas_evidence_items
from app.services.ttas_rule_loader import TTASRuleLoadError, load_ttas_rules
from app.services.schedule_filter import TAIPEI_ZONE
from app.services.time_preference_service import apply_time_preferences, validate_and_resolve_time_preferences

logger = logging.getLogger(__name__)
_EVIDENCE_ASSERTIONS = {"present", "absent", "uncertain"}
_SAFETY_NET_CHOICES = {"family_medicine", "continue"}
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
    time_preferences: list[TimePreference] | None = None
    interpretation_complete: bool = True
    # The repair call itself failed (timeout, provider error), unlike an answer that broke the rules.
    provider_failed: bool = False
    dialogue_act: DialogueActInterpretation | None = None
    prior_answer_reference: PriorAnswerReference | None = None
    department_preference: DepartmentPreferenceInterpretation | None = None


def _log_time_preferences(case_id: str, proposed: object, accepted: list[TimePreference]) -> None:
    """One line per turn with time data: what the AI proposed was accepted, with its rank."""
    if not proposed and not accepted:
        return
    logger.info(
        "time preferences case_id=%s proposed=%s accepted=%s",
        case_id, len(proposed) if isinstance(proposed, list) else 0,
        json.dumps([dict(kind=item.kind, relation=item.relation, priority=item.priority,
                         target=item.weekday or item.session or item.weekdays or item.date_value or item.day_offset or item.week_offset,
                         dates=item.resolved_dates, source=item.source_text) for item in accepted],
                   ensure_ascii=False),
    )


async def refine_case_with_ai(
    case: TriageCase, *, user_sources: list[str] | None = None,
) -> RagTriageSuggestion | None:
    """Extract grounded evidence from the current free-text turn."""
    if not _ai_available():
        logger.info("rag_triage_adapter refine skipped: runtime_unavailable deterministic_fallback=true")
        return None

    if user_sources is None:
        user_sources = [message.content for message in case.history_records if message.role == "user"]
    prior_extractions = tuple(item.model_copy(deep=True) for item in case.semantic_extractions)
    prior_user_sources = _prior_user_sources(case, user_sources)
    time_preference_reference_date = datetime.now(TAIPEI_ZONE).date()
    prompt = _build_symptom_collection_prompt(case, user_sources)
    try:
        raw = await complete_prompt(prompt)
        logger.debug("rag_triage_adapter response received case_id=%s chars=%s", case.case_id, len(raw))
        try:
            data = _parse_json_object(raw)
        except (ValueError, json.JSONDecodeError):
            # Malformed top-level output must use the same one-repair gate.
            data = {}
    except Exception as exc:
        logger.warning(
            "rag_triage_adapter refine failed case_id=%s error_type=%s",
            case.case_id,
            "invalid_json" if isinstance(exc, (ValueError, json.JSONDecodeError)) else "provider_timeout" if isinstance(exc, TimeoutError) else type(exc).__name__,
        )
        return None

    pending_intent = _active_pending_intent(case)
    clue_concepts = _pending_clue_concepts(case)
    dialogue_act = _dialogue_act_from_ai(data.get("dialogue_act"), user_sources)
    semantic_extractions, pending_answer, ttas_evidence = _validated_turn_interpretation(
        data,
        pending_intent=pending_intent,
        user_sources=user_sources,
        clue_concepts=clue_concepts,
    )
    prior_reference = validate_prior_answer_reference(
        data.get("prior_answer_reference"), dialogue_act, pending_intent, prior_extractions, prior_user_sources,
    )
    time_preferences = validate_and_resolve_time_preferences(
        data.get("time_preferences"),
        user_sources=user_sources,
        reference_date=time_preference_reference_date,
    )
    _log_time_preferences(case.case_id, data.get("time_preferences"), time_preferences)
    department_preference = validate_department_preference(data.get("department_preference"), user_sources)
    profile_sources = [message.content for message in case.history_records if message.role == "user"]
    patient_profile = validate_patient_profile(data.get("patient_profile"), profile_sources)
    repair_reason = _turn_interpretation_rejection_reason(
        data,
        pending_intent=pending_intent,
        user_sources=user_sources,
        dialogue_act=dialogue_act,
        semantic_extractions=semantic_extractions,
        pending_answer=pending_answer,
        ttas_evidence=ttas_evidence,
        time_preferences=time_preferences,
        department_preference=department_preference,
        clue_concepts=clue_concepts,
        patient_profile=patient_profile,
    )

    if repair_reason is not None:
        logger.info("turn interpretation accepted=false reason=%s repair_attempted=true", repair_reason)
        repair_prompt = _build_turn_interpretation_repair_prompt(prompt, repair_reason)
        try:
            repair_raw = await complete_prompt(repair_prompt)
            logger.debug(
                "rag_triage_adapter repair response received case_id=%s chars=%s",
                case.case_id,
                len(repair_raw),
            )
            repair_data = _parse_json_object(repair_raw)
            data = repair_data
            dialogue_act = _dialogue_act_from_ai(data.get("dialogue_act"), user_sources)
            semantic_extractions, pending_answer, ttas_evidence = _validated_turn_interpretation(
                repair_data,
                pending_intent=pending_intent,
                user_sources=user_sources,
                clue_concepts=clue_concepts,
            )
            prior_reference = validate_prior_answer_reference(
                data.get("prior_answer_reference"), dialogue_act, pending_intent, prior_extractions, prior_user_sources,
            )
            time_preferences = validate_and_resolve_time_preferences(
                repair_data.get("time_preferences"),
                user_sources=user_sources,
                reference_date=time_preference_reference_date,
            )
        except Exception as exc:
            logger.warning(
                "rag_triage_adapter repair failed case_id=%s error_type=%s",
                case.case_id,
                type(exc).__name__,
            )
            return RagTriageSuggestion(
                semantic_extractions=[], pending_answer=None, ttas_evidence=[], interpretation_complete=False,
                provider_failed=not isinstance(exc, (ValueError, json.JSONDecodeError)),
            )

        department_preference = validate_department_preference(data.get("department_preference"), user_sources)
        patient_profile = validate_patient_profile(data.get("patient_profile"), profile_sources)
        remaining_rejection = _turn_interpretation_rejection_reason(
            repair_data,
            pending_intent=pending_intent,
            user_sources=user_sources,
            dialogue_act=dialogue_act,
            semantic_extractions=semantic_extractions,
            pending_answer=pending_answer,
            ttas_evidence=ttas_evidence,
            time_preferences=time_preferences,
            department_preference=department_preference,
            clue_concepts=clue_concepts,
            patient_profile=patient_profile,
        )
        if remaining_rejection is not None:
            logger.warning(
                "rag_triage_adapter repair incomplete case_id=%s reason=%s",
                case.case_id,
                remaining_rejection,
            )
            return RagTriageSuggestion(
                semantic_extractions=[], pending_answer=None, ttas_evidence=[], interpretation_complete=False,
            )
        logger.info("turn interpretation accepted=true repair_attempted=true")
    replace_existing_fields = {
        item.field for item in semantic_extractions
        if dialogue_act.act == "correction" and _is_accepted_clinical_evidence(item)
        and item.semantic_status in {"available", "partial", "unavailable"}
        and item.assertion != "uncertain"
    }
    previous_values = {
        field: value.copy() if isinstance(value, list) else value
        for field in replace_existing_fields
        for value in [getattr(case.patient_input, field)]
    }
    if semantic_extractions:
        # Import locally to keep the adapter independent at module load time.
        # The rule engine performs field allow-listing, confidence checks and
        # merge protection. Generative AI may never complete red-flag safety.
        from app.services.rule_engine import apply_semantic_extractions

        apply_semantic_extractions(
            case,
            semantic_extractions,
            allow_red_flag_completion=False,
            replace_existing_fields=replace_existing_fields,
        )
    corrected_fields = {
        field for field, previous in previous_values.items()
        if previous and previous != getattr(case.patient_input, field)
    }
    if corrected_fields:
        invalidate_corrected_department_context(case)
        for field in sorted(corrected_fields):
            logger.info("semantic correction case_id=%s field=%s accepted=true reason=correction_replacement_applied",
                        case.case_id, field)
    if ttas_evidence:
        apply_ttas_evidence(case, ttas_evidence)
    if time_preferences:
        apply_time_preferences(case.availability, time_preferences)

    logger.info("patient profile case_id=%s present=%s accepted=%s subject=%s age_stated=%s",
                case.case_id, data.get("patient_profile") is not None, patient_profile is not None,
                patient_profile["subject"] if patient_profile else None,
                bool(patient_profile and patient_profile["age_text"]))
    apply_patient_profile(case.conversation_state, patient_profile)
    if department_preference is not None:
        case.patient_input.requested_department_id = department_preference.dept_id
        case.patient_input.requested_department_name = department_preference.department_name
    reply = data.get("reply")
    return RagTriageSuggestion(
        triage=None,
        reply=str(reply).strip() if isinstance(reply, str) and reply.strip() else None,
        semantic_extractions=semantic_extractions,
        pending_answer=pending_answer,
        ttas_evidence=ttas_evidence,
        time_preferences=time_preferences,
        dialogue_act=dialogue_act,
        prior_answer_reference=prior_reference,
        department_preference=department_preference,
    )


def _prior_user_sources(case: TriageCase, current_user_sources: list[str]) -> list[str]:
    """Exclude the appended current request, including multi-message requests."""
    user_texts = [item.content for item in case.history_records if item.role == "user"]
    count = len(current_user_sources)
    if count and user_texts[-count:] == [text.strip() for text in current_user_sources]:
        return user_texts[:-count]
    return user_texts


def _dialogue_act_rejection_reason(value: Any, user_sources: list[str]) -> str | None:
    """Validate the AI dialogue act without interpreting patient wording in Python."""
    if value is None:
        return "missing_dialogue_act"
    if not isinstance(value, dict) or set(value) != {"act", "source_text", "confidence"}:
        return "invalid_dialogue_act_schema"
    source, confidence = value.get("source_text"), value.get("confidence")
    if not isinstance(source, str) or not source.strip():
        return "missing_dialogue_act_source"
    if not any(source.strip() in text for text in user_sources):
        return "ungrounded_dialogue_act_source"
    if (type(confidence) not in {int, float} or not math.isfinite(confidence)
            or not ACCEPT_THRESHOLD <= confidence <= 1):
        return "invalid_dialogue_act_confidence"
    try:
        DialogueActInterpretation(**value)
    except (TypeError, ValueError):
        return "invalid_dialogue_act_value"
    return None


def _dialogue_act_from_ai(value: Any, user_sources: list[str]) -> DialogueActInterpretation:
    """Dialogue interpretation has no authority over accepted medical facts or state."""
    reason = _dialogue_act_rejection_reason(value, user_sources)
    if reason is not None:
        logger.info("turn dialogue act accepted=false reason=%s", reason)
        return DialogueActInterpretation()
    return DialogueActInterpretation(**value)


def _turn_interpretation_rejection_reason(
    data: dict[str, Any],
    *,
    pending_intent: str | None,
    user_sources: list[str],
    dialogue_act: DialogueActInterpretation,
    semantic_extractions: list[SemanticExtraction],
    pending_answer: PendingAnswerInterpretation | None,
    ttas_evidence: list[TTASEvidence],
    time_preferences: list[TimePreference],
    department_preference: DepartmentPreferenceInterpretation | None,
    clue_concepts: list[str] | None = None,
    patient_profile: dict | None = None,
) -> str | None:
    """Reject structurally incomplete AI turns; never infer user semantics here."""
    required = {
        "semantic_extractions", "pending_answer", "ttas_evidence", "time_preferences",
        "dialogue_act", "prior_answer_reference", "department_preference",
    }
    # patient_profile is optional: an older or terse reply without it still parses.
    if not required <= set(data) <= required | {"patient_profile"}:
        return "invalid_turn_interpretation_fields"
    if any(not isinstance(data[key], list) for key in (
        "semantic_extractions", "ttas_evidence", "time_preferences",
    )) or any(data.get(key) is not None and not isinstance(data[key], dict) for key in (
        "pending_answer", "prior_answer_reference", "department_preference", "patient_profile",
    )):
        return "invalid_turn_interpretation_shape"
    dialogue_reason = _dialogue_act_rejection_reason(data.get("dialogue_act"), user_sources)
    if dialogue_reason is not None:
        return dialogue_reason
    # A correction may change the visit time or the department, not only a symptom
    # (例：修訂後「我想要下週三不然就下週二」).
    if dialogue_act.act == "correction" and not (
        any(_is_accepted_clinical_evidence(item) for item in semantic_extractions)
        or time_preferences or department_preference is not None
    ):
        return "correction_without_grounded_evidence"
    if dialogue_act.act == "new_information" and not (
        any(accepted(item.confidence, item.semantic_status) for item in semantic_extractions)
        or ttas_evidence or time_preferences or department_preference is not None or patient_profile is not None
    ):
        return "new_information_without_grounded_evidence"
    if pending_intent is not None and dialogue_act.act == "answer" and pending_answer is None:
        return "answer_without_pending_answer"
    # A clue question claimed answered must say which official concepts were answered;
    # partial/unclear without concept answers is a legitimate miss that Backend re-asks.
    if (clue_concepts and dialogue_act.act == "answer" and pending_answer is not None
            and pending_answer.answer_status == "answered" and not pending_answer.concept_answers):
        return "answer_without_concept_answers"
    if (
        pending_intent is not None
        and pending_answer is None
        and not any(_is_accepted_clinical_evidence(item) for item in semantic_extractions)
        and dialogue_act.act in {"unknown", "off_topic"}
    ):
        return "unresolved_pending_turn"
    return None


def _validated_turn_interpretation(
    data: dict[str, Any],
    *,
    pending_intent: str | None,
    user_sources: list[str],
    clue_concepts: list[str] | None = None,
) -> tuple[list[SemanticExtraction], PendingAnswerInterpretation | None, list[TTASEvidence]]:
    return (
        _semantic_extractions_from_ai(data.get("semantic_extractions"), user_sources=user_sources),
        _pending_answer_from_ai(
            data.get("pending_answer"),
            pending_intent=pending_intent,
            user_sources=user_sources,
            clue_concepts=clue_concepts,
        ),
        validate_ttas_evidence_items(data.get("ttas_evidence"), user_sources=user_sources),
    )


def _active_pending_intent(case: TriageCase) -> str | None:
    if case.conversation_state.clarification_status == "safety_check":
        return SAFETY_PENDING_INTENT
    return case.conversation_state.pending_clarification_intent


def _pending_clue_concepts(case: TriageCase) -> list[str]:
    if _active_pending_intent(case) == SAFETY_PENDING_INTENT:
        return []
    return pending_clue_concepts(case.conversation_state)


def _is_accepted_clinical_evidence(extraction: SemanticExtraction) -> bool:
    if extraction.field not in _CLINICAL_EVIDENCE_FIELDS:
        return False
    if (
        extraction.field in _MEDICAL_ASSERTION_FIELDS
        and extraction.assertion in {"absent", "uncertain"}
    ):
        return extraction.confidence >= ACCEPT_THRESHOLD
    return accepted(extraction.confidence, extraction.semantic_status)


def _ai_available() -> bool:
    return runtime_ai_available(get_settings())


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
        normalized_value = item.get("normalized_value")
        if field == "body_part" and _renamed_body_part(normalized_value, status, source_text):
            # Anatomy stays in the patient's words: an AI rename (「屁股」→「臀部」) keeps the stated
            # part instead of dropping it, so later steps still see 「屁股」 next to its symptom.
            logger.info("semantic extraction body_part renamed_by_ai=true kept=source_text")
            normalized_value = source_text
        if (
            field not in {
                "symptom", "body_part", "duration", "severity", "onset",
                "accompanying_symptoms", "preferred_days", "preferred_sessions",
            }
            or status not in {"available", "unavailable", "unknown", "partial", "ambiguous"}
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
            or (medical_concept and assertion is None)
            or not _semantic_value_valid(field, normalized_value, status, source_text)
            or not _source_text_grounded(source_text, user_sources or [])
        ):
            continue
        try:
            extractions.append(
                SemanticExtraction(
                    field=field,
                    normalized_value=normalized_value,
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
    clue_concepts: list[str] | None = None,
) -> PendingAnswerInterpretation | None:
    if pending_intent is None or not isinstance(value, dict):
        return None
    base_keys = {"answered_intent", "answer_status", "answer_source_text", "answer_confidence"}
    if (not base_keys <= set(value) or set(value) - base_keys - {"answer_assertion", "concept_answers", "answer_choice"}
            or not isinstance(value.get("concept_answers", []), list)):
        logger.info("turn pending answer accepted=false reason=invalid_schema")
        return None
    intent = value.get("answered_intent")
    status = value.get("answer_status")
    source = value.get("answer_source_text")
    confidence = value.get("answer_confidence")
    assertion = value.get("answer_assertion")
    # Only the safety-net question carries a choice; other intents drop it.
    choice = value.get("answer_choice") if pending_intent == SAFETY_NET_PENDING_INTENT else None
    if (
        intent != pending_intent
        or not isinstance(status, str) or status not in {"answered", "partial", "unclear"}
        or not isinstance(source, str)
        or not source.strip()
        or not any(source.strip() in user_text for user_text in user_sources)
        or type(confidence) not in {int, float}
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
        or (assertion is not None and (not isinstance(assertion, str) or assertion not in _EVIDENCE_ASSERTIONS))
        or (pending_intent == SAFETY_PENDING_INTENT and assertion not in _EVIDENCE_ASSERTIONS)
        or (choice is not None and choice not in _SAFETY_NET_CHOICES)
    ):
        reason = (
            "wrong_intent" if intent != pending_intent else
            "ungrounded_source" if not isinstance(source, str) or not source.strip()
            or not any(source.strip() in text for text in user_sources) else
            "invalid_status_confidence_or_assertion"
        )
        logger.info("turn pending answer accepted=false reason=%s", reason)
        return None
    return PendingAnswerInterpretation(
        answered_intent=intent,
        answer_status=status,
        answer_source_text=source.strip(),
        answer_confidence=float(confidence),
        answer_assertion=assertion,
        concept_answers=_concept_answers_from_ai(value.get("concept_answers", []), clue_concepts or [], user_sources),
        answer_choice=choice,
    )


def _concept_answers_from_ai(value: list, clue_concepts: list[str], user_sources: list[str]) -> list[ConceptAnswer]:
    """Keep only per-concept answers naming an asked official concept, grounded in this turn."""
    accepted: list[ConceptAnswer] = []
    for item in value:
        concept = item.get("concept") if isinstance(item, dict) else None
        source = item.get("source_text") if isinstance(item, dict) else None
        confidence = item.get("confidence") if isinstance(item, dict) else None
        assertion = item.get("answer_assertion") if isinstance(item, dict) else None
        reason = (
            "invalid_schema" if not isinstance(item, dict)
            or set(item) != {"concept", "answer_assertion", "source_text", "confidence"} else
            "concept_not_asked" if concept not in clue_concepts else
            "duplicate_concept" if any(answer.concept == concept for answer in accepted) else
            "invalid_assertion" if not isinstance(assertion, str) or assertion not in _EVIDENCE_ASSERTIONS else
            "ungrounded_source" if not isinstance(source, str) or not source.strip()
            or not any(source.strip() in text for text in user_sources) else
            "invalid_confidence" if type(confidence) not in {int, float} or not math.isfinite(confidence)
            or not ACCEPT_THRESHOLD <= confidence <= 1 else None
        )
        if reason is not None:
            logger.info("turn concept answer accepted=false reason=%s", reason)
            continue
        accepted.append(ConceptAnswer(
            concept=concept, answer_assertion=assertion, source_text=source.strip(), confidence=float(confidence),
        ))
    return accepted


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


def _renamed_body_part(value: Any, status: str, source_text: str) -> bool:
    """A well-formed body part the AI renamed away from its own verbatim source_text."""
    return (
        ai_normalized_value_rejection_reason("body_part", value, status, source_text)
        == "normalized_value_not_supported_by_source"
        and ai_normalized_value_valid("body_part", source_text, status, source_text)
    )


def _build_turn_interpretation_repair_prompt(original_prompt: str, rejection_reason: str) -> str:
    return (
        original_prompt
        + "\n\n修復要求：上一份 Turn Interpretation 未通過 Backend 結構驗證。"
        f" rejection_reason={rejection_reason}。"
        "請重新解析同一份 current_user_text；不要改寫患者事實，也不要控制 workflow。"
        "這次必須完整輸出 semantic_extractions、pending_answer、ttas_evidence、time_preferences、"
        "dialogue_act、prior_answer_reference、department_preference 七個頂層欄位，以及 patient_profile；不適用時用 [] 或 null，不得省略。"
        "若 dialogue_act=request_explanation，pending_answer 可以為 null，且不得捏造 medical evidence。"
        "若 rejection_reason=answer_without_concept_answers，pending_answer 必須依規則 15b 補上 concept_answers。"
        "若 dialogue_act=correction，必須輸出本輪逐字原文可支持的更正 evidence；"
        "correction act 本身不能取代 evidence。"
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
    pending_question = case.conversation_state.pending_clarification_question or next(
        (message.content for message in reversed(case.history_records) if message.role == "assistant"),
        None,
    ) if pending_intent else None
    pending_context = {
        "pending_clarification_intent": pending_intent,
        "pending_question": pending_question,
        "current_user_text": current_user_text or [],
        "complete_safety_screen": QUESTION_TEXTS[RED_FLAG_QUESTION_KEY] if pending_intent == SAFETY_PENDING_INTENT else None,
        "prior_accepted_semantic_evidence": [
            item.model_dump() for item in accepted_prior_answer_evidence(
                tuple(case.semantic_extractions), _prior_user_sources(case, current_user_text or []),
            )
        ] if pending_intent and pending_intent != SAFETY_PENDING_INTENT else [],
        "pending_clue_concepts": _pending_clue_concepts(case),
        "pending_availability_dimensions": list(case.conversation_state.availability_pending_dimensions)
        if pending_intent == AVAILABILITY_PENDING_INTENT else [],
        "pending_safety_net_choice": pending_intent == SAFETY_NET_PENDING_INTENT,
    }
    state = case.conversation_state
    known_patient_profile = dict(
        subject=state.patient_subject, relation_text=state.patient_relation_text, age_text=state.patient_age_text,
    ) if state.patient_subject else None
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
  7. 不要輸出 patient_input、triage、TTAS level、urgency_score、warning_required、red_flags_checked、next_question、stage、科別判斷、確認狀態或掛號資訊；明確指定科別只能使用 department_preference。
8. duration 必須正規化為「數字+天／週／個月／年」，例如 3天、2週、6個月、1年；半年轉為 6個月，一年半轉為 18個月。
9. source_text 必須是直接支持該 extraction 的最短連續逐字原文（患者原話），不可改寫；不得包含與該 extraction 無關的前後症狀或其他子句。若 assertion=absent，source_text 應保留直接表達否定的最短原文，不得移除否定詞。
10. symptom 與 accompanying_symptoms 只要 normalized_value 含有 medical concept，就必須輸出 assertion：present 表示患者陳述存在、absent 表示患者否認、uncertain 表示患者不確定；其他欄位可省略 assertion 或輸出 null。
11. 同一 extraction 中的所有 normalized concept 必須具有相同 assertion；若同一句包含不同 polarity，必須拆成多筆 extraction，且每筆各自引用最短的 grounded source_text。不得把 mixed-polarity 整句共用為單一 medical extraction。
12. severity 的 normalized_value 只能輸出字串 "mild"、"moderate" 或 "severe"：輕微／還好 → mild，普通／中等／中度 → moderate，嚴重／很嚴重／痛到無法睡覺 → severe；不得輸出「輕微」「中等」「嚴重程度低」等其他字串。
13. onset 為簡短的發作描述字串；accompanying_symptoms 為伴隨症狀字串陣列，可使用 grounded source_text 的語意正規化概念，不可添加原文未表達的症狀。本輪表達多個 present 症狀時，最主要的一個放 symptom，其餘各自以 accompanying_symptoms 輸出並引用各自最短的 grounded source_text。
14. 同一次 interpretation 也要判斷本輪是否回答 pending clarification。pending context 如下：
{json.dumps(pending_context, ensure_ascii=False, indent=2)}
15. 若 pending_clarification_intent 為 null，pending_answer 必須為 null。否則 pending_answer 包含 answered_intent、answer_status、answer_source_text、answer_confidence，並可包含 answer_assertion；answered_intent 只是 Backend/AI 間的 opaque correlation key，必須原樣 echo pending intent。是否回答應只比較實際 pending_question 與 current_user_text，不得解析 key 名稱或要求患者回答 key 中看似列出的全部概念。answer_status 只能是 answered、partial、unclear，source 必須是 current_user_text 的最短連續逐字片段，confidence 為 0 到 1。明確否定仍可構成 answered。
  15a. pending intent 為 safety_screen 時，answer_assertion 必填：present 表示患者明確回報安全問題中的狀況，absent 只表示患者明確否認 complete_safety_screen 整份問題，uncertain 表示無法確定。即使 pending_question 已縮小成聚焦追問，僅否認這一項也不能當成整份陰性。部分否認應為 partial，不知道或離題應為 unclear；抱怨跳針不得當成陰性。保留本輪回報的頭痛、頭暈等 evidence，不得由「其他沒有」否定已回報的症狀。不得輸出 red_flags_checked；Backend 只會把 grounded、高信心的 answered+absent 視為整份 safety negative。明確陽性仍須抽對應 catalog 內 TTAS evidence，不能由 answer_assertion 決定級數。
  15b. 若 pending_clue_concepts 非空，pending_question 是詢問這些官方症狀概念的線索題。pending_answer 必須另含 concept_answers 陣列，每項只含 concept、answer_assertion、source_text、confidence：concept 必須逐字取自 pending_clue_concepts（可把患者用語對應到官方概念，但輸出官方字串）；answer_assertion 為 present、absent 或 uncertain，患者說「醫師說過我有 ○○」「以前被說有 ○○」也視為 present；source_text 為 current_user_text 中支持該回答的最短連續逐字片段，簡短回答如「有」「沒有」也可以，但只在能明確判斷它回答了哪些概念時才輸出；confidence 為 0 到 1。患者沒有回答到的概念不要輸出，不得猜測。患者對整題籠統回答「不知道」「不清楚」「不確定」時，這是回答而不是答非所問：這題全部 pending_clue_concepts 都輸出 answer_assertion=uncertain，answer_status=answered。若全部概念都沒有回答，answer_status 用 partial 或 unclear，concept_answers 用 []。pending_clue_concepts 為空時 concept_answers 用 [] 或省略。
  15c. 若 pending_question 是詢問最近哪幾天、哪個時段方便看診（pending_availability_dimensions 列出這題問的面向：date＝日期、session＝時段；只列一個時是只追問該面向），患者表示「都可以」「沒有限制」也是 answered；提到具體可以或不行的日期、時段時，pending_answer 為 answered，且這些時間必須同時依規則 26 輸出 time_preferences。答非所問或只談症狀時，不得把它當成已回答時間。
  15d. 若 pending_safety_net_choice=true，pending_question 是詢問患者要繼續描述症狀，還是先掛家庭醫學科。pending_answer 必須另含 answer_choice：患者明確選擇先掛家庭醫學科時為 family_medicine；這題能掛的只有家庭醫學科，所以「先決定掛號」「先掛號」「先掛就好」「直接掛號」這類想先掛號、不再描述的回答也是 family_medicine，明確選擇繼續描述時為 continue，兩者皆 answer_status=answered；無法判斷、離題或只談症狀時 answer_status 用 unclear，answer_choice 用 null。answer_choice 只依實際 pending_question 與 current_user_text 的意思判斷；同一句另外提到的症狀仍必須照常輸出 semantic_extractions。pending_safety_net_choice=false 時不要輸出 answer_choice。
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
  26. 同一次 interpretation 另以 time_preferences 表達本輪明確的看診時間限制與順位。kind 只能是 date、date_range、relative_day、relative_week、relative_weekday、weekday、weekday_group、session；relation 只能是 exclude、prefer、acceptable。prefer/acceptable 必須有正整數 priority，exclude 的 priority 必須是 null。相對星期只輸出 week_offset 與英文 weekday，不得自行輸出實際日期；Backend 會以 Asia/Taipei 當輪日期換算。「今天／明天／後天／大後天」「N 天後」用 kind=relative_day，只輸出 day_offset（今天 0、明天 1、後天 2、大後天 3、N 天後 N），不得輸出 date_value。kind=date 只用於原文說出幾月幾號的日期，而且必須有 date_value；你不知道今天的日期，不得用 kind=date 表示相對日期。沒有「這／本／下／下下」等字的星期幾（例「最好週三」「週五不行」）是指每週的那一天，必須用 kind=weekday，不得當成本週的 relative_weekday；只有原文明確說「這週三」「本週三」「下週三」時才用 relative_weekday。session 只能是 morning、afternoon、evening。每筆 source_text 必須是本輪最短連續逐字原文，confidence 為 0 到 1。沒有提到的時段不得輸出 exclude；「最好下午」是 prefer，不是只能下午；「只能下午」須把其他時段分別輸出 exclude。AI 不得選 SQL Schedule 或推薦排名。
  27. 同一次輸出 dialogue_act，僅含 act、source_text、confidence。act 只能是 answer、request_explanation、claims_already_answered、correction、new_information、decline_further_questions、unknown、off_topic。decline_further_questions 只用於使用者明確表示不想再回答問題、或已無法再提供更多資訊（例如「不想再回答了，直接幫我選科」「我真的說不出更多了」）；使用者表示想直接掛號、不要再問（例如「我要先掛號」「直接幫我掛號」「我想要掛號不要再問」）也是 decline_further_questions，即使句子以「我說」開頭也一樣：那是在重申要求，不是 claims_already_answered；單純回答「沒有」「不知道」某一題是 answer，不是 decline_further_questions。以本輪逐字原文支持對話行為，confidence 為有限 0 到 1。要求解釋問題不是未回答的醫療事實；表示已回答不等於已有可接受證據。dialogue_act 不得取代 semantic_extractions、pending_answer 或 TTAS evidence；一句話要求解釋並新增症狀時，兩者都要抽取。不得透過 act 設定任何 workflow 或安全狀態。
  28. 新增 prior_answer_reference，預設為 null。只有 dialogue_act=claims_already_answered、有 active pending intent 且不是 safety_screen 時，才可引用 prior_accepted_semantic_evidence 中真正回答該 pending question 的既有證據；只輸出 answered_intent、answer_source_text、answer_confidence。answered_intent 必須完全等於 pending intent，source 必須逐字等於一筆 prior accepted extraction 的 source_text，confidence 為 0 到 1。不得只引用 raw history、不得引用本輪剛抽的新證據、不得把 uncertain 當已回答；找不到可靠答案就保持 null。此 reference 不代表科別已收斂，也絕不可完成 safety screen。
  29. dialogue_act=correction 時，仍須逐欄抽出本輪直接支持的更正 evidence；若修正 medical concept，分別輸出舊概念的 grounded absent 與新概念的 grounded present，不得刪掉無關症狀。correction act 本身不會改值；Backend 只接受 grounded、高信心且 schema 合法的本輪欄位。不得輸出 revision_mode 或其他 workflow 指令。
  30. 每次回覆都必須包含八個頂層欄位：semantic_extractions、pending_answer、ttas_evidence、time_preferences、dialogue_act、prior_answer_reference、department_preference、patient_profile。沒有內容時分別使用 [] 或 null，不得省略。dialogue_act 一律必須輸出。

31. department_preference 預設為 null，只有本輪明確要求、指定或改指定科別才輸出；不得由症狀推論偏好，不可塞進 symptom。僅含 dept_id（整數）、department_name（同 ID 的正式名稱）、source_text（本輪連續逐字原文）、confidence（有限 0～1）。只能選下方 active SQL Department 中唯一存在的正式 ID/name pair；不得別名補救或發明 ID。這只是偏好，不能跳過安全檢查或決定科別結果、stage、confirmed。無候選或無明確要求時必須 null。
目前 active SQL Department：
{json.dumps(active_department_candidates(), ensure_ascii=False, indent=2)}
32. patient_profile 記錄「要看病的人（就診者）」是誰與年齡。Backend 目前記錄的就診者：{json.dumps(known_patient_profile, ensure_ascii=False)}。本輪原文表達了就診者是誰或就診者年齡時，必須輸出；Backend 目前記錄為 null、而先前任一則使用者訊息已表達就診者是誰時，本輪也必須輸出（source_text 可取自那則先前使用者訊息）。都沒有時才是 null。僅含 subject、relation_text、age_text、source_text、confidence。subject 為 self（使用者本人看病）或 other（替家人或他人描述病況）。relation_text 是 other 時對就診者的逐字稱呼（例如「小女兒」「我家寶寶」「我媽」），self 時必須 null。age_text 是「就診者本人」年齡或學齡的最短逐字原文（例如「7歲」「6個月大」「國小三年級」「高二」），不是其他人的年齡；沒說就 null。source_text 是支持這個判斷的最短連續逐字使用者原文，relation_text 與 age_text 都必須包含在其中；confidence 為有限 0～1。只依語意判斷誰在生病：「我是幼稚園老師，喉嚨痛」是 self 且 age_text=null（幼稚園是工作地點）；「我小孩感冒傳染給我，我現在咳嗽」是 self；「我女兒幫我掛號，我頭暈」是 self；「我小女兒感冒很多天」是 other、relation_text=「小女兒」。不得猜年齡，也不得由症狀推論年齡。

就診者正例（替家人描述病況）：
current_user_text:「我兒子咳嗽好幾天了」
{{
  "semantic_extractions": [
    {{
      "field": "symptom",
      "normalized_value": "咳嗽",
      "semantic_status": "available",
      "assertion": "present",
      "confidence": 0.97,
      "source_text": "咳嗽",
      "needs_clarification": false,
      "follow_up_reason": null
    }}
  ],
  "pending_answer": null,
  "ttas_evidence": [],
  "time_preferences": [],
  "dialogue_act": {{
    "act": "new_information",
    "source_text": "我兒子咳嗽好幾天了",
    "confidence": 0.97
  }},
  "department_preference": null,
  "prior_answer_reference": null,
  "patient_profile": {{
    "subject": "other",
    "relation_text": "兒子",
    "age_text": null,
    "source_text": "我兒子咳嗽",
    "confidence": 0.97
  }}
}}

對話行為正例 A（要求解釋，不是 medical answer）：
current_user_text:「我有點搞不懂，你現在主要是想確認什麼狀況？」
{{
  "semantic_extractions": [],
  "pending_answer": null,
  "ttas_evidence": [],
  "time_preferences": [],
  "dialogue_act": {{
    "act": "request_explanation",
    "source_text": "我有點搞不懂",
    "confidence": 0.98
  }},
  "department_preference": null,
  "prior_answer_reference": null,
  "patient_profile": null
}}

對話行為正例 B（更正舊症狀，同時保留未被更正的新資訊）：
current_user_text:「我剛剛講錯了，主要不是頭暈，是頭痛，右耳嗡嗡叫還是有。」
{{
  "semantic_extractions": [
    {{
      "field": "symptom",
      "normalized_value": "頭暈",
      "semantic_status": "unavailable",
      "assertion": "absent",
      "confidence": 0.99,
      "source_text": "不是頭暈",
      "needs_clarification": false,
      "follow_up_reason": null
    }},
    {{
      "field": "symptom",
      "normalized_value": "頭痛",
      "semantic_status": "available",
      "assertion": "present",
      "confidence": 0.99,
      "source_text": "頭痛",
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
      "source_text": "右耳嗡嗡叫還是有",
      "needs_clarification": false,
      "follow_up_reason": null
    }}
  ],
  "pending_answer": null,
  "ttas_evidence": [],
  "time_preferences": [],
  "dialogue_act": {{
    "act": "correction",
    "source_text": "我剛剛講錯了",
    "confidence": 0.99
  }},
  "department_preference": null,
  "prior_answer_reference": null,
  "patient_profile": null
}}

時間偏好正例：
current_user_text:「這週都不行，下週一或下週四都可以，下週四比較好，其他平日也可以。上午不行，下午最好。」
time_preferences 應輸出（priority 依意思決定，不是依說出的先後：這句先說下週一，但下週四比較好，所以下週四是 1）：
[
  {{"kind":"relative_week","relation":"exclude","priority":null,"week_offset":0,"source_text":"這週都不行","confidence":0.99}},
  {{"kind":"relative_weekday","relation":"prefer","priority":1,"week_offset":1,"weekday":"thursday","source_text":"下週四比較好","confidence":0.99}},
  {{"kind":"relative_weekday","relation":"prefer","priority":2,"week_offset":1,"weekday":"monday","source_text":"下週一","confidence":0.99}},
  {{"kind":"weekday_group","relation":"acceptable","priority":3,"weekdays":["monday","tuesday","wednesday","thursday","friday"],"source_text":"其他平日也可以","confidence":0.98}},
  {{"kind":"session","relation":"exclude","priority":null,"session":"morning","source_text":"上午不行","confidence":0.99}},
  {{"kind":"session","relation":"prefer","priority":1,"session":"afternoon","source_text":"下午最好","confidence":0.99}}
]
current_user_text:「明天下午可以，後天不行」→
[
  {{"kind":"relative_day","relation":"prefer","priority":1,"day_offset":1,"source_text":"明天","confidence":0.99}},
  {{"kind":"session","relation":"prefer","priority":1,"session":"afternoon","source_text":"下午可以","confidence":0.99}},
  {{"kind":"relative_day","relation":"exclude","priority":null,"day_offset":2,"source_text":"後天不行","confidence":0.99}}
]
此 AI 輸出不可包含 reference_date 或 resolved_dates；兩者由 Backend 在接受 grounded preference 時寫入。

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
  "ttas_evidence": [],
  "time_preferences": [],
  "dialogue_act": {{
    "act": "answer",
    "source_text": "沒有聽力下降",
    "confidence": 0.98
  }},
  "department_preference": null,
  "prior_answer_reference": null,
  "patient_profile": null
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
  ],
  "time_preferences": [],
  "dialogue_act": {{
    "act": "new_information",
    "source_text": "眼睛灼痛",
    "confidence": 0.99
  }},
  "department_preference": null,
  "prior_answer_reference": null,
  "patient_profile": null
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
  "ttas_evidence": [],
  "time_preferences": [],
  "dialogue_act": {{
    "act": "new_information",
    "source_text": "右眼很痛",
    "confidence": 0.99
  }},
  "department_preference": null,
  "prior_answer_reference": null,
  "patient_profile": null
}}"""
