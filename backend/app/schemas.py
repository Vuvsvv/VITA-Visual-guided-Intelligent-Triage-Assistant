from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConversationStage(str, Enum):
    COLLECTING = "collecting"
    WAITING_CONFIRMATION = "waiting_confirmation"
    RECOMMENDING = "recommending"
    SCRIPT_READY = "script_ready"
    DONE = "done"


class VisitType(str, Enum):
    INITIAL = "initial"
    FOLLOWUP = "followup"
    QUICK_SEARCH = "quick_search"
    RETURN_VISIT = "return_visit"


class QuickSearchPeriod(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"


ChecklistField = Literal[
    "symptom",
    "red_flags",
    "body_part",
    "duration",
    "severity",
    "preferred_days",
    "preferred_sessions",
]
DepartmentQuestionKey = Literal["department_clarification"]
DepartmentStateField = Literal["department_context"]
EvidenceAssertion = Literal["present", "absent", "uncertain"]
PendingAnswerStatus = Literal["answered", "partial", "unclear"]
TTASSemanticStatus = Literal["available", "unknown", "ambiguous"]
TTASEvaluationStatus = Literal["matched", "insufficient_information"]
TimePreferenceKind = Literal[
    "date",
    "date_range",
    "relative_day",
    "relative_week",
    "relative_weekday",
    "weekday",
    "weekday_group",
    "session",
]
TimePreferenceRelation = Literal["exclude", "prefer", "acceptable"]
TimePreferenceWeekday = Literal[
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
]
TimePreferenceSession = Literal["morning", "afternoon", "evening"]


class QuestionItem(BaseModel):
    key: ChecklistField | DepartmentQuestionKey
    question: str
    question_id: Optional[str] = None
    state_field: Optional[ChecklistField | DepartmentStateField] = None
    required: bool = True
    input_type: Optional[str] = "text"


class BatchAnswer(BaseModel):
    key: ChecklistField | DepartmentQuestionKey
    answer: str


class Message(BaseModel):
    role: str
    content: str


class SemanticExtraction(BaseModel):
    field: str
    normalized_value: Any = None
    semantic_status: str = "unknown"
    assertion: Optional[EvidenceAssertion] = None
    confidence: float = 0.0
    source_text: str = ""
    needs_clarification: bool = False
    follow_up_reason: Optional[str] = None
    extractor: str = "deterministic"


class DialogueActInterpretation(BaseModel):
    act: Literal["answer", "request_explanation", "claims_already_answered", "correction",
                 "new_information", "decline_further_questions", "unknown", "off_topic"] = "unknown"
    source_text: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1, allow_inf_nan=False, strict=True)


class ConceptAnswer(BaseModel):
    # One Backend-validated answer to an official concept asked by a clue question.
    concept: str
    answer_assertion: EvidenceAssertion
    source_text: str
    confidence: float


class PendingAnswerInterpretation(BaseModel):
    answered_intent: str
    answer_status: PendingAnswerStatus
    answer_source_text: str
    answer_confidence: float
    answer_assertion: Optional[EvidenceAssertion] = None
    concept_answers: List[ConceptAnswer] = Field(default_factory=list)
    # Only for the safety-net question: the patient's choice as judged by the Turn Interpreter.
    answer_choice: Optional[Literal["family_medicine", "continue"]] = None


class DepartmentPreferenceInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    dept_id: int = Field(gt=0)
    department_name: str = Field(min_length=1, max_length=100)
    source_text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class PriorAnswerReference(BaseModel):
    model_config = {"extra": "forbid"}

    answered_intent: str = Field(strict=True, min_length=1)
    answer_source_text: str = Field(strict=True, min_length=1)
    answer_confidence: float = Field(strict=True, ge=0, le=1, allow_inf_nan=False)


class TTASEvidence(BaseModel):
    field: str
    value: Any = None
    semantic_status: TTASSemanticStatus
    confidence: float
    source_text: str


class TTASRuleEvidenceReference(BaseModel):
    field: str
    value: Any = None
    source_text: str


class TTASMatchedRule(BaseModel):
    rule_id: str
    official_code: Optional[str] = None
    source_id: str
    official_locator: str
    level: int
    evidence_references: List[TTASRuleEvidenceReference] = Field(default_factory=list)


class TTASEvaluationResult(BaseModel):
    status: TTASEvaluationStatus = "insufficient_information"
    level_candidate: Optional[int] = None
    matched_rules: List[TTASMatchedRule] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class SeverityNormalization(BaseModel):
    severity_level: Optional[str] = None
    functional_impact: bool = False
    sleep_impact: bool = False
    confidence: float = 0.0
    semantic_status: str = "unknown"
    source_text: str = ""
    needs_clarification: bool = False
    follow_up_reason: Optional[str] = None


class UrgencyNormalization(BaseModel):
    urgency_level: str = "low"
    matched_red_flags: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    warning_required: bool = False
    semantic_status: str = "unknown"
    source_text: str = ""
    needs_clarification: bool = False
    follow_up_reason: Optional[str] = None
    answer_classification: Optional[str] = None


class PatientInput(BaseModel):
    symptom: str = ""
    body_part: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    onset: Optional[str] = None
    accompanying_symptoms: List[str] = Field(default_factory=list)
    department_context: Optional[str] = None
    requested_department_id: Optional[int] = None
    requested_department_name: Optional[str] = None
    red_flags: List[str] = Field(default_factory=list)
    collected_fields: List[str] = Field(default_factory=list)
    red_flags_checked: bool = False
    red_flags_status: str = "not_checked"
    severity_normalized: SeverityNormalization = Field(default_factory=SeverityNormalization)
    urgency_normalized: UrgencyNormalization = Field(default_factory=UrgencyNormalization)
    age_years: Optional[float] = None
    age_months: Optional[float] = None


class TimePreference(BaseModel):
    """Grounded time meaning plus its deterministic calendar resolution."""

    model_config = ConfigDict(extra="forbid")

    kind: TimePreferenceKind
    relation: TimePreferenceRelation
    priority: Optional[int] = Field(default=None, ge=1, le=99)
    source_text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    date_value: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    # 今天 0、明天 1、後天 2…；Backend 以當輪台北日期換算
    day_offset: Optional[int] = Field(default=None, ge=0, le=90)
    week_offset: Optional[int] = Field(default=None, ge=0, le=52)
    weekday: Optional[TimePreferenceWeekday] = None
    weekdays: List[TimePreferenceWeekday] = Field(default_factory=list)
    session: Optional[TimePreferenceSession] = None
    reference_date: Optional[str] = None
    resolved_dates: List[str] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence_type(cls, value: object) -> object:
        if type(value) not in {int, float}:
            raise ValueError("confidence must be a finite number")
        return value

    @field_validator("date_value", "start_date", "end_date", "reference_date")
    @classmethod
    def validate_optional_date(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return date.fromisoformat(value).isoformat()

    @field_validator("resolved_dates")
    @classmethod
    def validate_resolved_dates(cls, values: List[str]) -> List[str]:
        return [date.fromisoformat(value).isoformat() for value in values]

    @model_validator(mode="after")
    def validate_shape(self) -> "TimePreference":
        if self.relation == "exclude" and self.priority is not None:
            raise ValueError("exclude preferences cannot carry priority")
        if self.relation != "exclude" and self.priority is None:
            raise ValueError("prefer and acceptable preferences require priority")
        required = {
            "date": bool(self.date_value),
            "date_range": bool(self.start_date and self.end_date),
            "relative_day": self.day_offset is not None,
            "relative_week": self.week_offset is not None,
            "relative_weekday": self.week_offset is not None and self.weekday is not None,
            "weekday": self.weekday is not None,
            "weekday_group": bool(self.weekdays),
            "session": self.session is not None,
        }
        if not required[self.kind]:
            raise ValueError(f"missing value for time preference kind {self.kind}")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("date range start must not be after end")
        return self


class Availability(BaseModel):
    preferred_dates: List[str] = Field(default_factory=list)
    preferred_days: List[str] = Field(default_factory=list)
    preferred_sessions: List[str] = Field(default_factory=list)
    can_take_leave: bool = False
    time_preferences: List[TimePreference] = Field(default_factory=list)
    semantic_status: Dict[str, str] = Field(default_factory=dict)
    confidence: Dict[str, float] = Field(default_factory=dict)

    @field_validator("preferred_dates")
    @classmethod
    def validate_preferred_dates(cls, values: List[str]) -> List[str]:
        normalized = []
        for value in values:
            text = str(value or "").strip()
            try:
                parsed = date.fromisoformat(text)
            except ValueError as exc:
                raise ValueError("preferred_dates must use YYYY-MM-DD") from exc
            normalized.append(parsed.isoformat())
        return normalized


class Preferences(BaseModel):
    specialty_priority: bool = True
    doctor_preference: str = "不限"
    hospital_preference: str = "台北榮總"


class CandidateDepartmentEvidence(BaseModel):
    patient_source_text: str
    knowledge_source_id: str
    knowledge_concept: str


class SemanticOfficialLink(BaseModel):
    # An audited semantic retrieval link; rebuilt into official evidence only while still grounded.
    patient_source_text: str
    knowledge_concept: str
    # The audit's view of the concept's body part: same part, no part, or a wider/other part.
    body_part_match: Optional[Literal["same", "none", "other"]] = None


class ClueConceptRecord(BaseModel):
    # One official concept asked by a clue question, with the candidate department it targeted.
    concept: str
    dept_id: int
    question_index: int = Field(ge=1)
    answer: Literal["present", "absent", "uncertain", "unanswered"] = "unanswered"
    # Every candidate department holding the concept; older snapshots default to [dept_id].
    dept_ids: List[int] = Field(default_factory=list)
    answer_source_text: Optional[str] = None

    @model_validator(mode="after")
    def _default_dept_ids(self) -> "ClueConceptRecord":
        if not self.dept_ids:
            self.dept_ids = [self.dept_id]
        return self


class ClueDecisionRecord(BaseModel):
    # Diagnostic record of this turn's department_clues.decide() result.
    kind: Literal["ask_clue", "ask_companion", "decide", "tiebreak", "family_medicine",
                  "family_medicine_unavailable", "defer_to_safety_net"]
    reason: str = ""
    dept_id: Optional[int] = None
    tied_dept_ids: List[int] = Field(default_factory=list)
    # ask_companion only: patient text pieces that scored when the decision was made.
    scored_sources: List[str] = Field(default_factory=list)


class ClueOffer(BaseModel):
    # A ranked askable official concept and the candidate departments holding it.
    concept: str
    dept_ids: List[int] = Field(default_factory=list)


class CandidateDepartment(BaseModel):
    dept_id: int
    parentDept: str
    childDept: str
    confidence: float
    supporting_evidence: List[CandidateDepartmentEvidence] = Field(default_factory=list)


class ConversationState(BaseModel):
    stage: ConversationStage = ConversationStage.COLLECTING
    free_text_mode: bool = False
    is_complete: bool = False
    awaiting_confirmation: bool = False
    confirmed: bool = False
    revision_mode: bool = False
    asked_fields: List[str] = Field(default_factory=list)
    consumed_fields: List[str] = Field(default_factory=list)
    last_question_key: Optional[str] = None
    question_attempts: Dict[str, int] = Field(default_factory=dict)
    field_statuses: Dict[str, str] = Field(default_factory=dict)
    field_confidence: Dict[str, float] = Field(default_factory=dict)
    clarification_reasons: Dict[str, str] = Field(default_factory=dict)
    turn_count: int = 0
    consecutive_no_progress_turns: int = Field(default=0, ge=0)
    safety_answer_attempts: int = 0
    safety_answer_validation: Optional[str] = None
    # An accepted positive answer in the current safety screen; it ends the case and nothing clears it.
    safety_positive_reported: bool = False
    # New symptoms said after a completed safety screen; the reopened screen quotes them once.
    safety_rescreen_sources: List[str] = Field(default_factory=list)
    # Diagnostic only: Backend-validated dialogue act value of this turn, never its source text.
    last_dialogue_act: Optional[str] = None
    reply_source: Optional[str] = None
    question_rejection_reason: Optional[str] = None
    question_repair_attempted: bool = False
    clarification_status: str = "collecting"
    uncertainty_reasons: List[str] = Field(default_factory=list)
    next_information_needed: List[str] = Field(default_factory=list)
    asked_clarification_intents: List[str] = Field(default_factory=list)
    pending_clarification_intent: Optional[str] = None
    pending_clarification_question: Optional[str] = None
    clarification_evidence: Dict[str, str] = Field(default_factory=dict)
    department_status: Literal["unresolved", "ambiguous", "resolved"] = "unresolved"
    candidate_departments: List[CandidateDepartment] = Field(default_factory=list)
    department_uncertainty_reason: Optional[str] = None
    department_next_information_needed: Optional[str] = None
    department_next_question_intent: Optional[str] = None
    semantic_official_links: List[SemanticOfficialLink] = Field(default_factory=list)
    # Clue-question convergence; defaults keep older case snapshots loadable.
    clue_questions_asked: int = Field(default=0, ge=0)
    clue_concepts_asked: List[ClueConceptRecord] = Field(default_factory=list)
    clue_tiebreak_used: bool = False
    # The patient declined further questions; kept until revision so later turns settle on current evidence.
    clue_user_declined: bool = False
    # Recomputed every department turn, like candidate_departments.
    department_clue_offer: List[ClueOffer] = Field(default_factory=list)
    department_clue_candidates: List[int] = Field(default_factory=list)
    pending_clue_question_index: Optional[int] = None
    clue_reask_count: int = Field(default=0, ge=0)
    clue_decision: Optional[ClueDecisionRecord] = None
    # Safety net: the one-time turn-8 question and the patient's answer; revision resets both.
    safety_net_asked: bool = False
    safety_net_choice: Optional[Literal["family_medicine", "continue"]] = None
    # A symptom whose body part is unknown and which alone points away from the stated body part:
    # set by department reasoning each turn; sources already asked are never asked again.
    body_part_question_source: Optional[str] = None
    body_part_questions_asked: List[str] = Field(default_factory=list)
    # A single target department: the open 「還有其他不舒服嗎」 question, asked again only when
    # the previous answer added scored patient text (companion_question_sources is the snapshot).
    companion_questions_asked: int = Field(default=0, ge=0)
    companion_question_sources: List[str] = Field(default_factory=list)
    # Pediatric routing: who the patient is (validated turn-interpreter patient_profile), the
    # one-time age question for an age-less 「我女兒」 and its reply.
    patient_subject: Optional[Literal["self", "other"]] = None
    patient_relation_text: Optional[str] = None
    patient_age_text: Optional[str] = None
    patient_age_question_asked: bool = False
    patient_age_answer: Optional[str] = None
    # Appointment availability asked once the department converged; revision keeps it.
    availability_status: Literal["not_asked", "asked", "answered", "unspecified"] = "not_asked"
    availability_question_attempts: int = Field(default=0, ge=0)
    # 6-H-1: date and session settle separately; the full question counts once for each.
    availability_date_attempts: int = Field(default=0, ge=0)
    availability_session_attempts: int = Field(default=0, ge=0)
    availability_pending_dimensions: List[Literal["date", "session"]] = Field(default_factory=list)
    availability_dimension_outcomes: Dict[Literal["date", "session"], Literal["unrestricted", "exhausted"]] = Field(
        default_factory=dict,
    )


class UrgencyResult(BaseModel):
    urgency_score: Optional[int] = None
    urgency_level: Optional[str] = None
    warning_required: bool = False
    warning_message: Optional[str] = None
    need_more_info: bool = True
    next_question: Optional[str] = None
    reasons: List[str] = Field(default_factory=list)
    is_final: bool = False


class DepartmentResult(BaseModel):
    dept_id: Optional[int] = None
    parentDept: str = ""
    childDept: str = ""
    confidence: float = 0.0
    reason: List[str] = Field(default_factory=list)


class TriageCase(BaseModel):
    case_id: str
    visit_type: Optional[VisitType] = None
    history_records: List[Message] = Field(default_factory=list)
    patient_input: PatientInput = Field(default_factory=PatientInput)
    availability: Availability = Field(default_factory=Availability)
    preferences: Preferences = Field(default_factory=Preferences)
    triage: UrgencyResult = Field(default_factory=UrgencyResult)
    conversation_state: ConversationState = Field(default_factory=ConversationState)
    department_result: Optional[DepartmentResult] = None
    semantic_extractions: List[SemanticExtraction] = Field(default_factory=list)
    ttas_evidence: List[TTASEvidence] = Field(default_factory=list)
    ttas_result: TTASEvaluationResult = Field(default_factory=TTASEvaluationResult)
    confirmed: bool = False
    recommendation_generated: bool = False
    script_generated: bool = False
    selected_recommendation_id: Optional[str] = None


class ChatRequest(BaseModel):
    legacy_batch: bool = False
    case_id: Optional[str] = None
    message: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
    triage_case: Optional[TriageCase] = None
    visit_type: Optional[VisitType] = None
    answers: List[BatchAnswer] = Field(default_factory=list)
    confirmed: bool = False
    revision_requested: bool = False
    require_existing_case: bool = False


class ResumeDepartment(BaseModel):
    dept_id: Optional[int] = None
    parentDept: str
    childDept: str


class CaseResumeResult(BaseModel):
    case_id: str
    visit_type: Optional[VisitType] = None
    stage: ConversationStage
    confirmed: bool
    awaiting_confirmation: bool
    department_status: str
    clarification_status: str
    red_flags_checked: bool
    red_flags_status: str
    warning_required: bool
    warning_message: Optional[str] = None
    department_result: Optional[ResumeDepartment] = None
    next_question: Optional[str] = None
    last_question_key: Optional[str] = None
    question_batch: List[QuestionItem] = Field(default_factory=list)
    can_continue: bool = False
    reply_source: Optional[str] = None


class TriageResult(BaseModel):
    case_id: str
    triage_case: TriageCase
    conversation_state: ConversationState
    triage: UrgencyResult
    department_result: Optional[DepartmentResult] = None
    next_question: Optional[str] = None
    reply: Optional[str] = None
    needMoreInfo: bool = True
    triage_reasons: List[str] = Field(default_factory=list)
    question_batch: List[QuestionItem] = Field(default_factory=list)


class RecommendationItem(BaseModel):
    recommendation_id: str
    parentDept: str
    childDept: str
    doctor: str
    date: str
    session: str
    slot: str = ""
    doctor_id: Optional[str] = None
    schedule_id: Optional[str] = None
    session_time: Optional[str] = None
    room: Optional[str] = None
    visit_type: Optional[str] = None
    specialty_tags: Optional[str] = None
    specialty_score: Optional[float] = None
    time_score: Optional[float] = None
    match_reason: Optional[str] = None
    score: float
    reasons: List[str] = Field(default_factory=list)
    # Patient-facing sentences; reasons keeps the diagnostic detail.
    department_reason: str = ""
    time_reason: str = ""
    rank: Optional[int] = None
    is_best_match: bool = False
    dept_id: Optional[int] = None


class RecommendationColumns(BaseModel):
    # Up to 5 doctors x 3 slots each (appointment_service.SPECIALTY_DOCTOR_LIMIT / SPECIALTY_SLOTS_PER_DOCTOR).
    specialty_first: List[RecommendationItem] = Field(default_factory=list, max_length=15)
    time_first: List[RecommendationItem] = Field(default_factory=list, max_length=5)


class FallbackDepartment(BaseModel):
    parentDept: str
    childDept: str
    reason: str


class RecommendRequest(BaseModel):
    triage_case: Optional[TriageCase] = None
    case_id: Optional[str] = None
    userQuery: Optional[str] = None
    preference: Optional[str] = None
    confirmed: bool = False
    visit_type: Optional[Literal["initial", "followup", "quick_search", "return_visit"]] = None


class RecommendationResult(BaseModel):
    case_id: str
    recommendations: RecommendationColumns
    fallback_departments: List[FallbackDepartment] = Field(default_factory=list)
    total_count: int = 0


class ScriptRequest(BaseModel):
    case_id: Optional[str] = None
    recommendation_id: str
    recommendation: Optional[RecommendationItem] = None


class FollowupRecommendRequest(BaseModel):
    case_id: Optional[str] = None
    visit_type: Literal["return_visit"] = "return_visit"
    parentDept: Optional[str] = None
    childDept: Optional[str] = None
    dept_id: Optional[int] = None
    original_doctor: str
    original_doctor_id: Optional[int] = None
    followup_reason: Optional[str] = None
    availability: Availability = Field(default_factory=Availability)
    preferences: Preferences = Field(default_factory=Preferences)

    @field_validator("original_doctor")
    @classmethod
    def validate_original_doctor(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("original_doctor is required for return_visit")
        return normalized


class FollowupRecommendResponse(BaseModel):
    case_id: str
    department: Optional[DepartmentResult] = None
    recommendations: List[RecommendationItem] = Field(default_factory=list)
    fallback_departments: List[FallbackDepartment] = Field(default_factory=list)
    total_count: int = 0


class ReferenceDepartment(BaseModel):
    dept_id: str
    parentDept: str
    childDept: str


class ReferenceDepartmentResponse(BaseModel):
    departments: List[ReferenceDepartment] = Field(default_factory=list)


class ReferenceDoctor(BaseModel):
    doctor_id: str
    name: str


class ReferenceDoctorResponse(BaseModel):
    department: str
    doctors: List[ReferenceDoctor] = Field(default_factory=list)


class QuickSearchResponse(BaseModel):
    case_id: str
    dept_id: int
    parentDept: str
    childDept: str
    date: date
    period: QuickSearchPeriod
    results: List[RecommendationItem] = Field(default_factory=list)
    total_count: int = 0


class ScriptStep(BaseModel):
    action: str
    target: str
    resource_id: Optional[str] = None
    text: Optional[str] = None
    class_name: Optional[str] = None
    description: Optional[str] = None
    delay_ms: int = 0
    retry: int = 0


class ScriptResponse(BaseModel):
    isSuccess: bool
    script_id: str = "vgh_booking_001"
    recommendation_id: Optional[str] = None
    recommendation: Optional[RecommendationItem] = None
    steps: List[ScriptStep] = Field(default_factory=list)
    message: Optional[str] = None
    step_count: int = 0


# ─────────────────────────────────────────────
# 語音端點 schema（/voice/chat）
# 不影響現有 schema，只新增
# ─────────────────────────────────────────────

class VoiceChatResponse(BaseModel):
    """語音聊天回應：文字（字幕）+ 語音（播放）+ 對話狀態"""
    case_id: str
    user_text: str = ""                          # ASR 辨識出長輩說的話（給字幕）
    reply_text: str = ""                          # 系統回覆文字（給字幕）
    reply_audio_base64: str = ""                 # 回覆語音 base64（動態句才有）
    reply_audio_id: str = ""                     # 固定句 → 帶 id，Android 播本地檔（做法B）
    audio_format: str = "wav"                    # 音訊格式 wav / m4a
    needMoreInfo: bool = True                     # 是否還要繼續問
    stage: str = ""                               # 對話階段
    department_result: Optional[DepartmentResult] = None
    tts_failed: bool = False                      # TTS 失敗時 true，Android 改用系統 TTS 念 reply_text
    error: Optional[str] = None
