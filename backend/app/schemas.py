from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


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


class PendingAnswerInterpretation(BaseModel):
    answered_intent: str
    answer_status: PendingAnswerStatus
    answer_source_text: str
    answer_confidence: float
    answer_assertion: Optional[EvidenceAssertion] = None


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


class Availability(BaseModel):
    preferred_dates: List[str] = Field(default_factory=list)
    preferred_days: List[str] = Field(default_factory=list)
    preferred_sessions: List[str] = Field(default_factory=list)
    can_take_leave: bool = False
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
    clarification_status: str = "collecting"
    uncertainty_reasons: List[str] = Field(default_factory=list)
    next_information_needed: List[str] = Field(default_factory=list)
    asked_clarification_intents: List[str] = Field(default_factory=list)
    pending_clarification_intent: Optional[str] = None
    clarification_evidence: Dict[str, str] = Field(default_factory=dict)
    department_status: Literal["unresolved", "ambiguous", "resolved"] = "unresolved"
    candidate_departments: List[CandidateDepartment] = Field(default_factory=list)
    department_uncertainty_reason: Optional[str] = None
    department_next_information_needed: Optional[str] = None
    department_next_question_intent: Optional[str] = None


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
    case_id: Optional[str] = None
    message: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
    triage_case: Optional[TriageCase] = None
    visit_type: Optional[VisitType] = None
    answers: List[BatchAnswer] = Field(default_factory=list)
    confirmed: bool = False
    revision_requested: bool = False


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
    rank: Optional[int] = None
    is_best_match: bool = False
    dept_id: Optional[int] = None


class RecommendationColumns(BaseModel):
    specialty_first: List[RecommendationItem] = Field(default_factory=list, max_length=5)
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
