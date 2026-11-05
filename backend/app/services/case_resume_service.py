from app.schemas import CaseResumeResult, ConversationStage, QuestionItem, ResumeDepartment, TriageCase, VisitType
from app.services.conversation_service import safety_unresolved_stop
from app.services.question_specs import question_spec_for_field
from app.services.rule_engine import QUESTION_TEXTS


def build_case_resume(case: TriageCase) -> CaseResumeResult:
    """Project existing workflow state only; never infer new clinical conclusions."""
    state = case.conversation_state
    question = case.triage.next_question
    key = state.last_question_key
    batch = []
    supported_visit = case.visit_type in {VisitType.INITIAL, VisitType.FOLLOWUP}
    terminal = state.clarification_status in {"urgent", "unresolved"} or case.triage.warning_required
    can_continue = supported_visit and not terminal
    if state.stage == ConversationStage.COLLECTING:
        if state.clarification_status == "safety_check":
            # The primary safety-answer route accepts a free-text turn, not batch answers.
            if key != "red_flags":
                can_continue = False
            else:
                question = question or QUESTION_TEXTS[key]
        elif not state.free_text_mode:
            if key in QUESTION_TEXTS:
                spec = question_spec_for_field(key)
                question = question or spec.canonical_text
                batch = [QuestionItem(key=key, question=question, question_id=spec.question_id, state_field=key)]
            elif key == "department_clarification" and question:
                batch = [QuestionItem(key=key, question=question, state_field="department_context")]
            else:
                can_continue = False
        elif not question and not state.revision_mode:
            can_continue = False
    elif state.stage == ConversationStage.WAITING_CONFIRMATION:
        can_continue = can_continue and case.patient_input.red_flags_checked and not case.patient_input.red_flags and state.department_status == "resolved" and case.department_result is not None
        question = None
    elif state.stage in {ConversationStage.RECOMMENDING, ConversationStage.SCRIPT_READY, ConversationStage.DONE}:
        can_continue = can_continue and state.confirmed and case.patient_input.red_flags_checked and not case.patient_input.red_flags and state.department_status == "resolved" and case.department_result is not None
        question = None
    department = case.department_result
    return CaseResumeResult(
        case_id=case.case_id, visit_type=case.visit_type, stage=state.stage,
        confirmed=state.confirmed, awaiting_confirmation=state.awaiting_confirmation,
        department_status=state.department_status, clarification_status=state.clarification_status,
        red_flags_checked=case.patient_input.red_flags_checked,
        red_flags_status=case.patient_input.red_flags_status,
        warning_required=case.triage.warning_required, warning_message=case.triage.warning_message,
        department_result=ResumeDepartment(dept_id=department.dept_id, parentDept=department.parentDept,
                                           childDept=department.childDept) if department else None,
        next_question=question, last_question_key=key, question_batch=batch, can_continue=can_continue,
        reply_source="SAFETY_UNRESOLVED_STOP" if safety_unresolved_stop(case) else state.reply_source,
    )
