from __future__ import annotations

from app.schemas import QuestionItem, TriageCase
from app.services.rule_engine import (
    mark_questions_asked,
    missing_checklist_fields,
    question_text_for_field,
)
from app.services.question_specs import question_spec_for_field

MAX_BATCH_QUESTIONS = 1


def build_question_batch(
    case: TriageCase,
    *,
    max_questions: int = MAX_BATCH_QUESTIONS,
    mark_asked: bool = True,
    increment_existing: bool = False,
) -> list[QuestionItem]:
    """Build the next deterministic checklist question.

    The rule engine remains the source of truth for missing fields, wording
    order, max-attempt fallback, and completion. Returning one question lets
    the backend recompute missing fields after every answer.
    """
    limit = max(1, min(int(max_questions), MAX_BATCH_QUESTIONS))
    fields = missing_checklist_fields(case)[:limit]
    questions = [
        QuestionItem(
            key=field,
            question=question_text_for_field(case, field),
            question_id=question_spec_for_field(field).question_id,
            state_field=field,
            required=True,
            input_type="text",
        )
        for field in fields
    ]
    if mark_asked and fields:
        fields_to_mark = fields if increment_existing else [
            field for field in fields if field not in case.conversation_state.asked_fields
        ]
        mark_questions_asked(case, fields_to_mark)
    return questions
