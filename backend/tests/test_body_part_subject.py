"""A symptom said without its own body part can only add to departments the stated body part reached."""
from app.schemas import Message, SemanticExtraction, TriageCase
from app.services import department_reasoning_service as service
from app.services.conversation_service import body_part_question
from app.services.department_knowledge import load_department_knowledge
from app.services.rule_engine import apply_semantic_extractions

_, RECORDS = load_department_knowledge()
RESOLUTIONS = [dict(status="resolved", canonical_dept_id=r["canonical_dept_id"], db_dept_id=r["canonical_dept_id"],
                    db_parent_dept=r["canonical_parent_dept"], db_child_dept=r["canonical_child_dept"])
               for r in RECORDS]


def item(field, value, source, assertion="present"):
    if field == "accompanying_symptoms" and isinstance(value, str):
        value = [value]
    return SemanticExtraction(field=field, normalized_value=value, source_text=source, semantic_status="available",
                              assertion=None if field == "body_part" else assertion, confidence=.99,
                              extractor="ai")


def conversation(*turns):
    """turns: assistant question strings, or (patient text, extractions) pairs."""
    case = TriageCase(case_id="body-part-subject")
    for turn in turns:
        if isinstance(turn, str):
            case.history_records.append(Message(role="assistant", content=turn))
            continue
        text, extractions = turn
        case.history_records.append(Message(role="user", content=text))
        apply_semantic_extractions(case, list(extractions), allow_red_flag_completion=False)
    return case


def filtered(case):
    return service.subject_filtered_evidence(
        case, service.retrieve_official_evidence(case, RECORDS, RESOLUTIONS))


def depts(rows):
    return {row["childDept"] for row in rows}


EYE_CHIEF = ("眼睛紅腫酸痛", [item("body_part", "眼睛", "眼睛"), item("symptom", "紅腫", "紅腫"),
                             item("accompanying_symptoms", "酸痛", "酸痛")])
DISCHARGE = ("有淺黃色分泌物", [item("accompanying_symptoms", "分泌物", "淺黃色分泌物")])


def test_discharge_answering_an_eye_question_does_not_reach_gynecology():
    case = conversation(EYE_CHIEF, "請問眼睛是否有分泌物？", DISCHARGE)
    before = service.retrieve_official_evidence(case, RECORDS, RESOLUTIONS)
    assert "婦產科" in depts(before)
    rows = filtered(case)
    assert "婦產科" not in depts(rows)
    assert "眼科" in depts(rows)
    assert case.conversation_state.body_part_question_source is None


def test_discharge_after_generic_question_only_helps_the_eye_departments():
    case = conversation(EYE_CHIEF, "還有其他不適症狀嗎？", DISCHARGE)
    rows = filtered(case)
    assert "婦產科" not in depts(rows)
    assert "眼科" in depts(rows)
    assert case.conversation_state.body_part_question_source is None


def test_unknown_body_part_is_asked_when_nothing_else_remains():
    chief = ("眼皮怪怪的", [item("body_part", "眼皮", "眼皮"), item("symptom", "怪怪的", "怪怪的")])
    case = conversation(chief, "還有其他不適症狀嗎？", DISCHARGE)
    assert filtered(case) == []
    assert case.conversation_state.body_part_question_source == "淺黃色分泌物"
    case.conversation_state.body_part_questions_asked.append("淺黃色分泌物")
    assert filtered(case) == []
    assert case.conversation_state.body_part_question_source is None


def test_answer_with_a_new_body_part_counts_as_its_own_complaint():
    chief = ("眼皮怪怪的", [item("body_part", "眼皮", "眼皮"), item("symptom", "怪怪的", "怪怪的")])
    case = conversation(chief, "還有其他不適症狀嗎？", DISCHARGE, body_part_question("淺黃色分泌物"),
                        ("陰道", [item("body_part", "陰道", "陰道")]))
    assert "婦產科" in depts(filtered(case))


def test_answer_with_the_already_stated_body_part_stays_with_it():
    chief = ("眼皮怪怪的", [item("body_part", "眼皮", "眼皮"), item("symptom", "怪怪的", "怪怪的")])
    case = conversation(chief, "還有其他不適症狀嗎？", DISCHARGE, body_part_question("淺黃色分泌物"),
                        ("眼皮", [item("body_part", "眼皮", "眼皮")]))
    assert "婦產科" not in depts(filtered(case))


def test_without_any_stated_body_part_nothing_is_filtered():
    case = conversation(("有淺黃色分泌物", [item("symptom", "分泌物", "淺黃色分泌物")]))
    before = service.retrieve_official_evidence(case, RECORDS, RESOLUTIONS)
    assert filtered(case) == before
    assert "婦產科" in depts(before)


def test_chief_complaint_symptom_split_from_its_body_part_is_kept():
    case = conversation(EYE_CHIEF)
    before = service.retrieve_official_evidence(case, RECORDS, RESOLUTIONS)
    assert filtered(case) == before


def test_symptom_with_its_own_later_body_part_counts_on_its_own():
    case = conversation(EYE_CHIEF, "還有其他不適症狀嗎？",
                        ("鼻子也會流鼻水", [item("body_part", "鼻子", "鼻子"),
                                         item("accompanying_symptoms", "流鼻水", "流鼻水")]))
    rows = filtered(case)
    assert {"眼科"} <= depts(rows)
    assert any(row["patient_source_text"] == "流鼻水" for row in rows)
    assert case.patient_input.body_part == "眼睛"


def test_convergence_payload_for_reported_case_has_no_gynecology():
    import asyncio
    import json
    from unittest.mock import patch

    case = conversation(EYE_CHIEF, "請問眼睛是否有分泌物？", DISCHARGE)
    case.patient_input.red_flags_checked = True
    active = [dict(dept_id=r["canonical_dept_id"], parent_dept=r["canonical_parent_dept"],
                   child_dept=r["canonical_child_dept"])
              for r in {row["canonical_dept_id"]: row for row in RECORDS}.values()]
    prompts = []

    async def provider(prompt, *, purpose):
        prompts.append((purpose, prompt))
        return json.dumps({"status": "ambiguous", "candidates": []})

    with patch.object(service, "fetch_active_departments", return_value=active), \
            patch.object(service, "complete_runtime_json", new=provider):
        asyncio.run(service.reason_about_departments(case))
    data = json.loads(prompts[-1][1].split("\n資料：", 1)[1])
    assert "婦產科" not in depts(data["retrieved_official_evidence"])
    assert "眼科" in depts(data["retrieved_official_evidence"])


def test_body_part_question_is_asked_once_and_any_reply_releases_it():
    from app.services import conversation_service as conversation_flow

    case = conversation(EYE_CHIEF)
    conversation_flow._ask_body_part_question(case, "淺黃色分泌物")
    state = case.conversation_state
    assert case.triage.next_question == "請問您說的「淺黃色分泌物」是在哪個部位呢？"
    assert state.pending_clarification_intent == conversation_flow.BODY_PART_PENDING_INTENT
    assert state.body_part_questions_asked == ["淺黃色分泌物"]
    assert state.reply_source == "BODY_PART_QUESTION"
    assert conversation_flow.capture_pending_answer(case, None, ["眼睛"]) is False
    assert state.pending_clarification_intent is None
    assert state.pending_clarification_question is None
