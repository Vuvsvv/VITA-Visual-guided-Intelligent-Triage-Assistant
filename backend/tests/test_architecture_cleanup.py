"""Architecture boundaries, exercised offline without patient-text parsers."""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.schemas import Message, TriageCase
from app.services import conversation_service as cs
from app.services import department_preference_service as preference, rag_triage_adapter as adapter
from app.services.rule_engine import apply_user_message
from app.services.case_store import save_case
from test_phase7_natural_safety import runtime, evidence, pending, respond


ACTIVE = [{"dept_id": 1333, "parent_dept": "耳鼻喉部", "child_dept": "耳科"}]
QUESTION = "我想確認這次頭痛是否突然開始？"
CHECKLIST = cs.QUESTION_TEXTS[cs.RED_FLAG_QUESTION_KEY]
EXPLANATION = "突然開始是指頭痛在很短時間內變得很明顯，這能幫助確認是否需要優先處理。"


@pytest.mark.parametrize("success", [True, False])
def test_safety_explanation_has_ai_question_or_fixed_explanation(runtime, success):
    client, interpreter, planner, department = runtime
    case = TriageCase(case_id=f"cleanup-explanation-{success}")
    case.patient_input.symptom = "頭痛"
    case.history_records = [Message(role="user", content="頭痛")]
    cs._enter_safety_check(case)
    save_case(case)
    text = "我有點搞不懂，你現在主要是想確認什麼狀況？"
    interpreter.return_value = json.dumps(dict(semantic_extractions=[], pending_answer=None,
        ttas_evidence=[], time_preferences=[], prior_answer_reference=None, department_preference=None,
        dialogue_act=dict(act="request_explanation", source_text=text, confidence=0.99)))
    planner.return_value = json.dumps(dict(status="clarification_needed", intent="safety_screen",
        question=QUESTION if success else "", explanation=EXPLANATION, reason="安全資訊尚未確認", information_needs=["頭痛發作方式"]))
    result = client.post("/chat", json={"case_id": case.case_id, "message": text}).json()
    state = result["conversation_state"]
    assert state["safety_answer_attempts"] == 0
    assert state["clarification_status"] == "safety_check"
    assert not result["triage_case"]["patient_input"]["red_flags_checked"]
    # The AI question is never shown during the safety screen; the full checklist always is.
    assert result["reply"] == f"{EXPLANATION if success else cs.SAFETY_EXPLANATION_FALLBACK}\n{CHECKLIST}"
    assert state["reply_source"] == ("AI_PLANNER" if success else "SAFETY_CONTROLLED_FALLBACK")
    assert planner.await_count == (1 if success else 2)
    department.assert_not_awaited()


def test_unclear_medical_safety_answer_still_uses_three_attempt_cap(runtime):
    client, interpreter, planner, department = runtime
    case = TriageCase(case_id="cleanup-medical-cap")
    case.patient_input.symptom = "頭痛"
    cs._enter_safety_check(case)
    save_case(case)
    planner.return_value = "{}"
    for attempt in range(1, 4):
        respond(interpreter, answer=pending("不知道", "unclear", "uncertain"))
        result = client.post("/chat", json={"case_id": case.case_id, "message": "不知道"}).json()
        assert result["conversation_state"]["safety_answer_attempts"] == attempt
        assert not result["triage_case"]["patient_input"]["red_flags_checked"]
        if attempt == 1:
            assert result["reply"] == f"{cs.SAFETY_RETRY_REPLY}\n{CHECKLIST}"
        elif attempt == 2:
            # One more unclear answer stops the chat, so this retry says so first.
            assert result["reply"] == f"{cs.SAFETY_LAST_ATTEMPT_REPLY}\n{CHECKLIST}"
    assert result["conversation_state"]["clarification_status"] == "unresolved"
    assert result["conversation_state"]["reply_source"] == "SAFETY_UNRESOLVED_STOP"
    assert result["reply"] == cs.SAFETY_UNRESOLVED_REPLY
    department.assert_not_awaited()


def test_safety_explanation_provider_failure_repairs_once_then_fixed_explanation(runtime):
    client, interpreter, planner, department = runtime
    case = TriageCase(case_id="cleanup-explanation-timeout")
    cs._enter_safety_check(case)
    save_case(case)
    text = "請解釋這個問題"
    interpreter.return_value = json.dumps(dict(semantic_extractions=[], ttas_evidence=[], pending_answer=None,
        time_preferences=[], prior_answer_reference=None, department_preference=None,
        dialogue_act=dict(act="request_explanation", source_text=text, confidence=0.99)))
    planner.side_effect = TimeoutError("offline timeout")
    result = client.post("/chat", json={"case_id": case.case_id, "message": text}).json()
    assert planner.await_count == 2
    assert result["reply"] == f"{cs.SAFETY_EXPLANATION_FALLBACK}\n{CHECKLIST}"
    assert result["conversation_state"]["reply_source"] == "SAFETY_CONTROLLED_FALLBACK"
    assert result["conversation_state"]["question_repair_attempted"] is True
    assert result["conversation_state"]["safety_answer_attempts"] == 0
    department.assert_not_awaited()


@pytest.mark.parametrize("text", ["麻煩幫我安排耳科", "我想改指定耳科看看"])
@pytest.mark.parametrize("variant", ["valid", "null", "fake", "mismatch", "inactive", "duplicate", "ungrounded", "low", "extra", "string_id"])
def test_department_preference_requires_grounded_ai_and_current_exact_active_identity(text, variant):
    case = TriageCase(case_id="cleanup-preference")
    case.history_records.append(Message(role="user", content=text))
    value = dict(dept_id=1333, department_name="耳科", source_text=text, confidence=0.99)
    rows = ACTIVE
    if variant == "null": value = None
    elif variant == "fake": value["dept_id"] = 9999
    elif variant == "mismatch": value["department_name"] = "耳鼻喉科"
    elif variant == "inactive": rows = []
    elif variant == "duplicate": rows = ACTIVE * 2
    elif variant == "ungrounded": value["source_text"] = "不存在的原文"
    elif variant == "low": value["confidence"] = 0.1
    elif variant == "extra": value["confirmed"] = True
    elif variant == "string_id": value["dept_id"] = "1333"
    payload = dict(semantic_extractions=[], ttas_evidence=[], pending_answer=None, time_preferences=[],
        dialogue_act=dict(act="unknown" if value is None else "new_information", source_text=text, confidence=0.99),
        prior_answer_reference=None, department_preference=value)
    with patch.object(adapter, "_ai_available", return_value=True), patch.object(
        adapter, "complete_prompt", AsyncMock(return_value=json.dumps(payload)),
    ) as ai, patch.object(preference, "fetch_active_departments", return_value=rows):
        result = asyncio.run(adapter.refine_case_with_ai(case, user_sources=[text]))
    assert result is not None
    assert ai.await_count == (1 if variant in {"valid", "null"} else 2)
    assert result.interpretation_complete == (variant in {"valid", "null"})
    assert case.patient_input.requested_department_id == (1333 if variant == "valid" else None)
    assert case.patient_input.requested_department_name == ("耳科" if variant == "valid" else None)
    assert not case.patient_input.red_flags_checked
    assert case.department_result is None
    assert "department_preference" in ai.await_args.args[0]


@pytest.mark.parametrize("text", ["頭暈想看一般內科", "右腳痛三天很嚴重", "沒有胸痛，我說錯了", "下週二下午可以"])
def test_history_recording_does_not_interpret_any_patient_text(text):
    case = TriageCase(case_id="cleanup-raw")
    before = case.patient_input.model_dump()
    apply_user_message(case, text, semantic_first=False)
    assert case.patient_input.model_dump() == before
    assert case.semantic_extractions == []
    assert case.availability.preferred_days == []
    assert case.history_records[-1].content == text


def test_no_retired_patient_semantic_modules_or_production_callers():
    app = Path(__file__).parents[1] / "app"
    for name in ("project_smart_department_adapter", "semantic_normalizer", "semantic_refinement_gate", "negation_utils"):
        assert not (app / "services" / f"{name}.py").exists()
    retired = ("capture_department_preference", "detect_department_result", "_rule_based_department",
        "strip_negated_red_flags", "has_symptom_semantics", "plausible_semantic_target",
        "FALLBACK_QUESTIONS", "NEUTRAL_PENDING_RETRIES")
    for source in app.rglob("*.py"):
        content = source.read_text(encoding="utf-8-sig")
        for symbol in retired:
            assert symbol not in content, (source, symbol)


def test_department_insufficiency_never_selects_from_keywords():
    from app.services import department_reasoning_service as reasoning
    case = TriageCase(case_id="cleanup-no-keywords")
    apply_user_message(case, "膝蓋痛，頭暈，想看耳科")
    with patch.object(reasoning, "fetch_active_departments", return_value=ACTIVE), patch.object(
        reasoning, "complete_runtime_json", AsyncMock(side_effect=RuntimeError("offline")),
    ):
        asyncio.run(reasoning.reason_about_departments(case))
    assert case.department_result is None
    assert case.conversation_state.department_status in {"unresolved", "ambiguous"}


def test_preference_repair_contract_includes_new_nullable_field():
    prompt = adapter._build_turn_interpretation_repair_prompt("original", "invalid")
    assert "department_preference" in prompt


def test_all_full_interpreter_examples_include_nullable_department_preference():
    with patch.object(preference, "fetch_active_departments", return_value=ACTIVE):
        prompt = adapter._build_symptom_collection_prompt(TriageCase(case_id="prompt"), ["請幫我安排耳科"])
    assert prompt.count('"department_preference": null') == prompt.count('"prior_answer_reference": null')
    assert '"dept_id": 1333' in prompt


def test_changed_active_identity_is_rechecked_after_interpretation():
    case = TriageCase(case_id="stale-department")
    text = "請幫我安排耳科"
    data = dict(semantic_extractions=[], pending_answer=None, ttas_evidence=[], time_preferences=[],
        dialogue_act=dict(act="new_information", source_text=text, confidence=0.99), prior_answer_reference=None,
        department_preference=dict(dept_id=1333, department_name="耳科", source_text=text, confidence=0.99))
    with patch.object(preference, "fetch_active_departments", side_effect=[ACTIVE, [], []]) as db, patch.object(
        adapter, "_ai_available", return_value=True,
    ), patch.object(adapter, "complete_prompt", AsyncMock(return_value=json.dumps(data))):
        asyncio.run(adapter.refine_case_with_ai(case, user_sources=[text]))
    assert db.call_count == 3
    assert case.patient_input.requested_department_id is None


def test_inactive_row_and_database_failure_cannot_accept_preference():
    value = dict(dept_id=1333, department_name="耳科", source_text="耳科", confidence=0.99)
    for rows in [[dict(ACTIVE[0], is_active=False)], RuntimeError("offline DB unavailable")]:
        with patch.object(preference, "fetch_active_departments", side_effect=rows if isinstance(rows, Exception) else None,
                          return_value=rows if isinstance(rows, list) else None):
            assert preference.validate_department_preference(value, ["耳科"]) is None


def turn_payload(text="頭痛"):
    return dict(semantic_extractions=[evidence("symptom", "頭痛", "頭痛")],
        pending_answer=None, ttas_evidence=[], time_preferences=[],
        dialogue_act=dict(act="new_information", source_text=text, confidence=0.99),
        prior_answer_reference=None, department_preference=None)


@pytest.mark.parametrize("key", list(turn_payload()))
@pytest.mark.parametrize("repair_success", [True, False])
def test_every_required_field_uses_one_repair(key, repair_success):
    good = turn_payload()
    bad = dict(good)
    del bad[key]
    case = TriageCase(case_id="missing-contract")
    with patch.object(adapter, "_ai_available", return_value=True), patch.object(
        adapter, "complete_prompt", AsyncMock(side_effect=[json.dumps(bad), json.dumps(good if repair_success else bad)]),
    ) as ai, patch.object(preference, "fetch_active_departments", return_value=ACTIVE):
        result = asyncio.run(adapter.refine_case_with_ai(case, user_sources=["頭痛"]))
    assert ai.await_count == 2
    assert result.interpretation_complete is repair_success
    assert bool(case.patient_input.symptom) is repair_success


@pytest.mark.parametrize("key,value", [
    ("semantic_extractions", None), ("ttas_evidence", {}), ("time_preferences", "bad"),
    ("pending_answer", []), ("prior_answer_reference", []), ("department_preference", []),
    ("dialogue_act", None), ("dialogue_act", {}),
    ("dialogue_act", dict(act="fake", source_text="頭痛", confidence=0.99)),
])
def test_invalid_shape_or_act_cannot_hide_behind_semantic_evidence(key, value):
    bad = turn_payload()
    bad[key] = value
    case = TriageCase(case_id="invalid-contract")
    with patch.object(adapter, "_ai_available", return_value=True), patch.object(
        adapter, "complete_prompt", AsyncMock(return_value=json.dumps(bad)),
    ) as ai, patch.object(preference, "fetch_active_departments", return_value=ACTIVE):
        result = asyncio.run(adapter.refine_case_with_ai(case, user_sources=["頭痛"]))
    assert ai.await_count == 2
    assert not result.interpretation_complete
    assert not case.patient_input.symptom


@pytest.mark.parametrize("raw", ["[]", "null", "42", '"text"', "not json"])
def test_non_object_top_level_repairs_once(raw):
    with patch.object(adapter, "_ai_available", return_value=True), patch.object(
        adapter, "complete_prompt", AsyncMock(side_effect=[raw, json.dumps(turn_payload())]),
    ) as ai, patch.object(preference, "fetch_active_departments", return_value=ACTIVE):
        result = asyncio.run(adapter.refine_case_with_ai(TriageCase(case_id="shape"), user_sources=["頭痛"]))
    assert ai.await_count == 2
    assert result.interpretation_complete


def test_prevalidated_department_never_mutates_rejected_turn():
    text = "頭痛想看耳科"
    bad = turn_payload(text)
    bad["department_preference"] = dict(dept_id=1333, department_name="耳科", source_text="耳科", confidence=0.99)
    del bad["prior_answer_reference"]
    case = TriageCase(case_id="atomic-turn")
    async def respond_without_mutation(prompt):
        assert case.patient_input.requested_department_id is None
        assert case.patient_input.requested_department_name is None
        assert not case.patient_input.symptom
        return json.dumps(bad)
    with patch.object(adapter, "_ai_available", return_value=True), patch.object(
        adapter, "complete_prompt", AsyncMock(side_effect=respond_without_mutation),
    ) as ai, patch.object(preference, "fetch_active_departments", return_value=ACTIVE):
        result = asyncio.run(adapter.refine_case_with_ai(case, user_sources=[text]))
    assert ai.await_count == 2
    assert not result.interpretation_complete
    assert case.patient_input.requested_department_id is None


@pytest.mark.parametrize("kind", ["semantic", "ttas", "time"])
def test_invalid_preference_preserves_other_valid_evidence(kind):
    data = turn_payload()
    data["department_preference"] = dict(dept_id=9999, department_name="耳科", source_text="頭痛", confidence=0.99)
    if kind != "semantic":
        data["semantic_extractions"] = []
    if kind == "ttas":
        data["ttas_evidence"] = [dict(field="respiratory_distress", value="severe", semantic_status="available", source_text="頭痛", confidence=0.99)]
    if kind == "time":
        data["time_preferences"] = [dict(kind="session", relation="prefer", session="afternoon",
            priority=1, source_text="頭痛", confidence=0.99)]
    case = TriageCase(case_id="other-evidence")
    with patch.object(adapter, "_ai_available", return_value=True), patch.object(
        adapter, "complete_prompt", AsyncMock(return_value=json.dumps(data)),
    ) as ai, patch.object(preference, "fetch_active_departments", return_value=ACTIVE):
        result = asyncio.run(adapter.refine_case_with_ai(case, user_sources=["頭痛"]))
    ai.assert_awaited_once()
    assert result.interpretation_complete
    if kind == "semantic":
        assert case.patient_input.symptom == "頭痛"
    elif kind == "ttas":
        assert result.ttas_evidence
    else:
        assert result.time_preferences
    assert case.patient_input.requested_department_id is None


@pytest.mark.parametrize("extra_key", ["stage", "confirmed", "is_complete", "unexpected"])
@pytest.mark.parametrize("repair_success", [True, False])
def test_extra_top_level_field_repairs_once_then_requires_exact_contract(extra_key, repair_success):
    good = turn_payload()
    bad = dict(good, **{extra_key: "confirmed"})
    case = TriageCase(case_id="extra-top-level")

    async def repair(prompt):
        # No evidence or workflow may be applied before the entire turn passes.
        assert not case.patient_input.symptom
        assert case.semantic_extractions == []
        assert not case.confirmed
        assert "rejection_reason=invalid_turn_interpretation_fields" in prompt
        return json.dumps(good if repair_success else bad)

    ai = AsyncMock()
    async def respond(prompt):
        if ai.await_count == 1:
            return json.dumps(bad)
        return await repair(prompt)
    ai.side_effect = respond
    with patch.object(adapter, "_ai_available", return_value=True), patch.object(
        adapter, "complete_prompt", ai,
    ), patch.object(preference, "fetch_active_departments", return_value=ACTIVE):
        result = asyncio.run(adapter.refine_case_with_ai(case, user_sources=["頭痛"]))
    assert ai.await_count == 2
    assert result.interpretation_complete is repair_success
    assert bool(case.patient_input.symptom) is repair_success
    assert bool(case.semantic_extractions) is repair_success
    assert not case.confirmed


@pytest.mark.parametrize("extra", [{}, {"message": "我頭痛"}, {"legacy_batch": True}, {"visit_type": "return_visit"}])
def test_keyed_answers_are_rejected_without_entering_any_triage_path(extra):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.routes import chat as route
    interpreter, planner, department = AsyncMock(), AsyncMock(), AsyncMock()
    with patch.object(route, "refine_case_with_ai", interpreter), patch.object(
        route, "request_clarification", planner,
    ), patch.object(route, "reason_about_departments", department):
        response = TestClient(app).post("/chat", json={**extra, "answers": [{"key": "symptom", "answer": "頭痛"}]})
    assert response.status_code == 400
    assert response.json() == {"detail": "已不支援固定問卷作答，請直接以文字描述症狀。"}
    interpreter.assert_not_awaited()
    planner.assert_not_awaited()
    department.assert_not_awaited()
