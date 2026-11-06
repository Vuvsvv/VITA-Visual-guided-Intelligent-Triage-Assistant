"""Conversation ownership regressions; all provider responses are offline proposals."""
import asyncio
import json
import logging
from unittest.mock import AsyncMock, patch

import pytest

from app.config import get_settings
from app.schemas import DepartmentResult, Message, PendingAnswerInterpretation, SemanticExtraction, TriageCase
from app.services import conversation_service as cs, rag_triage_adapter as adapter
from app.services.case_store import save_case
from app.services.rule_engine import QUESTION_TEXTS, RED_FLAG_QUESTION_KEY
from test_phase7_natural_safety import runtime, respond, evidence, pending, route


FOCUSED = "我想確認頭暈的感受，是周圍在旋轉，還是快昏倒、眼前發黑？"
INTENT = "dizziness_quality"
TECHNICAL = cs.TECHNICAL_CLARIFICATION_REPLY
# The provider answered but nothing was usable: the patient gets a way forward, not a retry.
STUCK = cs.SAFETY_NET_QUESTION


def plan(question=FOCUSED, intent=INTENT):
    return dict(status="clarification_needed", question=question, intent=intent,
                reason="需要釐清頭暈感受", information_needs=["頭暈的感受"])


def clinical_case():
    case = TriageCase(case_id="ownership-test")
    case.patient_input.symptom = "頭暈"
    case.patient_input.red_flags_checked = True
    case.history_records = [Message(role="user", content="我頭暈"),
                            Message(role="assistant", content="你頭暈時有什麼感受？")]
    state = case.conversation_state
    state.department_status = "ambiguous"
    state.department_next_question_intent = INTENT
    state.pending_clarification_intent = INTENT
    state.asked_clarification_intents = [INTENT]
    state.next_information_needed = ["頭暈的感受"]
    # Time was already answered; this test covers the medical confirmation gate, not 6-H availability.
    state.availability_status = "answered"
    return case


def act(text, kind):
    return dict(act=kind, source_text=text, confidence=0.99)


def test_opaque_pending_answer_without_clinical_extraction_reaches_planner(runtime, caplog):
    client, interpreter, planner, department = runtime
    intent = "knee_swelling_presence"
    question = "左膝是否有腫脹或積液"
    text = "沒有明顯腫脹，也沒有感覺膝蓋裡有積液。"
    case = clinical_case()
    case.patient_input.symptom = "左膝痛"
    case.history_records = [Message(role="user", content="左膝痛"),
                            Message(role="assistant", content=question)]
    state = case.conversation_state
    state.pending_clarification_intent = intent
    state.pending_clarification_question = question
    state.asked_clarification_intents = [intent]
    state.department_next_question_intent = intent
    state.next_information_needed = [question]
    save_case(case)
    answer = dict(answered_intent=intent, answer_status="answered",
                  answer_source_text=text, answer_confidence=0.99)
    respond(interpreter, answer=answer)

    async def recompute(current):
        assert current.conversation_state.clarification_evidence[intent] == text
        assert current.conversation_state.pending_clarification_intent is None
        current.conversation_state.department_next_question_intent = "knee_function_impact"

    department.side_effect = recompute
    next_question = "左膝的不舒服對走路或上下樓梯有什麼影響？"
    planner.return_value = json.dumps(plan(next_question, "knee_function_impact"), ensure_ascii=False)
    with caplog.at_level(logging.INFO):
        result = client.post("/chat", json=dict(case_id=case.case_id, message=text)).json()
    interpreter.assert_awaited_once()
    department.assert_awaited_once()
    planner.assert_awaited_once()
    assert "turn interpretation accepted=false" not in caplog.text
    assert result["triage_case"]["semantic_extractions"] == []
    assert result["conversation_state"]["clarification_evidence"][intent] == text
    assert result["conversation_state"]["pending_clarification_intent"] != intent
    assert result["conversation_state"]["reply_source"] == "AI_PLANNER"
    assert result["reply"] == next_question
    assert text in planner.await_args.args[0]


def test_complete_negative_exits_safety_before_department_and_planner(runtime):
    client, interpreter, planner, department = runtime
    first_text = "我今天頭很暈而且心情不好一直想哭。"
    respond(interpreter, [evidence("symptom", "頭暈", "頭很暈"),
        evidence("duration", "1天", "今天", None),
        evidence("accompanying_symptoms", ["一直想哭"], "一直想哭")])
    first = client.post("/chat", json=dict(message=first_text)).json()
    assert first["conversation_state"]["clarification_status"] == "safety_check"
    assert first["conversation_state"]["next_information_needed"] == []

    async def recompute(case):
        assert case.conversation_state.clarification_status == "collecting"
        assert case.conversation_state.pending_clarification_intent is None
        assert case.conversation_state.next_information_needed == []
        case.conversation_state.department_status = "ambiguous"
        case.conversation_state.department_next_question_intent = INTENT

    department.reset_mock()
    department.side_effect = recompute
    respond(interpreter, answer=pending("都沒有"))
    planner.return_value = json.dumps(plan(), ensure_ascii=False)
    second = client.post("/chat", json=dict(case_id=first["case_id"], message="都沒有。")).json()
    department.assert_awaited_once()
    state = second["conversation_state"]
    assert second["triage_case"]["patient_input"]["red_flags_status"] == "negative"
    assert state["clarification_status"] == "clarifying"
    assert state["pending_clarification_intent"] == INTENT
    assert second["reply"] == FOCUSED
    assert "安全篩檢" not in "".join(state["next_information_needed"])
    assert '"safety_planning": false' in planner.await_args.args[0]


def test_safety_scope_clears_stale_pending_and_resets_only_fresh_attempts():
    case = clinical_case()
    case.patient_input.red_flags_checked = False
    case.conversation_state.safety_answer_attempts = 2
    case.conversation_state.safety_answer_validation = "old_cycle"
    cs._enter_safety_check(case)
    state = case.conversation_state
    assert state.pending_clarification_intent is None
    assert state.next_information_needed == []
    assert state.safety_answer_attempts == 0
    assert state.safety_answer_validation is None
    answer = PendingAnswerInterpretation(**pending("沒有胸痛，但其他我不知道。", "partial", "uncertain"))
    assert not cs.capture_safety_screen_answer(case, answer, [answer.answer_source_text])
    cs.continue_safety_check(case, None)
    assert state.safety_answer_attempts == 1
    assert not case.patient_input.red_flags_checked
    assert state.pending_clarification_intent is None
    answer = PendingAnswerInterpretation(**pending("都沒有"))
    assert cs.capture_safety_screen_answer(case, answer, ["都沒有"])
    assert state.clarification_status == "collecting"
    assert state.last_question_key is None
    assert state.department_next_question_intent is None


@pytest.mark.parametrize("kind,text", [
    ("request_explanation", "我聽不懂，你要問什麼細節？"),
    ("claims_already_answered", "我說過了"),
])
def test_grounded_dialogue_act_reaches_focused_planner_without_fabrication(runtime, kind, text):
    client, interpreter, planner, department = runtime
    case = clinical_case()
    save_case(case)
    interpreter.return_value = json.dumps(dict(time_preferences=[], prior_answer_reference=None, department_preference=None, semantic_extractions=[], ttas_evidence=[],
        pending_answer=None, dialogue_act=act(text, kind)), ensure_ascii=False)
    planner.return_value = json.dumps(plan(), ensure_ascii=False)
    result = client.post("/chat", json=dict(case_id=case.case_id, message=text)).json()
    assert result["reply"] == FOCUSED
    assert result["triage_case"]["semantic_extractions"] == []
    assert result["conversation_state"]["clarification_evidence"] == {}
    assert f'"act": "{kind}"' in planner.await_args.args[0]
    interpreter.assert_awaited_once()


def test_claims_already_answered_uses_accepted_answer_before_recomputation(runtime):
    client, interpreter, planner, department = runtime
    case = clinical_case()
    case.history_records.append(Message(role="user", content="房子在轉"))
    case.conversation_state.clarification_evidence[INTENT] = "房子在轉"
    save_case(case)

    async def recompute(current):
        assert current.conversation_state.pending_clarification_intent is None
        current.conversation_state.department_next_question_intent = "hearing_changes"

    department.side_effect = recompute
    interpreter.return_value = json.dumps(dict(time_preferences=[], prior_answer_reference=None, department_preference=None, semantic_extractions=[], ttas_evidence=[],
        pending_answer=None, dialogue_act=act("我說過了", "claims_already_answered")))
    planner.return_value = json.dumps(plan("頭暈時聽到的聲音有變化嗎？", "hearing_changes"))
    result = client.post("/chat", json=dict(case_id=case.case_id, message="我說過了")).json()
    department.assert_awaited_once()
    assert result["conversation_state"]["pending_clarification_intent"] == "hearing_changes"
    assert result["conversation_state"]["clarification_evidence"][INTENT] == "房子在轉"


def test_invalid_question_repaired_once_with_structured_reason():
    case = clinical_case()
    invalid = plan("你確診了，現在要掛哪個科別？")
    provider = AsyncMock(side_effect=[json.dumps(invalid), json.dumps(plan())])
    with patch.object(cs, "complete_runtime_json", provider):
        suggestion = asyncio.run(cs.request_clarification(case, ["不懂"]))
    assert provider.await_count == 2
    assert suggestion.question == FOCUSED
    assert suggestion.reply_source == "AI_REPAIR"
    repair = provider.await_args.args[0]
    for key in ("rejection_reason", "original_proposal", "required_next_question_intent",
                "grounded_patient_evidence", "pending_question", "dialogue_act"):
        assert key in repair


def test_invalid_repair_is_stuck_fallback_not_normal_retry():
    case = clinical_case()
    provider = AsyncMock(return_value=json.dumps(plan("你確診了，現在要掛哪個科別？")))
    with patch.object(cs, "complete_runtime_json", provider):
        suggestion = asyncio.run(cs.request_clarification(case, ["不懂"]))
    cs.advance_conversation(case, suggestion, user_sources=["不懂"], current_extractions=[])
    assert provider.await_count == 2
    assert case.triage.next_question == STUCK
    assert not case.conversation_state.confirmed
    assert case.conversation_state.department_status == "ambiguous"
    assert case.semantic_extractions == []


def test_answered_intent_different_wording_cannot_be_reasked():
    case = clinical_case()
    case.conversation_state.clarification_evidence[INTENT] = "房子在轉"
    case.conversation_state.pending_clarification_intent = None
    provider = AsyncMock(return_value=json.dumps(plan("暈的時候會覺得周圍旋轉嗎？")))
    with patch.object(cs, "complete_runtime_json", provider):
        suggestion = asyncio.run(cs.request_clarification(case, ["我說過了"]))
    cs.advance_conversation(case, suggestion, user_sources=["我說過了"], current_extractions=[])
    assert suggestion.question is None
    assert suggestion.rejection_reason == "intent_already_answered"
    assert case.conversation_state.pending_clarification_intent == cs.SAFETY_NET_PENDING_INTENT
    assert case.triage.next_question == STUCK


@pytest.mark.parametrize("question", [
    "你說的頭暈，是否像周圍在旋轉？",
    "頭暈和周圍旋轉的感覺，是同時出現嗎？",
])
def test_single_focused_question_with_normal_chinese_connector_is_valid(question):
    provider = AsyncMock(return_value=json.dumps(plan(question)))
    with patch.object(cs, "complete_runtime_json", provider):
        suggestion = asyncio.run(cs.request_clarification(clinical_case(), ["不確定"], focused_answer_status="partial"))
    assert suggestion.question == question
    provider.assert_awaited_once()


@pytest.mark.parametrize("question", ["程度如何，另外是否有疼痛？", "程度如何？持續多久？"])
def test_multiple_independent_requests_still_rejected(question):
    provider = AsyncMock(return_value=json.dumps(plan(question)))
    with patch.object(cs, "complete_runtime_json", provider):
        suggestion = asyncio.run(cs.request_clarification(clinical_case(), ["不確定"]))
    assert suggestion.question is None


@pytest.mark.parametrize("proposal", [
    act("原文不存在", "request_explanation"), act("不懂", "workflow_done"),
    {**act("不懂", "answer"), "confidence": float("nan")},
    {**act("不懂", "answer"), "confidence": True},
    {**act("不懂", "answer"), "confidence": 0.1},
    {**act("不懂", "answer"), "stage": "done"},
])
def test_invalid_dialogue_act_becomes_unknown(proposal):
    provider = AsyncMock(return_value=json.dumps(dict(pending_answer=None, ttas_evidence=[], time_preferences=[], prior_answer_reference=None, department_preference=None, semantic_extractions=[], dialogue_act=proposal)))
    with patch.object(adapter, "_ai_available", return_value=True), patch.object(adapter, "complete_prompt", provider):
        result = asyncio.run(adapter.refine_case_with_ai(clinical_case(), user_sources=["不懂"]))
    assert adapter._dialogue_act_from_ai(proposal, ["不懂"]).act == "unknown"
    assert result.interpretation_complete is False
    assert not clinical_case().conversation_state.confirmed


def test_explanation_act_and_new_medical_evidence_coexist(runtime):
    client, interpreter, planner, _ = runtime
    case = clinical_case()
    save_case(case)
    text = "我聽不懂，不過我還有一點耳鳴"
    interpreter.return_value = json.dumps(dict(time_preferences=[], prior_answer_reference=None, department_preference=None, semantic_extractions=[evidence(
        "accompanying_symptoms", ["耳鳴"], "還有一點耳鳴")], ttas_evidence=[],
        pending_answer=None, dialogue_act=act("我聽不懂", "request_explanation")))
    planner.return_value = json.dumps(plan())
    result = client.post("/chat", json=dict(case_id=case.case_id, message=text)).json()
    assert result["triage_case"]["semantic_extractions"][0]["normalized_value"] == ["耳鳴"]
    assert '"act": "request_explanation"' in planner.await_args.args[0]


def test_turn_log_has_ownership_and_reply_origin_but_no_patient_text(runtime, caplog):
    client, interpreter, planner, _ = runtime
    case = clinical_case()
    save_case(case)
    text = "SENTINEL_PRIVATE_我聽不懂"
    interpreter.return_value = json.dumps(dict(pending_answer=None, ttas_evidence=[], time_preferences=[], prior_answer_reference=None, department_preference=None, semantic_extractions=[], dialogue_act=act(text, "request_explanation")))
    planner.return_value = json.dumps({**plan(), "explanation": "頭暈的感受是指暈的時候身體或周圍的感覺"})
    with caplog.at_level(logging.INFO):
        client.post("/chat", json=dict(case_id=case.case_id, message=text))
    assert text not in caplog.text
    for field in ("clarification_before", "clarification_after", "pending_before", "pending_after",
                  "safety_completed_this_turn", "dialogue_act", "final_reply_source=AI_PLANNER"):
        assert field in caplog.text


def test_live_four_turn_sequence_never_rebinds_safety_need(runtime):
    client, interpreter, planner, department = runtime
    text = "我今天頭很暈而且心情不好一直想哭。"
    respond(interpreter, [evidence("symptom", "頭暈", "頭很暈"), evidence("duration", "1天", "今天", None)])
    first = client.post("/chat", json=dict(message=text)).json()
    case_id = first["case_id"]

    async def recompute(case):
        assert case.conversation_state.clarification_status != "safety_check"
        case.conversation_state.department_status = "ambiguous"
        case.conversation_state.department_next_question_intent = INTENT

    department.side_effect = recompute
    for text, interpretation, question in [
        ("都沒有。", dict(pending_answer=pending("都沒有"), dialogue_act=act("都沒有", "answer")), FOCUSED),
        ("我聽不懂，你要問什麼細節？", dict(pending_answer=None, dialogue_act=act("我聽不懂", "request_explanation")),
         "我想了解頭暈時周圍景物是否像在轉動？"),
        ("我說過了……", dict(pending_answer=None, dialogue_act=act("我說過了", "claims_already_answered")),
         "我還需要確認暈的感覺，是景物轉動還是眼前發黑？"),
    ]:
        interpreter.return_value = json.dumps(dict(semantic_extractions=[], ttas_evidence=[], time_preferences=[], prior_answer_reference=None, department_preference=None, **interpretation))
        planner.return_value = json.dumps(plan(question))
        result = client.post("/chat", json=dict(case_id=case_id, message=text)).json()
        assert result["reply"] == question
        assert result["conversation_state"]["pending_clarification_intent"] == INTENT
        assert result["conversation_state"]["next_information_needed"] == ["頭暈的感受"]
        assert result["triage_case"]["patient_input"]["red_flags_checked"]
        assert result["conversation_state"]["clarification_evidence"] == {}
        assert result["conversation_state"]["confirmed"] is False
    assert interpreter.await_count == 4


def test_partial_same_intent_uses_concrete_ai_question_without_clearing_answer(runtime):
    client, interpreter, planner, _ = runtime
    case = clinical_case()
    save_case(case)
    text = "好像有點轉但不確定"
    interpreter.return_value = json.dumps(dict(time_preferences=[], prior_answer_reference=None, department_preference=None, semantic_extractions=[], ttas_evidence=[],
        dialogue_act=act(text, "answer"), pending_answer=dict(answered_intent=INTENT,
            answer_status="partial", answer_source_text=text, answer_confidence=0.98)))
    planner.return_value = json.dumps(plan())
    result = client.post("/chat", json=dict(case_id=case.case_id, message=text)).json()
    assert result["reply"] == FOCUSED
    assert result["conversation_state"]["pending_clarification_intent"] == INTENT
    assert result["conversation_state"]["clarification_evidence"] == {}


def test_fresh_revision_safety_cycle_does_not_inherit_old_attempt_cap(runtime):
    client, interpreter, _, _ = runtime
    # 2026-11-06: a stopped screen can no longer be revised (test_safety_gaps.py); a fresh cycle now
    # comes from a symptom first said after a completed screen, which needed two unclear replies.
    case = clinical_case()
    case.patient_input.red_flags_status = "negative"
    case.conversation_state.clarification_status = "clarifying"
    case.conversation_state.safety_answer_attempts = cs.SAFETY_ANSWER_CAP - 1
    save_case(case)
    respond(interpreter, [evidence("accompanying_symptoms", ["耳鳴"], "耳鳴"),
                          evidence("duration", "3天", "3天", None)])
    fresh = client.post("/chat", json=dict(case_id=case.case_id, message="還有耳鳴3天")).json()
    assert fresh["conversation_state"]["clarification_status"] == "safety_check"
    assert fresh["conversation_state"]["safety_answer_attempts"] == 0
    respond(interpreter, answer=pending("不確定", "unclear", "uncertain"))
    retry = client.post("/chat", json=dict(case_id=case.case_id, message="不確定")).json()
    assert retry["conversation_state"]["safety_answer_attempts"] == 1
    assert not retry["triage_case"]["patient_input"]["red_flags_checked"]


def test_fresh_reasoning_may_propose_pre_safety_unanswered_intent_but_not_restore_automatically():
    case = clinical_case()
    case.patient_input.red_flags_checked = False
    cs._enter_safety_check(case)
    assert cs.capture_safety_screen_answer(case, PendingAnswerInterpretation(**pending("都沒有")), ["都沒有"])
    assert case.conversation_state.pending_clarification_intent is None
    assert case.conversation_state.department_next_question_intent is None
    case.conversation_state.department_next_question_intent = INTENT  # Fresh reasoning result.
    cs.advance_conversation(case, cs.ClarificationSuggestion("clarification_needed", FOCUSED, INTENT, "需確認感受"),
                            user_sources=["都沒有"], current_extractions=[])
    assert case.triage.next_question == FOCUSED
    assert case.conversation_state.pending_clarification_intent == INTENT
    assert case.conversation_state.asked_clarification_intents == [INTENT]


def test_technical_reply_does_not_replace_actual_pending_question_for_interpreter():
    case = clinical_case()
    cs.advance_conversation(case, cs.ClarificationSuggestion("clarification_needed", FOCUSED, INTENT, "需確認感受"),
                            user_sources=["不確定"], current_extractions=[])
    cs.advance_conversation(case, None, user_sources=["我不懂"], current_extractions=[])
    case.history_records.append(Message(role="assistant", content=TECHNICAL))
    prompt = adapter._build_symptom_collection_prompt(case, ["再試一次"])
    assert f'"pending_question": "{FOCUSED}"' in prompt


@pytest.mark.parametrize("invalid", ["{broken", [], dict(status=[], reason="invalid"),
    {**plan(), "information_needs": ["感受", "持續時間"]},
    {**plan(), "stage": "done"}, plan("可以再補充和剛才問題相關的症狀細節嗎？"),
    plan("是否直接改看耳科？")])
def test_malformed_and_untrusted_planner_never_gain_state_authority(invalid):
    raw = invalid if isinstance(invalid, str) else json.dumps(invalid)
    provider = AsyncMock(return_value=raw)
    case = clinical_case()
    with patch.object(cs, "complete_runtime_json", provider):
        suggestion = asyncio.run(cs.request_clarification(case, ["不懂"]))
    cs.advance_conversation(case, suggestion, user_sources=["不懂"], current_extractions=[])
    assert case.triage.next_question == STUCK
    assert not case.conversation_state.confirmed
    assert provider.await_count == 2


@pytest.mark.parametrize("field,value", [("answer_status", []), ("answer_assertion", [])])
def test_pending_answer_malformed_enum_fails_closed(field, value):
    item = dict(answered_intent=INTENT, answer_status="answered", answer_source_text="不懂", answer_confidence=0.99)
    item[field] = value
    assert adapter._pending_answer_from_ai(item, pending_intent=INTENT, user_sources=["不懂"]) is None


def test_revision_continues_the_case_and_never_reasks_an_answered_question():
    case = clinical_case()
    case.history_records += [Message(role="assistant", content=FOCUSED), Message(role="user", content="房子在轉")]
    case.conversation_state.clarification_evidence[INTENT] = "房子在轉"
    route._begin_revision(case)
    assert case.conversation_state.clarification_evidence == {INTENT: "房子在轉"}
    assert INTENT in case.conversation_state.asked_clarification_intents
    case.conversation_state.department_status = "ambiguous"
    case.conversation_state.department_next_question_intent = INTENT
    tinnitus = "耳鳴是單邊還是兩邊都有？"
    provider = AsyncMock(side_effect=[json.dumps(plan()), json.dumps(plan(tinnitus, "tinnitus_side"))])
    with patch.object(cs, "complete_runtime_json", provider):
        suggestion = asyncio.run(cs.request_clarification(case, ["我還有耳鳴"]))
    cs.advance_conversation(case, suggestion, user_sources=["我還有耳鳴"], current_extractions=[])
    assert suggestion.rejection_reason == "intent_already_answered"
    assert case.triage.next_question == tinnitus
    assert case.conversation_state.pending_clarification_intent == "tinnitus_side"


def test_second_a1_missing_information_need_rejected_then_repaired():
    invalid = plan("可以再詳細說明一下嗎？")
    invalid.pop("information_needs")
    provider = AsyncMock(side_effect=[json.dumps(invalid), json.dumps(plan())])
    with patch.object(cs, "complete_runtime_json", provider):
        suggestion = asyncio.run(cs.request_clarification(clinical_case(), ["不懂"]))
    assert provider.await_count == 2
    assert suggestion.rejection_reason == "invalid_information_need"
    assert suggestion.reply_source == "AI_REPAIR"
    assert suggestion.question == FOCUSED


def test_second_a2_missing_need_after_repair_is_stuck_fallback():
    invalid = plan("可以再詳細說明一下嗎？")
    invalid.pop("information_needs")
    provider = AsyncMock(return_value=json.dumps(invalid))
    case = clinical_case()
    with patch.object(cs, "complete_runtime_json", provider):
        suggestion = asyncio.run(cs.request_clarification(case, ["不懂"]))
    cs.advance_conversation(case, suggestion, user_sources=["不懂"], current_extractions=[])
    assert provider.await_count == 2
    assert case.triage.next_question == STUCK
    assert case.conversation_state.reply_source == "SAFETY_NET_QUESTION"
    assert not case.conversation_state.confirmed
    assert case.conversation_state.department_status == "ambiguous"


@pytest.mark.parametrize("question", ["可以再詳細說明一下嗎？", "可以再補充一下嗎？",
    "可以換個方式說明嗎？", "可以再描述目前情況嗎？"])
def test_second_a3_generic_question_with_concrete_need_requires_repair(question):
    provider = AsyncMock(side_effect=[json.dumps(plan(question)), json.dumps(plan())])
    with patch.object(cs, "complete_runtime_json", provider):
        suggestion = asyncio.run(cs.request_clarification(clinical_case(), ["不懂"]))
    assert suggestion.reply_source == "AI_REPAIR"
    assert suggestion.rejection_reason == "generic_question"
    assert provider.await_count == 2


@pytest.mark.parametrize("need", ["更多資訊", "更多細節", "症狀細節", "目前情況", "剛才問題", "上述問題", "其他資訊"])
def test_second_a_generic_information_need_is_not_concrete(need):
    proposal = {**plan(), "information_needs": [need]}
    assert cs._question_rejection(clinical_case(), proposal, INTENT, False) == "generic_information_need"


def prior_case():
    case = clinical_case()
    case.history_records.insert(1, Message(role="user", content="我頭暈，感覺房子一直在轉，而且耳朵嗡嗡叫。"))
    case.semantic_extractions = [SemanticExtraction(**evidence(
        "symptom", "眩暈", "房子一直在轉"), extractor="ai")]
    return case


def prior_reference(source="房子一直在轉", intent=INTENT, confidence=0.99):
    return dict(answered_intent=intent, answer_source_text=source, answer_confidence=confidence)


def test_second_b1_prior_accepted_reference_captured_before_department_recompute(runtime):
    client, interpreter, planner, department = runtime
    case = prior_case()
    save_case(case)

    async def recompute(current):
        assert current.conversation_state.pending_clarification_intent is None
        assert current.conversation_state.clarification_evidence[INTENT] == "房子一直在轉"
        current.conversation_state.department_next_question_intent = "hearing_changes"

    department.side_effect = recompute
    interpreter.return_value = json.dumps(dict(time_preferences=[], department_preference=None, semantic_extractions=[], ttas_evidence=[], pending_answer=None,
        dialogue_act=act("我說過了", "claims_already_answered"), prior_answer_reference=prior_reference()))
    planner.return_value = json.dumps(plan("這次聽到的聲音有改變嗎？", "hearing_changes"))
    result = client.post("/chat", json=dict(case_id=case.case_id, message="我說過了")).json()
    department.assert_awaited_once()
    assert result["conversation_state"]["pending_clarification_intent"] == "hearing_changes"
    assert len(result["triage_case"]["semantic_extractions"]) == 1
    assert "prior_answer_reference" in interpreter.await_args.args[0]


@pytest.mark.parametrize("change", ["fabricated", "history_only", "low_confidence", "uncertain",
    "low_prior_confidence", "wrong_intent", "wrong_act", "ungrounded_history", "extra_authority", "current_only"])
def test_second_b2_invalid_prior_reference_preserves_pending_and_evidence(runtime, change):
    client, interpreter, planner, department = runtime
    case = prior_case()
    reference = prior_reference()
    kind = "claims_already_answered"
    extractions = []
    text = "我說過了"
    if change == "fabricated":
        reference["answer_source_text"] = "我以前說過會昏倒"
    elif change == "history_only":
        case.semantic_extractions = []
    elif change == "low_confidence":
        reference["answer_confidence"] = 0.1
    elif change == "uncertain":
        case.semantic_extractions[0].assertion = "uncertain"
    elif change == "low_prior_confidence":
        case.semantic_extractions[0].confidence = 0.1
    elif change == "wrong_intent":
        reference["answered_intent"] = "wrong_intent"
    elif change == "wrong_act":
        kind = "request_explanation"
    elif change == "ungrounded_history":
        case.history_records = case.history_records[:1] + case.history_records[-1:]
    elif change == "extra_authority":
        reference["stage"] = "done"
    elif change == "current_only":
        text = "我說過了，房子一直在轉"
        case.semantic_extractions = []
        case.history_records = case.history_records[:1] + case.history_records[-1:]
        extractions = [evidence("accompanying_symptoms", ["眩暈"], "房子一直在轉")]
    save_case(case)
    interpreter.return_value = json.dumps(dict(time_preferences=[], department_preference=None, semantic_extractions=extractions, ttas_evidence=[], pending_answer=None,
        dialogue_act=act("我說過了", kind), prior_answer_reference=reference))
    planner.return_value = json.dumps(plan())
    result = client.post("/chat", json=dict(case_id=case.case_id, message=text)).json()
    assert result["conversation_state"]["pending_clarification_intent"] == INTENT
    assert result["conversation_state"]["clarification_evidence"] == {}
    assert result["reply"] == FOCUSED
    assert not result["conversation_state"]["confirmed"]


def test_second_b3_prior_reference_never_completes_safety(runtime):
    client, interpreter, planner, _ = runtime
    case = prior_case()
    case.patient_input.red_flags_checked = False
    cs._enter_safety_check(case)
    save_case(case)
    interpreter.return_value = json.dumps(dict(time_preferences=[], department_preference=None, semantic_extractions=[], ttas_evidence=[], pending_answer=None,
        dialogue_act=act("我說過了", "claims_already_answered"),
        prior_answer_reference=prior_reference(intent="safety_screen")))
    planner.return_value = json.dumps(plan("這次不適是否突然開始？", "safety_screen"))
    result = client.post("/chat", json=dict(case_id=case.case_id, message="我說過了")).json()
    assert not result["triage_case"]["patient_input"]["red_flags_checked"]
    assert result["conversation_state"]["clarification_status"] == "safety_check"
    assert result["conversation_state"]["safety_answer_attempts"] == 1
    assert result["conversation_state"]["clarification_evidence"] == {}


CORRECTION_TEXT = "我剛剛說錯，不是頭暈，是頭痛"


def correction_payload(dialogue=None, extractions=None):
    return dict(time_preferences=[], prior_answer_reference=None, department_preference=None, semantic_extractions=extractions if extractions is not None else [
        evidence("symptom", "頭暈", "不是頭暈", "absent"), evidence("symptom", "頭痛", "頭痛")],
        pending_answer=None, ttas_evidence=[], dialogue_act=dialogue or act("我剛剛說錯", "correction"))


def test_second_c1_collecting_correction_replaces_scalar_and_invalidates_before_reasoning(runtime):
    client, interpreter, planner, department = runtime
    case = clinical_case()
    case.semantic_extractions = [SemanticExtraction(**evidence("symptom", "頭暈", "頭暈"), extractor="ai")]
    case.department_result = DepartmentResult(dept_id=1333, parentDept="耳鼻喉部", childDept="耳科")
    case.conversation_state.department_status = "resolved"
    case.availability.preferred_sessions = ["下午"]
    save_case(case)

    async def recompute(current):
        assert current.patient_input.symptom == "頭痛"
        assert current.department_result is None
        state = current.conversation_state
        assert state.department_status == "unresolved"
        assert state.pending_clarification_intent is None
        assert state.pending_clarification_question is None
        assert state.next_information_needed == []
        assert state.candidate_departments == []
        assert not state.revision_mode
        state.department_status = "ambiguous"
        state.department_next_question_intent = "headache_quality"

    department.side_effect = recompute
    interpreter.return_value = json.dumps(correction_payload())
    planner.return_value = json.dumps({**plan("這次頭痛的感覺是怎樣的？", "headache_quality"),
                                      "information_needs": ["頭痛的感受"]})
    from app.services import rule_engine
    assert not hasattr(rule_engine, "_looks_like_symptom_revision")
    result = client.post("/chat", json=dict(case_id=case.case_id, message=CORRECTION_TEXT)).json()
    department.assert_awaited_once()
    assert result["triage_case"]["patient_input"]["symptom"] == "頭痛"
    assert result["triage_case"]["availability"]["preferred_sessions"] == ["下午"]
    from app.services.department_reasoning_service import accepted_semantic_evidence
    current = accepted_semantic_evidence(route.get_case(case.case_id))
    assert next(item for item in current if item["normalized_value"] == "頭暈")["assertion"] == "absent"
    assert next(item for item in current if item["normalized_value"] == "頭痛")["assertion"] == "present"


@pytest.mark.parametrize("kind,confidence,source", [("correction", 0.1, "我剛剛說錯"),
    ("correction", 0.99, "不在本輪"), ("new_information", 0.99, "我剛剛說錯")])
def test_second_c3_c4_no_valid_correction_no_scalar_replacement(kind, confidence, source):
    case = clinical_case()
    payload = correction_payload(dialogue=dict(act=kind, confidence=confidence, source_text=source),
        extractions=[evidence("symptom", "頭痛", "頭痛")])
    provider = AsyncMock(return_value=json.dumps(payload))
    with patch.object(adapter, "_ai_available", return_value=True), patch.object(adapter, "complete_prompt", provider):
        asyncio.run(adapter.refine_case_with_ai(case, user_sources=[CORRECTION_TEXT]))
    assert case.patient_input.symptom == "頭暈"
    # A grounded new_information turn keeps its additional symptom as evidence only;
    # invalid correction turns are rejected before any evidence is applied.
    assert [(item.normalized_value, item.assertion) for item in case.semantic_extractions] == (
        [("頭痛", "present")] if kind == "new_information" else []
    )
    assert case.conversation_state.pending_clarification_intent == INTENT


@pytest.mark.parametrize("extractions", [[], [evidence("symptom", "頭痛", "原文沒有")],
    [{**evidence("symptom", "頭痛", "頭痛"), "confidence": 0.1}]])
def test_second_c2_correction_without_accepted_replacement_has_no_state_authority(extractions):
    case = clinical_case()
    case.department_result = DepartmentResult(dept_id=1333)
    provider = AsyncMock(return_value=json.dumps(correction_payload(extractions=extractions)))
    with patch.object(adapter, "_ai_available", return_value=True), patch.object(adapter, "complete_prompt", provider):
        asyncio.run(adapter.refine_case_with_ai(case, user_sources=[CORRECTION_TEXT]))
    assert case.patient_input.symptom == "頭暈"
    assert case.department_result.dept_id == 1333
    assert case.conversation_state.pending_clarification_intent == INTENT
    assert not case.conversation_state.revision_mode


def test_second_c_companion_correction_removes_only_grounded_absent_concept():
    case = clinical_case()
    case.patient_input.accompanying_symptoms = ["耳鳴", "噁心"]
    payload = correction_payload(extractions=[evidence("accompanying_symptoms", ["耳鳴"], "沒有耳鳴", "absent"),
        evidence("accompanying_symptoms", ["頭痛"], "頭痛")])
    payload["dialogue_act"] = act("我說錯了", "correction")
    provider = AsyncMock(return_value=json.dumps(payload))
    case.history_records.append(Message(role="user", content="我說錯了，沒有耳鳴，還有頭痛"))
    with patch.object(adapter, "_ai_available", return_value=True), patch.object(adapter, "complete_prompt", provider):
        asyncio.run(adapter.refine_case_with_ai(case, user_sources=["我說錯了，沒有耳鳴，還有頭痛"]))
    assert case.patient_input.accompanying_symptoms == ["噁心", "頭痛"]


@pytest.mark.parametrize("required,expected", [(INTENT, "unresolved_information_remaining"),
    (None, "department_still_ambiguous")])
def test_second_d1_d2_sufficient_with_unresolved_rejected_then_repaired(required, expected):
    case = clinical_case()
    if required is None:
        case.conversation_state.pending_clarification_intent = None
        case.conversation_state.department_next_question_intent = None
        case.conversation_state.asked_clarification_intents = []
    provider = AsyncMock(side_effect=[json.dumps(dict(status="sufficient", reason="enough")), json.dumps(plan())])
    with patch.object(cs, "complete_runtime_json", provider):
        suggestion = asyncio.run(cs.request_clarification(case, ["不懂"]))
    assert suggestion.rejection_reason == expected
    assert suggestion.reply_source == "AI_REPAIR"
    assert suggestion.question == FOCUSED
    assert provider.await_count == 2


def test_second_d3_truly_sufficient_does_not_need_information_needs():
    case = clinical_case()
    state = case.conversation_state
    state.pending_clarification_intent = None
    state.department_next_question_intent = None
    state.department_status = "resolved"
    case.department_result = DepartmentResult(dept_id=1333)
    case.semantic_extractions = [SemanticExtraction(**evidence("duration", "3天", "3天", None), extractor="ai")]
    case.history_records.insert(1, Message(role="user", content="頭暈3天"))
    provider = AsyncMock(return_value=json.dumps(dict(status="sufficient", reason="enough")))
    with patch.object(cs, "complete_runtime_json", provider):
        suggestion = asyncio.run(cs.request_clarification(case, ["沒有補充"]))
    cs.advance_conversation(case, suggestion, user_sources=["沒有補充"], current_extractions=[])
    provider.assert_awaited_once()
    assert state.is_complete
    assert state.clarification_status == "sufficient"


@pytest.mark.parametrize("field,old,new,span", [("body_part", "右耳", "左耳", "左耳"),
    ("duration", "3天", "2週", "兩週"), ("severity", "mild", "severe", "嚴重"),
    ("onset", "逐漸開始", "突然開始", "突然開始")])
def test_second_c_scalar_replacement_is_field_scoped(field, old, new, span):
    case = clinical_case()
    setattr(case.patient_input, field, old)
    text = "我說錯了，" + span
    case.history_records.append(Message(role="user", content=text))
    payload = correction_payload(dialogue=act("我說錯了", "correction"),
        extractions=[evidence(field, new, span, None)])
    provider = AsyncMock(return_value=json.dumps(payload))
    with patch.object(adapter, "_ai_available", return_value=True), patch.object(adapter, "complete_prompt", provider):
        asyncio.run(adapter.refine_case_with_ai(case, user_sources=[text]))
    assert getattr(case.patient_input, field) == new
    assert case.patient_input.symptom == "頭暈"
    assert not case.conversation_state.revision_mode
    assert case.conversation_state.department_status == "unresolved"


def test_second_c_safety_correction_preserves_attempts_validation_and_ownership(runtime):
    client, interpreter, planner, department = runtime
    case = clinical_case()
    case.patient_input.red_flags_checked = False
    cs._enter_safety_check(case)
    case.conversation_state.safety_answer_attempts = 1
    save_case(case)
    interpreter.return_value = json.dumps({**correction_payload(),
        "pending_answer": pending(CORRECTION_TEXT, "partial", "uncertain")})
    planner.return_value = json.dumps({**plan("這次頭痛是否突然開始？", "safety_screen"),
        "information_needs": ["頭痛是否突然開始"]})
    result = client.post("/chat", json=dict(case_id=case.case_id, message=CORRECTION_TEXT)).json()
    assert result["triage_case"]["patient_input"]["symptom"] == "頭痛"
    assert not result["triage_case"]["patient_input"]["red_flags_checked"]
    assert result["conversation_state"]["clarification_status"] == "safety_check"
    assert result["conversation_state"]["safety_answer_attempts"] == 2
    assert result["conversation_state"]["safety_answer_validation"] == "partial_uncertain"
    department.assert_not_awaited()


@pytest.mark.parametrize("confidence", [True, "0.99", -0.1, 1.1, float("nan"), float("inf")])
def test_second_b_prior_reference_confidence_is_strict_finite_bounded(confidence):
    case = prior_case()
    interpretation = adapter._dialogue_act_from_ai(act("我說過了", "claims_already_answered"), ["我說過了"])
    assert cs.validate_prior_answer_reference(prior_reference(confidence=confidence), interpretation, INTENT,
        tuple(case.semantic_extractions), [item.content for item in case.history_records if item.role == "user"]) is None


def test_second_b_valid_prior_reference_does_not_trigger_extra_interpretation_or_capture_meta_answer(runtime):
    client, interpreter, planner, _ = runtime
    case = prior_case()
    save_case(case)
    interpreter.return_value = json.dumps(dict(time_preferences=[], department_preference=None, semantic_extractions=[], ttas_evidence=[],
        pending_answer=dict(answered_intent=INTENT, answer_status="answered",
                            answer_source_text="我說過了", answer_confidence=0.99),
        dialogue_act=act("我說過了", "claims_already_answered"), prior_answer_reference=prior_reference()))
    planner.return_value = json.dumps(plan("這次聽到的聲音有變化嗎？", "hearing_changes"))
    result = client.post("/chat", json=dict(case_id=case.case_id, message="我說過了")).json()
    interpreter.assert_awaited_once()
    assert result["conversation_state"]["clarification_evidence"][INTENT] == "房子一直在轉"


def resolved_after_answer(expected_source):
    async def recompute(case):
        state = case.conversation_state
        assert state.clarification_evidence[INTENT] == expected_source
        assert state.pending_clarification_intent is None
        state.department_status = "resolved"
        state.department_next_question_intent = None
        case.department_result = DepartmentResult(dept_id=1333, parentDept="五官科", childDept="耳科")
    return recompute


def test_control_prior_reference_precedes_meta_pending_answer_with_new_evidence(runtime, caplog):
    client, interpreter, planner, department = runtime
    case = prior_case()
    save_case(case)
    text = "我說過了，而且今天還有耳鳴"
    interpreter.return_value = json.dumps(dict(time_preferences=[], department_preference=None, 
        semantic_extractions=[evidence("accompanying_symptoms", ["耳鳴"], "還有耳鳴")],
        ttas_evidence=[], dialogue_act=act("我說過了", "claims_already_answered"),
        prior_answer_reference=prior_reference(), pending_answer=dict(answered_intent=INTENT,
            answer_status="answered", answer_source_text="我說過了", answer_confidence=0.99)))
    planner.return_value = json.dumps(dict(status="sufficient", reason="資訊已足夠"))
    department.side_effect = resolved_after_answer("房子一直在轉")
    with caplog.at_level(logging.INFO):
        result = client.post("/chat", json=dict(case_id=case.case_id, message=text)).json()
    assert result["conversation_state"]["clarification_evidence"][INTENT] == "房子一直在轉"
    assert result["conversation_state"]["clarification_evidence"][INTENT] != "我說過了"
    assert result["conversation_state"]["pending_clarification_intent"] is None
    assert any(item["normalized_value"] == ["耳鳴"] for item in result["triage_case"]["semantic_extractions"])
    interpreter.assert_awaited_once()
    planner.assert_awaited_once()
    department.assert_awaited_once()
    assert "prior_answer_reference_accepted" in caplog.text
    assert text not in caplog.text


@pytest.mark.parametrize("reference", [None, prior_reference("我以前說過會昏倒")])
def test_control_invalid_prior_reference_falls_back_to_valid_direct_answer(runtime, reference):
    client, interpreter, planner, department = runtime
    case = prior_case()
    save_case(case)
    text = "我說過了，房子在旋轉，而且有耳鳴"
    interpreter.return_value = json.dumps(dict(time_preferences=[], department_preference=None, 
        semantic_extractions=[evidence("accompanying_symptoms", ["耳鳴"], "有耳鳴")],
        ttas_evidence=[], dialogue_act=act("我說過了", "claims_already_answered"),
        prior_answer_reference=reference, pending_answer=dict(answered_intent=INTENT,
            answer_status="answered", answer_source_text="房子在旋轉", answer_confidence=0.99)))
    planner.return_value = json.dumps(dict(status="sufficient", reason="資訊已足夠"))
    department.side_effect = resolved_after_answer("房子在旋轉")
    result = client.post("/chat", json=dict(case_id=case.case_id, message=text)).json()
    assert result["conversation_state"]["clarification_evidence"][INTENT] == "房子在旋轉"
    assert result["conversation_state"]["pending_clarification_intent"] is None
    interpreter.assert_awaited_once()
    department.assert_awaited_once()


def active_safety_case():
    case = clinical_case()
    case.patient_input.red_flags_checked = False
    cs._enter_safety_check(case)
    case.conversation_state.safety_answer_attempts = 1
    case.history_records.append(Message(role="assistant", content=QUESTION_TEXTS[RED_FLAG_QUESTION_KEY]))
    return case


SAFETY_EXPLANATION_QUESTION = "我想確認這次不舒服是否突然出現？"
SAFETY_EXPLANATION = "突然出現是指不舒服在很短時間內發生，這能幫助確認是否需要優先處理。"


def safety_explanation_plan():
    return {**plan(SAFETY_EXPLANATION_QUESTION, "safety_screen"), "information_needs": ["不舒服是否突然出現"],
            "explanation": SAFETY_EXPLANATION}


def test_control_safety_explanation_does_not_consume_answer_attempt(runtime, caplog):
    client, interpreter, planner, department = runtime
    case = active_safety_case()
    save_case(case)
    text = "我聽不懂，你在問什麼？"
    interpreter.return_value = json.dumps(dict(time_preferences=[], prior_answer_reference=None, department_preference=None, semantic_extractions=[], ttas_evidence=[], pending_answer=None,
        dialogue_act=act("我聽不懂", "request_explanation")))
    planner.return_value = json.dumps(safety_explanation_plan())
    with caplog.at_level(logging.INFO):
        result = client.post("/chat", json=dict(case_id=case.case_id, message=text)).json()
    state = result["conversation_state"]
    assert state["safety_answer_attempts"] == 1
    assert state["safety_answer_validation"] == "request_explanation"
    assert state["clarification_status"] == "safety_check"
    assert not result["triage_case"]["patient_input"]["red_flags_checked"]
    assert result["triage_case"]["patient_input"]["red_flags_status"] == "not_checked"
    assert result["reply"] == f"{SAFETY_EXPLANATION}\n{QUESTION_TEXTS[RED_FLAG_QUESTION_KEY]}"
    assert state["reply_source"] == "AI_PLANNER"
    department.assert_not_awaited()
    planner.assert_awaited_once()
    interpreter.assert_awaited_once()
    assert "safety_request_explanation_no_attempt" in caplog.text
    assert text not in caplog.text


@pytest.mark.parametrize("status", ["partial", "unclear"])
def test_control_safety_medical_answer_still_consumes_attempt(runtime, status):
    client, interpreter, planner, department = runtime
    case = active_safety_case()
    save_case(case)
    text = "不確定"
    interpreter.return_value = json.dumps(dict(time_preferences=[], prior_answer_reference=None, department_preference=None, semantic_extractions=[], ttas_evidence=[],
        pending_answer=pending(text, status, "uncertain"), dialogue_act=act(text, "answer")))
    planner.return_value = json.dumps(safety_explanation_plan())
    result = client.post("/chat", json=dict(case_id=case.case_id, message=text)).json()
    assert result["conversation_state"]["safety_answer_attempts"] == 2
    assert not result["triage_case"]["patient_input"]["red_flags_checked"]
    department.assert_not_awaited()


@pytest.mark.parametrize("interpretation", [
    {**act("我聽不懂", "request_explanation"), "confidence": 0.1},
    act("本輪沒有這段原文", "request_explanation"),
])
def test_control_invalid_explanation_act_does_not_get_attempt_exemption(runtime, interpretation):
    client, interpreter, planner, department = runtime
    case = active_safety_case()
    save_case(case)
    interpreter.return_value = json.dumps(dict(time_preferences=[], prior_answer_reference=None, department_preference=None, semantic_extractions=[], ttas_evidence=[], pending_answer=None,
        dialogue_act=interpretation))
    planner.return_value = json.dumps(safety_explanation_plan())
    result = client.post("/chat", json=dict(case_id=case.case_id, message="我聽不懂，你在問什麼？")).json()
    assert result["conversation_state"]["safety_answer_attempts"] == 2
    planner.assert_not_awaited()  # Invalid interpretation stays fail-closed, including after repair.
    assert not result["triage_case"]["patient_input"]["red_flags_checked"]
    department.assert_not_awaited()


def test_control_safety_explanations_remain_bounded_by_existing_turn_cap(runtime):
    client, interpreter, planner, department = runtime
    case = active_safety_case()
    case.conversation_state.turn_count = get_settings().hard_turn_cap - 1
    save_case(case)
    interpreter.return_value = json.dumps(dict(time_preferences=[], prior_answer_reference=None, department_preference=None, semantic_extractions=[], ttas_evidence=[], pending_answer=None,
        dialogue_act=act("我聽不懂", "request_explanation")))
    planner.return_value = json.dumps(safety_explanation_plan())
    result = client.post("/chat", json=dict(case_id=case.case_id, message="我聽不懂")).json()
    assert result["conversation_state"]["clarification_status"] == "unresolved"
    assert result["conversation_state"]["safety_answer_attempts"] == 1
    assert not result["triage_case"]["patient_input"]["red_flags_checked"]
    assert result["conversation_state"]["confirmed"] is False
    assert result["next_question"] is None
    department.assert_not_awaited()

