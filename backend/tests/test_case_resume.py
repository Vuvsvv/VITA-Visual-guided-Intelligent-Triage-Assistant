from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ConversationStage, DepartmentResult, Message, VisitType
from app.services import case_store
import importlib

chat_route = importlib.import_module("app.routes.chat")


@pytest.fixture(autouse=True)
def isolated_store():
    case_store._CASES.clear()
    case_store._LAST_TOUCHED.clear()
    case_store._RECOMMENDATIONS_BY_CASE.clear()
    yield
    case_store._CASES.clear()
    case_store._LAST_TOUCHED.clear()
    case_store._RECOMMENDATIONS_BY_CASE.clear()


def test_resume_is_read_only_minimal_and_does_not_refresh_ttl(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(case_store, "_clock", lambda: now[0])
    case = case_store.create_case("case_resume")
    case.visit_type = VisitType.INITIAL
    case.patient_input.symptom = "PRIVATE_PATIENT_INPUT"
    case.history_records = [Message(role="user", content="PRIVATE_HISTORY")]
    case.conversation_state.last_question_key = "duration"
    case.conversation_state.question_attempts = {"duration": 2}
    case.triage.next_question = "已詢問的持續時間問題"
    before = case.model_dump(mode="json")
    touched = dict(case_store._LAST_TOUCHED)
    forbidden = Mock(side_effect=AssertionError("resume must not mutate or query providers"))
    for name in ("create_case", "save_case", "apply_ttas_evaluation", "apply_user_message"):
        monkeypatch.setattr(chat_route, name, forbidden)
    for name in ("refine_case_with_ai", "reason_about_departments", "request_clarification", "generate_triage_reply"):
        monkeypatch.setattr(chat_route, name, AsyncMock(side_effect=AssertionError(name)))
    now[0] += 60
    response = TestClient(app).get("/chat/case_resume/resume")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["can_continue"] is True
    assert body["last_question_key"] == "duration"
    assert body["question_batch"][0]["key"] == "duration"
    assert body["question_batch"][0]["question"] == case.triage.next_question
    assert set(body) == {
        "case_id", "visit_type", "stage", "confirmed", "awaiting_confirmation", "department_status",
        "clarification_status", "red_flags_checked", "red_flags_status", "warning_required", "warning_message",
        "department_result", "next_question", "last_question_key", "question_batch", "can_continue",
        "reply_source",
    }
    assert "PRIVATE" not in response.text
    assert case.model_dump(mode="json") == before
    assert case_store._LAST_TOUCHED == touched
    forbidden.assert_not_called()
    for name in ("refine_case_with_ai", "reason_about_departments", "request_clarification", "generate_triage_reply"):
        getattr(chat_route, name).assert_not_called()


def test_missing_and_expired_resume_never_create_or_extend_case(monkeypatch):
    now = [1.0]
    monkeypatch.setattr(case_store, "_clock", lambda: now[0])
    client = TestClient(app)
    assert client.get("/chat/missing/resume").status_code == 404
    assert not case_store._CASES
    case_store.create_case("expired")
    now[0] += case_store.CASE_TTL_SECONDS
    assert client.get("/chat/expired/resume").status_code == 404
    assert case_store._LAST_TOUCHED["expired"] == 1.0


@pytest.mark.parametrize("stage,confirmed", [
    (ConversationStage.WAITING_CONFIRMATION, False),
    (ConversationStage.RECOMMENDING, True),
    (ConversationStage.SCRIPT_READY, True),
])
def test_resume_preserves_confirmation_and_minimal_department(stage, confirmed):
    case = case_store.create_case("confirmed")
    case.visit_type = VisitType.FOLLOWUP
    case.patient_input.red_flags_checked = True
    case.conversation_state.stage = stage
    case.conversation_state.confirmed = confirmed
    case.conversation_state.department_status = "resolved"
    case.department_result = DepartmentResult(dept_id=1333, parentDept="五官科", childDept="耳科", reason=["private reason"])
    body = TestClient(app).get("/chat/confirmed/resume").json()
    assert body["can_continue"] is True
    assert body["stage"] == stage.value
    assert body["confirmed"] is confirmed
    assert body["department_result"] == {"dept_id": 1333, "parentDept": "五官科", "childDept": "耳科"}
    assert body["question_batch"] == []


def test_free_text_pending_intent_does_not_become_guessed_batch_key():
    case = case_store.create_case("free_text")
    case.visit_type = VisitType.INITIAL
    case.conversation_state.free_text_mode = True
    case.conversation_state.pending_clarification_intent = "hearing_context"
    case.conversation_state.last_question_key = "hearing_context"
    case.triage.next_question = "實際上一題"
    body = TestClient(app).get("/chat/free_text/resume").json()
    assert body["can_continue"] is True
    assert body["next_question"] == "實際上一題"
    assert body["question_batch"] == []


def test_unknown_batch_key_fails_closed_and_urgent_never_resumes_registration():
    case = case_store.create_case("unsafe")
    case.visit_type = VisitType.INITIAL
    case.conversation_state.last_question_key = "not_a_batch_key"
    case.triage.next_question = "未知題目"
    client = TestClient(app)
    assert client.get("/chat/unsafe/resume").json()["can_continue"] is False
    case.conversation_state.clarification_status = "urgent"
    case.triage.warning_required = True
    assert client.get("/chat/unsafe/resume").json()["can_continue"] is False


def test_safety_unresolved_resume_carries_stop_marker():
    case = case_store.create_case("safety_unresolved")
    case.visit_type = VisitType.INITIAL
    case.conversation_state.clarification_status = "unresolved"
    case.conversation_state.reply_source = "SAFETY_CONTROLLED_FALLBACK"
    client = TestClient(app)
    body = client.get("/chat/safety_unresolved/resume").json()
    assert (body["can_continue"], body["reply_source"]) == (False, "SAFETY_UNRESOLVED_STOP")
    # A department-unresolved case already passed the safety screen and is not a safety stop.
    case.patient_input.red_flags_checked = True
    assert client.get("/chat/safety_unresolved/resume").json()["reply_source"] == "SAFETY_CONTROLLED_FALLBACK"


def test_peek_snapshot_cannot_mutate_stored_case():
    case = case_store.create_case("copy")
    snapshot = case_store.peek_case("copy")
    snapshot.conversation_state.confirmed = True
    assert case.conversation_state.confirmed is False


def test_safety_resume_preserves_free_text_primary_answer_path():
    case = case_store.create_case("safety")
    case.visit_type = VisitType.INITIAL
    case.conversation_state.clarification_status = "safety_check"
    case.conversation_state.last_question_key = "red_flags"
    body = TestClient(app).get("/chat/safety/resume").json()
    assert body["can_continue"] is True
    assert body["last_question_key"] == "red_flags"
    assert body["next_question"]
    assert body["question_batch"] == []


def test_resumed_continuation_cannot_recreate_case_after_backend_restart(monkeypatch):
    case = case_store.create_case("lost_after_resume")
    case.visit_type = VisitType.INITIAL
    case.conversation_state.last_question_key = "symptom"
    client = TestClient(app)
    assert client.get("/chat/lost_after_resume/resume").status_code == 200
    case_store._CASES.clear()
    case_store._LAST_TOUCHED.clear()
    forbidden = Mock(side_effect=AssertionError("must not create case"))
    monkeypatch.setattr(chat_route, "create_case", forbidden)
    response = client.post("/chat", json={"case_id": "lost_after_resume", "require_existing_case": True, "message": "answer"})
    assert response.status_code == 404
    forbidden.assert_not_called()
    assert not case_store._CASES


def test_batch_question_reconstruction_is_deterministic_without_attempt_increment():
    case = case_store.create_case("reconstruct")
    case.visit_type = VisitType.INITIAL
    case.conversation_state.last_question_key = "severity"
    case.conversation_state.question_attempts = {"severity": 1}
    before = case.model_dump(mode="json")
    client = TestClient(app)
    first = client.get("/chat/reconstruct/resume").json()
    second = client.get("/chat/reconstruct/resume").json()
    assert first == second
    assert first["question_batch"][0]["key"] == "severity"
    assert case.model_dump(mode="json") == before
