"""Phase 6 Backend API contract (AUDIT_PHASE6 T1): synthetic, anonymous data only."""
from __future__ import annotations

import importlib
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ConversationStage, DepartmentResult, TriageCase, VisitType
from app.services import appointment_service, specialty_scoring
from app.services.case_store import save_case

ITEM_FIELDS = {
    "recommendation_id", "parentDept", "childDept", "doctor", "date", "session", "slot", "doctor_id",
    "schedule_id", "session_time", "room", "visit_type", "specialty_tags", "specialty_score", "time_score",
    "match_reason", "score", "reasons", "department_reason", "time_reason", "rank", "is_best_match", "dept_id",
}
RESULT_FIELDS = {"case_id", "recommendations", "fallback_departments", "total_count"}


class NoAiSettings:
    cerebras_api_key = ""
    ai_doctor_scoring_enabled = False


@pytest.fixture(autouse=True)
def isolated_services(monkeypatch):
    monkeypatch.setattr(specialty_scoring, "get_settings", lambda: NoAiSettings())
    monkeypatch.setattr(appointment_service, "fetch_active_departments", lambda: [
        {"dept_id": 1298, "parent_dept": "外科系", "child_dept": "一般骨科"},
    ])


def future(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def contract_case(case_id: str = "contract-case") -> TriageCase:
    case = TriageCase(case_id=case_id, visit_type=VisitType.INITIAL)
    case.department_result = DepartmentResult(dept_id=1298, parentDept="外科系", childDept="一般骨科", confidence=0.9)
    case.patient_input.symptom = "膝蓋痛"
    case.patient_input.red_flags_checked = True
    case.patient_input.red_flags_status = "negative"
    case.triage.need_more_info = False
    case.triage.is_final = True
    case.conversation_state.is_complete = True
    case.conversation_state.stage = ConversationStage.RECOMMENDING
    case.conversation_state.department_status = "resolved"
    case.conversation_state.clarification_status = "sufficient"
    case.confirmed = True
    case.conversation_state.confirmed = True
    return case


def schedule_row(index: int, *, session: str = "上午", tags: str | None = "膝關節、運動傷害") -> dict:
    return {
        "dept_id": 1298, "parent_dept": "外科系", "child_dept": "一般骨科",
        "doctor_id": f"D{index:03d}", "doctor": f"測試醫師{index}", "schedule_id": f"S{index:04d}",
        "date": future(index + 1), "session": session, "slot": "3201診", "room": "3201診",
        "specialty_tags": tags, "status": "open", "visit_type": "初診",
    }


def recommend(monkeypatch, rows: list[dict], case: TriageCase | None = None):
    monkeypatch.setattr(appointment_service, "fetch_available_slots", lambda *_a, **_k: rows)
    case = case or contract_case()
    save_case(case)
    return TestClient(app).post("/recommend", json={"case_id": case.case_id})


def items_of(body: dict) -> list[dict]:
    return body["recommendations"]["specialty_first"] + body["recommendations"]["time_first"]


def test_recommend_success_contract(monkeypatch):
    response = recommend(monkeypatch, [schedule_row(index, session="下午" if index % 2 else "上午")
                                       for index in range(8)])
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESULT_FIELDS
    assert set(body["recommendations"]) == {"specialty_first", "time_first"}
    columns = body["recommendations"]
    assert 1 <= len(columns["specialty_first"]) <= 15
    assert 1 <= len(columns["time_first"]) <= 5
    # total_count is the number of items actually returned in both columns together; a
    # schedule present in both columns is counted twice. It is not the number of SQL rows.
    assert body["total_count"] == len(columns["specialty_first"]) + len(columns["time_first"]) == 10
    for item in items_of(body):
        assert set(item) == ITEM_FIELDS
        assert isinstance(item["recommendation_id"], str) and item["recommendation_id"]
        assert isinstance(item["doctor_id"], str) and item["doctor_id"]
        assert isinstance(item["schedule_id"], str) and item["schedule_id"]
        assert type(item["dept_id"]) is int
        assert date.fromisoformat(item["date"]).isoformat() == item["date"]
        assert isinstance(item["session"], str) and item["session"]
        assert item["specialty_tags"] is None or isinstance(item["specialty_tags"], str)
        assert isinstance(item["score"], (int, float)) and 0 <= item["score"] <= 100
        assert 0 <= item["specialty_score"] <= 1 and 0 <= item["time_score"] <= 1
        assert isinstance(item["reasons"], list) and all(isinstance(reason, str) for reason in item["reasons"])
        assert isinstance(item["department_reason"], str) and isinstance(item["time_reason"], str)
        assert isinstance(item["is_best_match"], bool)
    ids = [item["recommendation_id"] for item in items_of(body)]
    assert len(ids) == len(set(ids))


def test_recommend_without_schedule_returns_503(monkeypatch):
    response = recommend(monkeypatch, [])
    assert response.status_code == 503
    assert set(response.json()) == {"detail"} and isinstance(response.json()["detail"], str)


@pytest.mark.parametrize("mutate,detail_fragment", [
    (lambda case: setattr(case.conversation_state, "clarification_status", "urgent"), "急迫性篩檢"),
    (lambda case: setattr(case.conversation_state, "is_complete", False), "尚未完成"),
    (lambda case: (setattr(case.patient_input, "red_flags_checked", False),
                   setattr(case.patient_input, "red_flags_status", "not_checked")), "急迫症狀篩檢尚未完成"),
    (lambda case: (setattr(case, "confirmed", False), setattr(case.conversation_state, "confirmed", False)),
     "尚未確認"),
])
def test_recommend_blocked_states_return_400(monkeypatch, mutate, detail_fragment):
    case = contract_case()
    mutate(case)
    response = recommend(monkeypatch, [schedule_row(1)], case)
    assert response.status_code == 400
    assert detail_fragment in response.json()["detail"]


def test_ai_specialty_scoring_failure_makes_whole_set_neutral(monkeypatch):
    settings = SimpleNamespace(ai_doctor_scoring_enabled=True, cerebras_api_key="test-key")
    monkeypatch.setattr(specialty_scoring, "get_settings", lambda: settings)
    monkeypatch.setattr(specialty_scoring, "_ai_available", lambda: True)
    provider = AsyncMock(side_effect=TimeoutError("provider timeout"))
    monkeypatch.setattr(specialty_scoring, "complete_prompt", provider)
    response = recommend(monkeypatch, [schedule_row(index) for index in range(3)])
    assert response.status_code == 200
    provider.assert_awaited()
    assert {item["specialty_score"] for item in items_of(response.json())} == {0.5}


def test_generate_script_success_contract(monkeypatch):
    response = recommend(monkeypatch, [schedule_row(1)])
    selected = response.json()["recommendations"]["specialty_first"][0]
    route = importlib.import_module("app.routes.generate_script")
    from app.schemas import RecommendationItem
    monkeypatch.setattr(route, "revalidate_schedule",
                        MagicMock(side_effect=lambda item, **_kwargs: RecommendationItem(**selected)))
    script = TestClient(app).post("/generate_script", json={
        "case_id": "contract-case", "recommendation_id": selected["recommendation_id"],
    })
    assert script.status_code == 200
    body = script.json()
    assert set(body) == {"isSuccess", "script_id", "recommendation_id", "recommendation", "steps", "message",
                         "step_count"}
    assert body["isSuccess"] is True and body["recommendation_id"] == selected["recommendation_id"]
    assert body["step_count"] == len(body["steps"]) > 0
    assert set(body["recommendation"]) == ITEM_FIELDS
