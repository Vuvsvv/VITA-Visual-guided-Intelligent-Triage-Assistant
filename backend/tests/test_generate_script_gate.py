"""/generate_script safety gate and stale recommendation clearing (AUDIT_PHASE6 T2, T3, T13)."""
from __future__ import annotations

import asyncio
import importlib
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ConversationStage, DepartmentResult, TriageCase, VisitType
from app.services import appointment_service, quick_search_service, specialty_scoring
from app.services.appointment_service import recommend_appointments
from app.services.case_store import get_recommendations_for_case, save_case, save_recommendation_result

chat_route = importlib.import_module("app.routes.chat")
DAY = (date.today() + timedelta(days=3)).isoformat()


class NoAiSettings:
    cerebras_api_key = ""
    ai_doctor_scoring_enabled = False


def schedule_row(visit_type: str = "初診") -> dict:
    return {
        "dept_id": 1298, "parent_dept": "外科系", "child_dept": "一般骨科", "doctor_id": "D001",
        "doctor": "測試醫師", "schedule_id": "S0001", "date": DAY, "session": "下午", "slot": "3201診",
        "room": "3201診", "specialty_tags": "膝關節", "status": "open", "is_placeholder": False,
        "visit_type": visit_type, "supports_followup": False,
    }


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(specialty_scoring, "get_settings", lambda: NoAiSettings())
    monkeypatch.setattr(appointment_service, "fetch_active_departments", lambda: [
        {"dept_id": 1298, "parent_dept": "外科系", "child_dept": "一般骨科"},
    ])
    monkeypatch.setattr(appointment_service, "fetch_available_slots", lambda *_a, **_k: [schedule_row()])
    monkeypatch.setattr(quick_search_service, "fetch_quick_search_slot_by_id", lambda _sid: schedule_row())
    monkeypatch.setattr(chat_route, "runtime_ai_available", lambda _settings: False)


def ready_case(case_id: str) -> TriageCase:
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


def recommended(case_id: str) -> tuple[TriageCase, str]:
    case = ready_case(case_id)
    save_case(case)
    result = asyncio.run(recommend_appointments(case))
    save_recommendation_result(result)
    return case, result.recommendations.specialty_first[0].recommendation_id


def script(recommendation_id: str, case_id: str | None):
    body = {"recommendation_id": recommendation_id}
    if case_id is not None:
        body["case_id"] = case_id
    return TestClient(app).post("/generate_script", json=body)


def test_ready_case_generates_script_after_real_revalidation():
    case, rid = recommended("gate-ok")
    response = script(rid, case.case_id)
    assert response.status_code == 200 and response.json()["isSuccess"] is True


BLOCKED = {
    "urgent": lambda case: setattr(case.conversation_state, "clarification_status", "urgent"),
    "ttas_level_2": lambda case: (setattr(case.ttas_result, "status", "matched"),
                                  setattr(case.ttas_result, "level_candidate", 2)),
    "incomplete": lambda case: setattr(case.conversation_state, "is_complete", False),
    "safety_unresolved": lambda case: (setattr(case.patient_input, "red_flags_checked", False),
                                       setattr(case.patient_input, "red_flags_status", "not_checked")),
    "unconfirmed": lambda case: (setattr(case, "confirmed", False),
                                 setattr(case.conversation_state, "confirmed", False)),
}


@pytest.mark.parametrize("state", sorted(BLOCKED))
@pytest.mark.parametrize("with_case_id", [True, False])
def test_blocked_states_match_recommend_status_and_message(state, with_case_id):
    case, rid = recommended(f"gate-{state}-{with_case_id}")
    BLOCKED[state](case)
    save_case(case)
    blocked = script(rid, case.case_id if with_case_id else None)
    recommend = TestClient(app).post("/recommend", json={"case_id": case.case_id})
    assert blocked.status_code == recommend.status_code == 400
    assert blocked.json()["detail"] == recommend.json()["detail"]


def test_revision_clears_old_recommendations():
    case, rid = recommended("gate-revision")
    response = TestClient(app).post("/chat", json={"case_id": case.case_id, "revision_requested": True})
    assert response.status_code == 200
    assert get_recommendations_for_case(case.case_id) == {}
    assert script(rid, case.case_id).status_code == 404


def test_becoming_urgent_blocks_and_clears(monkeypatch):
    case, rid = recommended("gate-urgent-chat")
    monkeypatch.setattr(chat_route, "ttas_requires_immediate_action", lambda _case: True)
    response = TestClient(app).post("/chat", json={"case_id": case.case_id, "message": "突然胸口很痛", "confirmed": True})
    assert response.json()["conversation_state"]["clarification_status"] == "urgent"
    assert get_recommendations_for_case(case.case_id) == {}
    assert script(rid, case.case_id).status_code == 404
    assert script(rid, None).status_code == 404


def test_department_change_after_confirmation_clears():
    case, rid = recommended("gate-department")
    # Without AI the department is recomputed from explicit preferences only, so it changes to none.
    response = TestClient(app).post("/chat", json={"case_id": case.case_id, "message": "其實是肚子痛", "confirmed": True})
    assert response.json()["department_result"] is None
    assert get_recommendations_for_case(case.case_id) == {}
    assert script(rid, case.case_id).status_code == 404


def test_other_case_id_cannot_reach_recommendation():
    _, rid = recommended("gate-owner")
    other, _ = recommended("gate-other")
    assert script(rid, other.case_id).status_code == 404


def test_schedule_visit_type_must_match_case(monkeypatch):
    case, rid = recommended("gate-visit-type")
    monkeypatch.setattr(quick_search_service, "fetch_quick_search_slot_by_id", lambda _sid: schedule_row("複診"))
    response = script(rid, case.case_id)
    assert response.status_code == 409
    assert "掛號別" in response.json()["detail"]


def test_revalidation_without_expected_visit_type_is_unchanged(monkeypatch):
    case, rid = recommended("gate-search-compat")
    item = get_recommendations_for_case(case.case_id)[rid]
    monkeypatch.setattr(quick_search_service, "fetch_quick_search_slot_by_id", lambda _sid: schedule_row("複診"))
    assert quick_search_service.revalidate_schedule(item).schedule_id == "S0001"
