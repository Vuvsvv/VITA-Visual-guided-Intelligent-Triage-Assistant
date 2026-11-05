from __future__ import annotations

import asyncio
import importlib
from datetime import date, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import (
    ConversationStage,
    DepartmentResult,
    TimePreference,
    TriageCase,
    VisitType,
)
from app.services import appointment_service, specialty_scoring
from app.services.appointment_service import _build_recommendations, recommend_appointments, weighted_score
from app.services.case_store import save_case, save_recommendation_result
from app.services.specialty_scoring import SpecialtyScore
from app.services.time_preference_service import pareto_time_tiers


class NoAiSettings:
    cerebras_api_key = ""
    ai_doctor_scoring_enabled = False


@pytest.fixture(autouse=True)
def isolated_services(monkeypatch):
    monkeypatch.setattr(specialty_scoring, "get_settings", lambda: NoAiSettings())
    monkeypatch.setattr(appointment_service, "fetch_active_departments", lambda: [
        {"dept_id": 1298, "parent_dept": "外科系", "child_dept": "一般骨科"},
    ])


def _future(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def _case() -> TriageCase:
    case = TriageCase(case_id="phase6-ranking", visit_type=VisitType.INITIAL)
    case.department_result = DepartmentResult(
        dept_id=1298, parentDept="外科系", childDept="一般骨科", confidence=0.9
    )
    case.patient_input.symptom = "膝蓋痛"
    case.patient_input.red_flags_checked = True
    case.patient_input.red_flags_status = "negative"
    case.triage.need_more_info = False
    case.triage.is_final = True
    case.conversation_state.is_complete = True
    case.conversation_state.stage = ConversationStage.RECOMMENDING
    case.conversation_state.department_status = "resolved"
    case.confirmed = True
    case.conversation_state.confirmed = True
    return case


def _row(
    doctor_id: str | None,
    schedule_id: str | None,
    day: str,
    session: str,
    tags: str = "膝關節",
) -> dict:
    return {
        "dept_id": 1298,
        "parent_dept": "外科系",
        "child_dept": "一般骨科",
        "doctor_id": doctor_id,
        "doctor": f"醫師 {doctor_id or 'name-only'}",
        "schedule_id": schedule_id,
        "date": day,
        "session": session,
        "slot": "3201診",
        "room": "3201診",
        "specialty_tags": tags,
        "status": "open",
        "visit_type": "初診",
    }


def test_missing_doctor_id_or_schedule_id_never_enters_formal_recommendation(monkeypatch):
    rows = [
        _row(None, "s-name", _future(2), "上午"),
        _row("doc-no-schedule", None, _future(2), "上午"),
        _row("doc-ok", "s-ok", _future(2), "上午"),
    ]
    monkeypatch.setattr(appointment_service, "fetch_available_slots", lambda *_a, **_k: rows)
    result = asyncio.run(recommend_appointments(_case()))
    assert [item.doctor_id for item in result.recommendations.specialty_first] == ["doc-ok"]


def test_wrong_department_closed_placeholder_and_past_rows_are_excluded(monkeypatch):
    rows = [
        {**_row("wrong", "s-wrong", _future(2), "上午"), "dept_id": 9999},
        {**_row("closed", "s-closed", _future(2), "上午"), "status": "closed"},
        {**_row("placeholder", "s-placeholder", _future(2), "上午"), "is_placeholder": True},
        _row("past", "s-past", _future(-2), "上午"),
        _row("ok", "s-ok", _future(2), "上午"),
    ]
    monkeypatch.setattr(appointment_service, "fetch_available_slots", lambda *_a, **_k: rows)
    result = asyncio.run(recommend_appointments(_case()))
    assert [item.doctor_id for item in result.recommendations.specialty_first] == ["ok"]


def test_specialty_first_is_lexicographic_not_70_30_weighted():
    case = _case()
    rows = [_row("a", "s-a", _future(4), "下午"), _row("b", "s-b", _future(1), "上午")]
    scores = {
        "a": SpecialtyScore("a", "A", "一般骨科", 0.9, "ai", "ai"),
        "b": SpecialtyScore("b", "B", "一般骨科", 0.8, "ai", "ai"),
    }
    items = _build_recommendations(case, case.case_id, rows, "rec", True, scores)
    assert [item.doctor_id for item in items] == ["a", "b"]
    assert weighted_score(0.9, 0.0, True) == 90.0


def test_specialty_tie_uses_time_tier_then_datetime():
    case = _case()
    case.availability.time_preferences = [_session("afternoon", 1)]
    rows = [_row("am", "s-am", _future(1), "上午"), _row("pm", "s-pm", _future(2), "下午")]
    scores = {
        key: SpecialtyScore(key, key, "一般骨科", 0.8, "ai", "ai") for key in ("am", "pm")
    }
    tiers = pareto_time_tiers(rows, case.availability)
    items = _build_recommendations(case, case.case_id, rows, "rec", True, scores, tiers)
    assert [item.doctor_id for item in items] == ["pm", "am"]


def test_time_first_uses_tier_then_datetime_before_specialty():
    case = _case()
    case.availability.time_preferences = [_session("afternoon", 1)]
    rows = [_row("high", "s-high", _future(1), "上午"), _row("low", "s-low", _future(2), "下午")]
    scores = {
        "high": SpecialtyScore("high", "H", "一般骨科", 0.99, "ai", "ai"),
        "low": SpecialtyScore("low", "L", "一般骨科", 0.2, "ai", "ai"),
    }
    tiers = pareto_time_tiers(rows, case.availability)
    items = _build_recommendations(case, case.case_id, rows, "rec", False, scores, tiers)
    assert [item.doctor_id for item in items] == ["low", "high"]


def test_time_tier_and_datetime_tie_then_uses_specialty_and_doctor_id():
    case = _case()
    rows = [_row("b", "s-b", _future(2), "上午"), _row("a", "s-a", _future(2), "上午")]
    scores = {
        "a": SpecialtyScore("a", "A", "一般骨科", 0.7, "ai", "ai"),
        "b": SpecialtyScore("b", "B", "一般骨科", 0.9, "ai", "ai"),
    }
    items = _build_recommendations(case, case.case_id, rows, "rec", False, scores)
    assert [item.doctor_id for item in items] == ["b", "a"]


def test_equal_everything_has_stable_doctor_id_sort():
    case = _case()
    rows = [_row("b", "s-b", _future(2), "上午"), _row("a", "s-a", _future(2), "上午")]
    scores = {key: SpecialtyScore(key, key, "一般骨科", 0.5, "neutral") for key in ("a", "b")}
    items = _build_recommendations(case, case.case_id, rows, "rec", True, scores)
    assert [item.doctor_id for item in items] == ["a", "b"]


def test_same_doctor_is_scored_once_but_concrete_schedule_contract_is_preserved(monkeypatch):
    rows = [_row("same", f"s-{index}", _future(index + 1), "上午") for index in range(5)]
    scorer = AsyncMock(return_value={
        "same": SpecialtyScore("same", "Same", "一般骨科", 0.8, "ai", "ai")
    })
    monkeypatch.setattr(appointment_service, "fetch_available_slots", lambda *_a, **_k: rows)
    monkeypatch.setattr(appointment_service, "score_doctor_specialties", scorer)
    result = asyncio.run(recommend_appointments(_case()))
    scorer.assert_awaited_once()
    assert len(scorer.await_args.args[2]) == 5
    assert [item.schedule_id for item in result.recommendations.specialty_first] == ["s-0", "s-1", "s-2"]
    assert {item.doctor_id for item in result.recommendations.specialty_first} == {"same"}
    assert len(result.recommendations.time_first) == 5


def test_each_column_is_limited_to_five_concrete_schedule_rows(monkeypatch):
    rows = [_row(f"doc-{index}", f"s-{index}", _future(index + 1), "上午") for index in range(8)]
    monkeypatch.setattr(appointment_service, "fetch_available_slots", lambda *_a, **_k: rows)
    result = asyncio.run(recommend_appointments(_case()))
    assert len(result.recommendations.specialty_first) == 5
    assert len(result.recommendations.time_first) == 5


def test_specialty_column_takes_five_doctors_with_up_to_three_slots_each():
    case = _case()
    case.availability.time_preferences = [_session("afternoon", 1)]
    rows = [
        _row(doctor, f"s-{doctor}-{index}", _future(index + 1), "下午" if index % 2 else "上午")
        for doctor in ("a", "b", "c", "d", "e", "f")
        for index in range(5)
    ] + [_row("g", "s-g-0", _future(1), "下午")]
    levels = {"a": 1.0, "b": 0.75, "c": 0.75, "d": 0.5, "e": 0.5, "f": 0.25, "g": 0.5}
    scores = {key: SpecialtyScore(key, key, "一般骨科", value, "ai", "ai") for key, value in levels.items()}
    tiers = pareto_time_tiers(rows, case.availability)
    items = _build_recommendations(case, case.case_id, rows, "rec", True, scores, tiers)

    doctors = list(dict.fromkeys(item.doctor_id for item in items))
    # Equal levels fall back to each doctor's best slot: g's earliest afternoon beats d and e.
    assert doctors == ["a", "b", "c", "g", "d"]
    per_doctor = {doctor: [item for item in items if item.doctor_id == doctor] for doctor in doctors}
    assert [len(per_doctor[doctor]) for doctor in doctors] == [3, 3, 3, 1, 3]
    # Each doctor's slots stay together, preferred afternoon first, then by date.
    assert [item.session for item in per_doctor["a"]] == ["下午", "下午", "上午"]
    assert [item.schedule_id for item in per_doctor["a"]] == ["s-a-1", "s-a-3", "s-a-0"]
    assert [item.rank for item in items] == list(range(1, 14))
    assert [item.is_best_match for item in items].count(True) == 1


def test_complex_hard_exclusion_is_never_relaxed(monkeypatch):
    case = _case()
    case.availability.can_take_leave = True
    excluded = _future(2)
    case.availability.time_preferences = [TimePreference(
        kind="date", relation="exclude", date_value=excluded, resolved_dates=[excluded],
        reference_date=date.today().isoformat(), source_text="那天不行", confidence=0.99,
    )]
    rows = [_row("excluded", "s-ex", excluded, "上午"), _row("ok", "s-ok", _future(3), "上午")]
    monkeypatch.setattr(appointment_service, "fetch_available_slots", lambda *_a, **_k: rows)
    result = asyncio.run(recommend_appointments(case))
    assert [item.doctor_id for item in result.recommendations.specialty_first] == ["ok"]


def test_sql_absence_naturally_means_no_evening_or_weekend_recommendation(monkeypatch):
    rows = [_row("weekday", "s-weekday", _future(2), "下午")]
    monkeypatch.setattr(appointment_service, "fetch_available_slots", lambda *_a, **_k: rows)
    result = asyncio.run(recommend_appointments(_case()))
    assert all(item.session != "晚上" for item in result.recommendations.specialty_first)
    assert {item.schedule_id for item in result.recommendations.specialty_first} == {"s-weekday"}


def test_legacy_preferred_days_and_sessions_remain_compatible(monkeypatch):
    target = date.today() + timedelta(days=1)
    weekday = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"][target.weekday()]
    case = _case()
    case.availability.preferred_days = [weekday]
    case.availability.preferred_sessions = ["下午"]
    rows = [_row("am", "s-am", target.isoformat(), "上午"), _row("pm", "s-pm", target.isoformat(), "下午")]
    monkeypatch.setattr(appointment_service, "fetch_available_slots", lambda *_a, **_k: rows)
    result = asyncio.run(recommend_appointments(case))
    assert [item.doctor_id for item in result.recommendations.specialty_first] == ["pm"]


def test_main_recommendation_keeps_navigation_identity(monkeypatch):
    monkeypatch.setattr(appointment_service, "fetch_available_slots", lambda *_a, **_k: [
        _row("doc", "schedule", _future(2), "下午")
    ])
    item = asyncio.run(recommend_appointments(_case())).recommendations.specialty_first[0]
    assert (item.doctor_id, item.schedule_id, item.dept_id, item.date, item.session) == (
        "doc", "schedule", 1298, _future(2), "下午"
    )


def test_recommend_route_preserves_both_columns(monkeypatch):
    monkeypatch.setattr(appointment_service, "fetch_available_slots", lambda *_a, **_k: [
        _row("doc", "schedule", _future(2), "下午")
    ])
    case = _case()
    save_case(case)
    response = TestClient(app).post("/recommend", json={"case_id": case.case_id})
    assert response.status_code == 200
    assert set(response.json()["recommendations"]) == {"specialty_first", "time_first"}


def test_generated_recommendation_remains_revalidatable(monkeypatch):
    monkeypatch.setattr(appointment_service, "fetch_available_slots", lambda *_a, **_k: [
        _row("doc", "schedule", _future(2), "下午")
    ])
    case = _case()
    save_case(case)
    result = asyncio.run(recommend_appointments(case))
    save_recommendation_result(result)
    selected = result.recommendations.specialty_first[0]
    route = importlib.import_module("app.routes.generate_script")
    revalidate = MagicMock(return_value=selected)
    monkeypatch.setattr(route, "revalidate_schedule", revalidate)
    response = TestClient(app).post(
        "/generate_script",
        json={"case_id": case.case_id, "recommendation_id": selected.recommendation_id},
    )
    assert response.status_code == 200
    assert revalidate.call_count == 1


def _session(value: str, priority: int) -> TimePreference:
    return TimePreference(
        kind="session", relation="prefer", priority=priority, session=value,
        source_text=f"{value} preferred", confidence=0.99,
    )
