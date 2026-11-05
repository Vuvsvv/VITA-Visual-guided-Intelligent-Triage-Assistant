from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.schemas import Availability, Message, TimePreference, TriageCase
from app.services import rag_triage_adapter
from app.services.time_preference_service import (
    filter_hard_exclusions,
    pareto_time_tiers,
    select_effective_feasible_rows,
    validate_and_resolve_time_preferences,
)


COMPLEX_TEXT = "這週都不行，最好下週二，不然下週三，再沒有其他平日都行。上午起不來，最好下午。"


def _complex_proposals() -> list[dict]:
    return [
        {"kind": "relative_week", "relation": "exclude", "priority": None, "week_offset": 0, "source_text": "這週都不行", "confidence": 0.99},
        {"kind": "relative_weekday", "relation": "prefer", "priority": 1, "week_offset": 1, "weekday": "tuesday", "source_text": "最好下週二", "confidence": 0.98},
        {"kind": "relative_weekday", "relation": "prefer", "priority": 2, "week_offset": 1, "weekday": "wednesday", "source_text": "不然下週三", "confidence": 0.97},
        {"kind": "weekday_group", "relation": "acceptable", "priority": 3, "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday"], "source_text": "其他平日都行", "confidence": 0.96},
        {"kind": "session", "relation": "exclude", "priority": None, "session": "morning", "source_text": "上午起不來", "confidence": 0.99},
        {"kind": "session", "relation": "prefer", "priority": 1, "session": "afternoon", "source_text": "最好下午", "confidence": 0.98},
    ]


def test_complex_preferences_are_grounded_and_relative_dates_are_frozen():
    accepted = validate_and_resolve_time_preferences(
        _complex_proposals(), user_sources=[COMPLEX_TEXT], reference_date=date(2026, 10, 7)
    )
    assert len(accepted) == 6
    assert accepted[0].resolved_dates == [f"2026-10-{day:02d}" for day in range(5, 12)]
    assert accepted[1].resolved_dates == ["2026-10-13"]
    assert accepted[2].resolved_dates == ["2026-10-14"]
    assert all(item.reference_date == "2026-10-07" for item in accepted)


def test_relative_weekday_crosses_month_deterministically():
    proposal = [_complex_proposals()[1]]
    first = validate_and_resolve_time_preferences(
        proposal, user_sources=[COMPLEX_TEXT], reference_date=date(2026, 10, 30)
    )
    second = validate_and_resolve_time_preferences(
        proposal, user_sources=[COMPLEX_TEXT], reference_date=date(2026, 10, 30)
    )
    assert first[0].resolved_dates == ["2026-11-03"]
    assert first == second


def test_relative_weekday_crosses_year():
    accepted = validate_and_resolve_time_preferences(
        [_complex_proposals()[1]], user_sources=[COMPLEX_TEXT], reference_date=date(2026, 12, 30)
    )
    assert accepted[0].resolved_dates == ["2027-01-05"]


def test_ungrounded_and_low_confidence_preferences_are_rejected():
    ungrounded = {**_complex_proposals()[1], "source_text": "下個月星期二"}
    low = {**_complex_proposals()[1], "confidence": 0.2}
    assert validate_and_resolve_time_preferences(
        [ungrounded, low], user_sources=[COMPLEX_TEXT], reference_date=date(2026, 10, 7)
    ) == []


def test_hard_exclusions_remove_current_week_and_morning_but_not_evening():
    availability = Availability(time_preferences=validate_and_resolve_time_preferences(
        _complex_proposals(), user_sources=[COMPLEX_TEXT], reference_date=date(2026, 10, 7)
    ))
    rows = [
        _row("current", "2026-10-08", "下午"),
        _row("morning", "2026-10-13", "上午"),
        _row("afternoon", "2026-10-13", "下午"),
        _row("evening", "2026-10-13", "晚上"),
    ]
    assert [row["schedule_id"] for row in filter_hard_exclusions(rows, availability)] == [
        "afternoon", "evening"
    ]


def test_only_afternoon_is_represented_by_explicit_other_session_exclusions():
    text = "只能下午"
    proposals = [
        {"kind": "session", "relation": "prefer", "priority": 1, "session": "afternoon", "source_text": text, "confidence": 0.99},
        {"kind": "session", "relation": "exclude", "priority": None, "session": "morning", "source_text": text, "confidence": 0.99},
        {"kind": "session", "relation": "exclude", "priority": None, "session": "evening", "source_text": text, "confidence": 0.99},
    ]
    availability = Availability(time_preferences=validate_and_resolve_time_preferences(
        proposals, user_sources=[text], reference_date=date(2026, 10, 7)
    ))
    assert [row["session"] for row in filter_hard_exclusions([
        _row("am", "2026-10-13", "上午"),
        _row("pm", "2026-10-13", "下午"),
        _row("eve", "2026-10-13", "晚上"),
    ], availability)] == ["下午"]


def test_best_afternoon_does_not_exclude_unmentioned_sessions():
    preference = TimePreference(
        kind="session", relation="prefer", priority=1, session="afternoon",
        source_text="下午最好", confidence=0.99,
    )
    availability = Availability(time_preferences=[preference])
    rows = [_row("am", "2026-10-13", "上午"), _row("eve", "2026-10-13", "晚上")]
    assert filter_hard_exclusions(rows, availability) == rows


def test_date_session_tradeoff_is_same_non_dominated_tier_without_weights():
    availability = Availability(time_preferences=validate_and_resolve_time_preferences(
        _complex_proposals()[1:3] + [_complex_proposals()[-1]],
        user_sources=[COMPLEX_TEXT], reference_date=date(2026, 10, 7),
    ))
    rows = [
        _row("tue-evening", "2026-10-13", "晚上"),
        _row("wed-afternoon", "2026-10-14", "下午"),
    ]
    tiers = pareto_time_tiers(rows, availability)
    assert tiers == {"tue-evening": 0, "wed-afternoon": 0}


def test_time_preferences_share_the_existing_turn_interpreter_call():
    case = TriageCase(case_id="phase6-time-turn")
    case.history_records = [Message(role="user", content=COMPLEX_TEXT)]
    provider = AsyncMock(return_value=json.dumps({
        "semantic_extractions": [],
        "pending_answer": None,
        "ttas_evidence": [],
        "time_preferences": _complex_proposals(),
        "dialogue_act": {"act": "new_information", "source_text": COMPLEX_TEXT, "confidence": 0.99},
        "prior_answer_reference": None, "department_preference": None,
    }, ensure_ascii=False))
    with patch.object(
        rag_triage_adapter,
        "get_settings",
        return_value=SimpleNamespace(cerebras_api_key="test-key"),
    ), patch.object(rag_triage_adapter, "complete_prompt", new=provider):
        result = asyncio.run(rag_triage_adapter.refine_case_with_ai(case, user_sources=[COMPLEX_TEXT]))

    provider.assert_awaited_once()
    assert result is not None and len(result.time_preferences or []) == 6
    assert len(case.availability.time_preferences) == 6
    assert all(item.reference_date for item in case.availability.time_preferences)
    assert case.availability.time_preferences[1].resolved_dates


def test_turn_interpreter_prompt_contains_complete_time_preference_example():
    case = TriageCase(case_id="phase6-time-prompt")
    prompt = rag_triage_adapter._build_symptom_collection_prompt(case, [COMPLEX_TEXT])
    example = prompt.split("時間偏好正例：", 1)[1].split("此 AI 輸出不可包含", 1)[0]
    assert '"kind":"relative_week"' in example
    # The ranked days are said in the opposite order of their priority, so the AI learns to
    # read the meaning; 310 twice got 「最好下週三不然下週二」 back as 週二 first.
    assert '"priority":1,"week_offset":1,"weekday":"thursday"' in example
    assert '"priority":2,"week_offset":1,"weekday":"monday"' in example
    assert "最好下週二" not in example
    assert '"kind":"weekday_group"' in example
    assert '"session":"morning"' in example
    assert '"session":"afternoon"' in example
    assert "reference_date" not in example
    assert "resolved_dates" not in example


def test_new_date_dimension_keeps_legacy_session_dimension():
    availability = Availability(
        preferred_sessions=["下午"],
        time_preferences=[_preference(
            kind="relative_weekday", relation="prefer", priority=1,
            week_offset=1, weekday="tuesday", resolved_dates=["2026-10-13"],
        )],
    )
    rows = [
        _row("tue-am", "2026-10-13", "上午"),
        _row("tue-pm", "2026-10-13", "下午"),
        _row("wed-pm", "2026-10-14", "下午"),
    ]
    feasible, relaxed = select_effective_feasible_rows(rows, availability, now=_now())
    assert relaxed is False
    assert [row["schedule_id"] for row in feasible] == ["tue-pm", "wed-pm"]
    tiers = pareto_time_tiers(feasible, availability)
    assert tiers["tue-pm"] < tiers["wed-pm"]


def test_new_session_dimension_keeps_legacy_date_dimension():
    availability = Availability(
        preferred_days=["週三"],
        time_preferences=[_preference(
            kind="session", relation="prefer", priority=1, session="afternoon",
        )],
    )
    rows = [
        _row("tue-pm", "2026-10-13", "下午"),
        _row("wed-am", "2026-10-14", "上午"),
        _row("wed-pm", "2026-10-14", "下午"),
    ]
    feasible, _ = select_effective_feasible_rows(rows, availability, now=_now())
    assert [row["schedule_id"] for row in feasible] == ["wed-am", "wed-pm"]
    tiers = pareto_time_tiers(feasible, availability)
    assert tiers["wed-pm"] < tiers["wed-am"]


def test_new_session_dimension_supersedes_legacy_session_dimension():
    availability = Availability(
        preferred_sessions=["下午"],
        time_preferences=[_preference(
            kind="session", relation="prefer", priority=1, session="morning",
        )],
    )
    rows = [
        _row("am", "2026-10-13", "上午"),
        _row("pm", "2026-10-13", "下午"),
    ]
    feasible, _ = select_effective_feasible_rows(rows, availability, now=_now())
    assert [row["schedule_id"] for row in feasible] == ["am", "pm"]
    tiers = pareto_time_tiers(feasible, availability)
    assert tiers["am"] < tiers["pm"]


def test_new_hard_exclusion_is_not_relaxed_by_can_take_leave():
    availability = Availability(
        preferred_sessions=["上午"],
        can_take_leave=True,
        time_preferences=[_preference(
            kind="session", relation="exclude", priority=None, session="morning",
        )],
    )
    rows = [
        _row("am", "2026-10-13", "上午"),
        _row("pm", "2026-10-13", "下午"),
    ]
    feasible, _ = select_effective_feasible_rows(rows, availability, now=_now())
    assert [row["schedule_id"] for row in feasible] == ["pm"]


def test_empty_new_preferences_preserve_legacy_feasibility_exactly():
    availability = Availability(preferred_sessions=["下午"], can_take_leave=False)
    rows = [
        _row("am", "2026-10-13", "上午"),
        _row("pm", "2026-10-13", "下午"),
    ]
    feasible, relaxed = select_effective_feasible_rows(rows, availability, now=_now())
    assert relaxed is False
    assert [row["schedule_id"] for row in feasible] == ["pm"]


def test_new_preferences_only_keep_existing_pareto_behavior():
    availability = Availability(time_preferences=[
        _preference(
            kind="relative_weekday", relation="prefer", priority=1,
            week_offset=1, weekday="tuesday", resolved_dates=["2026-10-13"],
        ),
        _preference(kind="session", relation="prefer", priority=1, session="afternoon"),
    ])
    rows = [
        _row("tue-am", "2026-10-13", "上午"),
        _row("wed-pm", "2026-10-14", "下午"),
    ]
    feasible, _ = select_effective_feasible_rows(rows, availability, now=_now())
    assert feasible == rows
    assert pareto_time_tiers(feasible, availability) == {"tue-am": 0, "wed-pm": 0}


def _preference(**kwargs) -> TimePreference:
    return TimePreference(
        source_text="測試結構化時間偏好",
        confidence=0.99,
        reference_date="2026-10-07",
        **kwargs,
    )


def _now() -> datetime:
    return datetime(2026, 10, 7, 9, 0)


def _row(schedule_id: str, value: str, session: str) -> dict:
    return {"schedule_id": schedule_id, "date": value, "session": session}


TOMORROW_TEXT = "明天下午可以，後天不行"


def test_relative_day_is_counted_from_the_backend_date():
    proposals = [
        {"kind": "relative_day", "relation": "prefer", "priority": 1, "day_offset": 1, "source_text": "明天", "confidence": 0.99},
        {"kind": "relative_day", "relation": "exclude", "priority": None, "day_offset": 2, "source_text": "後天不行", "confidence": 0.99},
    ]
    accepted = validate_and_resolve_time_preferences(
        proposals, user_sources=[TOMORROW_TEXT], reference_date=date(2026, 11, 4)
    )
    assert [item.resolved_dates for item in accepted] == [["2026-11-05"], ["2026-11-06"]]
    across_year = validate_and_resolve_time_preferences(
        proposals[:1], user_sources=[TOMORROW_TEXT], reference_date=date(2026, 12, 31)
    )
    assert across_year[0].resolved_dates == ["2027-01-01"]


def test_relative_date_without_a_value_is_still_rejected():
    # 310: the AI sent 「明天」 as kind=date with no date_value, and the date was lost.
    proposal = [{"kind": "date", "relation": "prefer", "priority": 1, "source_text": "明天", "confidence": 0.99}]
    assert validate_and_resolve_time_preferences(
        proposal, user_sources=[TOMORROW_TEXT], reference_date=date(2026, 11, 4)
    ) == []


def test_relative_day_answers_the_date_dimension():
    from app.services.conversation_service import availability_dimension_has_data

    case = TriageCase(case_id="relative-day-dimension")
    case.availability.time_preferences = validate_and_resolve_time_preferences(
        [{"kind": "relative_day", "relation": "prefer", "priority": 1, "day_offset": 1, "source_text": "明天", "confidence": 0.99}],
        user_sources=[TOMORROW_TEXT], reference_date=date(2026, 11, 4),
    )
    assert availability_dimension_has_data(case, "date")


def test_turn_interpreter_prompt_teaches_relative_day():
    case = TriageCase(case_id="relative-day-prompt")
    prompt = rag_triage_adapter._build_symptom_collection_prompt(case, [TOMORROW_TEXT])
    assert "kind=relative_day" in prompt
    assert '"kind":"relative_day","relation":"prefer","priority":1,"day_offset":1' in prompt
    assert '"kind":"relative_day","relation":"exclude","priority":null,"day_offset":2' in prompt
