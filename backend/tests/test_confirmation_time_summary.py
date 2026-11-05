"""Confirmation reply echoes validated time_preferences as backend-resolved dates (問題 14)."""
from datetime import date

import logging

from app.schemas import DepartmentResult, Message, SemanticExtraction, TriageCase
from app.services import rag_triage_adapter
from app.services.ai_reply_generator import build_availability_preference_summary, build_department_confirmation_reply
from app.services.time_preference_service import validate_and_resolve_time_preferences

TUESDAY = date(2026, 11, 3)
WEDNESDAY = date(2026, 11, 4)


def pref(kind, relation, text, priority=None, **extra):
    if relation != "exclude" and priority is None:
        priority = 1
    return dict(kind=kind, relation=relation, priority=priority, source_text=text, confidence=0.95, **extra)


def summary(text, proposals, today=TUESDAY, **outcomes):
    case = TriageCase(case_id="time-summary")
    case.availability.time_preferences = validate_and_resolve_time_preferences(
        proposals, user_sources=[text], reference_date=today,
    )
    assert len(case.availability.time_preferences) == len(proposals)
    case.conversation_state.availability_dimension_outcomes.update(outcomes)
    return build_availability_preference_summary(case)


def test_relative_weekday_shows_the_resolved_date_and_session():
    text = "下週三早上可以，星期一不行"
    assert summary(text, [
        pref("relative_weekday", "prefer", "下週三", week_offset=1, weekday="wednesday"),
        pref("session", "prefer", "早上", session="morning"),
        pref("weekday", "exclude", "星期一不行", weekday="monday"),
    ]) == "就醫時間偏好：11 月 11 日（週三）上午；週一不方便。"


def test_ranked_weekdays_exclusion_and_unrestricted_session():
    text = "最好週三，不然就週二，反正週五不行，其他隨便"
    assert summary(text, [
        pref("weekday", "prefer", "最好週三", 1, weekday="wednesday"),
        pref("weekday", "prefer", "不然就週二", 2, weekday="tuesday"),
        pref("weekday", "exclude", "週五不行", weekday="friday"),
    ], session="unrestricted") == "就醫時間偏好：週三優先，其次週二；週五不方便；時段不限。"


def test_ranked_weekdays_with_one_session():
    text = "最好週三，不然就週二，反正週五不行，下午"
    assert summary(text, [
        pref("weekday", "prefer", "最好週三", 1, weekday="wednesday"),
        pref("weekday", "prefer", "不然就週二", 2, weekday="tuesday"),
        pref("weekday", "exclude", "週五不行", weekday="friday"),
        pref("session", "prefer", "下午", session="afternoon"),
    ], today=WEDNESDAY) == "就醫時間偏好：週三優先，其次週二；週五不方便；時段下午。"


def test_this_week_reading_shows_todays_date_so_the_patient_can_correct_it():
    # 週三下午 4 點說「週三」，AI 若誤當成本週：確認句寫出今天的日期，病人看得出來。
    text = "最好週三"
    assert summary(text, [pref("relative_weekday", "prefer", "最好週三", week_offset=0, weekday="wednesday")],
                   today=WEDNESDAY) == "就醫時間偏好：11 月 4 日（週三）。"


def test_preferred_and_acceptable_sessions_with_any_date():
    text = "最好下午，上午也可以"
    assert summary(text, [
        pref("session", "prefer", "最好下午", 1, session="afternoon"),
        pref("session", "acceptable", "上午也可以", 2, session="morning"),
    ], date="unrestricted") == "就醫時間偏好：日期不限；時段下午優先，上午也可以。"


def test_week_ranges_groups_and_session_exclusions():
    text = "這週都不行，最好下週二，不然下週三，再沒有其他平日都行。上午起不來，最好下午。"
    assert summary(text, [
        pref("relative_week", "exclude", "這週都不行", week_offset=0),
        pref("relative_weekday", "prefer", "最好下週二", 1, week_offset=1, weekday="tuesday"),
        pref("relative_weekday", "prefer", "不然下週三", 2, week_offset=1, weekday="wednesday"),
        pref("weekday_group", "acceptable", "其他平日都行", 3,
             weekdays=["monday", "tuesday", "wednesday", "thursday", "friday"]),
        pref("session", "exclude", "上午起不來", session="morning"),
        pref("session", "prefer", "最好下午", 1, session="afternoon"),
    ]) == ("就醫時間偏好：11 月 10 日（週二）優先，其次11 月 11 日（週三），平日也可以；"
           "11 月 2 日（週一）到11 月 8 日（週日）、上午不方便；時段下午。")


def test_both_dimensions_unrestricted_without_data_shows_nothing_new():
    case = TriageCase(case_id="time-summary-empty")
    case.conversation_state.availability_dimension_outcomes.update(date="unrestricted", session="unrestricted")
    assert build_availability_preference_summary(case) == ""


def test_confirmation_reply_carries_the_time_summary():
    case = TriageCase(case_id="time-summary-confirm")
    case.patient_input.symptom = "頭暈"
    case.department_result = DepartmentResult(dept_id=1333, parentDept="耳鼻喉部", childDept="耳科", confidence=0.9)
    case.availability.time_preferences = validate_and_resolve_time_preferences(
        [pref("relative_weekday", "prefer", "下週三", week_offset=1, weekday="wednesday")],
        user_sources=["下週三"], reference_date=TUESDAY,
    )
    reply = build_department_confirmation_reply(case)
    assert reply.endswith("請確認後取得推薦掛號方案。\n就醫時間偏好：11 月 11 日（週三）。")


def test_turn_interpreter_prompt_treats_a_bare_weekday_as_every_week():
    prompt = rag_triage_adapter._build_symptom_collection_prompt(TriageCase(case_id="bare-weekday"), ["最好週三"])
    assert "沒有「這／本／下／下下」等字的星期幾" in prompt
    assert "必須用 kind=weekday，不得當成本週的 relative_weekday" in prompt


def body_part(value):
    return SemanticExtraction(field="body_part", normalized_value=value, source_text=value,
                              semantic_status="available", confidence=0.95, extractor="ai")


def test_confirmation_lists_every_stated_body_part_in_order():
    # 310：「我的屁股好痛」…「我的大腿有點酸」，確認句只寫了「不適部位「大腿」」。
    case = TriageCase(case_id="body-parts")
    case.history_records = [Message(role="user", content="我的屁股好痛"), Message(role="user", content="我的大腿有點酸")]
    case.patient_input.symptom = "屁股好痛"
    case.patient_input.body_part = "大腿"
    case.semantic_extractions = [body_part("屁股"), body_part("大腿")]
    case.department_result = DepartmentResult(dept_id=1298, parentDept="外科系", childDept="一般骨科", confidence=0.9)
    reply = build_department_confirmation_reply(case)
    assert "症狀「屁股好痛」、不適部位「屁股、大腿」" in reply


def test_stored_body_part_without_evidence_still_shows():
    case = TriageCase(case_id="body-part-stored")
    case.patient_input.symptom = "頭痛"
    case.patient_input.body_part = "頭"
    case.department_result = DepartmentResult(dept_id=1274, parentDept="神經醫學中心", childDept="神經內科", confidence=0.9)
    assert "不適部位「頭」" in build_department_confirmation_reply(case)


def test_accepted_time_preferences_are_logged_with_their_rank(caplog):
    text = "最好下週三不然下週二"
    case = TriageCase(case_id="time-log")
    case.history_records = [Message(role="user", content=text)]
    raw = [pref("relative_weekday", "prefer", "最好下週三", 1, week_offset=1, weekday="wednesday"),
           pref("relative_weekday", "prefer", "不然下週二", 2, week_offset=1, weekday="tuesday")]
    accepted = validate_and_resolve_time_preferences(raw, user_sources=[text], reference_date=TUESDAY)
    with caplog.at_level(logging.INFO, logger=rag_triage_adapter.logger.name):
        rag_triage_adapter._log_time_preferences(case.case_id, raw, accepted)
    line = next(record.getMessage() for record in caplog.records if "time preferences" in record.getMessage())
    assert "case_id=time-log proposed=2" in line
    assert '"priority": 1, "target": "wednesday", "dates": ["2026-11-11"], "source": "最好下週三"' in line


def test_tomorrow_and_day_after_show_backend_dates():
    text = "明天下午可以，後天不行"
    assert summary(text, [
        pref("relative_day", "prefer", "明天", day_offset=1),
        pref("session", "prefer", "下午可以", session="afternoon"),
        pref("relative_day", "exclude", "後天不行", day_offset=2),
    ], today=WEDNESDAY) == "就醫時間偏好：11 月 5 日（週四）下午；11 月 6 日（週五）不方便。"
