from __future__ import annotations

from app.schemas import (
    CandidateDepartment,
    CandidateDepartmentEvidence,
    ClueConceptRecord,
    DepartmentResult,
    TimePreference,
    TriageCase,
)
from app.services.patient_reason_text import department_reason, time_reason
from app.services.specialty_scoring import SpecialtyScore

EYE = 1300
FAMILY = 1233


def _case(dept_id: int = EYE, child: str = "眼科") -> TriageCase:
    case = TriageCase(case_id="reason-text")
    case.department_result = DepartmentResult(dept_id=dept_id, parentDept="內科系", childDept=child, confidence=0.9)
    return case


def _slot(day: str = "2026-11-10", session: str = "下午", child: str = "眼科", tags: str = "乾眼症、結膜炎") -> dict:
    return {"date": day, "session": session, "child_dept": child, "doctor": "王小明", "specialty_tags": tags}


def _neutral() -> SpecialtyScore:
    return SpecialtyScore("d", "王小明", "眼科", 0.5, "neutral", "neutral")


def _weekday(weekday: str, priority: int, text: str, relation: str = "prefer") -> TimePreference:
    return TimePreference(kind="weekday", relation=relation, priority=priority, weekday=weekday,
                          source_text=text, confidence=0.99)


def _ranked_week_case() -> TriageCase:
    # 「最好週二，不然週三，除了週一都可以」
    case = _case()
    case.availability.time_preferences = [
        _weekday("tuesday", 1, "最好週二"),
        _weekday("wednesday", 2, "不然週三"),
        TimePreference(kind="weekday", relation="exclude", weekday="monday", source_text="除了週一都可以",
                       confidence=0.99),
    ]
    return case


def test_ranked_weekdays_name_each_choice():
    case = _ranked_week_case()
    assert time_reason(case, _slot("2026-11-10", "下午")) == "這個門診在11月10日（週二）下午，符合您說的「最好週二」，是您的第一選擇。"
    assert time_reason(case, _slot("2026-11-11", "上午")) == "這個門診在11月11日（週三）上午，符合您說的「不然週三」，是您的第二選擇。"
    assert time_reason(case, _slot("2026-11-12", "上午")) == (
        "這個門診在11月12日（週四）上午，不是您優先的「最好週二」「不然週三」，"
        "但依您說的「除了週一都可以」，這個時間可以看診。"
    )


def test_acceptable_group_is_matched_without_rank():
    case = _ranked_week_case()
    case.availability.time_preferences.append(TimePreference(
        kind="weekday_group", relation="acceptable", priority=3,
        weekdays=["tuesday", "wednesday", "thursday", "friday"], source_text="其他平日都行", confidence=0.99,
    ))
    assert time_reason(case, _slot("2026-11-12", "上午")) == "這個門診在11月12日（週四）上午，符合您說的「其他平日都行」。"
    assert "第一選擇" in time_reason(case, _slot("2026-11-10", "上午"))


def test_date_and_session_dimensions_are_reported_together():
    case = _case()
    case.availability.time_preferences = [
        _weekday("tuesday", 1, "星期二"),
        TimePreference(kind="session", relation="prefer", priority=1, session="afternoon", source_text="下午",
                       confidence=0.99),
    ]
    assert time_reason(case, _slot("2026-11-10", "下午")) == "這個門診在11月10日（週二）下午，符合您說的「星期二」和「下午」。"
    assert time_reason(case, _slot("2026-11-10", "上午")) == "這個門診在11月10日（週二）上午，符合您說的「星期二」，但不是您說的「下午」。"


def test_no_time_preference_and_relaxed_fallback():
    case = _case()
    assert time_reason(case, _slot()) == "您沒有指定看診時間，這個門診在11月10日（週二）下午。"
    case.availability.preferred_days = ["週一"]
    assert time_reason(case, _slot(), relaxed_by_date=True) == (
        "這個門診在11月10日（週二）下午，不完全符合您方便的週一。您方便的時段近期沒有號，改推薦時間最接近的門診。"
    )


def test_department_reason_quotes_patient_words_and_confirmed_clues():
    case = _case()
    case.conversation_state.candidate_departments = [CandidateDepartment(
        dept_id=EYE, parentDept="內科系", childDept="眼科", confidence=0.9,
        supporting_evidence=[CandidateDepartmentEvidence(
            patient_source_text="眼睛紅", knowledge_source_id="k1", knowledge_concept="紅眼睛")],
    )]
    case.conversation_state.clue_concepts_asked = [
        ClueConceptRecord(concept="畏光", dept_id=EYE, question_index=1, answer="present"),
        ClueConceptRecord(concept="鼻塞", dept_id=999, question_index=2, answer="present"),
        ClueConceptRecord(concept="流淚", dept_id=EYE, question_index=3, answer="absent"),
    ]
    assert department_reason(case, _slot(), _neutral()) == (
        "您提到「眼睛紅」，並表示有「畏光」，官方衛教資料將這些症狀列在眼科，因此建議眼科。"
    )


def test_department_reason_single_evidence():
    case = _case()
    case.conversation_state.candidate_departments = [CandidateDepartment(
        dept_id=EYE, parentDept="內科系", childDept="眼科", confidence=0.9,
        supporting_evidence=[CandidateDepartmentEvidence(
            patient_source_text="眼睛癢", knowledge_source_id="k1", knowledge_concept="眼睛癢")],
    )]
    assert department_reason(case, _slot(), _neutral()) == "您提到「眼睛癢」，官方衛教資料將此症狀列在眼科，因此建議眼科。"


def test_department_reason_requested_and_family_medicine():
    case = _case()
    case.patient_input.requested_department_id = EYE
    assert department_reason(case, _slot(), _neutral()) == "依您指定的科別，安排眼科門診。"

    family = _case(FAMILY, "家庭醫學科")
    assert department_reason(family, _slot(child="家庭醫學科"), _neutral()) == (
        "目前的症狀還無法明確分到單一專科，建議先看家庭醫學科，由醫師初步評估。"
    )


def test_doctor_sentence_only_for_meaningful_ai_level():
    case = _case()
    case.patient_input.symptom = "眼睛癢"
    high = SpecialtyScore("d", "王小明", "眼科", 0.75, "ai", "ai")
    medium = SpecialtyScore("d", "王小明", "眼科", 0.5, "ai", "ai")
    low = SpecialtyScore("d", "王小明", "眼科", 0.25, "ai", "ai")
    assert department_reason(case, _slot(), high) == (
        "依您描述的「眼睛癢」，建議眼科。王小明醫師的專長包含「乾眼症、結膜炎」，和您的症狀高度相關。"
    )
    assert department_reason(case, _slot(), medium).endswith("和您的症狀相關。")
    assert department_reason(case, _slot(), low) == "依您描述的「眼睛癢」，建議眼科。"
    assert department_reason(case, _slot(tags=""), high) == "依您描述的「眼睛癢」，建議眼科。"
