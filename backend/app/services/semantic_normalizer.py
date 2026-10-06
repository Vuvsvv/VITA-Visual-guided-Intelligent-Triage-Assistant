from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.schemas import SemanticExtraction, SeverityNormalization
from app.services.confidence_scoring import (
    accepted,
    clamp_confidence,
    confidence_with_uncertainty,
    needs_clarification,
)

PREFERRED_DAYS_KEY = "preferred_days"
PREFERRED_DATES_KEY = "preferred_dates"
PREFERRED_SESSIONS_KEY = "preferred_sessions"

ALL_WEEKDAYS = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]
WEEKDAY_DAYS = ["週一", "週二", "週三", "週四", "週五"]
WEEKEND_DAYS = ["週六", "週日"]
ALL_SESSIONS = ["上午", "下午", "夜間"]
SESSION_ALIASES = {
    "上午": ("上午", "早上", "早晨", "一早", "中午前"),
    "下午": ("下午", "午後", "中午後"),
    "夜間": ("夜間", "晚上", "晚間", "夜診", "下班後"),
}

ANY_SESSION_TERMS = [
    "都可以", "我都可以", "隨便", "隨便安排", "都行", "皆可", "任何時段都可以",
    "上午下午晚上都可以", "上午下午夜間都可以", "時間沒差", "什麼時段都行",
]
ANY_DAY_TERMS = [
    "哪天都可以", "每天都可以", "日期都可以", "日期沒差", "日期都沒差",
    "任何一天都行", "任何一天都可以",
]
NO_WORK_TERMS = ["沒上班", "不用上班", "沒有上班", "最近沒上班", "近期沒上班"]
UNKNOWN_TERMS = [
    "不知道", "我真的不知道", "我就不知道", "不確定", "再看看", "看看", "不清楚",
    "沒辦法判斷", "無法判斷", "還不確定", "之後再說",
]
UNAVAILABLE_TERMS = ["沒空", "不能", "不行", "不方便", "無法", "沒辦法"]

@dataclass
class NormalizationResult:
    extractions: list[SemanticExtraction] = field(default_factory=list)
    severity: SeverityNormalization | None = None


def normalize_message(text: str, last_question_key: str | None = None) -> NormalizationResult:
    result = NormalizationResult()
    result.extractions.extend(_normalize_availability(text, last_question_key))

    severity = normalize_severity(text)
    if severity.semantic_status != "unknown" or last_question_key == "severity":
        result.severity = severity
        result.extractions.append(
            _extraction(
                field="severity",
                normalized_value={
                    "severity_level": severity.severity_level,
                    "functional_impact": severity.functional_impact,
                    "sleep_impact": severity.sleep_impact,
                },
                semantic_status=severity.semantic_status,
                confidence=severity.confidence,
                source_text=text,
                follow_up_reason=severity.follow_up_reason,
            )
        )

    return result


def normalize_severity(text: str) -> SeverityNormalization:
    explicit_mild_terms = [
        "沒有影響",
        "沒有影響生活",
        "沒有影響日常活動",
        "沒有到影響日常活動",
        "沒有明顯影響日常生活",
        "沒有到影響日常生活",
        "都可以正常生活",
        "可以正常生活",
        "可以正常作息",
        "不影響作息",
        "不影響日常生活",
        "不影響睡眠",
        "還能正常上班",
        "還能正常走路",
        "沒什麼影響",
        "沒有很嚴重",
    ]
    explicit_severe_terms = [
        "痛到睡不著",
        "已經影響睡眠",
        "沒辦法正常生活",
        "沒辦法上班",
        "不能正常走路",
        "嚴重影響生活",
        "無法正常活動",
    ]
    sleep_impact = any(
        term in text
        for term in ["睡不著", "睡不好", "痛醒", "不能睡", "影響睡覺", "影響睡眠"]
    ) and not any(term in text for term in ["不影響睡覺", "不影響睡眠"])
    severe_impact = any(term in text for term in ["快受不了", "受不了", "無法工作", "不能工作", "無法走路", "痛到冒冷汗", *explicit_severe_terms])
    functional_impact = severe_impact or any(term in text for term in ["影響生活", "走路困難", "爬樓梯很吃力", "不能活動"])
    mild_terms = ["有點痛", "一點痛", "微痛", "有點不舒服", "還好", "還能工作", "輕微"]
    moderate_terms = ["普通", "中等", "明顯", "不舒服", "痠痛", "會痛"]
    severe_terms = ["很痛", "劇痛", "嚴重", "快受不了", "受不了", "痛到睡不著", "痛到冒冷汗"]

    if sleep_impact or severe_impact or any(term in text for term in severe_terms if term != "嚴重") or (
        "嚴重" in text and "沒有很嚴重" not in text
    ):
        confidence = 0.92 if sleep_impact or severe_impact else 0.84
        return SeverityNormalization(
            severity_level="severe",
            functional_impact=functional_impact,
            sleep_impact=sleep_impact,
            confidence=confidence,
            semantic_status="available",
            source_text=text,
        )

    if any(term in text for term in explicit_mild_terms):
        return SeverityNormalization(
            severity_level="mild",
            functional_impact=False,
            sleep_impact=False,
            confidence=0.9,
            semantic_status="available",
            source_text=text,
        )

    if any(term in text for term in mild_terms):
        confidence = confidence_with_uncertainty(0.78, text)
        return SeverityNormalization(
            severity_level="mild",
            functional_impact=False if "還能工作" in text else functional_impact,
            sleep_impact=False,
            confidence=confidence,
            semantic_status="available" if accepted(confidence, "available") else "ambiguous",
            source_text=text,
            needs_clarification=needs_clarification(confidence, "available"),
            follow_up_reason="severity confidence below threshold" if confidence < 0.55 else None,
        )

    if any(term in text for term in moderate_terms):
        confidence = confidence_with_uncertainty(0.72, text)
        return SeverityNormalization(
            severity_level="moderate",
            functional_impact=functional_impact,
            sleep_impact=False,
            confidence=confidence,
            semantic_status="available" if accepted(confidence, "available") else "ambiguous",
            source_text=text,
            needs_clarification=needs_clarification(confidence, "available"),
            follow_up_reason="severity confidence below threshold" if confidence < 0.55 else None,
        )

    if any(term in text for term in UNKNOWN_TERMS):
        return SeverityNormalization(
            confidence=0.2,
            semantic_status="unknown",
            source_text=text,
            needs_clarification=True,
            follow_up_reason="使用者無法判斷嚴重程度",
        )

    if any(term in text for term in ["吧", "好像", "可能", "應該"]):
        return SeverityNormalization(
            severity_level=None,
            confidence=0.42,
            semantic_status="ambiguous",
            source_text=text,
            needs_clarification=True,
            follow_up_reason="嚴重程度語意模糊",
        )

    return SeverityNormalization(source_text=text)


def _normalize_availability(text: str, last_question_key: str | None) -> list[SemanticExtraction]:
    extractions: list[SemanticExtraction] = []
    dates = normalize_preferred_dates(text, last_question_key)
    if dates is not None:
        extractions.append(dates)

    days = normalize_preferred_days(text, last_question_key)
    if days is not None:
        extractions.append(days)

    sessions = normalize_preferred_sessions(text, last_question_key)
    if sessions is not None:
        extractions.append(sessions)

    leave = normalize_can_take_leave(text)
    if leave is not None:
        extractions.append(leave)

    return extractions


def normalize_preferred_days(text: str, last_question_key: str | None = None) -> SemanticExtraction | None:
    normalized = _normalize_weekday_text(text)

    if _is_unknown_answer(normalized, last_question_key, PREFERRED_DAYS_KEY):
        return _extraction(PREFERRED_DAYS_KEY, [], "unknown", 0.2, text, "日期偏好不確定")
    excluded_days = _extract_excluded_weekdays(normalized)
    if excluded_days:
        days = [day for day in ALL_WEEKDAYS if day not in excluded_days]
        return _extraction(
            PREFERRED_DAYS_KEY,
            days,
            "partial",
            confidence_with_uncertainty(0.92, normalized),
            text,
        )
    if _is_unavailable_answer(normalized, last_question_key, PREFERRED_DAYS_KEY):
        return _extraction(PREFERRED_DAYS_KEY, [], "unavailable", 0.82, text, "使用者表示最近沒有可就醫日期")

    days: list[str] = []
    confidence = 0.0
    exact_dates = normalize_preferred_dates(text, last_question_key)
    if exact_dates is not None:
        for value in exact_dates.normalized_value:
            parsed = datetime.strptime(value, "%Y-%m-%d").date()
            days.append(ALL_WEEKDAYS[parsed.weekday()])
        confidence = max(confidence, exact_dates.confidence)

    if last_question_key == PREFERRED_DAYS_KEY and any(term in normalized for term in NO_WORK_TERMS):
        days.extend(ALL_WEEKDAYS)
        confidence = max(confidence, confidence_with_uncertainty(0.72, normalized))

    if any(term in normalized for term in ["週一到週五", "週一至週五", "週一-週五", "平日", "工作日", "週間"]):
        days.extend(WEEKDAY_DAYS)
        confidence = max(confidence, confidence_with_uncertainty(0.88, normalized))

    if any(term in normalized for term in ["週末", "假日", "六日", "週六日", "週六週日"]):
        days.extend(WEEKEND_DAYS)
        confidence = max(confidence, confidence_with_uncertainty(0.86, normalized))

    range_match = re.search(r"週([一二三四五六日])\s*(?:到|至|-|～|~)\s*週([一二三四五六日])", normalized)
    if range_match:
        days.extend(_weekday_range(f"週{range_match.group(1)}", f"週{range_match.group(2)}"))
        confidence = max(confidence, confidence_with_uncertainty(0.9, normalized))

    for day in ALL_WEEKDAYS:
        if day in normalized:
            days.append(day)
            confidence = max(confidence, confidence_with_uncertainty(0.9, normalized))

    # Generic availability language must never broaden an explicit date set.
    if not days and _is_any_days_answer(normalized, last_question_key):
        days.extend(ALL_WEEKDAYS)
        confidence = confidence_with_uncertainty(0.8, normalized)

    if not days:
        if last_question_key == PREFERRED_DAYS_KEY and any(term in normalized for term in ["吧", "可能", "應該"]):
            return _extraction(PREFERRED_DAYS_KEY, [], "ambiguous", 0.42, text, "日期偏好語意模糊")
        return None

    status = "available" if len(set(days)) == len(ALL_WEEKDAYS) else "partial"
    return _extraction(PREFERRED_DAYS_KEY, _unique(days), status, confidence, text)


def normalize_preferred_dates(
    text: str,
    last_question_key: str | None = None,
) -> SemanticExtraction | None:
    """Extract explicit appointment dates without treating symptom dates as availability."""
    if not _has_explicit_appointment_date_context(text, last_question_key):
        return None

    today = _taipei_today()
    parsed_dates: list[str] = []
    occupied_spans: list[tuple[int, int]] = []
    full_date_pattern = re.compile(
        r"(?<!\d)(?P<year>\d{4})\s*(?:年|[-/])\s*(?P<month>\d{1,2})"
        r"\s*(?:月|[-/])\s*(?P<day>\d{1,2})\s*(?:日|號)?"
    )
    month_day_pattern = re.compile(
        r"(?<![\d/-])(?P<month>\d{1,2})\s*(?:月|/)\s*(?P<day>\d{1,2})\s*(?:日|號)?"
    )
    day_only_pattern = re.compile(
        r"(?<!\d)(?P<day>\d{1,2})\s*(?:日|號)(?:的)?"
        r"(?=\s*(?:上午|早上|早晨|下午|午後|晚上|晚間|夜間|夜診))"
    )

    def append_date(year: int | None, month: int, day: int, span: tuple[int, int]) -> None:
        try:
            if year is not None:
                candidate = datetime(year, month, day).date()
            else:
                candidate = datetime(today.year, month, day).date()
                if candidate < today:
                    candidate = datetime(today.year + 1, month, day).date()
        except ValueError:
            return
        parsed_dates.append(candidate.isoformat())
        occupied_spans.append(span)

    for match in full_date_pattern.finditer(text):
        append_date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
            match.span(),
        )
    for match in month_day_pattern.finditer(text):
        if any(start <= match.start() < end for start, end in occupied_spans):
            continue
        append_date(None, int(match.group("month")), int(match.group("day")), match.span())
    for match in day_only_pattern.finditer(text):
        if any(start <= match.start() < end for start, end in occupied_spans):
            continue
        try:
            candidate = datetime(today.year, today.month, int(match.group("day"))).date()
            if candidate < today:
                next_month = 1 if today.month == 12 else today.month + 1
                next_year = today.year + 1 if today.month == 12 else today.year
                candidate = datetime(next_year, next_month, int(match.group("day"))).date()
        except ValueError:
            continue
        parsed_dates.append(candidate.isoformat())

    values = _unique(parsed_dates)
    if not values:
        return None
    return _extraction(PREFERRED_DATES_KEY, values, "partial", 0.94, text)


def normalize_preferred_sessions(text: str, last_question_key: str | None = None) -> SemanticExtraction | None:
    if _is_unknown_answer(text, last_question_key, PREFERRED_SESSIONS_KEY):
        return _extraction(PREFERRED_SESSIONS_KEY, [], "unknown", 0.2, text, "看診時段不確定")
    if not _has_appointment_session_context(text, last_question_key):
        return None

    excluded_sessions = _extract_excluded_sessions(text)
    if not excluded_sessions and _is_unavailable_answer(text, last_question_key, PREFERRED_SESSIONS_KEY):
        return _extraction(PREFERRED_SESSIONS_KEY, [], "unavailable", 0.75, text, "使用者表示時段不方便")

    sessions: list[str] = []
    confidence = 0.0
    if last_question_key == PREFERRED_SESSIONS_KEY and any(term in text for term in NO_WORK_TERMS):
        sessions.extend(ALL_SESSIONS)
        confidence = max(confidence, confidence_with_uncertainty(0.72, text))

    for session, aliases in SESSION_ALIASES.items():
        if session not in excluded_sessions and any(term in text for term in aliases):
            sessions.append(session)
            confidence = max(confidence, confidence_with_uncertainty(0.86, text))
    if any(term in text for term in ["全天", "整天", "任何時段"]):
        sessions.extend(ALL_SESSIONS)
        confidence = max(confidence, confidence_with_uncertainty(0.88, text))

    # Generic "都可以" must not broaden an explicitly named subset such as
    # "上午跟下午都可以" into all three sessions.
    if (
        not sessions
        and last_question_key == PREFERRED_SESSIONS_KEY
        and any(term in text for term in ANY_SESSION_TERMS)
    ):
        sessions.extend(ALL_SESSIONS)
        confidence = confidence_with_uncertainty(0.78, text)

    if excluded_sessions:
        sessions = [session for session in sessions if session not in excluded_sessions]
        confidence = max(confidence, confidence_with_uncertainty(0.92, text))
        if not sessions:
            sessions = [session for session in ALL_SESSIONS if session not in excluded_sessions]

    if not sessions:
        return None

    status = "available" if len(set(sessions)) == len(ALL_SESSIONS) else "partial"
    return _extraction(PREFERRED_SESSIONS_KEY, _unique(sessions), status, confidence, text)


def normalize_can_take_leave(text: str) -> SemanticExtraction | None:
    if "請假" not in text:
        return None
    if any(term in text for term in ["不能", "不行", "不方便", "沒辦法", "無法", "不太方便"]):
        return _extraction("can_take_leave", False, "unavailable", 0.84, text)
    if any(term in text for term in ["可以", "能", "可", "願意"]):
        return _extraction("can_take_leave", True, "available", 0.86, text)
    return _extraction("can_take_leave", None, "ambiguous", 0.45, text, "請假意願不明確")


def _extraction(
    field: str,
    normalized_value: Any,
    semantic_status: str,
    confidence: float,
    source_text: str,
    follow_up_reason: str | None = None,
) -> SemanticExtraction:
    confidence = clamp_confidence(confidence)
    return SemanticExtraction(
        field=field,
        normalized_value=normalized_value,
        semantic_status=semantic_status,
        confidence=confidence,
        source_text=source_text,
        needs_clarification=needs_clarification(confidence, semantic_status),
        follow_up_reason=follow_up_reason,
    )


def _normalize_weekday_text(text: str) -> str:
    normalized = text
    replacements = {
        "星期一": "週一",
        "星期二": "週二",
        "星期三": "週三",
        "星期四": "週四",
        "星期五": "週五",
        "星期六": "週六",
        "星期日": "週日",
        "星期天": "週日",
        "禮拜一": "週一",
        "禮拜二": "週二",
        "禮拜三": "週三",
        "禮拜四": "週四",
        "禮拜五": "週五",
        "禮拜六": "週六",
        "禮拜日": "週日",
        "禮拜天": "週日",
        "周一": "週一",
        "周二": "週二",
        "周三": "週三",
        "周四": "週四",
        "周五": "週五",
        "周六": "週六",
        "周日": "週日",
        "周天": "週日",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    if "明天" in text:
        tomorrow = _taipei_today() + timedelta(days=1)
        normalized = f"{normalized} {ALL_WEEKDAYS[tomorrow.weekday()]}"
    if "大後天" in text:
        three_days_later = _taipei_today() + timedelta(days=3)
        normalized = f"{normalized} {ALL_WEEKDAYS[three_days_later.weekday()]}"
    if "後天" in text.replace("大後天", ""):
        after_tomorrow = _taipei_today() + timedelta(days=2)
        normalized = f"{normalized} {ALL_WEEKDAYS[after_tomorrow.weekday()]}"
    return normalized


def _is_any_days_answer(text: str, last_question_key: str | None) -> bool:
    if any(term in text for term in ["每天", "整週", "全週", *ANY_DAY_TERMS]):
        return True
    return (
        last_question_key == PREFERRED_DAYS_KEY
        and any(term in text for term in ["都可以", "都有空", "都行", "皆可"])
        and not any(term in text for term in ["前半", "後半"])
    )


def _extract_excluded_weekdays(text: str) -> list[str]:
    if "除了" in text:
        segment = text.split("除了", 1)[1]
        boundaries = [
            index
            for marker in ("之外", "其他", "其餘", "都可以", "都行", "皆可")
            if (index := segment.find(marker)) >= 0
        ]
        if boundaries:
            excluded = segment[: min(boundaries)]
            days = [day for day in ALL_WEEKDAYS if day in excluded]
            if days:
                return days

    complement_markers = ("其他都可以", "其他都行", "其他皆可", "其餘都可以", "其餘都行", "其餘皆可")
    matching_markers = [marker for marker in complement_markers if marker in text]
    if matching_markers and any(term in text for term in ("不行", "不能", "沒空", "不方便")):
        boundary = min(text.find(marker) for marker in matching_markers)
        excluded = text[:boundary]
        return [day for day in ALL_WEEKDAYS if day in excluded]
    return []


def _extract_excluded_sessions(text: str) -> list[str]:
    def mentioned(segment: str) -> list[str]:
        return [
            session
            for session, terms in SESSION_ALIASES.items()
            if any(term in segment for term in terms)
        ]

    excluded_sessions: set[str] = set()

    if "除了" in text:
        segment = text.split("除了", 1)[1]
        boundaries = [
            index
            for marker in ("之外", "其他", "其餘", "都可以", "都行", "皆可")
            if (index := segment.find(marker)) >= 0
        ]
        if boundaries:
            excluded = mentioned(segment[: min(boundaries)])
            if excluded:
                excluded_sessions.update(excluded)

    complement_markers = (
        "其他都可以",
        "其他都行",
        "其他皆可",
        "其他時段都可以",
        "其他時段都行",
        "其餘都可以",
        "其餘都行",
        "其餘皆可",
    )
    matching_markers = [marker for marker in complement_markers if marker in text]
    if matching_markers and any(term in text for term in ("不行", "不能", "沒空", "不方便")):
        boundary = min(text.find(marker) for marker in matching_markers)
        excluded_sessions.update(mentioned(text[:boundary]))

    direct_negations = ("不要", "不想要", "不想", "不希望", "不方便", "不能", "不行", "沒空", "無法")
    negation_pattern = "|".join(map(re.escape, direct_negations))
    for session, aliases in SESSION_ALIASES.items():
        alias_pattern = "|".join(map(re.escape, aliases))
        before_session = rf"(?:{negation_pattern})\s*(?:{alias_pattern})"
        after_session = rf"(?:{alias_pattern})\s*(?:{negation_pattern})"
        if re.search(before_session, text) or re.search(after_session, text):
            excluded_sessions.add(session)

    return [session for session in ALL_SESSIONS if session in excluded_sessions]


def _has_appointment_session_context(text: str, last_question_key: str | None) -> bool:
    if last_question_key in {PREFERRED_DAYS_KEY, PREFERRED_SESSIONS_KEY}:
        return True
    if last_question_key == "revision":
        has_calendar_date = bool(
            re.search(r"(?<!\d)\d{1,2}\s*(?:月|/)\s*\d{1,2}\s*(?:日|號)?", text)
        )
        has_explicit_time_change = bool(
            re.search(r"(?:改成|改為|改到|換成)\s*(?:上午|早上|下午|晚上|夜間)", text)
        )
        has_appointment_word = any(term in text for term in ("看診", "就醫", "掛號", "門診", "時段", "時間"))
        if has_calendar_date or has_explicit_time_change or has_appointment_word:
            return True
    session = r"(?:上午|早上|早晨|下午|午後|晚上|晚間|夜間|夜診)"
    explicit_preference_patterns = (
        rf"(?:想要|想|要|希望)\s*(?:掛|看)(?:診|門診)?\s*{session}",
        rf"(?:想要|想|要|希望)\s*(?:掛|看)(?:診|門診)?\s*"
        rf"[^，。！？!?\n]{{1,24}}?科[^，。！？!?\n]{{0,20}}?{session}",
        rf"(?:想要|想|要|希望)\s*{session}.{{0,3}}(?:看|看診|就醫|掛號|掛診|掛)",
        rf"{session}.{{0,4}}(?:有空|方便|可以(?:看診|就醫|掛號|掛診)?|能看|可看|門診|診次|時段)",
    )
    if any(re.search(pattern, text) for pattern in explicit_preference_patterns):
        return True
    if any(
        term in text
        for term in (
            "看診",
            "就醫",
            "掛號",
            "門診",
            "時段",
            "有空",
            "方便",
            "隨便安排",
            "時間沒差",
            "任何時段",
            "什麼時段",
            "全天",
            "整天",
            "哪天",
            "日期",
        )
    ):
        return True
    normalized = _normalize_weekday_text(text)
    if any(day in normalized for day in ALL_WEEKDAYS):
        return True
    return bool(
        re.search(
            r"(?:上午|早上|下午|晚上|夜間).{0,3}(?:都可以|都行|皆可|有空|方便)",
            text,
        )
    )


def _has_explicit_appointment_date_context(text: str, last_question_key: str | None) -> bool:
    if last_question_key in {PREFERRED_DAYS_KEY, PREFERRED_SESSIONS_KEY}:
        return True
    return any(
        term in text
        for term in (
            "看診", "就醫", "掛號", "門診", "有空", "方便",
            "上午", "早上", "早晨", "下午", "午後", "晚上", "晚間", "夜間", "夜診",
        )
    )


def _taipei_today():
    return datetime.now(ZoneInfo("Asia/Taipei")).date()


def _is_unknown_answer(text: str, last_question_key: str | None, field: str) -> bool:
    return last_question_key == field and any(term in text for term in UNKNOWN_TERMS)


def _is_unavailable_answer(text: str, last_question_key: str | None, field: str) -> bool:
    return last_question_key == field and any(term in text for term in UNAVAILABLE_TERMS)


def _weekday_range(start: str, end: str) -> list[str]:
    if start not in ALL_WEEKDAYS or end not in ALL_WEEKDAYS:
        return []
    start_index = ALL_WEEKDAYS.index(start)
    end_index = ALL_WEEKDAYS.index(end)
    if start_index <= end_index:
        return ALL_WEEKDAYS[start_index : end_index + 1]
    return [*ALL_WEEKDAYS[start_index:], *ALL_WEEKDAYS[: end_index + 1]]


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
