"""Patients under 18 go to pediatrics first.

Who the patient is comes from the turn interpreter's ``patient_profile`` (AI understanding);
code accepts it only when grounded and confident, then decides by fixed rules: the age is read
by code from the verbatim age text, the cut-off is under 18, and an age-less 「女兒」 gets one
fixed age question.

Official basis: Taichung VGH 症狀查詢「18歲以下兒童」— for any symptom of a child, 「若無法確認
小朋友的病症，可先掛『一般兒科』」(https://www.vghtc.gov.tw/SymptomQuery/605, retrieved 2026-11-03).
The user set the cut-off at under 18. A pediatric subspecialty the official evidence already
reached (e.g. 兒童腸胃科) is kept; otherwise 兒童內科.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Literal

from app.schemas import ConversationState
from app.services.confidence_scoring import ACCEPT_THRESHOLD

PEDIATRICS_DEPT_ID = 1320
PEDIATRIC_SUBSPECIALTY_DEPT_IDS = frozenset({1246, 1247, 1318, 1322})
ADULT_AGE = 18
AGE_PENDING_INTENT = "patient_age"
DECISION_REASON = "pediatric_default"
PROFILE_FIELDS = frozenset({"subject", "relation_text", "age_text", "source_text", "confidence"})

# Words that name a child whatever the age; school years up to 高中 are all under 18.
_CHILD_WORDS = ("寶寶", "寶貝", "嬰兒", "嬰幼兒", "新生兒", "幼兒", "小朋友", "小孩", "孩子", "小寶")
_SCHOOL = re.compile(r"幼稚園|幼兒園|托嬰|國小|小學|國中|初中|高中|高職|國[一二三]|高[一二三](?![點些下倍度])"
                     r"|小[一二三四五六]年級")
# Relations that do not tell the age: 「我兒子」 may be 5 or 30. 「我媽」 needs no question.
_KIN_WORDS = ("兒子", "女兒", "孫子", "孫女", "外孫", "姪子", "姪女", "外甥")
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_NUMBER = r"\d{1,3}|[零一二兩三四五六七八九十]{1,3}"
_AGE = re.compile(rf"({_NUMBER})\s*(?:足)?歲")
# Only 「月大」 is an age; a bare 「三個月」 is usually how long a symptom lasted.
_MONTHS = re.compile(rf"({_NUMBER})\s*(?:個)?月大")
_BARE_NUMBER = re.compile(rf"^\s*({_NUMBER})\s*$")
_OWNER_PREFIX = re.compile(r"^(?:我的|我家的|我家|我們家|我)")

Status = Literal["child", "adult", "ask"]


def _number(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text:
        tens, _, ones = text.partition("十")
        value = (_CN_DIGITS.get(tens, 0) if tens else 1) * 10 + (_CN_DIGITS.get(ones, 0) if ones else 0)
        return value if all(char in _CN_DIGITS for char in tens + ones) else None
    return _CN_DIGITS.get(text) if len(text) == 1 else None


def _age_status(text: str, *, bare_number: bool = False) -> Status | None:
    """child / adult from an age text read by code; None when it states no age."""
    ages = [value for match in _AGE.finditer(text) if (value := _number(match.group(1))) is not None]
    if _MONTHS.search(text):
        ages.append(0)
    bare = _BARE_NUMBER.match(text) if bare_number else None
    if bare and (value := _number(bare.group(1))) is not None:
        ages.append(value)
    if any(age < ADULT_AGE for age in ages) or _SCHOOL.search(text):
        return "child"
    return "adult" if ages else None


def validate_patient_profile(value: Any, user_sources: list[str]) -> dict | None:
    """Accept the AI's grounded, confident reading of who the patient is; never infer it here."""
    if not isinstance(value, dict) or set(value) != PROFILE_FIELDS:
        return None
    subject, relation, age, source, confidence = (
        value[key] for key in ("subject", "relation_text", "age_text", "source_text", "confidence"))
    # An empty string means "not said", like null.
    relation, age = (None if text == "" else text for text in (relation, age))
    if (subject not in {"self", "other"} or not isinstance(source, str) or not source.strip()
            or not any(source.strip() in text for text in user_sources)
            or type(confidence) not in {int, float} or not math.isfinite(confidence)
            or not ACCEPT_THRESHOLD <= confidence <= 1):
        return None
    for text in (relation, age):
        if text is not None and (not isinstance(text, str) or not text.strip() or text.strip() not in source):
            return None
    if subject == "self" and relation is not None:
        return None
    return dict(subject=subject, relation_text=relation.strip() if relation else None,
                age_text=age.strip() if age else None)


def apply_patient_profile(state: ConversationState, profile: dict | None) -> None:
    """A new subject replaces the stored one; the same subject keeps an earlier age when none is said."""
    if profile is None:
        return
    same = (state.patient_subject, state.patient_relation_text) == (profile["subject"], profile["relation_text"])
    state.patient_subject = profile["subject"]
    state.patient_relation_text = profile["relation_text"]
    if profile["age_text"] or not same:
        state.patient_age_text = profile["age_text"]


def patient_status(state: ConversationState) -> Status:
    """child / adult / ask (someone else with no age yet, whose one age question is still due)."""
    for text, bare in ((state.patient_age_text, False), (state.patient_age_answer, True)):
        decided = _age_status(text, bare_number=bare) if text else None
        if decided:
            return decided
    relation = state.patient_relation_text or ""
    if any(word in relation for word in _CHILD_WORDS) or _SCHOOL.search(relation):
        return "child"
    if (state.patient_subject == "other" and any(word in relation for word in _KIN_WORDS)
            and state.patient_age_answer is None):
        return "ask"
    return "adult"


def age_question(state: ConversationState) -> str:
    """Backend-owned fixed age question naming the person the user described."""
    who = _OWNER_PREFIX.sub("", state.patient_relation_text or "")
    return f"請問您{who}今年幾歲？" if who else "請問就診者今年幾歲？"


def pediatric_dept_id(candidate_dept_ids: Iterable[int]) -> int:
    """A pediatric subspecialty already reached by official evidence, else general pediatrics."""
    reached = sorted(set(candidate_dept_ids) & PEDIATRIC_SUBSPECIALTY_DEPT_IDS)
    return reached[0] if reached else PEDIATRICS_DEPT_ID
