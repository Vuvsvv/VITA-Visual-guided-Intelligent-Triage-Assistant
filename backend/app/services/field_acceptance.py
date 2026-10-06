from __future__ import annotations

import re
from typing import Any


SYMPTOM_TERMS = (
    "睡不著",
    "很難入睡",
    "難以入睡",
    "失眠",
    "一直醒來",
    "一直醒",
    "睡不好",
    "不舒服",
    "不太舒服",
    "疼",
    "痛",
    "癢",
    "麻",
    "腫",
    "暈",
    "咳",
    "喘",
    "發燒",
    "發熱",
    "噁心",
    "想吐",
    "嘔吐",
    "腹瀉",
    "拉肚子",
    "紅疹",
    "皮疹",
    "流鼻水",
    "鼻塞",
    "耳鳴",
    "出血",
    "無力",
    "痠",
    "酸",
    "刺",
    "卡卡",
    "感冒",
)

BODY_PART_TERMS = (
    "右上腹",
    "左上腹",
    "右下腹",
    "左下腹",
    "上腹",
    "下腹",
    "左大腿",
    "右大腿",
    "左小腿",
    "右小腿",
    "左腿",
    "右腿",
    "左邊屁股",
    "右邊屁股",
    "左側屁股",
    "右側屁股",
    "左屁股",
    "右屁股",
    "肛門周圍",
    "肛門口",
    "左臀部",
    "右臀部",
    "全身",
    "身上",
    "關節",
    "屁股",
    "臀部",
    "臀側",
    "尾椎",
    "髖部",
    "肛門",
    "膝蓋",
    "膝部",
    "大腿",
    "小腿",
    "手臂",
    "腳踝",
    "胸口",
    "肚子",
    "胃部",
    "腸胃",
    "胃",
    "喉嚨",
    "肩膀",
    "頭部",
    "腹部",
    "腰部",
    "背部",
    "皮膚",
    "耳朵",
    "眼睛",
    "膝",
    "胸",
    "頭",
    "腹",
    "眼",
    "耳",
    "腰",
    "背",
    "肩",
    "手",
    "腳",
    "腿",
    "臀",
    "髖",
)

_CONTEXTUAL_DISCOMFORT_TERMS = (
    "怪怪",
    "不對勁",
    "不太對勁",
    "卡卡",
    "酸酸",
    "痠痠",
    "麻麻",
    "悶悶",
)

_DURATION_PATTERN = re.compile(
    r"(?:\d+|[一二兩三四五六七八九十]+)\s*(?:(?:個)?(?:禮拜|星期)|天|週|周|個月|年)"
)
DURATION_ONSET_PATTERN = re.compile(
    r"(?:"
    r"(?:從)?(?:今天|昨天|前天|上週|上個月)"
    r"(?:早上|上午|中午|下午|晚上|半夜)?(?:就)?(?:開始|起(?!床|來|身|夜))"
    r"|"
    r"(?:從)?(?:早上|上午|中午|下午|晚上|半夜)(?:就)?開始"
    r")"
)
STRONG_UNCERTAINTY_TERMS = (
    "不知道",
    "不確定",
    "不清楚",
    "說不上來",
    "說不準",
    "無法判斷",
    "沒辦法判斷",
    "難以判斷",
)
COMPOUND_AMBIGUITY_TERMS = (
    "可是",
    "但是",
    "不過",
    "但有時",
    "，但",
    "還好但",
)
_DEPARTMENT_INTENT_TERMS = (
    "想看", "要看", "希望看", "直接看", "改看", "我要", "想要",
    "想掛", "要掛", "掛號", "掛診", "想改", "改成", "改為", "科別",
)
_BODY_LOCATION_FILLERS = (
    "附近",
    "那邊",
    "這邊",
    "部位",
    "位置",
    "邊",
    "側",
)


def has_symptom_semantics(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if any(term in text for term in SYMPTOM_TERMS):
        return True
    return any(part in text for part in BODY_PART_TERMS) and any(
        term in text for term in _CONTEXTUAL_DISCOMFORT_TERMS
    )


def has_body_part_semantics(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if looks_like_department_request(text) and not has_symptom_semantics(text):
        return False
    return any(term in text for term in BODY_PART_TERMS)


def normalize_body_part(value: str) -> str | None:
    text = str(value or "").strip()
    if not has_body_part_semantics(text):
        return None

    # These neighboring anatomical regions remain distinct because they may
    # lead to different downstream clinical considerations.
    if "肛門" in text:
        return "肛門周圍" if "肛門周圍" in text else "肛門"
    if "尾椎" in text:
        return "尾椎"
    if "髖部" in text or "髖" in text:
        return "髖部"
    if "屁股" in text or "臀部" in text or "臀側" in text or "臀" in text:
        if any(term in text for term in ("左邊屁股", "左側屁股", "左屁股", "左臀部", "左臀")):
            return "左臀部"
        if any(term in text for term in ("右邊屁股", "右側屁股", "右屁股", "右臀部", "右臀")):
            return "右臀部"
        return "臀部"

    for part in ("右上腹", "左上腹", "右下腹", "左下腹", "上腹", "下腹"):
        if part in text:
            return part
    if any(part in text for part in ("胃部", "腸胃", "胃")):
        return "腹"

    for part in BODY_PART_TERMS:
        if part in text:
            return {
                "膝蓋": "膝",
                "頭部": "頭",
                "胸口": "胸",
                "腹部": "腹",
                "肚子": "腹",
                "耳朵": "耳",
                "眼睛": "眼",
                "腰部": "腰",
                "背部": "背",
            }.get(part, part)
    return None


def has_strong_uncertainty(value: str) -> bool:
    text = str(value or "").strip()
    return any(term in text for term in STRONG_UNCERTAINTY_TERMS)


def has_compound_ambiguity(value: str) -> bool:
    text = str(value or "").strip()
    return any(term in text for term in COMPOUND_AMBIGUITY_TERMS)


def requires_semantic_refinement(value: str) -> bool:
    """Return true only for uncertainty that can invalidate a parsed core value."""
    return has_strong_uncertainty(value) or has_compound_ambiguity(value)


def has_duration_semantics(value: str) -> bool:
    text = str(value or "").strip()
    return bool(
        _DURATION_PATTERN.search(text)
        or DURATION_ONSET_PATTERN.search(text)
        or any(
            term in text
            for term in (
                "半天",
                "半年",
                "一年半",
                "幾天",
                "好幾天",
                "幾週",
                "好幾週",
                "幾個禮拜",
                "幾禮拜",
                "好幾個禮拜",
                "好幾禮拜",
                "幾個星期",
                "幾星期",
                "好幾個星期",
                "好幾星期",
                "一陣子",
                "一段時間",
            )
        )
    )


def has_severity_semantics(value: str) -> bool:
    text = str(value or "").strip().lower()
    if text in {"mild", "moderate", "severe"}:
        return True
    return any(
        term in text
        for term in (
            "輕微",
            "有點痛",
            "一點痛",
            "微痛",
            "不太影響",
            "沒有影響",
            "正常生活",
            "正常作息",
            "不影響日常生活",
            "不影響睡眠",
            "正常上班",
            "正常走路",
            "沒什麼影響",
            "沒有很嚴重",
            "還好但",
            "普通",
            "中等",
            "中度",
            "明顯",
            "很痛",
            "劇痛",
            "嚴重",
            "受不了",
            "影響生活",
            "影響睡覺",
            "影響睡眠",
            "睡不著",
            "痛醒",
            "無法工作",
            "不能工作",
            "無法走路",
            "走路困難",
            "沒辦法正常生活",
            "沒辦法上班",
            "不能正常走路",
            "無法正常活動",
        )
    )


def looks_like_department_request(value: str) -> bool:
    text = str(value or "").strip()
    return "科" in text and any(term in text for term in _DEPARTMENT_INTENT_TERMS)


def normalized_value_valid(field: str, value: Any, status: str) -> bool:
    if status in {"unknown", "ambiguous"}:
        return True
    if status == "unavailable" and field in {"preferred_days", "preferred_sessions"}:
        return isinstance(value, list)
    if field == "symptom":
        return isinstance(value, str) and has_symptom_semantics(value)
    if field == "onset":
        return isinstance(value, str) and 0 < len(value.strip()) <= 30
    if field == "accompanying_symptoms":
        return isinstance(value, list) and 0 < len(value) <= 10 and all(
            isinstance(item, str) and 0 < len(item.strip()) <= 30 for item in value
        )
    if field == "body_part":
        return isinstance(value, str) and has_body_part_semantics(value)
    if field == "duration":
        return isinstance(value, str) and has_duration_semantics(value)
    if field == "severity":
        level = value.get("severity_level") if isinstance(value, dict) else value
        return str(level or "").strip().lower() in {"mild", "moderate", "severe"}
    if field == "preferred_days":
        return isinstance(value, list) and bool(value) and all(
            str(item) in {"週一", "週二", "週三", "週四", "週五", "週六", "週日"}
            for item in value
        )
    if field == "preferred_sessions":
        return isinstance(value, list) and bool(value) and all(
            str(item) in {"上午", "下午", "夜間"} for item in value
        )
    return False


def ai_normalized_value_rejection_reason(
    field: str,
    value: Any,
    status: str,
    source_text: str,
) -> str | None:
    """Validate AI output shape without requiring free-text values to be in rule dictionaries."""
    if field == "symptom":
        text = str(value or "").strip() if isinstance(value, str) else ""
        if not text or len(text) > 24 or re.search(r"[\r\n，。！？；;]", text):
            return "invalid_normalized_value"
        if "科" in text or any(term in text for term in ("門診", "醫院", "診所", "掛號")):
            return "invalid_normalized_value"
        if re.search(r"\d{4}-\d{2}-\d{2}", text) or text in {
            "週一", "週二", "週三", "週四", "週五", "週六", "週日", "上午", "下午", "夜間",
        }:
            return "invalid_normalized_value"
        return None
    if field == "accompanying_symptoms":
        return None if (
            isinstance(value, list)
            and 0 < len(value) <= 10
            and all(isinstance(item, str) and 0 < len(item.strip()) <= 30 for item in value)
        ) else "invalid_normalized_value"
    if status in {"unknown", "ambiguous"}:
        return None
    if field == "onset":
        if not normalized_value_valid(field, value, status):
            return "invalid_normalized_value"
        if value.strip() not in source_text:
            return "normalized_value_not_supported_by_source"
        return None
    if field not in {"symptom", "body_part"}:
        return None if normalized_value_valid(field, value, status) else "invalid_canonical_value"

    text = str(value or "").strip()
    source = str(source_text or "").strip()
    max_length = 16
    if not text or len(text) > max_length or re.search(r"[\r\n，。！？；;]", text):
        return "invalid_normalized_value"
    if "科" in text or any(term in text for term in ("門診", "醫院", "診所", "掛號")):
        return "invalid_normalized_value"
    if re.search(r"\d{4}-\d{2}-\d{2}", text) or text in {
        "週一", "週二", "週三", "週四", "週五", "週六", "週日", "上午", "下午", "夜間",
    }:
        return "invalid_normalized_value"

    if text in {"左", "右", "左邊", "右邊", "附近", "這裡", "那裡", "身體"}:
        return "invalid_normalized_value"
    source_sides = {side for side in ("左", "右") if side in source}
    normalized_sides = {side for side in ("左", "右") if side in text}
    if (
        len(source_sides) == 1
        and len(normalized_sides) == 1
        and source_sides != normalized_sides
    ):
        return "laterality_conflict"
    if not source_sides and normalized_sides:
        return "laterality_not_grounded"
    if text in source:
        return None
    source_without_fillers = _remove_body_location_fillers(source)
    text_without_fillers = _remove_body_location_fillers(text)
    if text_without_fillers and text_without_fillers in source_without_fillers:
        return None
    # Known canonicalization (for example 腸胃 -> 腹) is allowed, while novel
    # anatomy remains open-ended as long as the normalized phrase is quoted
    # from the grounded source. This blocks unrelated replacements such as
    # 鎖骨附近 -> 膝蓋 without rebuilding an anatomy dictionary.
    if normalize_body_part(source) == text:
        return None
    if _has_grounded_abdominal_direction(source, text):
        return None
    return "normalized_value_not_supported_by_source"


def _remove_body_location_fillers(value: str) -> str:
    """Remove only non-anatomical location wording for lexical grounding."""
    result = str(value or "").strip()
    for filler in _BODY_LOCATION_FILLERS:
        result = result.replace(filler, "")
    return result


def _has_grounded_abdominal_direction(source: str, normalized: str) -> bool:
    """Allow a navel-relative quadrant without restoring arbitrary overlap."""
    match = re.fullmatch(r"([左右])([上下])腹", normalized)
    if not match or not any(anchor in source for anchor in ("肚臍", "腹", "肚子")):
        return False
    return f"{match.group(1)}{match.group(2)}" in source


def ai_normalized_value_valid(field: str, value: Any, status: str, source_text: str) -> bool:
    return ai_normalized_value_rejection_reason(field, value, status, source_text) is None


def plausible_semantic_target(field: str, text: str) -> bool:
    """Identify possible extra slots; never use this to block a keyed fallback."""
    if field == "symptom":
        return has_symptom_semantics(text)
    if field == "body_part":
        return has_body_part_semantics(text) or any(term in text for term in ("部位", "位置", "附近", "左邊", "右邊"))
    if field == "duration":
        return has_duration_semantics(text) or any(term in text for term in ("持續", "開始", "多久", "前陣子"))
    if field == "severity":
        return has_severity_semantics(text) or any(term in text for term in ("程度", "影響", "忍受"))
    if field == "preferred_days":
        return any(term in text for term in ("週", "星期", "禮拜", "日期", "哪天", "今天", "明天", "後天", "平日", "週末"))
    if field == "preferred_sessions":
        return any(term in text for term in ("上午", "早上", "下午", "晚上", "夜間", "時段", "全天", "整天", "任何時段"))
    return False
