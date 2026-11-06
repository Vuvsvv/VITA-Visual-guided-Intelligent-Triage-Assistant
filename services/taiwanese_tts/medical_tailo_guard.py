"""Conservative audit for Gemini-produced citation-tone Tailo.

This module NEVER silently edits a translation. It flags suspicious phonetic
forms and meaning-sensitive phrases for review. No Gemini or TTS dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import List


@dataclass(frozen=True)
class Finding:
    severity: str  # error | warning | info
    code: str
    message: str
    suggestion: str = ""

    def to_dict(self):
        return asdict(self)


# "大量出血" is preferred by the user after human listening verification.
# Other forms below are documented dictionary readings, not guarantees
# of semantic equivalence in all medical sentences.
VERIFIED_TERMS = {
    "大量出血": "tai7-liong7 tshut4-hueh4",
    "呼吸": "hoo1-khip4",
    "喘氣": "tshuan2-khui3",
}

# Scoped patterns are *warnings*, not blind substitution targets.
SUSPICIOUS_FORMS = {
    "khoo1-khip8": ("呼吸", "hoo1-khip4"),
    "tshuann3-khui3": ("喘氣", "tshuan2-khui3"),
}


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _syllables(tailo: str) -> List[str]:
    # Preserve double-hyphen convention by splitting on runs of hyphens.
    # Punctuation is removed only for *format* checks, not from output.
    text = re.sub(r"[，。！？?,.!;:()（）\[\]{}\n]", " ", tailo)
    return [p for w in text.split() for p in re.split(r"-+", w) if p]


def audit_translation(chinese: str, tailo_raw: str) -> List[Finding]:
    """Return findings; no automatic rewrite, no network calls.

    This is a guardrail rather than a proof of translation correctness.
    """
    findings: List[Finding] = []
    original = chinese.strip()
    tailo = _normalize_spaces(tailo_raw)

    if not tailo:
        return [Finding("error", "EMPTY_OUTPUT", "Gemini 沒有輸出臺羅；不可交給 TTS。")]

    # Strict whole-output check: do not silently strip unknown text such as
    # parentheses, Markdown, or embedded English commentary.
    group = re.compile(r"[a-zA-Z]+[0-9](?:-{1,2}[a-zA-Z]+[0-9])*")
    no_punctuation = re.sub(r"[，。！？?,.!;:、]", " ", tailo)
    for part in no_punctuation.split():
        if not group.fullmatch(part):
            findings.append(Finding("error", "UNEXPECTED_TEXT",
                f"輸出含無法識別為臺羅的片段 {part!r}。",
                "不可送入 TTS；請重新翻譯或人工審核。"))
            break

    for syllable in _syllables(tailo):
        # ASCII tone-number Tailo required by this particular TTS pipeline.
        if not re.fullmatch(r"[a-zA-Z]+[0-9]", syllable):
            findings.append(Finding("error", "TAILO_FORMAT",
                f"音節 {syllable!r} 缺少標準數字調或含不支援字元。",
                "請人工校對拼寫與每個音節的聲調數字。"))
            continue
        base, tone = syllable[:-1].lower(), int(syllable[-1])
        # In usual citation-tone Tailo, checked syllables end in p/t/k/h and carry tone 4 or 8.
        # This check deliberately excludes neutral tone 0.
        if base.endswith(("p", "t", "k", "h")) and tone not in (0, 4, 8):
            findings.append(Finding("error", "CHECKED_TONE",
                f"入聲韻尾音節 {syllable!r} 的聲調 {tone} 值得再次核對。",
                "確認它是本調（不是預先變調的輸出）。"))

    for wrong, (term, correct) in SUSPICIOUS_FORMS.items():
        if wrong in tailo:
            findings.append(Finding("error", "KNOWN_BAD_READING",
                f"偵測到曾出錯的臺羅 {wrong}；可能想表達「{term}」。",
                f"辭典讀音為 {correct}。先確認原文語意與詞彙對齊，不要直接全域取代。"))

    # EXTRA_MEDICAL_LEXICAL_CHECKS_V1
    # These checks are scoped to the intended Chinese expression; they do not
    # globally rewrite Gemini text or make any semantic equivalence claims.
    if "呼吸" in original and re.search(
        r"(?<![a-z0-9-])hu1-khip4(?![a-z0-9-])",
        tailo,
        re.IGNORECASE
        ):
        findings.append(Finding(
        "error",
        "KNOWN_WRONG_BREATHING_READING",
        "偵測到 hu1-khip4，不符合「呼吸」的詞典讀音。",
        "若此處確實表示「呼吸」，本調應為 hoo1-khip4；請重新翻譯並確認語意。",
    ))
    if "突然" in original and re.search(r"(?<![a-z0-9])thut4-lian5(?![a-z0-9])", tailo):
        findings.append(Finding(
            "error", "KNOWN_WRONG_SUDDEN_TONE",
            "『突然』的 thut4-lian5 和辭典本調不符。",
            "依所選讀法核對 tut8-jian5、tut8-lian5、thut8-jian5 或 thut8-lian5；不可直接全域替換。",
        ))
    if ("喘" in original or "呼吸困難" in original) and re.search(
        r"(?<![a-z0-9])tshuann[0-9](?![a-z0-9])", tailo
    ):
        findings.append(Finding(
            "error", "KNOWN_WRONG_PANTING_READING",
            "『喘』輸出 tshuann 加聲調數字（鼻化韻母），與已核對的 tshuan2 不符。",
            "辭典所收『喘』為 tshuan2；需核對實際用詞與整句語意。",
        ))

    if original == "大量出血":
        expected = VERIFIED_TERMS["大量出血"]
        # Accept punctuation after the standalone phrase.
        normalized = tailo.rstrip(".。!?！？ ")
        if normalized != expected:
            findings.append(Finding("warning", "HUMAN_BASELINE_MISMATCH",
                "「大量出血」和你人工聽測確認的本調基準不一致。",
                f"已確認基準：{expected}。請核對差異。"))
    elif "大量出血" in original and VERIFIED_TERMS["大量出血"] not in tailo:
        findings.append(Finding("warning", "MEDICAL_TERM_PARAPHRASE",
            "中文含「大量出血」，但臺羅未出現已驗證的詞組。可能是改寫或漏譯。",
            "請人工核對是否保留出血程度及否定語意。"))

    if "呼吸困難" in original and "tshuann3-khui3" in tailo:
        findings.append(Finding("warning", "SEMANTIC_REVIEW",
            "「呼吸困難」被表達成疑似錯讀的「喘氣」組合；可能同時有拼音與語意偏差。",
            "須由人工確認醫療語意；不要自動改為「喘氣」。"))

    if "不舒服" in original and "bo5 song2-khuai3" in tailo:
        findings.append(Finding("info", "SANDHI_BOUNDARY",
            "`bo5 song2-khuai3` 與先前 `bo5-song2-khuai3` 的變調域不同。",
            "請核對臺語詞組和 Sandhi 規則，不要自動串接連字號。"))

    if any(word in original for word in ("沒有", "沒", "不", "無", "未", "不要")):
        findings.append(Finding("info", "NEGATION_REVIEW",
            "原文含否定或否定問句，必須人工確認翻譯是否保留否定語意。",
            "單靠臺羅格式驗證無法判斷語意正確。"))

    # MEDICAL_LEXICON_GUARD_V1：只攔截已核對詞彙的疑似錯讀，不自動改寫。
    if __package__:
        from .medical_lexicon_guard import lexicon_findings
    else:
        from medical_lexicon_guard import lexicon_findings
    findings.extend(lexicon_findings(original, tailo))

    if not findings:
        findings.append(Finding("info", "FORMAT_OK",
            "未發現已知錯讀或基本格式異常；這不代表語意與讀音已驗證正確。"))
    return findings


def requires_review(findings: List[Finding]) -> bool:
    return any(f.severity in ("error", "warning") for f in findings)


if __name__ == "__main__":
    import json
    samples = [
        ("大量出血", "tai7-liong7 tshut4-hueh4"),
        ("你有沒有呼吸困難？", "li2 kam2 u7 khoo1-khip8 khun3-lan5?"),
        ("你有沒有呼吸困難？", "li2 kam2 u7 tshuann3-khui3 khun3-lan5?"),
        ("你現在哪裡不舒服？", "li2 tsit4-ma2 to2-ui7 bo5 song2-khuai3?"),
    ]
    for zh, tailo in samples:
        print(zh, "=>", tailo)
        print(json.dumps([f.to_dict() for f in audit_translation(zh, tailo)], ensure_ascii=False, indent=2))
