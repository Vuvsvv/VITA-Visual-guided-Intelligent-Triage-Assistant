"""Conservative, context-scoped Tailo lexical check (citation tones only).

This checks only candidate spellings of two verified words. It does not
rewrite Tailo, translate sentences, or assert overall medical correctness.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class LexiconFinding:
    severity: str
    code: str
    message: str
    suggestion: str = ""

    def to_dict(self):
        return dict(severity=self.severity, code=self.code,
                    message=self.message, suggestion=self.suggestion)


# The pattern matches likely attempted spellings of the same lexical item.
# It does NOT require Gemini to select that item when a valid paraphrase exists.
# Original-dictionary citation tones; not sandhi forms.
LEXICON = {
    "描述": {
        "expected": "biau5-sut8",
        "candidate": r"(?<![a-z0-9-])(?:biau|biat|miau)[0-9]-(?:su|sut)[0-9](?![a-z0-9-])",
        "dictionary": "https://sutian.moe.edu.tw/zh-hant/su/22460/",
    },
    "呼吸": {
        "expected": "hoo1-khip4",
        "candidate": r"(?<![a-z0-9-])(?:hoo|hu|khoo)[0-9]-khip[0-9](?![a-z0-9-])",
        "dictionary": "https://sutian.moe.edu.tw/zh-hant/su/3691/",
    },
}


def lexicon_findings(original: str, tailo: str):
    """Return only verified reading mismatches; never do replacement.

    If Gemini uses a different Taigi word, this checker does not guess
    its meaning or force the dictionary reading of the Chinese word.
    """
    if not isinstance(original, str) or not isinstance(tailo, str):
        return []
    text = tailo.lower()
    issues = []
    for chinese, rule in LEXICON.items():
        if chinese not in original:
            continue
        for m in re.finditer(rule["candidate"], text, flags=re.IGNORECASE):
            observed = m.group(0).lower()
            if observed == rule["expected"]:
                continue
            issues.append(LexiconFinding(
                severity="error",
                code="LEXICON_READING_MISMATCH",
                message=f"原文有「{chinese}」，疑似讀法 {observed}，已核對本調為 {rule['expected']}。",
                suggestion="請確認翻譯確實選用此詞；若選用其他合法臺語說法，不要強制全域取代。",
            ))
    return issues
