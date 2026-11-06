"""已審核的中文整句 -> 本調數字調臺羅。啟動 8003 後只載入一次。"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
import re
import unicodedata

JSON_FILE = Path(__file__).with_name("medical_phrase_bank.json")


def normalize_chinese(text: str) -> str:
    """僅統一 Unicode、空白和全半形標點；絕不做模糊／子字串比對。"""
    s = unicodedata.normalize("NFKC", text).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(?<=[\u3400-\u9fff]) (?=[\u3400-\u9fff])", "", s)
    return s


@lru_cache(maxsize=1)
def _approved_index() -> dict[str, str]:
    data = json.loads(JSON_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("phrases"), list):
        raise ValueError("medical_phrase_bank.json 必須包含 phrases 清單")
    seen: set[str] = set()
    approved: dict[str, str] = {}
    for index, item in enumerate(data["phrases"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 筆資料不是 JSON 物件")
        zh = item.get("chinese")
        status = item.get("status")
        if not isinstance(zh, str) or not zh.strip():
            raise ValueError(f"第 {index} 筆缺少中文原文")
        if status not in ("approved", "review"):
            raise ValueError(f"第 {index} 筆 status 必須是 approved 或 review")
        key = normalize_chinese(zh)
        if key in seen:
            raise ValueError(f"重複中文話術：{zh}")
        seen.add(key)
        if status != "approved":
            continue  # 待審核項目絕不跳過 Gemini
        tailo = item.get("tailo_raw")
        if not isinstance(tailo, str) or not tailo.strip():
            raise ValueError(f"已核准話術缺少臺羅：{zh}")
        if not re.fullmatch(r"[A-Za-z0-9\s,.?!;\-]+", tailo):
            raise ValueError(f"已核准話術包含非臺羅格式字元：{zh}")
        approved[key] = tailo.strip()
    return approved


def lookup_approved_phrase(text: str) -> str | None:
    return _approved_index().get(normalize_chinese(text))


def count_approved() -> int:
    return len(_approved_index())
