"""量「語意比對前先挑詞」會不會漏掉正確的官方詞（不呼叫 AI）。

每句病人口語都標了知識庫裡「算對」的官方詞（concept_shortlist_cases.json）；
用正式的 concept_shortlist 挑詞，任一個算對的詞進入清單就算找到。
擴建知識庫或換模型後重跑，確認找到率沒有下降。

用法（在 backend/ 目錄；第一次會下載模型）：
    .venv/Scripts/python.exe -m benchmark.concept_shortlist_recall
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.services import concept_shortlist
from app.services.department_knowledge import load_department_knowledge

CASES = Path(__file__).with_name("concept_shortlist_cases.json")


def main() -> int:
    _sources, records = load_department_knowledge()
    catalog = sorted({record["concept"] for record in records})
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    unknown = sorted({c for case in cases for c in case["acceptable_concepts"]} - set(catalog))
    if unknown:
        print(f"題目裡有知識庫沒有的詞：{unknown}")
        return 1
    started = time.perf_counter()
    concept_shortlist.warm_up(catalog)
    if not concept_shortlist.ready():
        print("模型沒有載入，無法量測。")
        return 1
    print(f"模型 {get_settings().concept_embedding_model}：準備 {time.perf_counter() - started:.1f} 秒")
    misses, sizes = [], []
    started = time.perf_counter()
    for case in cases:
        picked = asyncio.run(concept_shortlist.shortlist([case["patient_text"]], catalog)) or catalog
        sizes.append(len(picked))
        if not set(case["acceptable_concepts"]) & set(picked):
            misses.append(case["patient_text"])
    per_case_ms = (time.perf_counter() - started) * 1000 / len(cases)
    found = len(cases) - len(misses)
    print(f"知識庫 {len(catalog)} 個詞；每句挑 {max(sizes)} 個以內，每句 {per_case_ms:.0f} 毫秒")
    print(f"找到 {found}/{len(cases)}（{found / len(cases):.0%}）；漏：{misses}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
