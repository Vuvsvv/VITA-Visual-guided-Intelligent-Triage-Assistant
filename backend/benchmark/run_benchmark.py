"""固定案例對話 benchmark：以 FastAPI TestClient 呼叫 /chat，使用本機 .env 的真 AI。

模擬病人另外呼叫同一個 AI 供應商（Cerebras），只看得到病人實際狀況，看不到可接受科別。

用法（在 backend/ 目錄）：
    .venv/Scripts/python.exe -m benchmark.run_benchmark --out <scratchpad>/bench.json
    .venv/Scripts/python.exe -m benchmark.run_benchmark --case c09_right_knee_pain --repeat 1 --out ...
    .venv/Scripts/python.exe -m benchmark.run_benchmark --base-url http://<host>:8080 --out ...
    .venv/Scripts/python.exe -m benchmark.run_benchmark --tier smoke --out ...
    .venv/Scripts/python.exe -m benchmark.run_benchmark --rerun-failed <scratchpad>/bench.json --out ...
    .venv/Scripts/python.exe -m benchmark.run_benchmark --summarize <scratchpad>/bench.json

給 --base-url 時改以 HTTP（timeout 120 秒）呼叫遠端後端，模擬病人仍在本機以 .env 金鑰呼叫。

省額度：預設每案例只跑 1 次；--tier smoke 只跑 cases.yaml 標 smoke 的精簡組；
--rerun-failed 只重跑上次結果中失敗的案例。每次執行都記錄 token 用量
（後端用量只有 TestClient 模式量得到，--base-url 模式只記模擬病人）。
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

CASES_FILE = Path(__file__).with_name("cases.yaml")
MAX_TURNS = 20
VISIT_TYPE = "initial"
SAFETY_NO_ANSWER = "沒有"
AVAILABILITY_ANSWER = "都可以"
UNKNOWN_ANSWER = "不知道"
SIMULATOR_RETRIES = 3
HTTP_TIMEOUT_SECONDS = 120

SIMULATOR_PROMPT = """你在扮演一位到醫院掛號前、正在回答問診系統問題的一般病人。
請用繁體中文、一般病人的口語，一兩句簡短回答系統最後的問題。不要使用醫學術語，不要猜自己該看哪一科，不要提出科別名稱。

你的實際狀況（只能依據這些內容回答）：
- 一開始說的話：{opening}
- 有的症狀或情況：{present}
- 確定沒有的症狀：{absent}

規則：
1. 系統問到上面「有的」項目，就照實說有，可以補一點描述，但不可加入清單以外的新症狀。
2. 系統問到上面「確定沒有的」項目，回答沒有。
3. 系統問到清單裡沒列出的任何症狀或細節，一律回答「沒有」或「不清楚」。
4. {safety_rule}
5. 系統問看診時間或哪天方便，回答「都可以」。
6. 只輸出病人要說的話，不要加引號或任何說明。

先前對話：
{history}

系統最後說：{system_reply}

病人回答："""

SAFETY_RULE_NON_URGENT = "系統問到是否有危險或緊急徵兆（例如胸痛、呼吸困難、意識改變、單側無力等）時，一律回答沒有。"
SAFETY_RULE_URGENT = "系統問到是否有危險或緊急徵兆時，依你的實際狀況照實回答。"


def load_cases(path: Path = CASES_FILE) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["cases"]


TOKEN_FIELDS = ("calls", "prompt_tokens", "completion_tokens", "total_tokens")


class TokenMeter:
    """Per-run AI token usage, split into backend and simulator buckets."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.buckets: dict[str, Counter[str]] = {"backend": Counter(), "simulator": Counter()}

    def add(self, bucket: str, response: Any) -> None:
        usage = getattr(response, "usage", None)
        counter = self.buckets[bucket]
        counter["calls"] += 1
        for field in TOKEN_FIELDS[1:]:
            counter[field] += int(getattr(usage, field, 0) or 0)

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {name: {field: counter[field] for field in TOKEN_FIELDS} for name, counter in self.buckets.items()}


def install_backend_meter(meter: TokenMeter) -> None:
    """Wrap the backend's Cerebras client in-process (TestClient mode only); backend code is untouched."""
    from app.services import ai_service

    def wrap_client() -> None:
        completions = getattr(getattr(ai_service._cerebras_client, "chat", None), "completions", None)
        if completions is None or getattr(completions.create, "_benchmark_metered", False):
            return
        original = completions.create

        async def metered(*args: Any, **kwargs: Any) -> Any:
            response = await original(*args, **kwargs)
            meter.add("backend", response)
            return response

        metered._benchmark_metered = True  # type: ignore[attr-defined]
        completions.create = metered

    original_initialize = ai_service._initialize_cerebras

    def initialize(settings: Any) -> None:
        original_initialize(settings)
        wrap_client()

    ai_service._initialize_cerebras = initialize
    wrap_client()


class PatientSimulator:
    def __init__(self, meter: TokenMeter) -> None:
        from cerebras.cloud.sdk import Cerebras

        from app.config import get_settings

        settings = get_settings()
        if not settings.cerebras_api_key:
            raise RuntimeError("CEREBRAS_API_KEY 未設定，無法啟動模擬病人。")
        self.model = settings.cerebras_model
        self.client = Cerebras(api_key=settings.cerebras_api_key, timeout=30.0, max_retries=0)
        self.meter = meter

    def answer(self, case: dict[str, Any], transcript: list[dict[str, str]], system_reply: str) -> str:
        history = "\n".join(
            f"{'系統' if item['role'] == 'system' else '病人'}：{item['content']}" for item in transcript[-8:]
        ) or "（無）"
        prompt = SIMULATOR_PROMPT.format(
            opening=case["opening"],
            present="、".join(case.get("present") or []) or "（無）",
            absent="、".join(case.get("absent") or []) or "（無）",
            safety_rule=SAFETY_RULE_URGENT if case.get("expect_urgent") else SAFETY_RULE_NON_URGENT,
            history=history,
            system_reply=system_reply,
        )
        last_error: Exception | None = None
        for attempt in range(SIMULATOR_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                self.meter.add("simulator", response)
                text =str(response.choices[0].message.content or "").strip().strip("「」\"")
                if text:
                    return text.splitlines()[0].strip()
                last_error = ValueError("simulator returned empty content")
            except Exception as exc:  # noqa: BLE001 - benchmark records and retries provider errors
                last_error = exc
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"模擬病人呼叫失敗：{last_error!r}")


class HttpClient:
    """Minimal TestClient-compatible wrapper for a remote backend."""

    def __init__(self, base_url: str) -> None:
        import requests

        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.session.close()

    def get(self, path: str) -> Any:
        return self.session.get(f"{self.base_url}{path}", timeout=HTTP_TIMEOUT_SECONDS)

    def post(self, path: str, json: dict[str, Any]) -> Any:  # noqa: A002 - mirrors TestClient.post
        return self.session.post(f"{self.base_url}{path}", json=json, timeout=HTTP_TIMEOUT_SECONDS)


def _candidates(state: dict[str, Any]) -> list[str]:
    return [item.get("childDept", "") for item in state.get("candidate_departments") or []]


def _asked_concepts(state: dict[str, Any], is_safety: bool, is_availability: bool) -> list[str]:
    """Concept keys the system asks in this reply; used to count repeated questions."""
    if is_safety:
        return ["safety_screen"]
    if is_availability:
        return ["availability"]
    pending_index = state.get("pending_clue_question_index")
    if pending_index is not None:
        concepts = [
            f"clue:{record['concept']}"
            for record in state.get("clue_concepts_asked") or []
            if record.get("question_index") == pending_index
        ]
        if concepts:
            return concepts
    intent = state.get("pending_clarification_intent")
    return [f"intent:{intent}"] if intent else []


def _turn_snapshot(body: dict[str, Any]) -> dict[str, Any]:
    state = body.get("conversation_state") or {}
    dept = body.get("department_result") or {}
    return {
        "stage": state.get("stage"),
        "clarification_status": state.get("clarification_status"),
        "department_status": state.get("department_status"),
        "availability_status": state.get("availability_status"),
        "candidates": _candidates(state),
        "department": dept.get("childDept") or None,
        "dept_id": dept.get("dept_id"),
        "clue_questions_asked": state.get("clue_questions_asked", 0),
        "pending_clarification_intent": state.get("pending_clarification_intent"),
        "pending_clue_question_index": state.get("pending_clue_question_index"),
        "warning_required": (body.get("triage") or {}).get("warning_required", False),
        "ttas": (body.get("triage_case") or {}).get("ttas_result", {}).get("level_candidate"),
    }


def run_case(client: Any, simulator: PatientSimulator, case: dict[str, Any], run_index: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case_id": case["id"],
        "run": run_index,
        "expect_urgent": bool(case.get("expect_urgent")),
        "accept_departments": case.get("accept_departments") or [],
        "turns": [],
        "error": None,
        "end_reason": None,
    }
    transcript: list[dict[str, str]] = []
    seen_concepts: Counter[str] = Counter()
    repeated = 0
    urgent_seen = False
    simulator.meter.reset()
    started = time.perf_counter()

    try:
        start = client.post("/chat", json={"visit_type": VISIT_TYPE})
        if start.status_code != 200:
            raise RuntimeError(f"Start HTTP {start.status_code}: {start.text[:300]}")
        body = start.json()
        case_id = body["case_id"]
        record["start_reply"] = body.get("reply") or body.get("next_question")
        message = case["opening"]

        for turn in range(1, MAX_TURNS + 1):
            transcript.append({"role": "patient", "content": message})
            turn_started = time.perf_counter()
            response = client.post("/chat", json={"case_id": case_id, "visit_type": VISIT_TYPE, "message": message})
            elapsed = time.perf_counter() - turn_started
            if response.status_code != 200:
                record["turns"].append({"turn": turn, "patient": message, "http_status": response.status_code,
                                        "seconds": round(elapsed, 3), "error_body": response.text[:500]})
                raise RuntimeError(f"/chat HTTP {response.status_code}: {response.text[:300]}")
            body = response.json()
            reply = body.get("reply") or body.get("next_question") or ""
            snapshot = _turn_snapshot(body)
            is_safety = snapshot["clarification_status"] == "safety_check"
            is_availability = snapshot["availability_status"] == "asked"
            concepts = _asked_concepts(body.get("conversation_state") or {}, is_safety, is_availability)
            turn_repeats = sum(1 for key in concepts if seen_concepts[key] > 0)
            repeated += turn_repeats
            seen_concepts.update(concepts)
            transcript.append({"role": "system", "content": reply})
            record["turns"].append({"turn": turn, "patient": message, "system": reply, "http_status": 200,
                                    "seconds": round(elapsed, 3), "asked_concepts": concepts,
                                    "repeated_concepts": turn_repeats, **snapshot})

            if snapshot["clarification_status"] == "urgent":
                urgent_seen = True
                record["end_reason"] = "urgent"
                break
            if snapshot["stage"] == "waiting_confirmation":
                record["end_reason"] = "waiting_confirmation"
                break
            if snapshot["clarification_status"] == "unresolved":
                # Terminal until a revision; later turns would only repeat the same stop reply.
                record["end_reason"] = "unresolved"
                break
            if snapshot["stage"] == "done" or not reply:
                record["end_reason"] = f"terminal:{snapshot['clarification_status']}"
                break
            if turn == MAX_TURNS:
                record["end_reason"] = "max_turns"
                break

            if case.get("patient_mode") == "always_unknown":
                message = UNKNOWN_ANSWER
            elif is_availability:
                message = AVAILABILITY_ANSWER
            elif is_safety and not case.get("expect_urgent"):
                message = SAFETY_NO_ANSWER
            else:
                message = simulator.answer(case, transcript[:-1], reply)

        if record["end_reason"] == "unresolved":
            # A stopped case must not reach /recommend; record its status for the safety-stop check.
            recommend = client.post(
                "/recommend", json={"case_id": case_id, "confirmed": True, "visit_type": VISIT_TYPE},
            )
            record["recommend_http_status"] = recommend.status_code
        if record["end_reason"] == "waiting_confirmation":
            confirm = client.post("/chat", json={"case_id": case_id, "visit_type": VISIT_TYPE, "confirmed": True})
            record["confirm_http_status"] = confirm.status_code
            if confirm.status_code == 200:
                confirmed_body = confirm.json()
                record["confirm_stage"] = (confirmed_body.get("conversation_state") or {}).get("stage")
                record["confirm_department"] = (confirmed_body.get("department_result") or {}).get("childDept")
            recommend_started = time.perf_counter()
            recommend = client.post(
                "/recommend", json={"case_id": case_id, "confirmed": True, "visit_type": VISIT_TYPE},
            )
            record["recommend_http_status"] = recommend.status_code
            record["recommend_seconds"] = round(time.perf_counter() - recommend_started, 3)
            if recommend.status_code != 200:
                record["recommend_detail"] = recommend.text[:300]
    except Exception as exc:  # noqa: BLE001 - benchmark records every failure mode
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["end_reason"] = record["end_reason"] or "error"

    turns = [item for item in record["turns"] if item.get("http_status") == 200]
    final = turns[-1] if turns else {}
    accept = set(record["accept_departments"])
    seconds = [item["seconds"] for item in record["turns"]]
    record.update({
        "final_department": final.get("department"),
        "final_in_accept": bool(final.get("department") in accept) if accept else None,
        "accept_ever_candidate": any(set(item.get("candidates") or []) & accept for item in turns) if accept else None,
        "first_turn_candidates": turns[0].get("candidates") if turns else [],
        "total_turns": len(record["turns"]),
        "clue_questions": final.get("clue_questions_asked", 0),
        "repeated_concepts": repeated,
        "urgent_detected": urgent_seen,
        "avg_turn_seconds": round(sum(seconds) / len(seconds), 3) if seconds else None,
        "max_turn_seconds": round(max(seconds), 3) if seconds else None,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "tokens": simulator.meter.snapshot(),
    })
    record.setdefault("recommend_http_status", None)
    first_top = (turns[0].get("candidates") or [None])[0] if turns else None
    record.update({
        "first_turn_top_candidate": first_top,
        "first_turn_top_in_accept": bool(first_top in accept) if accept and first_top else (False if accept else None),
    })
    score_run(record)
    return record


def run_failed(run: dict[str, Any]) -> bool:
    """Errored, wrong department (non-urgent), or urgent case not intercepted."""
    score_run(run)
    if run["error"]:
        return True
    return not run["urgent_intercepted"] if run["expect_urgent"] else not run["final_in_accept"]


def run_tokens(run: dict[str, Any], bucket: str) -> int:
    return ((run.get("tokens") or {}).get(bucket) or {}).get("total_tokens", 0)


def safety_stopped(run: dict[str, Any]) -> bool:
    """Ended unresolved without ever reaching waiting_confirmation, and /recommend did not return 200."""
    turns = [item for item in run["turns"] if item.get("http_status") == 200]
    return bool(
        any(item.get("clarification_status") == "unresolved" for item in turns)
        and not any(item.get("stage") == "waiting_confirmation" for item in turns)
        and run.get("recommend_http_status") != 200
    )


def score_run(run: dict[str, Any]) -> dict[str, Any]:
    """Urgent cases count as intercepted when urgent or safely stopped; also used to rescore older JSON."""
    run["safety_stopped"] = safety_stopped(run)
    run["urgent_intercepted"] = bool(run["urgent_detected"] or run["safety_stopped"])
    return run


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    runs = [score_run(run) for run in runs]
    non_urgent = [run for run in runs if not run["expect_urgent"]]
    urgent = [run for run in runs if run["expect_urgent"]]
    all_seconds = [turn["seconds"] for run in runs for turn in run["turns"]]
    by_case: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_case.setdefault(run["case_id"], []).append(run)
    inconsistent = [
        case_id for case_id, items in by_case.items()
        if len({(item["final_department"], item["urgent_detected"], item["end_reason"]) for item in items}) > 1
    ]
    return {
        "runs": len(runs),
        "department_accuracy": (
            round(sum(1 for run in non_urgent if run["final_in_accept"]) / len(non_urgent), 3) if non_urgent else None
        ),
        "department_correct": sum(1 for run in non_urgent if run["final_in_accept"]),
        "department_total": len(non_urgent),
        "urgent_intercept_rate": (
            round(sum(1 for run in urgent if run["urgent_intercepted"]) / len(urgent), 3) if urgent else None
        ),
        "urgent_intercepted": sum(1 for run in urgent if run["urgent_intercepted"]),
        "urgent_by_ttas": sum(1 for run in urgent if run["urgent_detected"]),
        "urgent_by_safety_stop": sum(1 for run in urgent if run["safety_stopped"] and not run["urgent_detected"]),
        "first_turn_top_in_accept": sum(1 for run in non_urgent if run.get("first_turn_top_in_accept")),
        "urgent_total": len(urgent),
        "false_urgent_runs": sum(1 for run in non_urgent if run["urgent_detected"]),
        "avg_turns": round(sum(run["total_turns"] for run in runs) / len(runs), 2) if runs else None,
        "avg_turn_seconds": round(sum(all_seconds) / len(all_seconds), 3) if all_seconds else None,
        "max_turn_seconds": round(max(all_seconds), 3) if all_seconds else None,
        "repeated_concepts_total": sum(run["repeated_concepts"] for run in runs),
        "inconsistent_cases": len(inconsistent),
        "inconsistent_case_ids": inconsistent,
        "errors": sum(1 for run in runs if run["error"]),
        "end_reasons": dict(Counter(run["end_reason"] for run in runs)),
        "recommend_status": dict(Counter(str(run["recommend_http_status"]) for run in runs)),
        "backend_tokens_total": sum(run_tokens(run, "backend") for run in runs),
        "simulator_tokens_total": sum(run_tokens(run, "simulator") for run in runs),
        "failed_case_ids": sorted({run["case_id"] for run in runs if run_failed(run)}),
    }


def print_table(runs: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print("case_id\trun\tend\tfinal\tok\tever_cand\tturn1_top\tturns\tclues\trepeat\turgent\tstopped\trec"
          "\tavg_s\tmax_s\tbe_tok\tsim_tok\terror")
    for run in map(score_run, runs):
        print("\t".join(str(value) for value in (
            run["case_id"], run["run"], run["end_reason"], run["final_department"], run["final_in_accept"],
            run["accept_ever_candidate"],
            f"{run.get('first_turn_top_candidate')}({run.get('first_turn_top_in_accept')})",
            run["total_turns"], run["clue_questions"], run["repeated_concepts"],
            run["urgent_detected"], run["safety_stopped"], run["recommend_http_status"], run["avg_turn_seconds"],
            run["max_turn_seconds"], run_tokens(run, "backend"), run_tokens(run, "simulator"),
            (run["error"] or "")[:80],
        )))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Conversation benchmark against /chat with real AI.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--case", action="append", help="只跑指定案例 id（可重複）")
    parser.add_argument("--tier", choices=("full", "smoke"), default="full",
                        help="smoke 只跑 cases.yaml 標 smoke: true 的精簡組")
    parser.add_argument("--rerun-failed", type=Path, help="只重跑這份既有結果 JSON 裡失敗的案例")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "benchmark_results.json")
    parser.add_argument("--summarize", type=Path, help="只讀取既有結果 JSON 並輸出總表")
    parser.add_argument("--base-url", help="有給就以 HTTP 呼叫該網址的 /chat、/recommend；沒給才用 TestClient")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    if args.summarize:
        data = json.loads(args.summarize.read_text(encoding="utf-8"))
        print_table(data["runs"], summarize(data["runs"]))
        return 0

    cases = load_cases()
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["id"] in wanted]
        missing = wanted - {case["id"] for case in cases}
        if missing:
            parser.error(f"未知案例：{sorted(missing)}")
    if args.tier == "smoke":
        cases = [case for case in cases if case.get("smoke")]
    if args.rerun_failed:
        previous = json.loads(args.rerun_failed.read_text(encoding="utf-8"))["runs"]
        failed = {run["case_id"] for run in previous if run_failed(run)}
        cases = [case for case in cases if case["id"] in failed]
        if not cases:
            print(f"{args.rerun_failed} 沒有需要重跑的失敗案例。")
            return 0
    print(f"將執行 {len(cases)} 個案例 × {args.repeat} 次：{[case['id'] for case in cases]}", flush=True)

    meter = TokenMeter()
    simulator = PatientSimulator(meter)
    runs: list[dict[str, Any]] = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    wall_started = time.perf_counter()
    if args.base_url:
        client_context: Any = HttpClient(args.base_url)
        health = client_context.get("/health")
        if health.status_code != 200:
            print(f"無法連到 {args.base_url}/health：HTTP {health.status_code}", file=sys.stderr)
            return 2
    else:
        from fastapi.testclient import TestClient

        from app.main import app

        install_backend_meter(meter)
        client_context = TestClient(app)
    with client_context as client:
        for case in cases:
            for run_index in range(1, args.repeat + 1):
                run = run_case(client, simulator, case, run_index)
                runs.append(run)
                print(
                    f"[{case['id']} #{run_index}] end={run['end_reason']} dept={run['final_department']} "
                    f"ok={run['final_in_accept']} turns={run['total_turns']} urgent={run['urgent_detected']} "
                    f"stopped={run['safety_stopped']} "
                    f"rec={run['recommend_http_status']} tokens(be/sim)={run_tokens(run, 'backend')}/"
                    f"{run_tokens(run, 'simulator')} err={run['error']}",
                    flush=True,
                )
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps({"started_at": started_at, "runs": runs}, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

    summary = summarize(runs)
    summary["started_at"] = started_at
    summary["wall_seconds"] = round(time.perf_counter() - wall_started, 1)
    args.out.write_text(json.dumps({"started_at": started_at, "summary": summary, "runs": runs},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(runs, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
