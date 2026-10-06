"""網關路由 / 中介層的單元測試（不需要 GPU，也不需要下游服務真的存在）。

對應審查報告：C1、C7、C8、C9、C10、C11、C12、C13、C18。
"""
import asyncio
import copy
import time

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from conftest import BASE_CONFIG

AUTH = {"X-API-Key": "sk-test-key"}
ORIGIN = {"Origin": "https://app.example.com"}


# ══════════════════════════════ C1：CORS ══════════════════════════════

def test_preflight_is_not_blocked_by_auth(make_gateway):
    """C1：瀏覽器 preflight 不會帶 X-API-Key，必須放行並回 CORS header。"""
    gw = make_gateway()
    with TestClient(gw.app) as client:
        r = client.options(
            "/api/process",
            headers={
                **ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-API-Key",
            },
        )
    assert r.status_code == 200, f"preflight 被擋掉了：{r.status_code}"
    assert r.headers.get("access-control-allow-origin") == "https://app.example.com"


def test_auth_failure_still_carries_cors_headers(make_gateway):
    """C1：401 也要帶 CORS header，前端才看得到真正的錯誤訊息。"""
    gw = make_gateway()
    with TestClient(gw.app) as client:
        r = client.post(
            "/api/process",
            headers={**ORIGIN, "X-API-Key": "wrong"},
            data={"service_type": "taiwanese_tts", "text_input": "你好"},
        )
    assert r.status_code == 401
    assert r.headers.get("access-control-allow-origin") == "https://app.example.com"


# ══════════════════════════════ C9：permissions ══════════════════════════════

def test_reload_config_requires_admin_permission(make_gateway):
    """C9：一般 key 不能重載設定。"""
    gw = make_gateway(env={"API_KEY": "", "API_KEYS": "sk-user:後端A"})
    with TestClient(gw.app) as client:
        r = client.post("/api/reload-config", headers={"X-API-Key": "sk-user"})
    assert r.status_code == 403


def test_reload_config_allowed_for_admin_key(make_gateway):
    gw = make_gateway(env={"API_KEY": "", "API_KEYS": "sk-admin:維運:*"})
    with TestClient(gw.app) as client:
        r = client.post("/api/reload-config", headers={"X-API-Key": "sk-admin"})
    assert r.status_code == 200


# ══════════════════════════════ C8：速率限制分桶 ══════════════════════════════

def test_rate_limit_buckets_by_key_not_by_name(make_gateway):
    """C8：兩把同名的 key 不可以共用同一個配額桶。"""
    gw = make_gateway(
        env={
            "API_KEY": "",
            "API_KEYS": "sk-aaa:共用名稱,sk-bbb:共用名稱",
            "RATE_LIMIT": 2,
        }
    )
    with TestClient(gw.app) as client:
        for _ in range(2):
            assert client.get("/services", headers={"X-API-Key": "sk-aaa"}).status_code == 200
        assert client.get("/services", headers={"X-API-Key": "sk-aaa"}).status_code == 429
        # 第二把 key 名字一樣，但配額必須獨立
        assert client.get("/services", headers={"X-API-Key": "sk-bbb"}).status_code == 200


def test_rate_limit_uses_forwarded_for_when_proxy_trusted(make_gateway):
    """C8：Nginx 後面所有匿名請求不可以共用一桶。"""
    gw = make_gateway(env={"RATE_LIMIT": 1, "TRUSTED_PROXY_HOPS": 1})
    with TestClient(gw.app) as client:
        a1 = client.get("/health", headers={"X-Forwarded-For": "203.0.113.1"})
        a2 = client.get("/health", headers={"X-Forwarded-For": "203.0.113.1"})
        b1 = client.get("/health", headers={"X-Forwarded-For": "203.0.113.9"})
    assert a1.status_code == 200
    assert a2.status_code == 429
    assert b1.status_code == 200, "不同來源 IP 被錯誤地共用了同一個配額桶"


def test_forwarded_for_ignored_when_no_proxy_trusted(make_gateway):
    """C8：預設不信任 XFF，否則任何人都能偽造來源繞過限制。"""
    gw = make_gateway(env={"RATE_LIMIT": 1})
    with TestClient(gw.app) as client:
        assert client.get("/health", headers={"X-Forwarded-For": "203.0.113.1"}).status_code == 200
        assert client.get("/health", headers={"X-Forwarded-For": "203.0.113.2"}).status_code == 429


# ══════════════════════════════ C7：記憶體清理 ══════════════════════════════

def test_rate_limiter_cleanup_removes_stale_buckets(make_gateway):
    gw = make_gateway()
    limiter = gw.rate_limiter
    limiter.check("someone")
    assert limiter.bucket_count() == 1
    limiter.window = -1  # 讓所有紀錄立刻過期
    limiter.cleanup()
    assert limiter.bucket_count() == 0


def test_cleanup_task_is_started_and_stopped_by_lifespan(make_gateway):
    """C7：cleanup() 必須真的有人定期呼叫。"""
    gw = make_gateway(env={"RATE_LIMIT_CLEANUP_INTERVAL": 60})
    with TestClient(gw.app) as client:
        client.get("/")
        task = getattr(gw.app.state, "rate_limit_cleanup_task", None)
        assert task is not None and not task.done()
    assert task.cancelled() or task.done()


# ══════════════════════════════ C10：設定檔缺欄位 ══════════════════════════════

def test_missing_optional_field_does_not_500(make_gateway):
    """C10：少一個非必要欄位不該讓 /services 整個炸掉。"""
    cfg = copy.deepcopy(BASE_CONFIG)
    del cfg["services"]["taiwanese_asr"]["description"]
    del cfg["services"]["taiwanese_asr"]["health_check"]
    gw = make_gateway(config=cfg)
    with respx.mock:
        respx.get("http://asr.test/").mock(return_value=httpx.Response(200))
        respx.get("http://tts.test/").mock(return_value=httpx.Response(200))
        with TestClient(gw.app) as client:
            r = client.get("/services", headers=AUTH)
    assert r.status_code == 200
    assert {s["service_type"] for s in r.json()} == set(BASE_CONFIG["services"])


def test_invalid_config_is_rejected_at_load_time(make_gateway):
    """C10：設定打錯字應該在載入時就爆，而不是等某個請求進來才爆。"""
    from pydantic import ValidationError

    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["services"]["taiwanese_asr"]["input_type"] = "vidoe"  # typo
    gw = make_gateway(config=cfg)
    with pytest.raises(ValidationError):
        gw.load_config()


def test_reload_config_is_atomic(make_gateway, tmp_path):
    """C10/S5：重載失敗時要保留舊設定，而不是留下半個壞掉的狀態。"""
    gw = make_gateway(env={"API_KEY": "", "API_KEYS": "sk-admin:維運:*"})
    with TestClient(gw.app) as client:
        before = client.get("/", headers={"X-API-Key": "sk-admin"}).json()["available_services"]
        (tmp_path / "services_config.yaml").write_text("services: {oops: 1}", encoding="utf-8")
        r = client.post("/api/reload-config", headers={"X-API-Key": "sk-admin"})
        after = client.get("/", headers={"X-API-Key": "sk-admin"}).json()["available_services"]
    assert r.status_code == 500
    assert after == before, "重載失敗後設定被清空了"


# ══════════════════════════════ C11：並行健康檢查 ══════════════════════════════

def test_health_probes_run_concurrently(make_gateway):
    gw = make_gateway()

    async def slow(request):
        await asyncio.sleep(0.3)
        return httpx.Response(200)

    with respx.mock:
        respx.get("http://asr.test/").mock(side_effect=slow)
        respx.get("http://tts.test/").mock(side_effect=slow)
        with TestClient(gw.app) as client:
            t0 = time.perf_counter()
            r = client.get("/health")
            elapsed = time.perf_counter() - t0

    assert r.status_code == 200
    assert r.json()["services"] == {
        "taiwanese_asr": "healthy",
        "taiwanese_tts": "healthy",
        "chinese_tts": "disabled",
    }
    assert elapsed < 0.55, f"health probe 看起來還是串行的（{elapsed:.2f}s）"


# ══════════════════════════════ C12：共用 httpx client ══════════════════════════════

def test_shared_http_client_is_reused(make_gateway):
    gw = make_gateway()
    with TestClient(gw.app) as client:
        client.get("/")
        shared = gw.app.state.http_client
        assert isinstance(shared, httpx.AsyncClient)
        assert not shared.is_closed
    assert shared.is_closed, "lifespan 結束時應該關閉共用 client"


# ══════════════════════════════ C13：檔案大小 ══════════════════════════════

def test_oversized_upload_is_rejected_before_full_read(make_gateway, wav_bytes):
    gw = make_gateway(env={"MAX_FILE_SIZE_MB": 1})
    big = b"RIFF" + b"\x00" * (2 * 1024 * 1024)
    with TestClient(gw.app) as client:
        r = client.post(
            "/api/process",
            headers=AUTH,
            data={"service_type": "taiwanese_asr"},
            files={"file": ("big.wav", big, "audio/wav")},
        )
    assert r.status_code == 413
    assert r.json()["success"] is False


def test_valid_upload_is_forwarded(make_gateway, wav_bytes):
    gw = make_gateway()
    with respx.mock:
        route = respx.post("http://asr.test/api/transcribe").mock(
            return_value=httpx.Response(200, json={"text": "你好", "success": True})
        )
        with TestClient(gw.app) as client:
            r = client.post(
                "/api/process",
                headers=AUTH,
                data={"service_type": "taiwanese_asr", "language": "nan"},
                files={"file": ("a.wav", wav_bytes, "audio/wav")},
            )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True and body["data"]["text"] == "你好"
    sent = route.calls.last.request.content
    assert b'name="audio_file"' in sent, "file_field 設定沒有被套用"
    assert b"nan" in sent, "language 沒有轉發給下游"


# ══════════════════════════════ C18：錯誤語意 ══════════════════════════════

def test_disabled_service_returns_503(make_gateway):
    gw = make_gateway()
    with TestClient(gw.app) as client:
        r = client.post(
            "/api/process",
            headers=AUTH,
            data={"service_type": "chinese_tts", "text_input": "你好"},
        )
    assert r.status_code == 503
    assert r.json()["success"] is False


def test_unknown_service_returns_404_in_envelope(make_gateway):
    gw = make_gateway()
    with TestClient(gw.app) as client:
        r = client.post("/api/process", headers=AUTH, data={"service_type": "nope"})
    assert r.status_code == 404
    assert r.json()["success"] is False


def test_missing_text_returns_400(make_gateway):
    gw = make_gateway()
    with TestClient(gw.app) as client:
        r = client.post("/api/process", headers=AUTH, data={"service_type": "taiwanese_tts"})
    assert r.status_code == 400


def test_invalid_audio_returns_400(make_gateway):
    gw = make_gateway()
    with TestClient(gw.app) as client:
        r = client.post(
            "/api/process",
            headers=AUTH,
            data={"service_type": "taiwanese_asr"},
            files={"file": ("a.exe", b"MZ\x00\x00not audio", "application/octet-stream")},
        )
    assert r.status_code == 400


def test_downstream_timeout_returns_504(make_gateway):
    gw = make_gateway()
    with respx.mock:
        respx.post("http://tts.test/api/synthesize").mock(
            side_effect=httpx.TimeoutException("too slow")
        )
        with TestClient(gw.app) as client:
            r = client.post(
                "/api/process",
                headers=AUTH,
                data={"service_type": "taiwanese_tts", "text_input": "你好"},
            )
    assert r.status_code == 504
    assert r.json()["success"] is False


def test_downstream_connect_error_returns_502(make_gateway):
    gw = make_gateway()
    with respx.mock:
        respx.post("http://tts.test/api/synthesize").mock(
            side_effect=httpx.ConnectError("refused")
        )
        with TestClient(gw.app) as client:
            r = client.post(
                "/api/process",
                headers=AUTH,
                data={"service_type": "taiwanese_tts", "text_input": "你好"},
            )
    assert r.status_code == 502


def test_downstream_5xx_becomes_502(make_gateway):
    gw = make_gateway()
    with respx.mock:
        respx.post("http://tts.test/api/synthesize").mock(
            return_value=httpx.Response(500, text="boom")
        )
        with TestClient(gw.app) as client:
            r = client.post(
                "/api/process",
                headers=AUTH,
                data={"service_type": "taiwanese_tts", "text_input": "你好"},
            )
    assert r.status_code == 502


def test_downstream_200_with_success_false_becomes_502(make_gateway):
    """C18：下游若回 200 + success:false，網關不能把它當成成功往上送。"""
    gw = make_gateway()
    with respx.mock:
        respx.post("http://tts.test/api/synthesize").mock(
            return_value=httpx.Response(
                200, json={"success": False, "error": "模型未載入", "original_text": "你好"}
            )
        )
        with TestClient(gw.app) as client:
            r = client.post(
                "/api/process",
                headers=AUTH,
                data={"service_type": "taiwanese_tts", "text_input": "你好"},
            )
    assert r.status_code == 502
    body = r.json()
    assert body["success"] is False
    assert "模型未載入" in body["error"]
    assert body["data"]["error"] == "模型未載入", "下游原始回應應該保留給呼叫端除錯"


def test_non_preflight_options_still_requires_auth(make_gateway):
    """OPTIONS 的放行只該給真正的 preflight，不能變成免認證通道。"""
    gw = make_gateway()
    with TestClient(gw.app) as client:
        r = client.options("/api/process", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_downstream_4xx_is_passed_through(make_gateway):
    gw = make_gateway()
    with respx.mock:
        respx.post("http://tts.test/api/synthesize").mock(
            return_value=httpx.Response(400, text="文字不可為空")
        )
        with TestClient(gw.app) as client:
            r = client.post(
                "/api/process",
                headers=AUTH,
                data={"service_type": "taiwanese_tts", "text_input": "你好"},
            )
    assert r.status_code == 400


def test_extra_params_whitelist_is_enforced(make_gateway):
    gw = make_gateway()
    with respx.mock:
        route = respx.post("http://tts.test/api/synthesize").mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        with TestClient(gw.app) as client:
            r = client.post(
                "/api/process",
                headers=AUTH,
                data={
                    "service_type": "taiwanese_tts",
                    "text_input": "你好",
                    "extra_params": '{"speed": 0.85, "evil": "rm -rf"}',
                },
            )
    assert r.status_code == 200
    body = route.calls.last.request.content.decode()
    assert '"speed"' in body and "evil" not in body


def test_timestamp_is_timezone_aware(make_gateway):
    """S11 附帶修正：timestamp 必須帶時區 offset。"""
    gw = make_gateway()
    with TestClient(gw.app) as client:
        r = client.post("/api/process", headers=AUTH, data={"service_type": "taiwanese_tts"})
    ts = r.json()["timestamp"]
    assert ts.endswith("+00:00") or ts.endswith("Z")
