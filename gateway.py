"""
AI 模型服務統一網關 (API Gateway)
=================================
所有後端只需呼叫同一個端點 POST /api/process，
透過 service_type 欄位決定走哪個模型服務。

新增服務只需：
  1. 在 services_config.yaml 加一段設定
  2. 啟動對應的微服務
  網關會自動路由，不需改任何程式碼。

設定檔會在載入時用 Pydantic 驗證：少欄位、打錯字會在啟動（或 /api/reload-config）
當下就報錯，而不是等某個請求進來才 KeyError。
每個服務的 endpoint 都可以用環境變數 `<SERVICE_TYPE>_ENDPOINT` 覆蓋，
例如 `CHINESE_TTS_ENDPOINT=http://127.0.0.1:8004`，裸機/容器共用同一份設定檔。
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import httpx
import yaml
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from security import (
    setup_security,
    validate_audio_file,
    sanitize_text_input,
    MAX_FILE_SIZE,
)

__version__ = "2.2.0"

# ── 日誌 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gateway")

HEALTH_PROBE_TIMEOUT = float(os.getenv("HEALTH_PROBE_TIMEOUT", "5"))
UPLOAD_CHUNK_SIZE = 1024 * 1024


# ===================== 設定模型 =====================

class ServiceConfig(BaseModel):
    """一個下游微服務的設定。缺欄位 / 型別錯誤會在載入時就被擋下（C10）。"""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    version: str = "0.0.0"
    enabled: bool = True
    endpoint: str
    health_check: str = "/"
    timeout: int = 60
    input_type: Literal["audio", "image", "text"]
    output_type: Literal["text", "audio"]
    forward_path: str
    file_field: Optional[str] = None
    models: List[str] = Field(default_factory=list)


class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    gateway_port: int = 8000


class GatewayConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    services: Dict[str, ServiceConfig] = Field(default_factory=dict)
    global_: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")


# ── 全域狀態 ──
CONFIG: GatewayConfig = GatewayConfig()


def load_config() -> GatewayConfig:
    """
    載入並驗證設定檔。

    先完整驗證再原子替換 CONFIG —— 中途失敗時舊設定原封不動，
    不會留下「設定被清空一半」的壞狀態。
    """
    global CONFIG
    path = os.getenv("GATEWAY_CONFIG", "services_config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = GatewayConfig.model_validate(raw)

    # endpoint 環境變數覆蓋（讓裸機 / Docker 共用同一份 yaml）
    for key, service in config.services.items():
        override = os.getenv(f"{key.upper()}_ENDPOINT")
        if override:
            service.endpoint = override.rstrip("/")
            logger.info(f"{key}.endpoint 由環境變數覆蓋為 {service.endpoint}")

    CONFIG = config
    logger.info(f"載入設定: {len(config.services)} 個服務 ({path})")
    return CONFIG


def svc(service_type: str) -> Optional[ServiceConfig]:
    """取得某個服務的設定，找不到回傳 None。"""
    return CONFIG.services.get(service_type)


# ===================== Lifespan =====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_config()

    # 共用的 httpx client：每個請求重新建立等於每次都重做 TCP handshake（C12）
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    # 速率限制器的記憶體回收（C7）
    app.state.rate_limit_cleanup_task = rate_limiter.start_cleanup_task()

    logger.info(f"網關已啟動 (v{__version__})")
    try:
        yield
    finally:
        await rate_limiter.stop_cleanup_task()
        await app.state.http_client.aclose()
        logger.info("網關已關閉")


# ── FastAPI ──
app = FastAPI(
    title="AI 模型服務網關",
    description=(
        "統一入口：後端只需帶上 `service_type` 參數 + `X-API-Key` header，"
        "網關自動路由到對應的 AI 模型微服務。"
    ),
    version=__version__,
    lifespan=lifespan,
)

# ── 安全中間件（API Key 認證 + 權限 + 速率限制 + 日誌）──
# ⚠️ 順序很重要：Starlette 中「後 add 的在最外層」。
#    安全中間件要先加，CORS 後加，CORS 才會包在最外層；
#    否則瀏覽器 preflight（不帶 X-API-Key）會先被 401 擋掉，
#    而且 401 / 429 的回應也不會帶 CORS header（C1）。
auth_manager, rate_limiter = setup_security(app, {
    "rate_limit": int(os.getenv("RATE_LIMIT", "60")),
    "rate_window": int(os.getenv("RATE_WINDOW", "60")),
    "enable_logging": True,
})

# ── CORS 白名單（從環境變數讀取）──
allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",")
allowed_origins = [o.strip() for o in allowed_origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Authorization", "Content-Type"],
)


# ===================== Pydantic Models =====================

class ServiceInfo(BaseModel):
    service_type: str
    name: str
    description: str
    version: str
    enabled: bool
    endpoint: str
    input_type: str
    output_type: str
    models: List[str]
    status: str = "unknown"


class GatewayResponse(BaseModel):
    """後端拿到的統一回應格式"""
    success: bool
    service_type: str
    timestamp: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[int] = None


# ===================== Helpers =====================

def _now() -> str:
    """帶時區的 ISO 8601 時間戳（跨時區對帳用）。"""
    return datetime.now(timezone.utc).isoformat()


def ok(service_type: str, data: dict, elapsed_ms: int) -> JSONResponse:
    payload = GatewayResponse(
        success=True,
        service_type=service_type,
        timestamp=_now(),
        data=data,
        processing_time_ms=elapsed_ms,
    )
    return JSONResponse(status_code=200, content=payload.model_dump())


def fail(
    service_type: str,
    error: str,
    status_code: int,
    elapsed_ms: int = 0,
    data: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    """
    失敗回應。

    一律帶對應的 HTTP status code —— 全部回 200 + success:false 的話，
    呼叫端沒辦法用 status code 做重試 / 告警 / circuit breaker（C18）。
    """
    payload = GatewayResponse(
        success=False,
        service_type=service_type,
        timestamp=_now(),
        data=data,
        error=error,
        processing_time_ms=elapsed_ms,
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


async def health_probe(client: httpx.AsyncClient, endpoint: str, path: str) -> bool:
    try:
        r = await client.get(f"{endpoint}{path}", timeout=HEALTH_PROBE_TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False


async def probe_all(client: httpx.AsyncClient) -> Dict[str, str]:
    """
    並行探測所有服務的健康狀態。

    串行 for-await 的話，4 個服務全掛時 /health 要 20 秒才回（C11）。
    """
    enabled = [(k, c) for k, c in CONFIG.services.items() if c.enabled]
    results = await asyncio.gather(
        *[health_probe(client, c.endpoint, c.health_check) for _, c in enabled]
    )
    status = {k: "disabled" for k, c in CONFIG.services.items() if not c.enabled}
    status.update(
        {k: ("healthy" if up else "unhealthy") for (k, _), up in zip(enabled, results)}
    )
    return status


class DownstreamError(Exception):
    """下游服務的錯誤，帶著要往上回的 HTTP status code。"""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


async def forward(
    client: httpx.AsyncClient,
    cfg: ServiceConfig,
    path: str,
    *,
    files: Optional[Dict] = None,
    data: Optional[Dict] = None,
    json_body: Optional[Dict] = None,
) -> httpx.Response:
    """將請求轉發到下游微服務（用共用的 client，保留連線池）。"""
    url = f"{cfg.endpoint}{path}"
    timeout = cfg.timeout

    try:
        if files:
            return await client.post(url, files=files, data=data or {}, timeout=timeout)
        elif json_body:
            return await client.post(url, json=json_body, timeout=timeout)
        else:
            return await client.get(url, timeout=timeout)
    except httpx.TimeoutException:
        raise DownstreamError(504, f"下游服務逾時 ({timeout}s): {url}")
    except httpx.ConnectError:
        raise DownstreamError(502, f"無法連線到下游服務: {url}")
    except httpx.HTTPError as e:
        raise DownstreamError(502, f"轉發失敗: {e}")


async def read_upload(file: UploadFile, max_bytes: int) -> Optional[bytes]:
    """
    分塊讀取上傳檔案，超過上限就中止並回 None。

    先整包 `await file.read()` 再比大小，等於任何人都能逼網關吃下任意大小的檔案（C13）。
    """
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await file.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


# ===================== 端點 =====================

@app.get("/")
async def root():
    return {
        "service": "AI 模型服務網關",
        "version": __version__,
        "available_services": [
            k for k, v in CONFIG.services.items() if v.enabled
        ],
        "docs": "/docs",
    }


@app.get("/services", response_model=List[ServiceInfo])
async def list_services(request: Request):
    """列出所有已註冊的服務及其狀態。"""
    status = await probe_all(request.app.state.http_client)
    return [
        ServiceInfo(
            service_type=key,
            name=c.name,
            description=c.description,
            version=c.version,
            enabled=c.enabled,
            endpoint=c.endpoint,
            input_type=c.input_type,
            output_type=c.output_type,
            models=c.models,
            status=status.get(key, "unknown"),
        )
        for key, c in CONFIG.services.items()
    ]


@app.get("/health")
async def health(request: Request):
    """所有服務的健康檢查摘要。"""
    return {
        "gateway": "healthy",
        "services": await probe_all(request.app.state.http_client),
    }


# ─────────────────────────────────────────────
#  核心端點：統一處理入口
# ─────────────────────────────────────────────

@app.post("/api/process", response_model=GatewayResponse)
async def process(
    request: Request,
    service_type: str = Form(
        ...,
        description="要使用的服務名稱",
        examples=["taiwanese_asr", "taiwanese_tts", "chinese_tts"],
    ),
    file: Optional[UploadFile] = File(None, description="音訊 / 圖片等檔案"),
    text_input: Optional[str] = Form(None, description="文字輸入"),
    language: Optional[str] = Form(None, description="語言參數（部分服務需要）"),
    extra_params: Optional[str] = Form(None, description="其他參數（JSON 字串）"),
):
    """
    ## 統一處理入口

    後端只需呼叫這一個端點，透過 **service_type** 決定走哪個模型。

    | service_type    | 輸入     | 說明                                        |
    |-----------------|----------|---------------------------------------------|
    | taiwanese_asr   | 音訊檔案 | 台語 → 中文字（Breeze-ASR-26）                |
    | taiwanese_tts   | 文字     | 中文 → 台羅（Gemini）→ 台語語音（Chatterbox）     |
    | chinese_tts     | 文字     | 中文 → 台灣國語語音（BreezyVoice）            |

    ### 回應狀態碼
    成功一律 200；失敗會帶對應的 status code（400 輸入錯誤、404 未知服務、
    413 檔案過大、502/504 下游異常、503 服務未啟用），body 仍是同一個信封格式。

    ### 範例 (curl)
    ```bash
    # 台語語音識別
    curl -X POST http://localhost:8000/api/process \\
      -H 'X-API-Key: sk-...' \\
      -F "service_type=taiwanese_asr" \\
      -F "file=@taiwanese.wav"

    # 台語 TTS（中文文字 → 台語語音）
    curl -X POST http://localhost:8000/api/process \\
      -H 'X-API-Key: sk-...' \\
      -F "service_type=taiwanese_tts" \\
      -F "text_input=你好嗎" \\
      -F 'extra_params={"speed":1.0}'
    ```
    """
    t0 = time.time()
    client: httpx.AsyncClient = request.app.state.http_client

    cfg = svc(service_type)
    if cfg is None:
        return fail(
            service_type,
            f"未知的 service_type: '{service_type}'。可用: {list(CONFIG.services)}",
            404,
        )

    if not cfg.enabled:
        return fail(service_type, f"服務 '{service_type}' 目前未啟用", 503)

    try:
        # ── 需要檔案的服務（audio / image）──
        if cfg.input_type in ("audio", "image"):
            if file is None:
                return fail(
                    service_type,
                    f"此服務需要上傳檔案（input_type={cfg.input_type}）",
                    400,
                )

            contents = await read_upload(file, MAX_FILE_SIZE)
            if contents is None:
                return fail(
                    service_type,
                    f"檔案超過大小限制 ({MAX_FILE_SIZE // (1024 * 1024)}MB)",
                    413,
                )

            # 安全檢查：驗證檔案
            if cfg.input_type == "audio":
                valid, error_msg = validate_audio_file(contents, file.filename or "")
                if not valid:
                    return fail(service_type, f"檔案驗證失敗: {error_msg}", 400)

            files_dict = {
                (cfg.file_field or "file"): (file.filename, contents, file.content_type)
            }

            # 額外表單欄位
            form_data = {}
            if language:
                form_data["language"] = language

            resp = await forward(
                client, cfg, cfg.forward_path, files=files_dict, data=form_data
            )

        # ── 純文字服務（翻譯、TTS 等）──
        elif cfg.input_type == "text":
            if not text_input:
                return fail(service_type, "此服務需要提供 text_input", 400)

            # 安全檢查：清理輸入文字
            cleaned_text, warnings = sanitize_text_input(text_input)
            if not cleaned_text:
                return fail(service_type, "輸入文字無效", 400)
            if warnings:
                logger.info(f"[{service_type}] 輸入警告: {warnings}")

            body: Dict[str, Any] = {"text": cleaned_text}
            if language:
                body["language"] = language

            # 解析額外參數（如 speed, temperature 等）
            if extra_params:
                try:
                    extra = json.loads(extra_params)
                    if isinstance(extra, dict):
                        # 只允許白名單內的參數
                        allowed_params = {"speed", "temperature", "return_format", "format"}
                        body.update(
                            {k: v for k, v in extra.items() if k in allowed_params}
                        )
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"無法解析 extra_params: {extra_params}")

            resp = await forward(client, cfg, cfg.forward_path, json_body=body)

        else:  # pragma: no cover - 設定已被 Literal 限制，理論上到不了
            return fail(service_type, f"不支援的 input_type: {cfg.input_type}", 500)

        elapsed = int((time.time() - t0) * 1000)

        if resp.status_code == 200:
            data = resp.json()
            # 有些下游端點失敗時仍然回 200 + success:false（歷史包袱）。
            # 不能把它當成成功往上送，否則 C18 的 status code 語意在最常見的
            # 失敗路徑上仍然是壞的。
            if isinstance(data, dict) and data.get("success") is False:
                return fail(
                    service_type,
                    f"下游服務處理失敗: {data.get('error') or '未提供原因'}",
                    502,
                    elapsed,
                    data=data,
                )
            return ok(service_type, data, elapsed)

        # 下游 4xx 是呼叫端的問題，原樣往上帶；5xx 一律轉成 502 Bad Gateway。
        status_code = resp.status_code if 400 <= resp.status_code < 500 else 502
        return fail(
            service_type,
            f"下游服務回傳 HTTP {resp.status_code}: {resp.text[:300]}",
            status_code,
            elapsed,
        )

    except DownstreamError as e:
        elapsed = int((time.time() - t0) * 1000)
        logger.warning(f"[{service_type}] {e.message}")
        return fail(service_type, e.message, e.status_code, elapsed)
    except Exception as e:
        elapsed = int((time.time() - t0) * 1000)
        logger.exception(f"[{service_type}] 處理失敗")
        return fail(service_type, str(e), 500, elapsed)


# ─────────────────────────────────────────────
#  輔助端點（需要 admin 權限，見 security.ADMIN_PATHS）
# ─────────────────────────────────────────────

@app.post("/api/reload-config")
async def reload_config():
    """熱重載服務設定（不需重啟網關）。驗證失敗時保留原設定。"""
    try:
        load_config()
    except (ValidationError, yaml.YAMLError, OSError) as e:
        logger.error(f"設定重載失敗，維持原設定: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": f"重載失敗: {e}"},
        )
    return {"status": "ok", "services": list(CONFIG.services)}


# ===================== 入口 =====================

if __name__ == "__main__":
    import uvicorn

    load_config()
    port = int(os.getenv("GATEWAY_PORT", CONFIG.global_.gateway_port))
    try:
        trusted_hops = int(os.getenv("TRUSTED_PROXY_HOPS", "0"))
    except ValueError:
        trusted_hops = 0
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        # 在反向代理後面時，讓 uvicorn 依 X-Forwarded-* 還原來源資訊
        proxy_headers=trusted_hops > 0,
        forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
