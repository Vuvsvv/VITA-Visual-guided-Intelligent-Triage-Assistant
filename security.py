"""
安全中間件模組
==============
為 API 網關提供：
  1. API Key 認證（含權限）
  2. 速率限制（per-key + per-IP）
  3. 檔案上傳驗證（大小 + magic bytes）
  4. 輸入文字過濾（長度限制 + prompt injection 防護）
  5. 請求日誌與稽核

CORS 不在這裡處理 —— 由 gateway.py 的 CORSMiddleware 負責，且必須掛在本中間件
「之外」（後 add_middleware 的在外層），否則 preflight 會先被 401 擋掉。
"""

import asyncio
import os
import time
import hashlib
import logging
import secrets
from typing import Optional, Dict, Set, List
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("security")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# =====================================================
#  1. API Key 認證
# =====================================================

#: 未指定權限時的預設值 —— 只能打一般處理端點，不能做管理操作。
DEFAULT_PERMISSIONS = ["process"]

#: 需要 admin 權限的路徑（見 C9）。
ADMIN_PATHS: Set[str] = {"/api/reload-config"}


class APIKeyAuth:
    """
    API Key 認證管理器

    用法：
        auth = APIKeyAuth()
        auth.load_keys_from_env()               # 從環境變數載入
        auth.add_key("my-key", "後端A")          # 或手動加，預設 permissions=["process"]
        auth.add_key("ops-key", "維運", ["*"])   # 管理金鑰

    驗證：
        key_info = auth.validate("my-key")      # 成功回傳 dict，失敗回傳 None
        auth.has_permission(key_info, "admin")  # 權限判斷
    """

    def __init__(self):
        # key_hash → metadata
        self._keys: Dict[str, Dict] = {}
        # 不需要認證的路徑
        self.public_paths: Set[str] = {
            "/",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/health",
        }

    def _hash_key(self, key: str) -> str:
        """用 SHA-256 雜湊儲存 key（不存明文）"""
        return hashlib.sha256(key.encode()).hexdigest()

    def add_key(self, key: str, name: str, permissions: Optional[List[str]] = None):
        """新增一組 API key。permissions 預設只有 'process'（最小權限）。"""
        h = self._hash_key(key)
        self._keys[h] = {
            "key_id": h,
            "name": name,
            "permissions": list(permissions) if permissions else list(DEFAULT_PERMISSIONS),
            "created_at": _utcnow(),
        }
        logger.info(f"已新增 API key: {name} (permissions={self._keys[h]['permissions']})")

    def generate_key(self, name: str, permissions: Optional[List[str]] = None) -> str:
        """產生一組安全的隨機 API key"""
        key = f"sk-{secrets.token_urlsafe(32)}"
        self.add_key(key, name, permissions)
        return key

    def load_keys_from_env(self):
        """
        從環境變數載入 API keys

        單一 key：
            API_KEY=your-key
            API_KEY_PERMISSIONS=*          # 選填，預設 '*'（管理金鑰）

        多組 keys：
            API_KEYS=key1:name1,key2:name2:perm1|perm2
            （沒寫第三段就是最小權限 process，打不了 /api/reload-config）
        """
        single = os.getenv("API_KEY")
        if single:
            perms = _parse_permissions(os.getenv("API_KEY_PERMISSIONS", "*"))
            self.add_key(single, "default", perms)

        multi = os.getenv("API_KEYS", "")
        for pair in multi.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if ":" in pair:
                parts = pair.split(":", 2)
                key, name = parts[0].strip(), parts[1].strip()
                perms = _parse_permissions(parts[2]) if len(parts) > 2 else None
                self.add_key(key, name, perms)
            else:
                self.add_key(pair, "unnamed")

    def validate(self, key: str) -> Optional[Dict]:
        """驗證 API key，成功回傳 metadata，失敗回傳 None"""
        if not key:
            return None
        h = self._hash_key(key)
        return self._keys.get(h)

    @staticmethod
    def has_permission(key_info: Optional[Dict], required: str) -> bool:
        """檢查某把 key 是否具備指定權限。'*' 代表全部。"""
        if not key_info:
            return False
        perms = key_info.get("permissions") or []
        return "*" in perms or required in perms

    def has_keys(self) -> bool:
        """是否有設定任何 key"""
        return len(self._keys) > 0


def _parse_permissions(raw: str) -> List[str]:
    """把 'admin|process' 或 '*' 解析成 list。"""
    return [p.strip() for p in raw.replace(",", "|").split("|") if p.strip()] or list(
        DEFAULT_PERMISSIONS
    )


# =====================================================
#  2. 速率限制
# =====================================================

class RateLimiter:
    """
    簡易滑動視窗速率限制器

    用法：
        limiter = RateLimiter(max_requests=60, window_seconds=60)
        allowed, remaining = limiter.check("client-id")

    注意：`_requests` 是 defaultdict，未認證路徑人人可打，所以一定要有人定期呼叫
    `cleanup()`，否則長跑服務會慢性記憶體洩漏（見 C7）。gateway 的 lifespan 會啟動
    `start_cleanup_task()`。
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        # client_id → list of timestamps
        self._requests: Dict[str, list] = defaultdict(list)
        self._cleanup_task: Optional[asyncio.Task] = None

    def check(self, client_id: str) -> tuple[bool, int]:
        """
        檢查是否允許請求

        Returns:
            (allowed: bool, remaining: int)
        """
        now = time.time()
        cutoff = now - self.window

        # 清除過期紀錄
        timestamps = self._requests[client_id]
        self._requests[client_id] = [t for t in timestamps if t > cutoff]

        current_count = len(self._requests[client_id])

        if current_count >= self.max_requests:
            return False, 0

        self._requests[client_id].append(now)
        remaining = self.max_requests - current_count - 1
        return True, remaining

    def cleanup(self):
        """清理過期紀錄並移除空桶（記憶體管理）"""
        now = time.time()
        cutoff = now - self.window
        empty_keys = []
        for key, timestamps in self._requests.items():
            self._requests[key] = [t for t in timestamps if t > cutoff]
            if not self._requests[key]:
                empty_keys.append(key)
        for key in empty_keys:
            del self._requests[key]
        if empty_keys:
            logger.debug(f"RateLimiter 清掉 {len(empty_keys)} 個閒置桶")

    def bucket_count(self) -> int:
        """目前有多少個計數桶（測試 / 監控用）"""
        return len(self._requests)

    def start_cleanup_task(self, interval_seconds: Optional[int] = None) -> asyncio.Task:
        """啟動背景清理任務；重複呼叫只會有一個。"""
        if self._cleanup_task and not self._cleanup_task.done():
            return self._cleanup_task

        interval = interval_seconds or int(os.getenv("RATE_LIMIT_CLEANUP_INTERVAL", "60"))

        async def _loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    self.cleanup()
                except Exception:  # pragma: no cover - 清理失敗不該讓服務掛掉
                    logger.exception("RateLimiter 清理失敗")

        self._cleanup_task = asyncio.create_task(_loop(), name="rate-limiter-cleanup")
        return self._cleanup_task

    async def stop_cleanup_task(self):
        """停止背景清理任務（lifespan 結束時呼叫）。"""
        task = self._cleanup_task
        if not task or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# =====================================================
#  3. 檔案驗證
# =====================================================

# 常見音訊格式的 magic bytes
AUDIO_SIGNATURES = {
    b"RIFF": "wav",           # WAV
    b"\xff\xfb": "mp3",      # MP3 (MPEG)
    b"\xff\xf3": "mp3",      # MP3 (MPEG)
    b"\xff\xf2": "mp3",      # MP3 (MPEG)
    b"ID3": "mp3",           # MP3 with ID3 tag
    b"fLaC": "flac",         # FLAC
    b"OggS": "ogg",          # OGG
    b"\x00\x00\x00": "m4a",  # M4A/MP4 (partial)
}

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
#: multipart 封裝與其他表單欄位的額外空間
MAX_REQUEST_SIZE = MAX_FILE_SIZE + 1024 * 1024


def validate_audio_file(content: bytes, filename: str = "") -> tuple[bool, str]:
    """
    驗證音訊檔案

    Returns:
        (valid: bool, error_message: str)
    """
    # 檢查大小
    if len(content) == 0:
        return False, "檔案為空"

    if len(content) > MAX_FILE_SIZE:
        size_mb = len(content) / (1024 * 1024)
        return False, f"檔案過大: {size_mb:.1f}MB（上限 {MAX_FILE_SIZE // (1024*1024)}MB）"

    # 檢查 magic bytes
    is_valid_audio = False
    for signature, fmt in AUDIO_SIGNATURES.items():
        if content[:len(signature)] == signature:
            is_valid_audio = True
            break

    # 也檢查副檔名作為輔助判斷
    valid_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm"}
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            is_valid_audio = True

    if not is_valid_audio:
        return False, "不支援的檔案格式（支援: wav, mp3, flac, ogg, m4a）"

    return True, ""


# =====================================================
#  4. 輸入過濾
# =====================================================

# 常見 prompt injection 模式
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard your instructions",
    "forget your instructions",
    "你是一個",
    "假裝你是",
    "pretend you are",
    "act as if",
    "system prompt",
    "reveal your prompt",
]


def sanitize_text_input(text: str) -> tuple[str, list]:
    """
    清理文字輸入

    Returns:
        (cleaned_text: str, warnings: list)
    """
    warnings = []

    if not text or not text.strip():
        return "", ["輸入為空"]

    # 長度限制
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
        warnings.append(f"文字已截斷至 {MAX_TEXT_LENGTH} 字元")

    # 移除控制字元（保留換行和空格）
    cleaned = ""
    for ch in text:
        if ch in ("\n", "\r", "\t") or (ord(ch) >= 32):
            cleaned += ch
    text = cleaned

    # 檢查 prompt injection（只記錄警告，不阻擋）
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if pattern in text_lower:
            warnings.append(f"偵測到可疑輸入模式: {pattern[:20]}...")
            logger.warning(f"Prompt injection 嘗試: {text[:100]}")
            break

    return text.strip(), warnings


# =====================================================
#  5. 來源 IP
# =====================================================

def client_ip_from(request: Request, trusted_hops: int) -> str:
    """
    取得呼叫端 IP。

    直接讀 `request.client.host` 在 Nginx 後面永遠是 127.0.0.1，所有匿名請求會共用
    同一個速率限制桶（見 C8）。但 X-Forwarded-For 是使用者可偽造的 header，所以只有
    在明確設定 `TRUSTED_PROXY_HOPS`（前面有幾層自己信任的 proxy）時才採用，並且只取
    「最外層信任 proxy 交給我們的那一跳」。
    """
    direct = request.client.host if request.client else "unknown"
    if trusted_hops <= 0:
        return direct

    raw = request.headers.get("X-Forwarded-For", "")
    hops = [h.strip() for h in raw.split(",") if h.strip()]
    if not hops:
        return direct
    # hops[-1] 是最靠近我們的那一跳；往前數 trusted_hops 個就是最外層信任 proxy 看到的來源
    index = max(0, len(hops) - trusted_hops)
    return hops[index]


# =====================================================
#  6. 安全中間件（整合以上所有功能）
# =====================================================

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    FastAPI 安全中間件

    用法：
        app.add_middleware(SecurityMiddleware,
            api_key_auth=auth,
            rate_limiter=limiter)

    ⚠️ CORSMiddleware 必須在本中間件「之後」add_middleware（＝在外層），
       否則瀏覽器 preflight 會被這裡的 401 擋掉（見 C1）。
    """

    def __init__(
        self,
        app,
        api_key_auth: APIKeyAuth = None,
        rate_limiter: RateLimiter = None,
        enable_logging: bool = True,
        trusted_proxy_hops: Optional[int] = None,
        admin_paths: Optional[Set[str]] = None,
    ):
        super().__init__(app)
        self.auth = api_key_auth or APIKeyAuth()
        self.limiter = rate_limiter or RateLimiter()
        self.enable_logging = enable_logging
        self.trusted_proxy_hops = (
            int(os.getenv("TRUSTED_PROXY_HOPS", "0"))
            if trusted_proxy_hops is None
            else trusted_proxy_hops
        )
        self.admin_paths = set(admin_paths) if admin_paths is not None else set(ADMIN_PATHS)

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # ── 0. CORS preflight 直接放行 ──
        # 瀏覽器發 OPTIONS 時不會帶 X-API-Key，擋掉的話所有跨域請求都到不了（C1）。
        # 只放行「真的是 preflight」的 OPTIONS（帶 Access-Control-Request-Method），
        # 否則等於開了一條不用認證、也不算速率的萬用通道。
        if request.method == "OPTIONS" and "access-control-request-method" in request.headers:
            return await call_next(request)

        client_ip = client_ip_from(request, self.trusted_proxy_hops)
        path = request.url.path
        key_info = None

        # ── 0b. 請求體大小上限 ──
        # 在讀取 body 之前就擋掉，才不會先把整包 50MB+ 收進記憶體再說（C13）。
        # 沒有 Content-Length（chunked）時擋不到，端點裡還有一層分塊讀取的保險。
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_REQUEST_SIZE:
            logger.warning(f"請求體過大 | {declared} bytes | IP: {client_ip} | Path: {path}")
            return JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "error": f"請求體超過大小限制 ({MAX_FILE_SIZE // (1024 * 1024)}MB)",
                },
            )

        # ── 1. API Key 認證 ──
        if self.auth.has_keys() and path not in self.auth.public_paths:
            api_key = (
                request.headers.get("X-API-Key")
                or request.headers.get("Authorization", "").removeprefix("Bearer ")
            )

            key_info = self.auth.validate(api_key)
            if key_info is None:
                if self.enable_logging:
                    logger.warning(f"認證失敗 | IP: {client_ip} | Path: {path}")
                return JSONResponse(
                    status_code=401,
                    content={
                        "success": False,
                        "error": "無效的 API Key。請在 X-API-Key header 提供有效的金鑰。"
                    }
                )

            # 把 key info 存到 request state 供後續使用
            request.state.api_key_name = key_info["name"]
            request.state.api_key_id = key_info["key_id"]

            # ── 1b. 權限檢查（C9）──
            if path in self.admin_paths and not self.auth.has_permission(key_info, "admin"):
                logger.warning(f"權限不足 | Key: {key_info['name']} | Path: {path}")
                return JSONResponse(
                    status_code=403,
                    content={
                        "success": False,
                        "error": (
                            f"此金鑰沒有 admin 權限，無法存取 {path}。"
                            "請用 API_KEYS=<key>:<名稱>:admin 設定管理金鑰。"
                        ),
                    },
                )

        # ── 2. 速率限制 ──
        # 用 key 的 hash 分桶（不是 name —— 兩把同名的 key 會共用配額，見 C8），
        # 未認證的請求則用來源 IP。
        if key_info is not None:
            rate_key = f"key:{key_info['key_id'][:16]}"
        else:
            rate_key = f"ip:{client_ip}"
        allowed, remaining = self.limiter.check(rate_key)

        if not allowed:
            if self.enable_logging:
                logger.warning(f"速率限制 | Bucket: {rate_key} | IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error": "請求過於頻繁，請稍後再試。"
                },
                headers={
                    "Retry-After": str(self.limiter.window),
                    "X-RateLimit-Remaining": "0",
                }
            )

        # ── 3. 處理請求 ──
        response = await call_next(request)

        # ── 4. 加上安全 headers ──
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Cache-Control"] = "no-store"

        # ── 5. 請求日誌 ──
        if self.enable_logging:
            elapsed = int((time.time() - start_time) * 1000)
            key_name = getattr(request.state, "api_key_name", "anonymous")
            logger.info(
                f"{request.method} {path} | "
                f"{response.status_code} | "
                f"{elapsed}ms | "
                f"Key: {key_name} | "
                f"IP: {client_ip}"
            )

        return response


# =====================================================
#  7. 工具函式
# =====================================================

def setup_security(app, config: dict = None):
    """
    一鍵設定所有安全功能

    用法：
        from security import setup_security
        auth, limiter = setup_security(app, {
            "rate_limit": 60,           # 每分鐘最大請求數
            "rate_window": 60,          # 速率限制視窗（秒）
            "enable_logging": True,
        })

    ⚠️ 呼叫順序：先 setup_security(app)，再 app.add_middleware(CORSMiddleware, ...)，
       這樣 CORS 才會在最外層（見 C1）。
    """
    config = config or {}

    # 建立認證管理器
    auth = APIKeyAuth()
    auth.load_keys_from_env()

    # 如果沒有設定任何 key，自動產生一組並印出
    if not auth.has_keys():
        generated = auth.generate_key("auto-generated", ["*"])
        logger.warning("=" * 60)
        logger.warning("  未偵測到 API_KEY 環境變數")
        logger.warning(f"  已自動產生 API Key: {generated}")
        logger.warning("  請儲存此 key 並設定到你的後端")
        logger.warning("  設定方式: export API_KEY=<key>")
        logger.warning("=" * 60)

    # 建立速率限制器
    limiter = RateLimiter(
        max_requests=config.get("rate_limit", 60),
        window_seconds=config.get("rate_window", 60),
    )

    # 安裝中間件
    app.add_middleware(
        SecurityMiddleware,
        api_key_auth=auth,
        rate_limiter=limiter,
        enable_logging=config.get("enable_logging", True),
        trusted_proxy_hops=config.get("trusted_proxy_hops"),
    )

    return auth, limiter
