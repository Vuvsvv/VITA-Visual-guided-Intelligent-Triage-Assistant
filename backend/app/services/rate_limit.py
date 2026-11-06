"""Per-device request cap for the AI-costly chat endpoints.

Each /chat or /voice/chat request can trigger several paid AI calls, and the backend has
no login, so anyone who reaches it could drain the AI quota with a loop. A device (client
IP) may send at most ``CHAT_RATE_LIMIT_PER_MINUTE`` such requests in any 60 seconds.
Loopback callers (scripts run on the backend machine itself) are not limited.
The state is in memory, like the case store: it resets when the backend restarts.
"""
import threading
import time
from collections import deque

from fastapi import HTTPException, Request

CHAT_RATE_LIMIT_PER_MINUTE = 10
WINDOW_SECONDS = 60.0
RATE_LIMITED_DETAIL = "您傳送得太快了，請稍等一分鐘再傳。"
_EXEMPT_HOSTS = {"127.0.0.1", "::1", "localhost"}

_requests: dict[str, deque[float]] = {}
_lock = threading.Lock()
_clock = time.monotonic


def reset() -> None:
    with _lock:
        _requests.clear()


def allow(client_key: str, now: float | None = None) -> bool:
    current = _clock() if now is None else now
    with _lock:
        stamps = _requests.setdefault(client_key, deque())
        while stamps and current - stamps[0] >= WINDOW_SECONDS:
            stamps.popleft()
        if len(stamps) >= CHAT_RATE_LIMIT_PER_MINUTE:
            return False
        stamps.append(current)
        # Drop idle devices so the table cannot grow without bound.
        if len(_requests) > 10_000:
            for key in [key for key, value in _requests.items() if not value or current - value[-1] >= WINDOW_SECONDS]:
                _requests.pop(key, None)
        return True


async def enforce_chat_rate_limit(request: Request) -> None:
    host = request.client.host if request.client else "unknown"
    if host in _EXEMPT_HOSTS:
        return
    if not allow(host):
        raise HTTPException(status_code=429, detail=RATE_LIMITED_DETAIL)
