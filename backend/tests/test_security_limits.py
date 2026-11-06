"""Basic abuse limits: unguessable case ids, a 100-character message cap, a per-device chat rate."""
import re

from fastapi.testclient import TestClient

from app.main import app
from app.routes.chat import MAX_MESSAGE_CHARS, MESSAGE_TOO_LONG_REPLY
from app.services import ai_service, rate_limit
from app.services.case_store import get_case, new_case_id


def test_case_id_carries_full_uuid():
    assert re.fullmatch(r"case_[0-9a-f]{32}", new_case_id())


def test_over_long_message_is_answered_without_ai_and_not_recorded(monkeypatch):
    async def no_ai(*_args, **_kwargs):
        raise AssertionError("an over-long message must not reach the AI")

    monkeypatch.setattr(ai_service, "_complete_cerebras", no_ai)
    response = TestClient(app).post("/chat", json={"message": "頭" * (MAX_MESSAGE_CHARS + 1)})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == MESSAGE_TOO_LONG_REPLY
    assert body["needMoreInfo"] is True
    case = get_case(body["case_id"])
    assert case is not None
    assert not [item for item in case.history_records if item.role == "user"]


def test_message_at_the_limit_is_accepted():
    response = TestClient(app).post("/chat", json={"message": "頭" * MAX_MESSAGE_CHARS})

    assert response.status_code == 200
    assert response.json()["reply"] != MESSAGE_TOO_LONG_REPLY


def test_surrounding_spaces_do_not_count_toward_the_limit():
    padded = "  " + "頭" * MAX_MESSAGE_CHARS + "  "
    response = TestClient(app).post("/chat", json={"message": padded})

    assert response.json()["reply"] != MESSAGE_TOO_LONG_REPLY


def test_chat_rate_limit_blocks_the_eleventh_request_in_a_minute(monkeypatch):
    monkeypatch.setattr(rate_limit, "_EXEMPT_HOSTS", {"127.0.0.1", "::1", "localhost"})
    client = TestClient(app)
    statuses = [
        client.post("/chat", json={"message": "頭" * (MAX_MESSAGE_CHARS + 1)}).status_code
        for _ in range(rate_limit.CHAT_RATE_LIMIT_PER_MINUTE)
    ]
    blocked = client.post("/chat", json={"message": "頭痛"})

    assert statuses == [200] * rate_limit.CHAT_RATE_LIMIT_PER_MINUTE
    assert blocked.status_code == 429
    assert blocked.json()["detail"] == rate_limit.RATE_LIMITED_DETAIL


def test_rate_window_reopens_after_a_minute_and_is_per_device():
    limit = rate_limit.CHAT_RATE_LIMIT_PER_MINUTE
    assert all(rate_limit.allow("10.0.0.1", now=float(i)) for i in range(limit))
    assert not rate_limit.allow("10.0.0.1", now=30.0)
    assert rate_limit.allow("10.0.0.2", now=30.0)
    assert rate_limit.allow("10.0.0.1", now=60.0)


def test_health_route_is_not_rate_limited(monkeypatch):
    monkeypatch.setattr(rate_limit, "_EXEMPT_HOSTS", set())
    client = TestClient(app)
    assert all(client.get("/health").status_code == 200 for _ in range(rate_limit.CHAT_RATE_LIMIT_PER_MINUTE + 2))
