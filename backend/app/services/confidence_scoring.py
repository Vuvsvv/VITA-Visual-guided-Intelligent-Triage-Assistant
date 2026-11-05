from __future__ import annotations

ACCEPT_THRESHOLD = 0.55
MAX_QUESTION_ATTEMPTS = 2


def accepted(confidence: float, semantic_status: str) -> bool:
    return confidence >= ACCEPT_THRESHOLD and semantic_status not in {"unknown", "ambiguous"}
