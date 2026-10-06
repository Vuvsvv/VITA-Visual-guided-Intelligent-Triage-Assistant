from __future__ import annotations

CLARIFICATION_THRESHOLD = 0.55
ACCEPT_THRESHOLD = 0.55
MAX_QUESTION_ATTEMPTS = 2


def clamp_confidence(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(float(value), 1.0))


def needs_clarification(confidence: float, semantic_status: str) -> bool:
    return semantic_status in {"unknown", "ambiguous"} or confidence < CLARIFICATION_THRESHOLD


def accepted(confidence: float, semantic_status: str) -> bool:
    return confidence >= ACCEPT_THRESHOLD and semantic_status not in {"unknown", "ambiguous"}


def confidence_with_uncertainty(base: float, text: str) -> float:
    lowered = text.lower()
    uncertain_terms = ["吧", "可能", "應該", "大概", "不確定", "也許", "好像"]
    penalty = 0.15 if any(term in lowered for term in uncertain_terms) else 0.0
    return clamp_confidence(base - penalty)
