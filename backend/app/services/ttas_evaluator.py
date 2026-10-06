from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.schemas import (
    TTASEvaluationResult,
    TTASMatchedRule,
    TTASRuleEvidenceReference,
    TriageCase,
    UrgencyResult,
)
from app.services.ttas_evidence import effective_ttas_evidence
from app.services.ttas_rule_loader import TTASRuleLoadError, load_ttas_rules


TruthValue = Literal["true", "false", "unknown"]


@dataclass(frozen=True)
class _PredicateResult:
    truth: TruthValue
    fields: frozenset[str] = frozenset()


def evaluate_ttas(case: TriageCase) -> TTASEvaluationResult:
    try:
        ruleset = load_ttas_rules()
    except TTASRuleLoadError:
        return TTASEvaluationResult(
            status="insufficient_information",
            warnings=["TTAS official rule data is unavailable; no level was assigned."],
        )

    evidence = effective_ttas_evidence(case)
    values = {
        field: item.value
        for field, item in evidence.items()
        if item.semantic_status == "available"
    }
    population = _population(values)
    matched: list[TTASMatchedRule] = []
    missing: set[str] = set()

    for rule in ruleset.enabled_rules:
        required_population = rule.get("population")
        if required_population in {"adult", "pediatric"} and population is None:
            missing.add("age_years|age_months")
            continue
        if required_population in {"adult", "pediatric"} and population != required_population:
            continue
        result = _evaluate_predicate(rule["predicate"], values)
        if result.truth == "unknown":
            missing.update(field for field in result.fields if field not in values)
            missing.update(_missing_required_fields(rule.get("required_fields", []), values))
            continue
        if result.truth != "true":
            continue
        references = [
            TTASRuleEvidenceReference(
                field=field,
                value=evidence[field].value,
                source_text=evidence[field].source_text,
            )
            for field in sorted(result.fields)
            if field in evidence
        ]
        matched.append(
            TTASMatchedRule(
                rule_id=rule["rule_id"],
                official_code=rule.get("official_code"),
                source_id=rule["source_id"],
                official_locator=rule["official_locator"],
                level=rule["ttas_level"],
                evidence_references=references,
            )
        )

    matched.sort(key=lambda item: (item.level, item.rule_id))
    if not matched:
        return TTASEvaluationResult(
            status="insufficient_information",
            level_candidate=None,
            missing_evidence=sorted(missing),
            warnings=["No enabled official rule was completely matched; no TTAS level was assigned."],
        )
    return TTASEvaluationResult(
        status="matched",
        level_candidate=min(item.level for item in matched),
        matched_rules=matched,
        missing_evidence=sorted(missing),
    )


def apply_ttas_evaluation(case: TriageCase, *, preserve_question: bool = True) -> UrgencyResult:
    result = evaluate_ttas(case)
    case.ttas_result = result
    existing = case.triage
    warning = result.status == "matched" and result.level_candidate in {1, 2}
    reasons = [
        f"TTAS official rule matched: {item.rule_id} (level {item.level})."
        for item in result.matched_rules
    ]
    if result.status == "insufficient_information":
        reasons.append("TTAS evidence is insufficient; no urgency level was inferred or defaulted.")
    return UrgencyResult(
        urgency_score=None,
        urgency_level=None,
        warning_required=warning,
        warning_message=(
            "目前 TTAS 初步篩檢結果較急迫，請儘速由醫療人員評估；如症狀持續惡化請尋求緊急醫療協助。"
            if warning else None
        ),
        need_more_info=existing.need_more_info,
        next_question=existing.next_question if preserve_question else None,
        reasons=reasons,
        is_final=warning,
    )


def ttas_requires_immediate_action(case: TriageCase) -> bool:
    return bool(
        case.ttas_result.status == "matched"
        and case.ttas_result.level_candidate in {1, 2}
    )


def _population(values: dict[str, Any]) -> str | None:
    years = values.get("age_years")
    months = values.get("age_months")
    if type(years) in {int, float}:
        return "adult" if years >= 18 else "pediatric"
    if type(months) in {int, float}:
        return "adult" if months >= 216 else "pediatric"
    return None


def _evaluate_predicate(predicate: dict[str, Any], values: dict[str, Any]) -> _PredicateResult:
    operator, operands = next(iter(predicate.items()))
    if operator in {"all", "any"}:
        parts = [_evaluate_predicate(item, values) for item in operands]
        fields = frozenset().union(*(part.fields for part in parts))
        if operator == "all":
            if any(part.truth == "false" for part in parts):
                return _PredicateResult("false", fields)
            return _PredicateResult("unknown" if any(part.truth == "unknown" for part in parts) else "true", fields)
        if any(part.truth == "true" for part in parts):
            matched_fields = frozenset().union(*(part.fields for part in parts if part.truth == "true"))
            return _PredicateResult("true", matched_fields)
        return _PredicateResult("unknown" if any(part.truth == "unknown" for part in parts) else "false", fields)

    field = operands[0]
    if field not in values:
        return _PredicateResult("unknown", frozenset({field}))
    actual = values[field]
    try:
        if operator == "eq":
            matched = actual == operands[1] and type(actual) is type(operands[1])
        elif operator == "lt":
            matched = actual < operands[1]
        elif operator == "lte":
            matched = actual <= operands[1]
        elif operator == "gt":
            matched = actual > operands[1]
        elif operator == "gte":
            matched = actual >= operands[1]
        elif operator == "between":
            matched = operands[1] <= actual <= operands[2]
        else:
            return _PredicateResult("unknown", frozenset({field}))
    except TypeError:
        return _PredicateResult("false", frozenset({field}))
    return _PredicateResult("true" if matched else "false", frozenset({field}))


def _missing_required_fields(required: list[object], values: dict[str, Any]) -> set[str]:
    missing: set[str] = set()
    for item in required:
        if not isinstance(item, str):
            continue
        alternatives = item.split("|")
        if not any(field in values for field in alternatives):
            missing.add(item)
    return missing
