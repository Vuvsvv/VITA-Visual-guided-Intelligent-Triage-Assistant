from __future__ import annotations

import pytest

from app.schemas import Message, TTASEvidence, TriageCase
from app.services import ttas_evaluator
from app.services.ttas_evaluator import evaluate_ttas
from app.services.ttas_rule_loader import TTASRuleLoadError


def _case(*facts: tuple[str, object], age: float | None = None) -> TriageCase:
    case = TriageCase(case_id="ttas", history_records=[Message(role="user", content="reported")])
    case.patient_input.age_years = age
    case.ttas_evidence = [
        TTASEvidence(
            field=field,
            value=value,
            semantic_status="available",
            confidence=0.95,
            source_text="reported",
        )
        for field, value in facts
    ]
    return case


@pytest.mark.parametrize(
    ("facts", "expected_level"),
    [
        (("gcs", 10), 2),
        (("temperature_c", 42.0), 1),
    ],
)
def test_modifiers_match_official_levels(facts, expected_level) -> None:
    result = evaluate_ttas(_case(facts))
    assert result.status == "matched"
    assert result.level_candidate == expected_level


def test_multiple_matches_choose_most_urgent_and_keep_trace() -> None:
    result = evaluate_ttas(
        _case(("temperature_c", 42.0), ("chemical_eye_injury", True))
    )
    assert result.level_candidate == 1
    assert {item.rule_id for item in result.matched_rules} >= {
        "TTAS-MOD-TEMP-HIGH-EXTREME",
        "TTAS-E010509",
    }
    assert all(item.source_id and item.official_locator for item in result.matched_rules)


@pytest.mark.parametrize(
    ("official_code", "facts", "level"),
    [
        ("A040504", (("seizure_status", "ongoing"),), 1),
        ("A040511", (("seizure_status", "stopped_not_recovered"),), 2),
        ("A040516", (("seizure_status", "stopped_recovered"),), 3),
        ("A030713", (("vomiting_pattern", "acute_persistent"),), 3),
        ("E010509", (("chemical_eye_injury", True),), 2),
    ],
)
def test_official_code_regressions(official_code, facts, level) -> None:
    result = evaluate_ttas(_case(*facts, age=30))
    matches = [item for item in result.matched_rules if item.official_code == official_code]
    assert matches and matches[0].level == level


def test_toxic_gas_rule_is_reference_only_without_objective_respiration_derivation() -> None:
    result = evaluate_ttas(
        _case(("toxic_gas_exposure", True), ("respiratory_distress", "none"))
    )
    assert result.status == "insufficient_information"
    assert not any(item.official_code == "E010809" for item in result.matched_rules)


def test_open_fracture_and_suspected_deformity_require_exact_limb_region() -> None:
    open_result = evaluate_ttas(_case(("open_fracture", True), ("injury_region", "upper_limb")))
    suspected_result = evaluate_ttas(
        _case(
            ("suspected_fracture_or_dislocation_deformity", True),
            ("injury_region", "lower_limb"),
        )
    )
    assert open_result.level_candidate == 2
    assert suspected_result.level_candidate == 3
    assert {item.official_code for item in open_result.matched_rules} == {"T120110"}
    assert {item.official_code for item in suspected_result.matched_rules} == {"T120617"}


@pytest.mark.parametrize(
    ("field", "region", "expected_code", "forbidden_code"),
    [
        ("open_fracture", "upper_limb", "T120110", "T120610"),
        ("open_fracture", "lower_limb", "T120610", "T120110"),
        ("suspected_fracture_or_dislocation_deformity", "upper_limb", "T120116", "T120617"),
        ("suspected_fracture_or_dislocation_deformity", "lower_limb", "T120617", "T120116"),
    ],
)
def test_trauma_trace_never_crosses_limb_region(field, region, expected_code, forbidden_code) -> None:
    result = evaluate_ttas(_case((field, True), ("injury_region", region)))
    codes = {item.official_code for item in result.matched_rules}
    assert expected_code in codes
    assert forbidden_code not in codes


def test_fracture_without_region_cannot_emit_an_official_limb_code() -> None:
    result = evaluate_ttas(_case(("open_fracture", True)))
    assert result.status == "insufficient_information"
    assert not {"T120110", "T120610"} & {item.official_code for item in result.matched_rules}


def test_insect_rules_require_grounded_complaint_context() -> None:
    without_context = evaluate_ttas(_case(("generalized_rash_or_blisters", True)))
    with_context = evaluate_ttas(
        _case(("generalized_rash_or_blisters", True), ("insect_sting_exposure", True))
    )
    assert not any(item.official_code == "E010112" for item in without_context.matched_rules)
    assert any(item.official_code == "E010112" for item in with_context.matched_rules)


def test_penetration_codes_are_reference_only_without_exact_complaint_context() -> None:
    result = evaluate_ttas(
        _case(("penetrating_injury_high_region", True), ("injury_region", "upper_limb"))
    )
    assert not {"T010110", "T120207", "T120707"} & {
        item.official_code for item in result.matched_rules
    }


def test_subjective_respiratory_modifier_cannot_assign_level() -> None:
    result = evaluate_ttas(_case(("respiratory_distress", "severe")))
    assert result.status == "insufficient_information"
    assert result.level_candidate is None


@pytest.mark.parametrize(
    ("field", "official_code"),
    [
        ("tearing_pain", "A020211"),
        ("acute_visual_disturbance", "A040411"),
        ("genital_swelling_deformity", "T110107"),
        ("coffee_ground_emesis_or_melena", "A030712"),
    ],
)
def test_complaint_context_gaps_are_reference_only(field: str, official_code: str) -> None:
    """A fact without its official complaint context cannot emit that code.

    A030712 is also unsafe because the stored crosswalk distinguishes the
    A0307 nausea/vomiting complaint from black-stool rules under A0310, while
    the current evidence field merges coffee-ground emesis and melena.
    """
    result = evaluate_ttas(_case((field, True), age=30))
    assert all(item.official_code != official_code for item in result.matched_rules)


def test_spo2_has_no_invented_threshold_when_package_does_not_supply_one() -> None:
    result = evaluate_ttas(_case(("spo2_pct", 82)))
    assert result.status == "insufficient_information"
    assert result.level_candidate is None


def test_ai_classified_consciousness_cannot_replace_objective_gcs() -> None:
    result = evaluate_ttas(_case(("consciousness", "altered")))
    assert result.status == "insufficient_information"
    assert result.level_candidate is None


@pytest.mark.parametrize(
    "facts",
    [
        (("blood_glucose_mg_dl", 55), ("hypoglycemia_symptoms", True)),
        (("blood_glucose_mg_dl", 55), ("hypoglycemia_symptoms", False)),
        (("stroke_symptoms", True), ("stroke_onset_hours", 5.9)),
        (("stroke_symptoms", True), ("stroke_onset_hours", 6.0)),
    ],
)
def test_clinical_grouping_rules_are_reference_only(facts) -> None:
    result = evaluate_ttas(_case(*facts, age=30))
    assert result.status == "insufficient_information"
    assert result.level_candidate is None


def test_missing_vital_and_no_match_never_default_to_four_or_five() -> None:
    missing = evaluate_ttas(_case(("generalized_rash_or_blisters", True)))
    no_match = evaluate_ttas(_case(("respiratory_distress", "none")))
    assert missing.status == "insufficient_information"
    assert missing.level_candidate is None
    assert "insect_sting_exposure" in missing.missing_evidence
    assert no_match.status == "insufficient_information"
    assert no_match.level_candidate is None


def test_age_dependent_rules_do_not_assume_adult() -> None:
    unknown_age = evaluate_ttas(_case(("temperature_c", 39.0)))
    pediatric = evaluate_ttas(_case(("age_months", 2), ("temperature_c", 39.0)))
    assert unknown_age.level_candidate is None
    assert "age_years|age_months" in unknown_age.missing_evidence
    assert pediatric.level_candidate == 2


def test_reference_only_blood_pressure_rule_never_executes() -> None:
    result = evaluate_ttas(_case(("sbp_mmhg", 210), ("dbp_mmhg", 120), age=30))
    assert result.status == "insufficient_information"
    assert result.level_candidate is None
    assert not any((item.official_code or "").startswith("A0204") for item in result.matched_rules)


def test_raw_history_is_not_an_evaluator_input() -> None:
    case = TriageCase(
        case_id="raw-isolation",
        history_records=[Message(role="user", content="我喘得很嚴重而且正在抽搐")],
    )
    result = evaluate_ttas(case)
    assert result.status == "insufficient_information"
    assert result.level_candidate is None


def test_loader_failure_makes_evaluator_fail_closed(monkeypatch) -> None:
    def fail():
        raise TTASRuleLoadError("do not expose upstream detail")

    monkeypatch.setattr(ttas_evaluator, "load_ttas_rules", fail)
    result = evaluate_ttas(_case(("respiratory_distress", "severe")))
    assert result.status == "insufficient_information"
    assert result.level_candidate is None
    assert result.matched_rules == []
