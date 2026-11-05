from __future__ import annotations

import json

import pytest

from app.services import ttas_rule_loader
from app.services.ttas_rule_loader import TTASRuleLoadError, load_ttas_rules


_KNOWLEDGE_FILES = (
    "source_manifest.json",
    "ttas_evidence_schema.json",
    "active_rules.json",
    "evidence_catalog.json",
)


def _copy_knowledge_package(tmp_path) -> None:
    source = ttas_rule_loader.TTAS_KNOWLEDGE_DIR
    for name in _KNOWLEDGE_FILES:
        (tmp_path / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")


def test_official_ttas_package_integrity() -> None:
    ruleset = load_ttas_rules()
    assert len(ruleset.rules) == len({rule["rule_id"] for rule in ruleset.rules})
    assert all(rule["source_id"] in ruleset.sources for rule in ruleset.rules)
    assert all(1 <= rule["ttas_level"] <= 5 for rule in ruleset.rules)
    assert all(
        rule["implementation_status"] in {"enabled", "reference_only"}
        for rule in ruleset.rules
    )


def test_loader_builds_enabled_prompt_evidence_catalog() -> None:
    ruleset = load_ttas_rules()
    catalog = ruleset.prompt_evidence_catalog
    assert len(ruleset.enabled_rules) == 28
    assert len(catalog) == 24
    assert catalog["chemical_eye_injury"] == {
        "field": "chemical_eye_injury",
        "type": "true|false|unknown",
        "label_zh": "化學物質濺入／直接接觸眼睛",
        "meaning_zh": "患者原文明確表示化學性物質直接濺入、噴入或接觸眼睛。",
        "do_not_infer_zh": "只有眼痛、眼紅、灼熱、流淚或視力不適，但沒有明確化學物質接觸時，不得推測 chemical_eye_injury=true。",
        "evidence_kind": "direct_event_fact",
    }


def test_prompt_catalog_excludes_reference_only_only_fields() -> None:
    catalog = load_ttas_rules().prompt_evidence_catalog
    assert {
        "respiratory_distress",
        "hemodynamic_status",
        "cardiac_chest_pain_suspected",
        "high_risk_injury_mechanism",
    }.isdisjoint(catalog)


def test_reference_only_hypertension_rules_are_not_executable() -> None:
    ruleset = load_ttas_rules()
    reference_ids = {
        rule["rule_id"]
        for rule in ruleset.rules
        if rule["implementation_status"] == "reference_only"
    }
    enabled_ids = {rule["rule_id"] for rule in ruleset.enabled_rules}
    assert {"TTAS-A020407", "TTAS-A020409", "TTAS-A020410", "TTAS-A020411"} <= reference_ids
    assert reference_ids.isdisjoint(enabled_ids)


def test_clinical_modifier_and_untraceable_context_rules_are_reference_only() -> None:
    ruleset = load_ttas_rules()
    reference_ids = {
        rule["rule_id"]
        for rule in ruleset.rules
        if rule["implementation_status"] == "reference_only"
    }
    assert {
        "TTAS-MOD-RESP-SEVERE",
        "TTAS-MOD-RESP-MODERATE",
        "TTAS-MOD-RESP-MILD",
        "TTAS-MOD-SHOCK",
        "TTAS-MOD-HEMODYNAMIC-INSUFFICIENT",
        "TTAS-MOD-ADULT-HIGH-RISK-MECHANISM",
        "TTAS-A020210",
        "TTAS-A020211",
        "TTAS-A030712",
        "TTAS-A040411",
        "TTAS-T110107",
        "TTAS-A041011",
        "TTAS-A041017",
        "TTAS-A130409",
        "TTAS-A130413",
        "TTAS-E010809",
        "TTAS-T010109",
        "TTAS-T010110",
        "TTAS-T120207",
        "TTAS-T120707",
    } <= reference_ids


def test_stroke_rules_use_official_six_hour_boundary() -> None:
    rules = {rule["official_code"]: rule for rule in load_ttas_rules().rules}
    assert rules["A041011"]["predicate"]["all"][1] == {"lt": ["stroke_onset_hours", 6]}
    assert rules["A041017"]["predicate"]["all"][1] == {"gte": ["stroke_onset_hours", 6]}
    assert rules["A041011"]["implementation_status"] == "reference_only"
    assert rules["A041017"]["implementation_status"] == "reference_only"


def test_malformed_rules_fail_closed(tmp_path, monkeypatch) -> None:
    _copy_knowledge_package(tmp_path)
    data = json.loads((tmp_path / "active_rules.json").read_text(encoding="utf-8"))
    data["rules"][0]["source_id"] = "missing-source"
    (tmp_path / "active_rules.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(ttas_rule_loader, "TTAS_KNOWLEDGE_DIR", tmp_path)
    ttas_rule_loader.clear_ttas_rule_cache()
    with pytest.raises(TTASRuleLoadError):
        ttas_rule_loader.load_ttas_rules()
    ttas_rule_loader.clear_ttas_rule_cache()


def test_enabled_field_missing_from_catalog_fails_closed(tmp_path, monkeypatch) -> None:
    _copy_knowledge_package(tmp_path)
    data = json.loads((tmp_path / "evidence_catalog.json").read_text(encoding="utf-8"))
    del data["fields"]["chemical_eye_injury"]
    (tmp_path / "evidence_catalog.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(ttas_rule_loader, "TTAS_KNOWLEDGE_DIR", tmp_path)
    ttas_rule_loader.clear_ttas_rule_cache()
    with pytest.raises(TTASRuleLoadError, match="chemical_eye_injury"):
        ttas_rule_loader.load_ttas_rules()
    ttas_rule_loader.clear_ttas_rule_cache()


def test_catalog_unknown_source_id_fails_closed(tmp_path, monkeypatch) -> None:
    _copy_knowledge_package(tmp_path)
    data = json.loads((tmp_path / "evidence_catalog.json").read_text(encoding="utf-8"))
    data["fields"]["chemical_eye_injury"]["source_basis"]["source_ids"] = ["missing-source"]
    (tmp_path / "evidence_catalog.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(ttas_rule_loader, "TTAS_KNOWLEDGE_DIR", tmp_path)
    ttas_rule_loader.clear_ttas_rule_cache()
    with pytest.raises(TTASRuleLoadError, match="source_id"):
        ttas_rule_loader.load_ttas_rules()
    ttas_rule_loader.clear_ttas_rule_cache()


def test_catalog_unknown_rule_id_fails_closed(tmp_path, monkeypatch) -> None:
    _copy_knowledge_package(tmp_path)
    data = json.loads((tmp_path / "evidence_catalog.json").read_text(encoding="utf-8"))
    data["fields"]["chemical_eye_injury"]["source_basis"]["rule_ids"] = ["missing-rule"]
    (tmp_path / "evidence_catalog.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(ttas_rule_loader, "TTAS_KNOWLEDGE_DIR", tmp_path)
    ttas_rule_loader.clear_ttas_rule_cache()
    with pytest.raises(TTASRuleLoadError, match="rule_id"):
        ttas_rule_loader.load_ttas_rules()
    ttas_rule_loader.clear_ttas_rule_cache()


def test_malformed_catalog_fails_closed(tmp_path, monkeypatch) -> None:
    _copy_knowledge_package(tmp_path)
    data = json.loads((tmp_path / "evidence_catalog.json").read_text(encoding="utf-8"))
    data["fields"] = []
    (tmp_path / "evidence_catalog.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(ttas_rule_loader, "TTAS_KNOWLEDGE_DIR", tmp_path)
    ttas_rule_loader.clear_ttas_rule_cache()
    with pytest.raises(TTASRuleLoadError, match="catalog fields"):
        ttas_rule_loader.load_ttas_rules()
    ttas_rule_loader.clear_ttas_rule_cache()
