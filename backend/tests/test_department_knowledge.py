from copy import deepcopy
import json
from pathlib import Path

import pytest

from app.services.department_knowledge import (
    canonical_department_id,
    exact_db_department_ids,
    load_department_canonical_mapping,
    load_department_knowledge,
    lookup_concept,
    resolve_department_names,
    validate_department_canonical_mapping,
    validate_department_knowledge,
)


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = BACKEND_ROOT / "AI_Source" / "phase3_kb_migration_package"
REQUIRED_PACKAGE_FILES = {
    "phase3_kb_migration_763.json",
    "phase3_kb_migration_763.csv",
    "phase3_department_canonical_mapping.json",
    "phase3_source_registry.json",
    "phase3_kb_production_seed.json",
    "PHASE3_KB_MIGRATION_REPORT.md",
}


@pytest.fixture
def knowledge():
    return load_department_knowledge()


@pytest.fixture
def mappings():
    return load_department_canonical_mapping()


def test_required_reviewed_migration_package_is_complete():
    assert PACKAGE_DIR.is_dir(), (
        f"reviewed Phase 3 migration package is missing: {PACKAGE_DIR}"
    )
    assert REQUIRED_PACKAGE_FILES <= {path.name for path in PACKAGE_DIR.iterdir()}

def test_production_knowledge_has_exact_package_lineage(knowledge, mappings):
    assert PACKAGE_DIR.is_dir(), (
        f"reviewed Phase 3 migration package is missing: {PACKAGE_DIR}"
    )
    sources, records = knowledge
    source_package = json.loads(
        (PACKAGE_DIR / "phase3_source_registry.json").read_text(encoding="utf-8")
    )
    mapping_package = json.loads(
        (PACKAGE_DIR / "phase3_department_canonical_mapping.json").read_text(encoding="utf-8")
    )
    seed = json.loads(
        (PACKAGE_DIR / "phase3_kb_production_seed.json").read_text(encoding="utf-8")
    )

    assert mappings == mapping_package["mappings"]
    package_sources = {source["source_id"]: source for source in source_package["sources"]}
    assert {source["source_id"] for source in sources} == {
        source_id for source_id, source in package_sources.items()
        if source["source_url"].startswith("https://")
    }
    for source in sources:
        packaged = package_sources[source["source_id"]]
        assert all(source[field] == packaged[field] for field in (
            "source_hospital", "source_type", "source_title", "source_url",
            "source_priority", "retrieved_at",
        ))

    expanded_seed = {
        (
            row["legacy_department"], row["canonical_dept_id"],
            row["canonical_concept"], row["classification"], source_id,
        )
        for row in seed["routing_records"]
        for source_id in row["supporting_source_ids"]
    }
    production = {
        (
            record["department_name"], record["canonical_dept_id"],
            record["concept"], record["evidence_type"], record["source_id"],
        )
        for record in records
    }
    assert production == expanded_seed
    assert len(seed["semantic_synonyms"]) == 18
    assert len(seed["legacy_unverified"]) == 222


def test_migration_package_production_inventory_is_loaded(knowledge, mappings):
    sources, records = knowledge

    assert len(sources) == 18
    assert len(records) == 541
    assert len(mappings) == 35
    assert len({source["source_id"] for source in sources}) == len(sources)
    assert {record["evidence_type"] for record in records} == {
        "direct_routing_evidence",
        "department_specialty_evidence",
    }
    assert all(record["source_hospital"] in {"vghtpe", "vghtc"} for record in records)
    assert all(record["source_url"].startswith("https://") for record in records)
    assert all(record["source_id"] != "app_sql_inventory_2026_09_29" for record in records)


def test_records_match_reviewed_canonical_mapping(knowledge, mappings):
    _, records = knowledge
    by_name = {mapping["legacy_department_name"]: mapping for mapping in mappings}

    for record in records:
        mapping = by_name[record["department_name"]]
        assert record["canonical_dept_id"] == mapping["canonical_dept_id"]
        assert record["canonical_parent_dept"] == mapping["canonical_parent_dept"]
        assert record["canonical_child_dept"] == mapping["canonical_child_dept"]
        assert record["mapping_method"] == mapping["mapping_method"]
        assert set(record["official_department_labels"]) == set(
            mapping["official_source_department_labels"]
        )


def test_only_official_registered_sources_are_accepted(knowledge, mappings):
    sources, records = deepcopy(knowledge)
    records[0]["source_id"] = "legacy_dept_keywords"
    with pytest.raises(ValueError, match="not registered"):
        validate_department_knowledge(sources, records, mappings)

    sources, records = deepcopy(knowledge)
    records[0].pop("source_id")
    with pytest.raises(ValueError, match="incomplete"):
        validate_department_knowledge(sources, records, mappings)

    sources, records = deepcopy(knowledge)
    records[0]["source_id"] = ["not-a-source-id"]
    with pytest.raises(ValueError, match="must be a string"):
        validate_department_knowledge(sources, records, mappings)


def test_source_policy_rejects_unknown_hospital_host_and_non_https(knowledge, mappings):
    for mutation in (
        {"source_hospital": "unknown"},
        {"source_url": "https://example.com/not-official"},
        {"source_url": "http://www.vghtpe.gov.tw/not-https"},
    ):
        sources, records = deepcopy(knowledge)
        sources[0].update(mutation)
        with pytest.raises(ValueError):
            validate_department_knowledge(sources, records, mappings)


def test_provenance_mismatch_and_nonrouting_evidence_are_rejected(knowledge, mappings):
    sources, records = deepcopy(knowledge)
    records[0]["source_priority"] = 99
    with pytest.raises(ValueError, match="provenance"):
        validate_department_knowledge(sources, records, mappings)

    sources, records = deepcopy(knowledge)
    records[0]["evidence_type"] = "legacy_synonym"
    with pytest.raises(ValueError, match="non-routing evidence"):
        validate_department_knowledge(sources, records, mappings)

    sources, records = deepcopy(knowledge)
    records[0]["evidence_type"] = "legacy_unverified"
    with pytest.raises(ValueError, match="non-routing evidence"):
        validate_department_knowledge(sources, records, mappings)


def test_duplicate_sources_records_and_mapping_names_are_rejected(knowledge, mappings):
    sources, records = deepcopy(knowledge)
    sources.append(deepcopy(sources[0]))
    with pytest.raises(ValueError, match="duplicate source_id"):
        validate_department_knowledge(sources, records, mappings)

    sources, records = deepcopy(knowledge)
    duplicate_source = deepcopy(sources[0])
    duplicate_source["source_id"] = "vghtpe_duplicate_url"
    sources.append(duplicate_source)
    with pytest.raises(ValueError, match="duplicate source URL"):
        validate_department_knowledge(sources, records, mappings)

    sources, records = deepcopy(knowledge)
    records.append(deepcopy(records[0]))
    with pytest.raises(ValueError, match="duplicate normalized"):
        validate_department_knowledge(sources, records, mappings)

    duplicate_mappings = deepcopy(mappings)
    duplicate_mappings.append(deepcopy(duplicate_mappings[0]))
    with pytest.raises(ValueError, match="duplicate canonical mapping"):
        validate_department_canonical_mapping(duplicate_mappings)


def test_mapping_and_record_canonical_identity_must_be_reviewed(knowledge, mappings):
    invalid_mappings = deepcopy(mappings)
    invalid_mappings[0]["canonical_dept_id"] = "123"
    with pytest.raises(ValueError, match="invalid canonical department ID"):
        validate_department_canonical_mapping(invalid_mappings)

    sources, records = deepcopy(knowledge)
    records[0]["canonical_dept_id"] = 999999
    with pytest.raises(ValueError, match="differs from reviewed mapping"):
        validate_department_knowledge(sources, records, mappings)

    sources, records = deepcopy(knowledge)
    records[0]["canonical_child_dept"] = "自行猜測科別"
    with pytest.raises(ValueError, match="differs from reviewed mapping"):
        validate_department_knowledge(sources, records, mappings)


def test_taichung_records_require_fallback_marker(knowledge, mappings):
    sources, records = deepcopy(knowledge)
    fallback = next(record for record in records if record["source_hospital"] == "vghtc")
    fallback["fallback_for_vghtpe"] = False
    with pytest.raises(ValueError, match="fallback"):
        validate_department_knowledge(sources, records, mappings)


def test_lookup_keeps_different_departments_and_prefers_taipei_within_same_department(knowledge):
    _, records = knowledge

    heart_candidates = lookup_concept("心悸", records)
    assert {item["canonical_dept_id"] for item in heart_candidates} == {1232, 1237, 1241}

    ear_candidates = lookup_concept("耳鳴", records)
    ear = next(item for item in ear_candidates if item["canonical_dept_id"] == 1333)
    assert ear["source_hospital"] == "vghtpe"


@pytest.mark.parametrize(("raw", "expected"), [
    (123, 123), ("123", 123), ("00123", 123),
    (True, None), (False, None), (None, None), ("", None),
    ("abc", None), ("1.5", None), (" 123", None),
    (1.5, None), (0, None), (-1, None), ("0", None), ("-1", None),
    ("1" * 5000, None),
])
def test_live_api_department_id_accepts_only_positive_decimal_ids(raw, expected):
    assert canonical_department_id(raw) == expected


def test_official_label_can_differ_from_live_child_name(knowledge):
    _, records = knowledge
    cardiac_records = [record for record in records if record["canonical_dept_id"] == 1241]
    assert cardiac_records
    assert any("心臟科" in record["official_department_labels"] for record in cardiac_records)

    resolution = resolve_department_names(cardiac_records, [{
        "dept_id": "1241", "parentDept": "內科", "childDept": "心臟內科",
    }])[0]

    assert resolution["status"] == "resolved"
    assert resolution["canonical_dept_id"] == 1241
    assert resolution["db_dept_id"] == 1241
    assert resolution["db_parent_dept"] == "內科"
    assert resolution["db_child_dept"] == "心臟內科"
    assert resolution["resolution_method"] == "verified_canonical_dept_id"


def test_resolution_uses_exact_canonical_id_not_department_name(knowledge):
    _, records = knowledge
    cardiac_records = [record for record in records if record["canonical_dept_id"] == 1241]

    missing = resolve_department_names(cardiac_records, [{
        "dept_id": 9999, "parentDept": "內科", "childDept": "心臟內科",
    }])[0]
    assert missing["status"] == "unresolved"
    assert missing["reason"] == "canonical_dept_id_not_in_live_db"

    resolved = resolve_department_names(cardiac_records, [{
        "dept_id": 1241, "parentDept": "Live parent", "childDept": "Live canonical name",
    }])[0]
    assert resolved["status"] == "resolved"
    assert resolved["db_parent_dept"] == "Live parent"
    assert resolved["db_child_dept"] == "Live canonical name"


def test_duplicate_live_rows_for_canonical_id_are_unresolved(knowledge):
    _, records = knowledge
    general_records = [record for record in records if record["canonical_dept_id"] == 1232]
    resolution = resolve_department_names(general_records, [
        {"dept_id": "1232", "parentDept": "內科", "childDept": "一般內科"},
        {"dept_id": 1232, "parentDept": "內科", "childDept": "一般內科"},
    ])[0]
    assert resolution["status"] == "unresolved"
    assert resolution["reason"] == "duplicate_canonical_dept_id_in_live_db"


def test_exact_db_department_ids_is_compatibility_view_over_canonical_mapping(knowledge):
    _, records = knowledge
    ids = exact_db_department_ids(records, [
        {"dept_id": 1241, "parentDept": "內科", "childDept": "心臟內科"},
        {"dept_id": "1242", "parentDept": "內科", "childDept": "腎臟科"},
    ])
    assert ids["心臟內科"] == 1241
    assert ids["腎臟科"] == 1242
    assert ids["一般內科"] is None


def test_production_knowledge_files_are_not_loaded_from_docs_or_package():
    service = (
        Path(__file__).resolve().parents[1]
        / "app" / "services" / "department_knowledge.py"
    ).read_text(encoding="utf-8")
    assert "PHASE3_310_DEPARTMENT_INVENTORY" not in service
    assert "phase3_kb_migration_package" not in service
