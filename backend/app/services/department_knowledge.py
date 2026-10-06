"""Read-only official guidance with reviewed APP/SQL canonical identities."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"
_SOURCE_POLICY = {
    ("vghtpe", "symptom_department_guidance"): (1, {"www.vghtpe.gov.tw", "wd.vghtpe.gov.tw"}),
    ("vghtpe", "doctor_specialty_keywords"): (2, {"www.vghtpe.gov.tw"}),
    ("vghtpe", "department_specialty"): (2, {"wd.vghtpe.gov.tw"}),
    ("vghtpe", "department_service"): (2, {"wd.vghtpe.gov.tw"}),
    ("vghtpe", "patient_education"): (2, {"ihealth.vghtpe.gov.tw"}),
    ("vghtpe", "clinic_service"): (2, {"wd.vghtpe.gov.tw"}),
    ("vghtc", "department_service"): (3, {"www.vghtc.gov.tw"}),
}
_RECORD_FIELDS = (
    "department_name", "official_department_labels", "concept", "evidence_type",
    "canonical_dept_id", "canonical_parent_dept", "canonical_child_dept", "mapping_method",
    "source_id", "source_hospital", "source_type",
    "source_title", "source_url", "source_priority", "retrieved_at",
    "evidence_text", "fallback_for_vghtpe",
)
_SOURCE_FIELDS = (
    "source_id", "source_hospital", "source_type", "source_title", "source_url",
    "source_priority", "retrieved_at",
)
_MAPPING_FIELDS = (
    "legacy_department_name", "canonical_dept_id", "canonical_parent_dept",
    "canonical_child_dept", "mapping_method", "official_source_department_labels",
    "identity_source_id", "note",
)
_ROUTING_EVIDENCE_TYPES = {"direct_routing_evidence", "department_specialty_evidence"}
_MAPPING_METHODS = {"exact_app_name", "app_existing_alias"}


def normalize_concept(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def load_department_knowledge(directory: Path = KNOWLEDGE_DIR) -> tuple[list[dict], list[dict]]:
    sources = json.loads((directory / "sources.json").read_text(encoding="utf-8"))["sources"]
    records = json.loads((directory / "vghtpe_department_guidance.json").read_text(encoding="utf-8"))["records"]
    mappings = load_department_canonical_mapping(directory)
    validate_department_knowledge(sources, records, mappings)
    return sources, records


def load_department_canonical_mapping(directory: Path = KNOWLEDGE_DIR) -> list[dict]:
    mappings = json.loads(
        (directory / "department_canonical_mapping.json").read_text(encoding="utf-8")
    )["mappings"]
    validate_department_canonical_mapping(mappings)
    return mappings


def validate_department_canonical_mapping(mappings: list[dict]) -> None:
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("canonical department mappings must be a nonempty list")
    seen_names: set[str] = set()
    for mapping in mappings:
        if not isinstance(mapping, dict) or not _has_fields(mapping, _MAPPING_FIELDS):
            raise ValueError("canonical department mapping is incomplete")
        name = mapping["legacy_department_name"]
        parent = mapping["canonical_parent_dept"]
        child = mapping["canonical_child_dept"]
        method = mapping["mapping_method"]
        labels = mapping["official_source_department_labels"]
        if not all(isinstance(value, str) and value.strip() for value in (name, parent, child)):
            raise ValueError("canonical department names are required")
        if name in seen_names:
            raise ValueError("duplicate canonical mapping department name")
        if type(mapping["canonical_dept_id"]) is not int or mapping["canonical_dept_id"] <= 0:
            raise ValueError("invalid canonical department ID")
        if method not in _MAPPING_METHODS:
            raise ValueError("unknown canonical mapping method")
        if not isinstance(labels, list) or not labels or any(
            not isinstance(label, str) or not label.strip() for label in labels
        ) or len(set(labels)) != len(labels):
            raise ValueError("official department labels must be unique nonempty strings")
        if not isinstance(mapping["identity_source_id"], str) or not mapping["identity_source_id"].strip():
            raise ValueError("canonical mapping identity source is required")
        if not isinstance(mapping["note"], str):
            raise ValueError("canonical mapping note must be a string")
        seen_names.add(name)


def validate_department_knowledge(
    sources: list[dict], records: list[dict], mappings: list[dict] | None = None,
) -> None:
    if not isinstance(sources, list) or not sources or not isinstance(records, list) or not records:
        raise ValueError("knowledge sources and records must be nonempty lists")
    canonical_mappings = mappings if mappings is not None else load_department_canonical_mapping()
    validate_department_canonical_mapping(canonical_mappings)
    mapping_by_name = {mapping["legacy_department_name"]: mapping for mapping in canonical_mappings}

    source_by_id = {}
    source_urls = set()
    for source in sources:
        if not isinstance(source, dict) or not _has_fields(source, _SOURCE_FIELDS):
            raise ValueError("source metadata is incomplete")
        source_id = source["source_id"]
        if not isinstance(source_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]+", source_id):
            raise ValueError("invalid source_id")
        if source_id in source_by_id:
            raise ValueError("duplicate source_id")
        _validate_source_policy(source)
        url_key = source["source_url"].strip().casefold()
        if url_key in source_urls:
            raise ValueError("duplicate source URL")
        source_urls.add(url_key)
        _validate_date(source["retrieved_at"])
        source_by_id[source_id] = source

    seen = set()
    for record in records:
        if not isinstance(record, dict) or not _has_fields(record, _RECORD_FIELDS):
            raise ValueError("medical record metadata is incomplete")
        if not isinstance(record["source_id"], str):
            raise ValueError("medical record source_id must be a string")
        source = source_by_id.get(record["source_id"])
        if source is None:
            raise ValueError("medical record source_id is not registered")
        if any(record[field] != source[field] for field in _SOURCE_FIELDS):
            raise ValueError("medical record provenance differs from source registry")
        department = record["department_name"]
        concept = record["concept"]
        evidence = record["evidence_text"]
        if not all(isinstance(value, str) and value.strip() for value in (department, concept, evidence)):
            raise ValueError("department, concept and evidence_text are required")
        if len(evidence) > 120 or normalize_concept(concept) not in normalize_concept(evidence):
            raise ValueError("evidence_text must be short and contain the concept")
        if record["evidence_type"] not in _ROUTING_EVIDENCE_TYPES:
            raise ValueError("non-routing evidence is disabled in production")
        mapping = mapping_by_name.get(department)
        if mapping is None:
            raise ValueError("medical record has no reviewed canonical mapping")
        if (
            type(record["canonical_dept_id"]) is not int
            or record["canonical_dept_id"] <= 0
            or record["canonical_dept_id"] != mapping["canonical_dept_id"]
            or record["canonical_parent_dept"] != mapping["canonical_parent_dept"]
            or record["canonical_child_dept"] != mapping["canonical_child_dept"]
            or record["mapping_method"] != mapping["mapping_method"]
        ):
            raise ValueError("medical record canonical identity differs from reviewed mapping")
        labels = record["official_department_labels"]
        if not isinstance(labels, list) or set(labels) != set(mapping["official_source_department_labels"]):
            raise ValueError("medical record official labels differ from reviewed mapping")
        if type(record["fallback_for_vghtpe"]) is not bool:
            raise ValueError("fallback_for_vghtpe must be boolean")
        if record["fallback_for_vghtpe"] != (record["source_hospital"] == "vghtc"):
            raise ValueError("Taichung records must be marked as fallback")
        key = (
            record["source_id"], record["canonical_dept_id"],
            normalize_concept(concept), record["evidence_type"],
        )
        if key in seen:
            raise ValueError("duplicate normalized source/canonical department/concept")
        seen.add(key)


def lookup_concept(concept: str, records: list[dict]) -> list[dict]:
    """Keep each department's best evidence for an exact concept, not a chosen department."""
    key = normalize_concept(concept)
    if not key:
        return []
    matches = [record for record in records if normalize_concept(record["concept"]) == key]
    best_by_department = {}
    evidence_rank = {"direct_routing_evidence": 0, "department_specialty_evidence": 1}
    for record in sorted(
        matches,
        key=lambda item: (
            item["source_priority"], evidence_rank[item["evidence_type"]], item["source_id"],
        ),
    ):
        best_by_department.setdefault(record["canonical_dept_id"], record)
    return sorted(best_by_department.values(), key=lambda record: record["canonical_dept_id"])


def canonical_department_id(value: object) -> int | None:
    """Accept only positive integer IDs or ASCII decimal strings from the reference API."""
    if type(value) is int:
        return value if value > 0 else None
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def resolve_department_names(records: list[dict], db_departments: list[dict]) -> list[dict]:
    """Resolve reviewed canonical IDs against the unique live DB row."""
    resolutions = []
    grouped: dict[int, list[dict]] = {}
    for record in records:
        dept_id = canonical_department_id(record.get("canonical_dept_id"))
        if dept_id is not None:
            grouped.setdefault(dept_id, []).append(record)
    for dept_id, department_records in sorted(grouped.items()):
        matches = [
            row for row in db_departments
            if canonical_department_id(row.get("dept_id")) == dept_id
        ]
        names = sorted({record["department_name"] for record in department_records})
        entry = {
            "canonical_dept_id": dept_id,
            "knowledge_department_names": names,
            "canonical_parent_snapshot": department_records[0]["canonical_parent_dept"],
            "canonical_child_snapshot": department_records[0]["canonical_child_dept"],
        }
        if not matches:
            entry.update(status="unresolved", reason="canonical_dept_id_not_in_live_db")
        elif len(matches) != 1:
            entry.update(status="unresolved", reason="duplicate_canonical_dept_id_in_live_db")
        else:
            row = matches[0]
            parent = row.get("parentDept", row.get("parent_dept"))
            child = _db_child_name(row)
            if not isinstance(parent, str) or not parent.strip():
                entry.update(status="unresolved", reason="missing_live_db_parent_name")
            elif not isinstance(child, str) or not child.strip():
                entry.update(status="unresolved", reason="missing_live_db_child_name")
            else:
                entry.update(
                    status="resolved", db_dept_id=dept_id, db_parent_dept=parent,
                    db_child_dept=child, resolution_method="verified_canonical_dept_id",
                )
        resolutions.append(entry)
    return resolutions


def exact_db_department_ids(records: list[dict], db_departments: list[dict]) -> dict[str, int | None]:
    """Compatibility helper keyed by knowledge mapping name."""
    by_id = {
        entry["canonical_dept_id"]: entry.get("db_dept_id")
        for entry in resolve_department_names(records, db_departments)
    }
    return {
        name: by_id.get(canonical_department_id(record.get("canonical_dept_id")))
        for name, record in {
            record["department_name"]: record for record in records
        }.items()
    }


def _db_child_name(row: dict) -> object:
    return row.get("childDept", row.get("child_dept"))


def _has_fields(item: dict, fields: tuple[str, ...]) -> bool:
    return all(field in item and item[field] is not None for field in fields)


def _validate_source_policy(source: dict) -> None:
    if not isinstance(source["source_hospital"], str) or not isinstance(source["source_type"], str):
        raise ValueError("unknown source hospital or source type")
    policy = _SOURCE_POLICY.get((source["source_hospital"], source["source_type"]))
    if policy is None:
        raise ValueError("unknown source hospital or source type")
    priority, hosts = policy
    url = source["source_url"]
    parsed = urlparse(url) if isinstance(url, str) else None
    if not parsed or parsed.scheme != "https" or parsed.hostname not in hosts or not parsed.path:
        raise ValueError("source URL must be an official HTTPS page")
    if type(source["source_priority"]) is not int or source["source_priority"] != priority:
        raise ValueError("source priority violates hospital policy")
    if not isinstance(source["source_title"], str) or not source["source_title"].strip():
        raise ValueError("source title is required")


def _validate_date(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("retrieved_at must be an ISO date")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("retrieved_at must be an ISO date") from exc
