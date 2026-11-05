"""Audited semantic retrieval links persist across turns without changing single-turn results."""
import asyncio
import importlib
import json
from unittest.mock import patch

from app.schemas import ConversationState, Message, SemanticExtraction, SemanticOfficialLink, TriageCase
from app.services import department_reasoning_service as reasoning
from app.services.case_store import sanitize_untrusted_snapshot
from app.services.conversation_service import invalidate_corrected_department_context
from app.services.department_knowledge import load_department_knowledge, lookup_concept, resolve_department_names

chat_route = importlib.import_module("app.routes.chat")
_, RECORDS = load_department_knowledge()
ACTIVE = [
    dict(dept_id=record["canonical_dept_id"], parent_dept=record["canonical_parent_dept"],
         child_dept=record["canonical_child_dept"])
    for record in {item["canonical_dept_id"]: item for item in RECORDS}.values()
]
DIZZY_DEPTS = {record["canonical_dept_id"] for record in lookup_concept("頭暈", RECORDS)}
EAR_DEPTS = {record["canonical_dept_id"] for record in lookup_concept("耳鳴", RECORDS)}


def extraction(value, source, assertion="present", field="symptom"):
    return SemanticExtraction(
        field=field, normalized_value=[value] if field == "accompanying_symptoms" else value,
        semantic_status="available" if assertion == "present" else "unavailable",
        assertion=assertion, confidence=0.95, source_text=source, extractor="ai",
    )


def dizzy_case():
    case = TriageCase(case_id="semantic-link-retention")
    case.history_records = [Message(role="user", content="頭腦昏沉、食慾不佳")]
    case.semantic_extractions = [extraction("頭腦昏沉", "頭腦昏沉")]
    return case


def add_turn(case, text, *items):
    case.history_records.append(Message(role="user", content=text))
    case.semantic_extractions.extend(items)


class Provider:
    """Answers by purpose and records the call order, like the live runtime seam."""

    def __init__(self):
        self.purposes, self.convergence_payloads = [], []

    async def __call__(self, prompt, *, purpose):
        self.purposes.append(purpose)
        if purpose == "department_semantic_retrieval":
            return json.dumps({"matches": [
                {"patient_source_text": "頭腦昏沉", "knowledge_concept": "頭暈", "confidence": 0.95},
            ]}, ensure_ascii=False)
        if purpose == "department_retrieval_entailment":
            return json.dumps({"judgments": [{"patient_source_text": "頭腦昏沉", "knowledge_concept": "頭暈",
                                              "supported": True, "reason": "語意相符"}]}, ensure_ascii=False)
        self.convergence_payloads.append(json.loads(prompt.split("\n資料：", 1)[1]))
        return json.dumps({"status": "ambiguous", "candidates": []})


def reason(case, provider):
    with patch.object(reasoning, "fetch_active_departments", return_value=ACTIVE), patch.object(
        reasoning, "complete_runtime_json", new=provider,
    ):
        asyncio.run(reasoning.reason_about_departments(case))


def payload_depts(provider, concept=None):
    return {item["dept_id"] for item in provider.convergence_payloads[-1]["retrieved_official_evidence"]
            if concept is None or item["knowledge_concept"] == concept}


def test_first_turn_payload_equals_semantic_retrieval_alone():
    case, provider = dizzy_case(), Provider()
    reason(case, provider)
    expected = asyncio.run(_direct_semantic(dizzy_case()))
    assert provider.convergence_payloads[0]["retrieved_official_evidence"] == expected
    assert provider.purposes == ["department_semantic_retrieval", "department_retrieval_entailment",
                                 "department_convergence"]
    assert [(link.patient_source_text, link.knowledge_concept)
            for link in case.conversation_state.semantic_official_links] == [("頭腦昏沉", "頭暈")]


async def _direct_semantic(case):
    with patch.object(reasoning, "complete_runtime_json", new=Provider()):
        return await reasoning.retrieve_semantic_official_evidence(
            case, RECORDS, resolve_department_names(RECORDS, ACTIVE),
        )


def test_lexical_only_payload_is_unchanged_and_stores_no_link():
    case, provider = TriageCase(case_id="lexical"), Provider()
    case.history_records = [Message(role="user", content="一直耳鳴")]
    case.semantic_extractions = [extraction("耳鳴", "耳鳴")]
    reason(case, provider)
    expected = reasoning.retrieve_official_evidence(case, RECORDS, resolve_department_names(RECORDS, ACTIVE))
    assert provider.convergence_payloads[0]["retrieved_official_evidence"] == expected
    # 2026-10-28: semantic retrieval runs alongside literal hits; its match is not grounded in this
    # case's evidence, so nothing is audited, merged or stored.
    assert provider.purposes == ["department_semantic_retrieval", "department_convergence"]
    assert case.conversation_state.semantic_official_links == []


def test_later_lexical_hit_keeps_earlier_semantic_evidence():
    case, provider = dizzy_case(), Provider()
    reason(case, provider)
    add_turn(case, "還有耳鳴", extraction("耳鳴", "耳鳴", field="accompanying_symptoms"))
    # The lexical-only path that existed before would have dropped every dizziness department.
    lexical = reasoning.retrieve_official_evidence(case, RECORDS, resolve_department_names(RECORDS, ACTIVE))
    assert {item["knowledge_concept"] for item in lexical} == {"耳鳴"}
    reason(case, provider)
    assert payload_depts(provider, "頭暈") == DIZZY_DEPTS
    assert payload_depts(provider, "耳鳴") == EAR_DEPTS
    # 2026-10-28: semantic retrieval now also runs on the turn with a lexical hit.
    assert provider.purposes.count("department_semantic_retrieval") == 2


def test_semantic_retrieval_call_pattern_is_unchanged_when_lexical_stays_empty():
    case, provider = dizzy_case(), Provider()
    reason(case, provider)
    add_turn(case, "大概三天", extraction("3天", "三天", field="duration"))
    reason(case, provider)
    assert provider.purposes.count("department_semantic_retrieval") == 2
    assert payload_depts(provider, "頭暈") == DIZZY_DEPTS
    assert len(case.conversation_state.semantic_official_links) == 1


def test_absent_official_concept_drops_retained_link():
    case, provider = dizzy_case(), Provider()
    reason(case, provider)
    add_turn(case, "其實沒有頭暈", extraction("頭暈", "沒有頭暈", assertion="absent"))
    retained = reasoning.retained_semantic_official_evidence(
        case, RECORDS, resolve_department_names(RECORDS, ACTIVE),
    )
    assert retained == []
    assert case.conversation_state.semantic_official_links == []


def test_superseded_source_drops_retained_link():
    case, provider = dizzy_case(), Provider()
    reason(case, provider)
    add_turn(case, "現在不會頭腦昏沉了", extraction("頭腦昏沉", "不會頭腦昏沉", assertion="absent"))
    assert reasoning.retained_semantic_official_evidence(
        case, RECORDS, resolve_department_names(RECORDS, ACTIVE),
    ) == []


def test_unresolved_live_department_is_not_rebuilt_from_link():
    case = dizzy_case()
    case.conversation_state.semantic_official_links = [
        SemanticOfficialLink(patient_source_text="頭腦昏沉", knowledge_concept="不存在的概念"),
    ]
    assert reasoning.retained_semantic_official_evidence(
        case, RECORDS, resolve_department_names(RECORDS, ACTIVE),
    ) == []
    assert case.conversation_state.semantic_official_links == []


def test_revision_correction_and_untrusted_snapshot_clear_links():
    link = SemanticOfficialLink(patient_source_text="頭腦昏沉", knowledge_concept="頭暈")
    for reset in (chat_route._begin_revision, invalidate_corrected_department_context, sanitize_untrusted_snapshot):
        case = dizzy_case()
        case.conversation_state.semantic_official_links = [link]
        result = reset(case) or case
        assert result.conversation_state.semantic_official_links == []


def test_old_state_without_links_loads_empty():
    state = ConversationState.model_validate({"turn_count": 3})
    assert state.semantic_official_links == []
