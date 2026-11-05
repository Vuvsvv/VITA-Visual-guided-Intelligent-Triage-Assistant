"""Semantic retrieval runs alongside literal matches; results merge without double scoring (c14)."""
import asyncio
import json
from unittest.mock import patch

from app.schemas import Message, SemanticExtraction, TriageCase
from app.services import department_clues as clues
from app.services import department_reasoning_service as reasoning
from app.services.department_knowledge import load_department_knowledge, lookup_concept, resolve_department_names

_, RECORDS = load_department_knowledge()
ACTIVE = [
    dict(dept_id=record["canonical_dept_id"], parent_dept=record["canonical_parent_dept"],
         child_dept=record["canonical_child_dept"])
    for record in {item["canonical_dept_id"]: item for item in RECORDS}.values()
]
RESOLUTIONS = resolve_department_names(RECORDS, ACTIVE)
EYE = {record["canonical_dept_id"] for record in lookup_concept("眼睛癢", RECORDS)}
ITCH = {record["canonical_dept_id"] for record in lookup_concept("癢", RECORDS)}


def extraction(value, source, field="symptom"):
    return SemanticExtraction(field=field, normalized_value=value, semantic_status="available", assertion="present",
                              confidence=0.95, source_text=source, extractor="ai")


def red_itchy_eyes():
    case = TriageCase(case_id="c14-red-itchy-eyes")
    case.history_records = [Message(role="user", content="眼睛紅紅的很癢")]
    case.semantic_extractions = [extraction("眼睛紅癢", "眼睛紅紅的很癢")]
    return case


class Provider:
    """Semantic retrieval and audit answer with the given links; convergence stays ambiguous."""

    def __init__(self, matches, fail_semantic=False):
        self.matches, self.fail_semantic = matches, fail_semantic
        self.purposes, self.convergence_payloads = [], []

    async def __call__(self, prompt, *, purpose):
        self.purposes.append(purpose)
        if purpose == "department_semantic_retrieval":
            if self.fail_semantic:
                raise TimeoutError()
            return json.dumps({"matches": [dict(patient_source_text=source, knowledge_concept=concept, confidence=0.95)
                                           for source, concept in self.matches]}, ensure_ascii=False)
        if purpose == "department_retrieval_entailment":
            return json.dumps({"judgments": [dict(patient_source_text=source, knowledge_concept=concept,
                                                  supported=True, reason="語意相符")
                                             for source, concept in self.matches]}, ensure_ascii=False)
        self.convergence_payloads.append(json.loads(prompt.split("\n資料：", 1)[1]))
        return json.dumps({"status": "ambiguous", "candidates": []})


def reason(case, provider):
    with patch.object(reasoning, "fetch_active_departments", return_value=ACTIVE), patch.object(
        reasoning, "complete_runtime_json", new=provider,
    ):
        asyncio.run(reasoning.reason_about_departments(case))
    return provider.convergence_payloads[-1]["retrieved_official_evidence"]


def test_c14_literal_itch_hit_still_runs_semantic_retrieval_and_adds_eye_department():
    case = red_itchy_eyes()
    literal = reasoning.retrieve_official_evidence(case, RECORDS, RESOLUTIONS)
    # 「眼睛紅」 (official red-eye page, 2026-11-02) now matches literally; 「眼睛癢」 still needs semantics.
    assert {row["knowledge_concept"] for row in literal} == {"癢", "眼睛紅"}
    assert {row["dept_id"] for row in literal} == ITCH | EYE
    provider = Provider([("眼睛紅紅的很癢", "眼睛癢")])
    retrieved = reason(case, provider)
    assert provider.purposes == ["department_semantic_retrieval", "department_retrieval_entailment",
                                 "department_convergence"]
    assert {row["dept_id"] for row in retrieved} == ITCH | EYE
    assert EYE <= set(case.conversation_state.department_clue_candidates)
    assert [(link.patient_source_text, link.knowledge_concept)
            for link in case.conversation_state.semantic_official_links] == [("眼睛紅紅的很癢", "眼睛癢")]


def test_literal_concept_is_not_scored_twice_by_a_semantic_duplicate():
    case = TriageCase(case_id="itch-twice")
    case.history_records = [Message(role="user", content="皮膚很癢，一直抓個不停")]
    case.semantic_extractions = [extraction("癢", "皮膚很癢"), extraction("抓癢", "一直抓個不停")]
    literal = reasoning.retrieve_official_evidence(case, RECORDS, RESOLUTIONS)
    literal_pairs = {(row["patient_source_text"], row["knowledge_concept"]) for row in literal}
    # The semantic pass repeats the literal concept, on the same and on another text piece.
    retrieved = reason(case, Provider([("皮膚很癢", "癢"), ("一直抓個不停", "癢")]))
    itch_rows = [row for row in retrieved if row["knowledge_concept"] == "癢"]
    assert {(row["patient_source_text"], row["knowledge_concept"]) for row in itch_rows} == {
        pair for pair in literal_pairs if pair[1] == "癢"}
    candidates = clues.candidate_dept_ids(retrieved)
    assert clues.department_scores(retrieved, candidates, []) == clues.department_scores(literal, candidates, [])
    assert case.conversation_state.semantic_official_links == []


def test_semantic_failure_keeps_the_literal_result():
    case = red_itchy_eyes()
    literal = reasoning.retrieve_official_evidence(case, RECORDS, RESOLUTIONS)
    provider = Provider([("眼睛紅紅的很癢", "眼睛癢")], fail_semantic=True)
    assert reason(case, provider) == literal
    # A failed call is retried once before semantic retrieval is given up.
    assert provider.purposes == ["department_semantic_retrieval", "department_semantic_retrieval",
                                 "department_convergence"]
    assert case.conversation_state.semantic_official_links == []
