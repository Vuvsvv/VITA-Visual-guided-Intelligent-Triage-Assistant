"""2026-11-04: with nothing official behind the chief complaint, an AI link may not widen a body part.

「我的屁股好痛」 matched no official concept; the later 「我的大腿有點酸」 was linked by the AI to
「下肢酸痛」 (Taipei VGH lists only 心臟血管外科), so the case ended in 心臟外科 with no basis
in the chief complaint.
"""
import asyncio
import json
from unittest.mock import patch

from app.schemas import Message, SemanticExtraction, SemanticOfficialLink, TriageCase
from app.services import department_reasoning_service as reasoning
from app.services.conversation_service import NO_OFFICIAL_EVIDENCE_REASON
from app.services.department_knowledge import lookup_concept
from tests.test_semantic_literal_merge import ACTIVE, RECORDS, RESOLUTIONS

CARDIAC = {record["canonical_dept_id"] for record in lookup_concept("下肢酸痛", RECORDS)}
COLORECTAL = {record["canonical_dept_id"] for record in lookup_concept("臀部腫塊", RECORDS)}
NUMB = {record["canonical_dept_id"] for record in lookup_concept("麻木", RECORDS)}
EAR = {record["canonical_dept_id"] for record in lookup_concept("耳痛", RECORDS)}
DIZZY = {record["canonical_dept_id"] for record in lookup_concept("頭暈", RECORDS)}
REHAB = {record["canonical_dept_id"] for record in lookup_concept("臀部疼痛", RECORDS)}


def extraction(value, source, field="symptom"):
    return SemanticExtraction(field=field, normalized_value=value, semantic_status="available", assertion="present",
                              confidence=0.95, source_text=source, extractor="ai")


class Provider:
    """Semantic retrieval and audit answer with the given (source, concept, body_part_match) links."""

    def __init__(self, links):
        self.links, self.convergence_payloads = links, []

    async def __call__(self, prompt, *, purpose):
        if purpose == "department_semantic_retrieval":
            return json.dumps({"matches": [dict(patient_source_text=source, knowledge_concept=concept, confidence=0.95)
                                           for source, concept, _ in self.links]}, ensure_ascii=False)
        if purpose == "department_retrieval_entailment":
            return json.dumps({"judgments": [
                dict(patient_source_text=source, knowledge_concept=concept, supported=True, reason="語意相符",
                     **({} if match is None else dict(body_part_match=match)))
                for source, concept, match in self.links]}, ensure_ascii=False)
        self.convergence_payloads.append(json.loads(prompt.split("\n資料：", 1)[1]))
        return json.dumps({"status": "ambiguous", "candidates": []})


def sore_buttock(case_id, later="我的大腿有點酸", later_extractions=None, chief="我的屁股好痛", chief_extractions=None):
    case = TriageCase(case_id=case_id)
    case.history_records = [Message(role="user", content=chief), Message(role="assistant", content="還有其他不舒服嗎？"),
                            Message(role="user", content=later)]
    case.semantic_extractions = [
        *(chief_extractions or [extraction("屁股", "屁股", "body_part"), extraction("疼痛", "屁股好痛")]),
        *(later_extractions or [extraction("大腿", "大腿", "body_part"), extraction("酸", "大腿有點酸")]),
    ]
    return case


def reason(case, provider):
    with patch.object(reasoning, "fetch_active_departments", return_value=ACTIVE), patch.object(
        reasoning, "complete_runtime_json", new=provider,
    ):
        asyncio.run(reasoning.reason_about_departments(case))
    payloads = provider.convergence_payloads
    return payloads[-1]["retrieved_official_evidence"] if payloads else []


def test_widened_thigh_link_is_dropped_when_the_chief_complaint_has_no_official_evidence():
    case = sore_buttock("buttock-thigh")
    assert reasoning.retrieve_official_evidence(case, RECORDS, RESOLUTIONS) == []
    assert CARDIAC
    assert reason(case, Provider([("大腿有點酸", "下肢酸痛", "other")])) == []
    state = case.conversation_state
    # No candidate at all, so the turn-3 family medicine offer applies (SAFETY_NET_NO_CANDIDATE_TURN).
    assert state.department_uncertainty_reason == NO_OFFICIAL_EVIDENCE_REASON
    assert not CARDIAC & set(state.department_clue_candidates)
    # The audited link is still remembered; it is only refused while the chief complaint has no basis.
    assert [(link.patient_source_text, link.knowledge_concept, link.body_part_match)
            for link in state.semantic_official_links] == [("大腿有點酸", "下肢酸痛", "other")]


def test_a_widened_chief_complaint_link_is_no_official_basis():
    # 310 replay 2026-11-04: 「我的屁股好痛」 was widened to 「下肢疼痛」 (疼痛控制科), which made the chief
    # complaint look covered and let that department be decided.
    case = sore_buttock("buttock-widened-chief")
    provider = Provider([("屁股好痛", "下肢疼痛", "other"), ("大腿有點酸", "下肢酸痛", "other")])
    assert reason(case, provider) == []
    assert case.conversation_state.department_uncertainty_reason == NO_OFFICIAL_EVIDENCE_REASON


def test_a_widened_link_is_refused_even_when_the_ai_split_off_no_part():
    # 310 replay 2026-11-04: the first turn held only the symptom 「痛」, no body part, and the
    # audit still marked 「痛」→「下肢疼痛」 as a wider part.
    case = sore_buttock("buttock-no-part", chief_extractions=[extraction("疼痛", "痛")])
    provider = Provider([("痛", "下肢疼痛", "other"), ("大腿有點酸", "下肢酸痛", "other")])
    assert reason(case, provider) == []
    case = sore_buttock("buttock-no-part-unlabelled", chief_extractions=[extraction("疼痛", "痛")])
    # Without the audit's view and with no stated part, the link stands (the subject filter's job).
    assert {row["knowledge_concept"] for row in reason(case, Provider([("痛", "下肢疼痛", None)]))} == {"下肢疼痛"}


def test_link_without_an_audit_part_view_needs_the_stated_part_in_the_concept():
    case = sore_buttock("buttock-thigh-unlabelled")
    assert reason(case, Provider([("大腿有點酸", "下肢酸痛", None)])) == []


def test_same_part_in_other_words_is_kept():
    case = sore_buttock("buttock-lump", "屁股摸到一顆腫塊",
                        [extraction("屁股", "屁股", "body_part"), extraction("腫塊", "摸到一顆腫塊")])
    retrieved = reason(case, Provider([("摸到一顆腫塊", "臀部腫塊", "same")]))
    assert COLORECTAL and {row["dept_id"] for row in retrieved} == COLORECTAL


def test_concept_naming_no_part_is_kept():
    case = sore_buttock("buttock-numb", "大腿有點麻",
                        [extraction("大腿", "大腿", "body_part"), extraction("麻", "大腿有點麻")])
    retrieved = reason(case, Provider([("大腿有點麻", "麻木", "none")]))
    assert NUMB and {row["dept_id"] for row in retrieved} == NUMB


def test_a_side_word_alone_never_blocks_a_link():
    case = sore_buttock("buttock-right-ear", "右耳也很痛",
                        [extraction("右耳", "右耳", "body_part"), extraction("耳朵痛", "右耳也很痛")])
    retrieved = reason(case, Provider([("右耳也很痛", "耳痛", None)]))
    assert EAR and {row["dept_id"] for row in retrieved} == EAR


def test_retained_links_follow_their_stored_part_view():
    case = sore_buttock("buttock-thigh-retained")
    case.conversation_state.semantic_official_links = [
        SemanticOfficialLink(patient_source_text="大腿有點酸", knowledge_concept="下肢酸痛", body_part_match="other")]
    assert reason(case, Provider([])) == []
    assert case.conversation_state.department_uncertainty_reason == NO_OFFICIAL_EVIDENCE_REASON
    case = sore_buttock("buttock-lump-retained", "屁股摸到一顆腫塊",
                        [extraction("屁股", "屁股", "body_part"), extraction("腫塊", "摸到一顆腫塊")])
    case.conversation_state.semantic_official_links = [
        SemanticOfficialLink(patient_source_text="摸到一顆腫塊", knowledge_concept="臀部腫塊", body_part_match="same")]
    assert {row["dept_id"] for row in reason(case, Provider([]))} == COLORECTAL


def test_a_chief_complaint_with_official_evidence_keeps_widened_links():
    case = sore_buttock("dizzy-thigh", chief="最近一直頭暈", chief_extractions=[extraction("頭暈", "頭暈")])
    retrieved = reason(case, Provider([("大腿有點酸", "下肢酸痛", "other")]))
    assert ("大腿有點酸", "下肢酸痛") in {(row["patient_source_text"], row["knowledge_concept"]) for row in retrieved}
    # The chief complaint filter still keeps a department that does not see 頭暈 out of the decision.
    assert not (CARDIAC - DIZZY) & set(case.conversation_state.department_clue_candidates)


def test_a_split_body_part_is_shortlisted_with_its_symptom():
    # Replay 2026-11-04: 「我的屁股好痛」 split into 「屁股」 and 「痛」 was shortlisted from 「痛」 alone,
    # so 「臀部疼痛」 (Taipei VGH 復健醫學部 doctors) never reached the AI.
    case = sore_buttock("buttock-split", chief_extractions=[extraction("屁股", "屁股", "body_part"),
                                                            extraction("疼痛", "痛")])
    queries = []

    async def shortlist(texts, catalog):
        queries.append(texts)
        return None

    with patch.object(reasoning.concept_shortlist, "shortlist", new=shortlist):
        retrieved = reason(case, Provider([("痛", "臀部疼痛", "same")]))
    # 「屁股」 still counts after 「大腿」 replaced it as the current part; only part and symptom of
    # one message are joined, and 「大腿有點酸」 already holds its part. 「屁股好痛」 is the degree-word anchor.
    assert set(queries[0]) - {"痛", "疼痛", "大腿有點酸", "酸"} == {"屁股痛", "屁股好痛"}
    # The anchor stays the patient's own 「痛」.
    assert REHAB and {(row["patient_source_text"], row["dept_id"]) for row in retrieved} == {("痛", dept) for dept in REHAB}


def test_touching_part_and_symptom_add_no_joined_query():
    case = sore_buttock("buttock-touching")
    assert reasoning.part_symptom_queries(case, reasoning.accepted_semantic_evidence(case)) == set()


def split_buttock(case_id):
    return sore_buttock(case_id, chief_extractions=[extraction("屁股", "屁股", "body_part"), extraction("疼痛", "痛")])


def test_a_degree_word_between_part_and_symptom_gives_one_verbatim_anchor():
    # Replay 2026-11-04: with 「屁股」 and 「痛」 split, both AI steps saw only 「痛」, so retrieval
    # wrote the non-verbatim 「屁股痛」 and the audit refused 「原文只說痛」.
    retrieved = reason(split_buttock("buttock-bridge"), Provider([("屁股好痛", "臀部疼痛", "same")]))
    assert REHAB and {(row["patient_source_text"], row["dept_id"]) for row in retrieved} == {
        ("屁股好痛", dept) for dept in REHAB}
    # Scored once, as its two parts.
    assert {tuple(row["patient_source_parts"]) for row in retrieved} == {("屁股", "痛")}


def test_the_degree_word_anchor_is_kept_on_later_turns():
    case = split_buttock("buttock-bridge-retained")
    case.conversation_state.semantic_official_links = [
        SemanticOfficialLink(patient_source_text="屁股好痛", knowledge_concept="臀部疼痛", body_part_match="same")]
    assert {row["dept_id"] for row in reason(case, Provider([]))} == REHAB


def test_only_degree_words_bridge_a_part_and_a_symptom():
    evidence = [extraction("屁股", "屁股", "body_part"), extraction("疼痛", "痛")]
    evidence = [item.model_dump() for item in evidence]

    def bridged(text):
        return [fragment for fragment, _ in reasoning.adjacent_present_spans([text], evidence, bridge_degree_words=True)]

    assert bridged("我的屁股好痛") == ["屁股好痛"]
    assert bridged("屁股真的很痛") == ["屁股真的很痛"]
    # A clause break, another word or a long gap is no bridge.
    assert bridged("屁股，好痛") == bridged("屁股坐久了痛") == bridged("屁股有點非常真的痛") == []
    # Without the bridge only touching spans join.
    assert reasoning.adjacent_present_spans(["我的屁股好痛"], evidence) == []


WAIST = {record["canonical_dept_id"] for record in lookup_concept("腰酸", RECORDS)}


def sore_waist(case_id, chief_extractions, message="我的腰好酸"):
    case = TriageCase(case_id=case_id)
    case.history_records = [Message(role="user", content=message)]
    case.semantic_extractions = chief_extractions
    return case


def test_degree_words_after_a_body_part_are_dropped_for_literal_matching():
    parts = {"腰"}
    assert reasoning.without_degree_words_after_parts("我的腰好酸", parts) == "我的腰酸"
    assert reasoning.without_degree_words_after_parts("腰真的很酸", parts) == "腰酸"
    # Negation, other words or a long gap stay.
    for text in ("我的腰不酸", "腰坐久了酸", "腰有點非常真的酸"):
        assert reasoning.without_degree_words_after_parts(text, parts) == text
    assert reasoning.without_degree_words_after_parts("額頭好痛", {"額頭", "頭"}) == "額頭痛"


def test_a_degree_word_after_the_part_no_longer_hides_a_literal_concept():
    # Real AI 2026-11-04: 「我的腰好酸」 was normalized to 「酸痛」 and the audit refused 「酸」→「腰部酸痛」,
    # so 2 of 5 runs ended with no department although 「腰酸」 is official.
    case = sore_waist("waist-literal", [extraction("腰", "腰", "body_part"), extraction("酸痛", "腰好酸")])
    literal = reasoning.retrieve_official_evidence(case, RECORDS, RESOLUTIONS)
    assert WAIST and {(row["patient_source_text"], row["knowledge_concept"], row["dept_id"]) for row in literal} == {
        ("腰好酸", "腰酸", dept) for dept in WAIST}


def test_a_split_part_and_symptom_across_a_degree_word_match_literally_once():
    case = sore_waist("waist-split", [extraction("腰", "腰", "body_part"), extraction("酸痛", "酸")])
    literal = reasoning.retrieve_official_evidence(case, RECORDS, RESOLUTIONS)
    assert {(row["patient_source_text"], row["knowledge_concept"]) for row in literal} == {("腰好酸", "腰酸")}
    assert {tuple(row["patient_source_parts"]) for row in literal} == {("腰", "酸")}


def test_dropping_degree_words_keeps_the_inside_a_word_rule_and_negation():
    case = sore_waist("forehead", [extraction("額頭", "額頭", "body_part"), extraction("疼痛", "額頭好痛")], "額頭好痛")
    assert "頭痛" not in {row["knowledge_concept"] for row in reasoning.retrieve_official_evidence(case, RECORDS, RESOLUTIONS)}
    absent = extraction("酸痛", "腰不酸")
    absent.assertion = "absent"
    case = sore_waist("waist-absent", [extraction("腰", "腰", "body_part"), absent], "我的腰不酸")
    assert reasoning.retrieve_official_evidence(case, RECORDS, RESOLUTIONS) == []


class FlakyProvider(Provider):
    """The first call for each AI step fails, as when Cerebras does not answer."""

    def __init__(self, links, failures=1):
        super().__init__(links)
        self.failures = {"department_semantic_retrieval": failures, "department_retrieval_entailment": failures}

    async def __call__(self, prompt, *, purpose):
        if self.failures.get(purpose):
            self.failures[purpose] -= 1
            raise TimeoutError
        return await super().__call__(prompt, purpose=purpose)


def test_a_failed_ai_step_is_retried_once():
    retrieved = reason(split_buttock("buttock-retry"), FlakyProvider([("屁股好痛", "臀部疼痛", "same")]))
    assert {row["dept_id"] for row in retrieved} == REHAB
    assert reason(split_buttock("buttock-retry-twice"), FlakyProvider([("屁股好痛", "臀部疼痛", "same")], 2)) == []


class Refused(Exception):
    status_code = 402


def test_a_refused_ai_request_is_not_retried():
    class RefusedProvider(Provider):
        calls = 0

        async def __call__(self, prompt, *, purpose):
            if purpose == "department_semantic_retrieval":
                self.calls += 1
                raise Refused
            return await super().__call__(prompt, purpose=purpose)

    provider = RefusedProvider([("屁股好痛", "臀部疼痛", "same")])
    assert reason(split_buttock("buttock-refused"), provider) == []
    assert provider.calls == 1


def test_invalid_audit_part_view_is_ignored():
    case = sore_buttock("buttock-thigh-invalid")
    assert reason(case, Provider([("大腿有點酸", "下肢酸痛", "maybe")])) == []
    assert [link.body_part_match for link in case.conversation_state.semantic_official_links] == [None]
