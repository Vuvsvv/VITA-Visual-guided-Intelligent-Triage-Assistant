"""Official concepts can match across touching present spans of one patient phrase."""
import asyncio
import json
from unittest.mock import AsyncMock, patch

from app.schemas import Message, SemanticExtraction, SemanticOfficialLink, TriageCase
from app.services import department_clues as clues
from app.services import department_reasoning_service as service
from app.services.department_knowledge import load_department_knowledge, lookup_concept
from app.services.rule_engine import apply_semantic_extractions

_, RECORDS = load_department_knowledge()
RESOLUTIONS = [dict(status="resolved", canonical_dept_id=r["canonical_dept_id"], db_dept_id=r["canonical_dept_id"],
                    db_parent_dept=r["canonical_parent_dept"], db_child_dept=r["canonical_child_dept"])
               for r in RECORDS]
ACTIVE = [dict(dept_id=r["canonical_dept_id"], parent_dept=r["canonical_parent_dept"],
               child_dept=r["canonical_child_dept"])
          for r in {item["canonical_dept_id"]: item for item in RECORDS}.values()]


def item(field, value, source, assertion="present"):
    if field == "accompanying_symptoms" and isinstance(value, str):
        value = [value]
    status = {"absent": "unavailable", "uncertain": "ambiguous"}.get(assertion, "available")
    return SemanticExtraction(field=field, normalized_value=value, source_text=source, semantic_status=status,
                              assertion=None if field == "body_part" else assertion, confidence=.99,
                              extractor="ai")


def interpreted(*texts_and_items):
    """Mimic Turn Interpreter output going through the same acceptance gate as live turns."""
    case = TriageCase(case_id="adjacent-spans")
    for text, extractions in texts_and_items:
        case.history_records.append(Message(role="user", content=text))
        apply_semantic_extractions(case, list(extractions), allow_red_flag_completion=False)
    return case


def split_back_pain(text="周一開始腰酸背痛", first="腰酸"):
    return interpreted((text, [item("symptom", first, first),
                               item("accompanying_symptoms", "背痛", "背痛"),
                               item("onset", "周一開始", "周一開始", assertion=None)]))


def literal(case):
    return service.retrieve_official_evidence(case, RECORDS, RESOLUTIONS)


def pairs(retrieved):
    return {(row["patient_source_text"], row["knowledge_concept"]) for row in retrieved}


def compound(retrieved):
    # 「背痛」alone has been an official concept since 2026-11-03 (Taipei VGH fid 6730);
    # these tests are about the joined 腰酸背痛 anchor, so they look only at those rows.
    return [row for row in retrieved if row["knowledge_concept"] == "腰酸背痛"]


def test_split_compound_phrase_matches_whole_official_concept():
    case = split_back_pain()
    retrieved = literal(case)
    # 「腰酸」 alone is an official 復健科 concept since 2026-11-04 (三軍總醫院, MND open data).
    assert pairs(retrieved) == {("腰酸背痛", "腰酸背痛"), ("背痛", "背痛"), ("腰酸", "腰酸")}
    retrieved = compound(retrieved)
    assert {row["childDept"] for row in retrieved} >= {"一般骨科", "神經內科", "神經外科", "復健醫學", "疼痛控制科"}
    assert {row["dept_id"] for row in retrieved} == {r["canonical_dept_id"] for r in lookup_concept("腰酸背痛", RECORDS)}


def test_split_compound_phrase_reaches_department_convergence_payload():
    case = split_back_pain()
    case.patient_input.red_flags_checked = True
    payloads = []

    async def provider(prompt, *, purpose):
        payloads.append((purpose, prompt))
        return json.dumps({"status": "ambiguous", "candidates": []})

    with patch.object(service, "fetch_active_departments", return_value=ACTIVE), \
            patch.object(service, "complete_runtime_json", new=provider):
        asyncio.run(service.reason_about_departments(case))
    # 2026-10-28: semantic retrieval also runs when literal matching hits; this provider's reply
    # is not a valid retrieval, so only the literal evidence reaches convergence.
    assert [purpose for purpose, _ in payloads] == ["department_semantic_retrieval", "department_convergence"]
    data = json.loads(payloads[-1][1].split("\n資料：", 1)[1])
    assert {row["childDept"] for row in data["retrieved_official_evidence"]} >= {
        "一般骨科", "神經內科", "神經外科", "復健醫學", "疼痛控制科"}


def test_body_part_and_touching_symptom_match_whole_concept():
    case = interpreted(("下背痛", [item("body_part", "下背", "下背"), item("symptom", "疼痛", "痛")]))
    # 「背痛」 starts inside the body part 下背 and inside 下背痛, so it is left to semantic retrieval.
    assert pairs(literal(case)) == {("下背痛", "下背痛")}


def test_concept_already_covered_by_shorter_chain_is_not_reanchored():
    case = interpreted(("腰酸背痛無力", [item("symptom", "腰酸", "腰酸"), item("accompanying_symptoms", "背痛", "背痛"),
                                       item("accompanying_symptoms", "無力", "無力")]))
    assert ("腰酸背痛無力", ("腰酸", "背痛", "無力")) in service.adjacent_present_spans(
        service._user_texts(case), service.accepted_semantic_evidence(case))
    assert {pair for pair in pairs(literal(case)) if pair[1] == "腰酸背痛"} == {("腰酸背痛", "腰酸背痛")}


def test_body_part_spans_alone_never_form_an_anchor():
    case = interpreted(("腰背", [item("body_part", "腰", "腰"), item("body_part", "背", "背")]))
    assert service.adjacent_present_spans(service._user_texts(case), service.accepted_semantic_evidence(case)) == []


def test_non_touching_spans_are_not_joined():
    case = interpreted(("腰酸，背痛", [item("symptom", "腰酸", "腰酸"), item("accompanying_symptoms", "背痛", "背痛")]))
    assert compound(literal(case)) == []


def test_absent_or_uncertain_span_never_takes_part():
    for assertion in ("absent", "uncertain"):
        case = interpreted(("腰酸背痛", [item("symptom", "腰酸", "腰酸"),
                                       item("accompanying_symptoms", "背痛", "背痛", assertion=assertion)]))
        # Only the present 「腰酸」 matches, on its own (an official concept since 2026-11-04).
        assert pairs(literal(case)) == {("腰酸", "腰酸")}, assertion


def test_current_absent_or_uncertain_concept_blocks_joined_anchor():
    for assertion in ("absent", "uncertain"):
        case = interpreted(
            ("周一開始腰酸背痛", [item("symptom", "腰酸", "腰酸"), item("accompanying_symptoms", "背痛", "背痛")]),
            ("應該不是腰酸背痛", [item("accompanying_symptoms", "腰酸背痛", "不是腰酸背痛", assertion=assertion)]),
        )
        assert compound(literal(case)) == [], assertion


def test_span_repeated_in_message_has_no_certain_position():
    case = interpreted(("腰酸背痛，腰酸到不行", [item("symptom", "腰酸", "腰酸"),
                                            item("accompanying_symptoms", "背痛", "背痛")]))
    assert compound(literal(case)) == []


def test_denied_compound_phrase_never_matches():
    case = interpreted(("沒有腰酸背痛", [item("symptom", "腰酸背痛", "沒有腰酸背痛", assertion="absent")]))
    assert literal(case) == []
    case = interpreted(("沒有腰酸背痛", [item("symptom", "腰酸", "沒有腰酸", assertion="absent"),
                                       item("accompanying_symptoms", "背痛", "背痛")]))
    assert compound(literal(case)) == []


def _semantic(case, matches):
    judgments = [dict(patient_source_text=m["patient_source_text"], knowledge_concept=m["knowledge_concept"],
                      supported=True, reason="整段原話相符") for m in matches]
    provider = AsyncMock(side_effect=[json.dumps(dict(matches=matches), ensure_ascii=False),
                                      json.dumps(dict(judgments=judgments), ensure_ascii=False)])
    with patch.object(service, "complete_runtime_json", provider):
        result = asyncio.run(service.retrieve_semantic_official_evidence(case, RECORDS, RESOLUTIONS))
    return result, provider


def _payload(prompt):
    return json.loads(prompt[prompt.index("\n{") + 1:])


def test_variant_character_split_phrase_offers_whole_anchor_to_semantic_retrieval():
    case = split_back_pain("周一開始腰痠背痛", first="腰痠")
    assert compound(literal(case)) == []
    result, provider = _semantic(case, [dict(patient_source_text="腰痠背痛", knowledge_concept="腰酸背痛",
                                             confidence=.97)])
    retrieval, audit = (call.args[0] for call in provider.await_args_list)
    assert _payload(retrieval)["combined_present_source_texts"] == ["腰痠背痛"]
    assert _payload(audit)["combined_present_source_texts"] == ["腰痠背痛"]
    assert _payload(audit)["proposed_matches"] == [dict(patient_source_text="腰痠背痛", knowledge_concept="腰酸背痛")]
    assert pairs(result) == {("腰痠背痛", "腰酸背痛")}


def test_semantic_anchor_outside_joined_fragments_is_still_rejected():
    case = split_back_pain("周一開始腰痠，背痛", first="腰痠")
    result, provider = _semantic(case, [dict(patient_source_text="腰痠，背痛", knowledge_concept="腰酸背痛",
                                             confidence=.97)])
    assert result == []
    assert "combined_present_source_texts" not in provider.await_args_list[0].args[0]


def test_retained_link_on_joined_anchor_survives_until_a_part_is_superseded():
    case = split_back_pain("周一開始腰痠背痛", first="腰痠")
    case.conversation_state.semantic_official_links = [
        SemanticOfficialLink(patient_source_text="腰痠背痛", knowledge_concept="腰酸背痛")]
    assert pairs(service.retained_semantic_official_evidence(case, RECORDS, RESOLUTIONS)) == {("腰痠背痛", "腰酸背痛")}
    case.history_records.append(Message(role="user", content="其實背不痛"))
    apply_semantic_extractions(case, [item("accompanying_symptoms", "背痛", "背不痛", assertion="absent")],
                               allow_red_flag_completion=False)
    assert service.retained_semantic_official_evidence(case, RECORDS, RESOLUTIONS) == []
    assert case.conversation_state.semantic_official_links == []


def _custom(*concepts_by_dept):
    records = [dict(department_name=f"科{dept_id}", concept=concept, canonical_dept_id=dept_id,
                    evidence_type="direct_routing_evidence", source_id=f"src-{dept_id}-{concept}",
                    evidence_text=concept, source_priority=1)
               for dept_id, concept in concepts_by_dept]
    resolutions = [dict(status="resolved", canonical_dept_id=dept_id, db_dept_id=dept_id, db_parent_dept="測試系",
                        db_child_dept=f"科{dept_id}") for dept_id in {dept_id for dept_id, _ in concepts_by_dept}]
    return records, resolutions


def test_two_independent_symptoms_touching_add_no_extra_result():
    touching = interpreted(("頭痛發燒", [item("symptom", "頭痛", "頭痛"), item("accompanying_symptoms", "發燒", "發燒")]))
    apart = interpreted(("頭痛，發燒", [item("symptom", "頭痛", "頭痛"), item("accompanying_symptoms", "發燒", "發燒")]))
    retrieved = literal(touching)
    assert retrieved == literal(apart)
    assert {row["patient_source_text"] for row in retrieved} == {"頭痛", "發燒"}
    assert not any("patient_source_parts" in row for row in retrieved)
    assert [(e.field, e.source_text) for e in touching.semantic_extractions] == [
        ("symptom", "頭痛"), ("accompanying_symptoms", "發燒")]
    assert touching.patient_input.symptom == "頭痛"
    assert touching.patient_input.accompanying_symptoms == ["發燒"]


def test_joined_match_leaves_part_evidence_and_fields_unchanged():
    case = split_back_pain()
    before = [e.model_dump() for e in case.semantic_extractions]
    evidence_before = service.accepted_semantic_evidence(case)
    retrieved = compound(literal(case))
    assert retrieved and all(row["patient_source_parts"] == ["腰酸", "背痛"] for row in retrieved)
    assert [e.model_dump() for e in case.semantic_extractions] == before
    assert service.accepted_semantic_evidence(case) == evidence_before
    assert [(e["field"], e["source_text"]) for e in evidence_before if e["assertion"] == "present"] == [
        ("symptom", "腰酸"), ("accompanying_symptoms", "背痛")]
    assert case.patient_input.symptom == "腰酸"
    assert case.patient_input.accompanying_symptoms == ["背痛"]


def test_joined_fragment_and_its_parts_score_once_per_department():
    records, resolutions = _custom((1, "腰酸背痛"), (1, "背痛"), (2, "腰酸背痛"), (3, "腰酸背痛"), (3, "腰酸"),
                                   (3, "背痛"))
    case = split_back_pain()
    retrieved = service.retrieve_official_evidence(case, records, resolutions)
    assert pairs(retrieved) == {("腰酸背痛", "腰酸背痛"), ("背痛", "背痛"), ("腰酸", "腰酸")}
    scores = clues.department_scores(retrieved, {1, 2, 3}, [])
    # 1: part + joined fragment -> 1; 2: joined fragment only -> 1; 3: both parts already count -> 2.
    assert scores == {1: 1, 2: 1, 3: 2}


def test_semantic_and_retained_joined_rows_also_score_once():
    case = split_back_pain("周一開始腰痠背痛", first="腰痠")
    result, _ = _semantic(case, [dict(patient_source_text="腰痠背痛", knowledge_concept="腰酸背痛", confidence=.97)])
    assert all(row["patient_source_parts"] == ["腰痠", "背痛"] for row in result)
    case.conversation_state.semantic_official_links = [
        SemanticOfficialLink(patient_source_text="腰痠背痛", knowledge_concept="腰酸背痛")]
    retained = service.retained_semantic_official_evidence(case, RECORDS, RESOLUTIONS)
    assert all(row["patient_source_parts"] == ["腰痠", "背痛"] for row in retained)
    plain_back = [dict(row, patient_source_text="背痛", knowledge_concept="背痛", knowledge_source_id="x")
                  for row in retained if "patient_source_parts" in row]
    for row in plain_back:
        row.pop("patient_source_parts")
    dept_ids = {row["dept_id"] for row in retained}
    assert clues.department_scores(retained + plain_back, dept_ids, []) == {dept_id: 1 for dept_id in dept_ids}
