"""Pure helpers for knowledge-based clue questions (docs/修改計畫.md 第 9 節「線索題收斂」).

No AI calls, no SQL and no settings reads: callers pass candidates, official records,
validated evidence and limits. Patient text is never parsed here; concepts are compared
only through official KB strings and AI-structured normalized values.

Scoring counts distinct grounded pieces of patient text per candidate department:
distinct ``patient_source_text`` in retrieved official evidence plus distinct clue
``question_index`` answered present for that department. Commit 3 把 concept_answers 轉成
語意證據時，必須標記來源是線索題回答，計分時從 retrieved 的文字計數中排除，避免和題號計數重複加分。
Such items carry ``evidence_origin == CLUE_ANSWER_ORIGIN`` and are skipped by the text count.
An item anchored on a joined fragment of touching spans lists them in ``patient_source_parts``;
it scores only when none of those parts is already counted for that department.
A concept held by several candidates credits every department in its ``dept_ids``.

``ConversationState.department_clue_offer`` holds only the top ``clue_planner_concept_limit``
ranked clues for the planner. 平手加問（Commit 5）必須從全部可問線索找，不能只用
department_clue_offer：tiebreak_clues() must receive the full clue_concepts() result.
2026-10-28 起平手改採收斂 AI 排名（decide 的 ai_order），決定流程不再呼叫 tiebreak_clues()。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from app.schemas import ClueConceptRecord, ClueOffer, ConversationState
from app.services.department_knowledge import lookup_concept, normalize_concept

CLUE_ANSWER_ORIGIN = "clue_answer"
MAX_CLUE_CONCEPT_RECORDS = 30
CLUE_INTENT_PREFIX = "department_clue_"
# The official routing tables list symptoms and diseases together, so wording covers both.
CLUE_TEMPLATE = "請問有沒有以下情況，或以前曾被告知有：{}？"
CLUE_REASK_TEMPLATE = "想再確認一次，請問有沒有以下情況，或以前曾被告知有：{}？"
_SYMPTOM_FIELDS = {"symptom", "accompanying_symptoms"}
# Sexually transmitted diseases, cancer diagnoses and pregnancy offend or frighten when asked
# unprompted; they still route when the patient raises them.
_NEVER_ASK_MARKERS = ("愛滋", "梅毒", "淋病", "菜花", "披衣菌", "性病", "癌", "腫瘤", "惡性", "白血病", "懷孕")
_CONCLUDED_ANSWERS = {"present", "absent"}

DecisionKind = Literal["ask_clue", "ask_companion", "decide", "tiebreak", "family_medicine", "defer_to_safety_net"]


@dataclass(frozen=True)
class ClueConcept:
    concept: str
    dept_ids: frozenset[int]
    unique: bool
    has_direct_routing: bool
    best_source_priority: int
    uncertain_before: bool


@dataclass(frozen=True)
class ClueDecision:
    kind: DecisionKind
    dept_id: int | None = None
    tied_dept_ids: tuple[int, ...] = ()
    reason: str = ""


def candidate_dept_ids(retrieved: list[dict]) -> set[int]:
    """All departments officially retrieved this turn, not only the convergence AI's top 3."""
    return {item["dept_id"] for item in retrieved}


def mentioned_concepts(retrieved: list[dict], accepted_evidence: list[dict]) -> set[str]:
    """Normalized concepts the patient already covered, from official matches and AI values."""
    mentioned = {normalize_concept(item["knowledge_concept"]) for item in retrieved}
    for evidence in accepted_evidence:
        value = evidence.get("normalized_value")
        if evidence.get("field") in _SYMPTOM_FIELDS and isinstance(value, str) and value.strip():
            mentioned.add(normalize_concept(value))
    return mentioned


def clue_concepts(
    records: list[dict], candidates: set[int], mentioned: set[str], asked: list[ClueConceptRecord],
    non_target_dept_ids: Iterable[int] = (),
) -> list[ClueConcept]:
    """Distinguishing official concepts of the candidates that the patient has not covered.

    Records are matched by ``canonical_dept_id``, which equals the live ``dept_id`` once
    resolved. A concept held by every candidate cannot distinguish them and is dropped.
    Concepts answered present/absent are never asked again; uncertain/unanswered stay.
    Only concepts with at least one candidate ``direct_routing_evidence`` record (the
    hospital's symptom-to-department guidance) are askable; their ``dept_ids`` still hold
    every candidate with the concept. Specialty-only concepts are never asked but keep
    their part in retrieval and scoring.
    A concept held only by ``non_target_dept_ids`` (the family-medicine end point) is never
    asked; other concepts keep those departments in ``dept_ids`` so scoring is unchanged.
    Sensitive concepts (``never_ask_concept``) are never asked either.
    """
    non_target = set(non_target_dept_ids)
    concluded = {normalize_concept(item.concept) for item in asked if item.answer in _CONCLUDED_ANSWERS}
    inconclusive = {normalize_concept(item.concept) for item in asked if item.answer not in _CONCLUDED_ANSWERS}
    grouped: dict[str, dict] = {}
    for record in records:
        dept_id = record["canonical_dept_id"]
        key = normalize_concept(record["concept"])
        if dept_id not in candidates or key in mentioned or key in concluded or never_ask_concept(key):
            continue
        entry = grouped.setdefault(key, {"names": set(), "depts": set(), "direct": False, "priority": None})
        entry["names"].add(record["concept"])
        entry["depts"].add(dept_id)
        entry["direct"] = entry["direct"] or record["evidence_type"] == "direct_routing_evidence"
        priority = record["source_priority"]
        entry["priority"] = priority if entry["priority"] is None else min(entry["priority"], priority)
    clues = []
    for key, entry in grouped.items():
        if entry["depts"] == candidates or not entry["direct"] or not entry["depts"] - non_target:
            continue
        clues.append(ClueConcept(
            concept=min(entry["names"]), dept_ids=frozenset(entry["depts"]),
            unique=len(entry["depts"]) == 1, has_direct_routing=entry["direct"],
            best_source_priority=entry["priority"], uncertain_before=key in inconclusive,
        ))
    return clues


def never_ask_concept(concept: str) -> bool:
    """STD, cancer and pregnancy concepts are matched from patient wording but never asked."""
    key = normalize_concept(concept)
    return any(marker in key for marker in _NEVER_ASK_MARKERS)


DepartmentPriority = tuple[int, int]


def department_priority(
    scores: dict[int, int], ai_order: list[int], non_target_dept_ids: Iterable[int] = (),
) -> dict[int, DepartmentPriority]:
    """Target-department priority: higher score first, then the convergence AI's order.

    Departments absent from ``ai_order`` come after listed ones; equal priorities fall
    through to the clue's own rank keys. Non-target departments get no priority.
    """
    non_target = set(non_target_dept_ids)
    position = {dept_id: index for index, dept_id in enumerate(dict.fromkeys(ai_order))}
    return {
        dept_id: (-score, position.get(dept_id, len(position)))
        for dept_id, score in scores.items() if dept_id not in non_target
    }


def rank_clue_concepts(
    clues: Iterable[ClueConcept], limit: int, priority: dict[int, DepartmentPriority] | None = None,
) -> list[ClueConcept]:
    """Best target-department priority, then unique, direct routing, source priority,
    inconclusive last, then text. A concept takes its best-priority department."""
    def target(clue: ClueConcept) -> DepartmentPriority:
        if not priority:
            return (0, 0)
        # Scores are >= 0, so (1, 0) sorts a clue without any prioritized department last.
        return min((priority[dept_id] for dept_id in clue.dept_ids if dept_id in priority), default=(1, 0))

    ranked = sorted(clues, key=lambda clue: (
        target(clue), not clue.unique, not clue.has_direct_routing, clue.best_source_priority,
        clue.uncertain_before, clue.concept,
    ))
    return ranked[:max(limit, 0)]


def common_clue_departments(
    selected: list[str], offered: Iterable[ClueConcept | ClueOffer], per_question: int,
) -> set[int]:
    """Departments shared by 1..per_question distinct offered concepts; empty when invalid."""
    by_concept = {clue.concept: clue for clue in offered}
    if not 1 <= len(selected) <= per_question or len(set(selected)) != len(selected):
        return set()
    if any(concept not in by_concept for concept in selected):
        return set()
    common = set(by_concept[selected[0]].dept_ids)
    for concept in selected[1:]:
        common &= set(by_concept[concept].dept_ids)
    return common


def clue_offer(ranked: Iterable[ClueConcept]) -> list[ClueOffer]:
    """Planner-facing form of ranked clues: official concept and opaque department ids only."""
    return [ClueOffer(concept=clue.concept, dept_ids=sorted(clue.dept_ids)) for clue in ranked]


def clue_intent(question_index: int) -> str:
    """Opaque Backend/AI correlation key of one clue question."""
    return f"{CLUE_INTENT_PREFIX}{question_index}"


def pending_clue_index(state: ConversationState) -> int | None:
    """The pending clue question's index, only while it still owns the pending intent."""
    index = state.pending_clue_question_index
    if index is not None and state.pending_clarification_intent == clue_intent(index):
        return index
    return None


def pending_clue_concepts(state: ConversationState) -> list[str]:
    index = pending_clue_index(state)
    if index is None:
        return []
    return [record.concept for record in state.clue_concepts_asked if record.question_index == index]


def concepts_missing_from_question(question: str, concepts: Iterable[str]) -> list[str]:
    """Official concepts not quoted verbatim in the planner's question (AI output only)."""
    return [concept for concept in concepts if concept not in question]


def template_clue_selection(offer: list[ClueOffer], per_question: int) -> tuple[list[str], int | None]:
    """Up to per_question top-ranked offered concepts sharing the first clue's department."""
    if not offer or not offer[0].dept_ids:
        return [], None
    dept_id = min(offer[0].dept_ids)
    selected = [item.concept for item in offer if dept_id in item.dept_ids][:max(per_question, 1)]
    return selected, dept_id


def clue_template_question(concepts: list[str], *, reask: bool = False) -> str:
    """Backend-owned fallback wording; it only lists already validated official concepts."""
    return (CLUE_REASK_TEMPLATE if reask else CLUE_TEMPLATE).format("、".join(concepts))


def append_clue_records(
    asked: list[ClueConceptRecord], concepts: list[str], dept_id: int, question_index: int,
    limit: int = MAX_CLUE_CONCEPT_RECORDS, dept_ids: dict[str, list[int]] | None = None,
) -> list[ClueConceptRecord]:
    """Return a new bounded list with one question's concepts appended as unanswered."""
    added = [ClueConceptRecord(concept=concept, dept_id=dept_id, question_index=question_index,
                               dept_ids=list((dept_ids or {}).get(concept) or [dept_id]))
             for concept in concepts]
    return (list(asked) + added)[-limit:]


def clue_answer_evidence(
    asked: list[ClueConceptRecord], records: list[dict], resolutions: list[dict], contradicted: set[str],
) -> list[dict]:
    """Official evidence rebuilt from present clue answers, marked with CLUE_ANSWER_ORIGIN.

    A concept that current semantic evidence marks absent/uncertain is skipped.
    """
    resolved = {item["canonical_dept_id"]: item for item in resolutions if item["status"] == "resolved"}
    found: dict[tuple, dict] = {}
    for item in asked:
        source = (item.answer_source_text or "").strip()
        if item.answer != "present" or not source or normalize_concept(item.concept) in contradicted:
            continue
        for record in lookup_concept(item.concept, records):
            row = resolved.get(record["canonical_dept_id"])
            if row is None:
                continue
            key = (row["db_dept_id"], source, record["source_id"], record["concept"])
            found[key] = dict(
                dept_id=row["db_dept_id"], parentDept=row["db_parent_dept"], childDept=row["db_child_dept"],
                patient_source_text=source, knowledge_source_id=record["source_id"],
                knowledge_concept=record["concept"], knowledge_evidence_text=record["evidence_text"],
                source_priority=record["source_priority"], evidence_origin=CLUE_ANSWER_ORIGIN,
            )
    return list(found.values())


def answered_concept_departments(record: ClueConceptRecord, records: list[dict] | None) -> set[int]:
    """Departments that currently hold an answered concept in the official knowledge base.

    The ``dept_ids`` snapshot taken when the question was asked is used only when no
    records are given or the knowledge base no longer lists the concept.
    """
    if records is not None:
        holders = {item["canonical_dept_id"] for item in lookup_concept(record.concept, records)}
        if holders:
            return holders
    return set(record.dept_ids)


def department_scores(
    retrieved: list[dict], candidates: set[int], asked: list[ClueConceptRecord],
    records: list[dict] | None = None,
) -> dict[int, int]:
    """Distinct retrieved text pieces plus distinct present clue questions, per candidate.

    Synonyms matched from one text piece score once; absent answers never subtract.
    A present concept credits every current candidate holding it in ``records``.
    A text piece that restates a present clue answer for a department (it matched that
    clue's concept, or it lies inside the answer's source text) scores only through the
    question: 病人答「有耳鳴」時 Turn Interpreter 另外抽出的「耳鳴」不會再加 1 分。
    """
    texts: dict[int, set[str]] = {dept_id: set() for dept_id in candidates}
    questions: dict[int, set[int]] = {dept_id: set() for dept_id in candidates}
    answered_concepts: dict[int, set[str]] = {dept_id: set() for dept_id in candidates}
    answered_sources: dict[int, set[str]] = {dept_id: set() for dept_id in candidates}
    for record in asked:
        if record.answer != "present":
            continue
        for dept_id in answered_concept_departments(record, records):
            if dept_id in questions:
                questions[dept_id].add(record.question_index)
                answered_concepts[dept_id].add(normalize_concept(record.concept))
                if (record.answer_source_text or "").strip():
                    answered_sources[dept_id].add(record.answer_source_text.strip())
    restated: dict[int, set[str]] = {dept_id: set() for dept_id in candidates}
    joined: list[dict] = []
    for item in retrieved:
        dept_id = item.get("dept_id")
        if dept_id not in texts or item.get("evidence_origin") == CLUE_ANSWER_ORIGIN:
            continue
        source = item["patient_source_text"]
        if (normalize_concept(item.get("knowledge_concept") or "") in answered_concepts[dept_id]
                or any(source in answer for answer in answered_sources[dept_id])):
            restated[dept_id].add(source)
        if item.get("patient_source_parts"):
            joined.append(item)
        else:
            texts[dept_id].add(source)
    for dept_id, found in restated.items():
        texts[dept_id] -= found
    # A joined fragment never adds a point beyond the text pieces it is made of.
    covered = {dept_id: texts[dept_id] | restated[dept_id] for dept_id in candidates}
    for item in sorted(joined, key=lambda row: row["patient_source_text"]):
        pieces = {item["patient_source_text"], *item["patient_source_parts"]}
        if not pieces & covered[item["dept_id"]]:
            texts[item["dept_id"]].add(item["patient_source_text"])
        covered[item["dept_id"]] |= pieces
    return {dept_id: len(texts[dept_id]) + len(questions[dept_id]) for dept_id in candidates}


def scored_patient_sources(retrieved: list[dict]) -> list[str]:
    """Distinct patient text pieces behind retrieved evidence, clue answers excluded."""
    return sorted({item["patient_source_text"] for item in retrieved
                   if item.get("evidence_origin") != CLUE_ANSWER_ORIGIN})


def clue_confirmed_departments(
    asked: list[ClueConceptRecord], candidates: set[int], records: list[dict] | None = None,
) -> set[int]:
    """Current candidates holding a concept the patient answered present (current KB holders)."""
    return {
        dept_id for record in asked if record.answer == "present"
        for dept_id in answered_concept_departments(record, records)
    } & candidates


def tiebreak_clues(
    clues: Iterable[ClueConcept], tied: set[int], non_target_dept_ids: Iterable[int] = (),
    priority: dict[int, DepartmentPriority] | None = None,
) -> list[ClueConcept]:
    """Clues held by exactly one tied department that is not a non-target, in rank order."""
    non_target = set(non_target_dept_ids)
    eligible = [clue for clue in clues
                if len(clue.dept_ids & tied) == 1 and not clue.dept_ids & tied & non_target]
    return rank_clue_concepts(eligible, len(eligible), priority)


def asked_question_counts(
    asked: list[ClueConceptRecord], candidates: set[int], records: list[dict] | None = None,
) -> dict[int, int]:
    """Distinct answered clue questions that asked a concept each candidate currently holds."""
    questions: dict[int, set[int]] = {dept_id: set() for dept_id in candidates}
    for record in asked:
        if record.answer == "unanswered":
            continue
        for dept_id in answered_concept_departments(record, records):
            if dept_id in questions:
                questions[dept_id].add(record.question_index)
    return {dept_id: len(found) for dept_id, found in questions.items()}


def ranked_targets(
    candidates: set[int], scores: dict[int, int], ai_order: Iterable[int] = (),
    non_target_dept_ids: Iterable[int] = (),
) -> list[int]:
    """Target departments by score, then the convergence AI's order, then id."""
    position = {dept_id: index for index, dept_id in enumerate(dict.fromkeys(ai_order))}
    targets = candidates - set(non_target_dept_ids)
    return sorted(targets, key=lambda dept_id: (-scores.get(dept_id, 0), position.get(dept_id, len(position)), dept_id))


def clue_target_department(
    ranked: list[int], asked_counts: dict[int, int], confirmed: set[int],
) -> int | None:
    """The department the next clue question asks about.

    Until a clue answer is present, ranked departments are asked in turn. Afterwards the
    leader and the strongest challenger alternate: the one asked fewer times goes next,
    the leader on a draw, so a department never wins only by being asked more often.
    """
    if not ranked:
        return None
    if not confirmed & set(ranked):
        return next((dept_id for dept_id in ranked if not asked_counts.get(dept_id)), ranked[0])
    if len(ranked) == 1:
        return ranked[0]
    leader, challenger = ranked[0], ranked[1]
    return challenger if asked_counts.get(challenger, 0) < asked_counts.get(leader, 0) else leader


def _settled_reason(
    ranked: list[int], scores: dict[int, int], asked_counts: dict[int, int], confirmed: set[int],
) -> str | None:
    """Why the clue answers already settle two or more target departments, or None.

    A leader strictly ahead settles when the challenger had no present answer, or was asked
    at least as often. A top tie settles only when every tied department had a present
    answer and was asked equally often; an all-absent tie keeps asking.
    """
    leader, challenger = ranked[0], ranked[1]
    if scores.get(leader, 0) > scores.get(challenger, 0):
        if challenger not in confirmed or asked_counts.get(challenger, 0) >= asked_counts.get(leader, 0):
            return "clue_confirmed_single_department" if confirmed & set(ranked) == {leader} else "leader_confirmed"
        return None
    tied = [dept_id for dept_id in ranked if scores.get(dept_id, 0) == scores.get(leader, 0)]
    if all(dept_id in confirmed for dept_id in tied) and len({asked_counts.get(dept_id, 0) for dept_id in tied}) == 1:
        return "balanced_tie"
    return None


def decide(
    *, ai_resolved_dept_id: int | None, candidates: set[int], scores: dict[int, int],
    askable: list[ClueConcept], questions_asked: int, tiebreak_used: bool,
    user_declined: bool, clue_confirmed_dept_ids: set[int], question_limit: int,
    non_target_dept_ids: Iterable[int] = (), ai_order: Iterable[int] = (),
    turn_limit_reached: bool = False, min_questions: int = 0,
    asked_counts: dict[int, int] | None = None, companion_due: bool = False,
) -> ClueDecision:
    """Apply rules A/C and the decision triggers; every decision still needs user confirmation.

    With two or more target departments nothing settles before ``min_questions`` clue
    questions (unless the patient declines or no clue is askable); the convergence AI's
    resolution waits for them too. A single target department (family medicine is never
    one) first gets the open accompanying-symptom question while ``companion_due``.
    A tied top score goes to the tied department ranked first in ``ai_order`` (this turn's
    validated convergence candidates); only when none of them is ranked does it fall back
    to family medicine. No extra tiebreak question is asked, so ``tiebreak_used`` is unused.
    At the safety-net turn (``turn_limit_reached``) only a single top-scoring target
    department is decided; otherwise clues continue and the safety-net question is asked.
    """
    if not candidates:
        return ClueDecision("defer_to_safety_net", reason="no_candidates")
    ranked = ranked_targets(candidates, scores, ai_order, non_target_dept_ids)
    confirmed = clue_confirmed_dept_ids & candidates
    single_target = len(ranked) <= 1
    enough_questions = single_target or not askable or questions_asked >= min_questions
    if user_declined:
        reason = "user_declined"
    elif single_target and companion_due:
        return ClueDecision("ask_companion", reason="single_candidate")
    elif ai_resolved_dept_id is not None and ai_resolved_dept_id in candidates and enough_questions:
        return ClueDecision("decide", dept_id=ai_resolved_dept_id, reason="ai_resolved")
    elif single_target:
        reason = "single_candidate"
    elif not askable:
        reason = "no_askable_clues"
    elif questions_asked >= question_limit:
        reason = "question_limit"
    elif enough_questions and (settled := _settled_reason(ranked, scores, asked_counts or {}, confirmed)):
        reason = settled
    elif turn_limit_reached:
        reason = "turn_limit_top_score"
    else:
        return ClueDecision("ask_clue", reason="continue")
    best = max(scores.get(dept_id, 0) for dept_id in candidates)
    tied = sorted(dept_id for dept_id in candidates if scores.get(dept_id, 0) == best)
    if reason == "turn_limit_top_score" and (len(tied) != 1 or tied[0] in set(non_target_dept_ids)):
        return ClueDecision("ask_clue", reason="continue")
    if len(tied) == 1:
        return ClueDecision("decide", dept_id=tied[0], reason=reason)
    ranked = [dept_id for dept_id in dict.fromkeys(ai_order) if dept_id in tied]
    if ranked:
        return ClueDecision("decide", dept_id=ranked[0], tied_dept_ids=tuple(tied), reason="tie_broken_by_ai_rank")
    return ClueDecision("family_medicine", tied_dept_ids=tuple(tied), reason=reason)
