"""Natural-language clarification planning with Backend-owned completion gates."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass

from app.config import get_settings
from app.schemas import (
    ClueDecisionRecord, ClueOffer, DialogueActInterpretation, PendingAnswerInterpretation, PriorAnswerReference,
    SemanticExtraction, TriageCase,
)
from app.services import department_clues as clues
from app.services import pediatric_routing as pediatric
from app.services.ai_service import complete_runtime_json
from app.services.confidence_scoring import ACCEPT_THRESHOLD
from app.services.field_acceptance import ai_normalized_value_valid
from app.services.rule_engine import QUESTION_TEXTS, RED_FLAG_QUESTION_KEY, mark_questions_asked

logger = logging.getLogger(__name__)
# Turn limits (safety_net_ask_turn, hard_turn_cap) live in config; neither establishes completion.
SAFETY_ANSWER_CAP = 3
SAFETY_NET_PENDING_INTENT = "safety_net_choice"
SAFETY_NET_QUESTION = "目前還沒辦法確定要看哪一科，要繼續描述，還是先掛家庭醫學科？"
SAFETY_NET_REASONS = frozenset({"safety_net_choice", "safety_net_turn_cap", "user_declined_no_candidates"})
# Set only when department reasoning ran and the knowledge base matched nothing the patient said.
NO_OFFICIAL_EVIDENCE_REASON = "目前沒有可精確對應正式科別的官方症狀證據"
BODY_PART_PENDING_INTENT = "symptom_body_part"
# Only one target department: an open question can surface symptoms pointing elsewhere.
COMPANION_PENDING_INTENT = "other_discomfort"
COMPANION_QUESTION_LIMIT = 2
COMPANION_FOLLOWUP_QUESTION = "還有其他不舒服的地方嗎？沒有的話直接說「沒有」就可以。"


def body_part_question(source: str) -> str:
    """Backend-owned fixed question for a symptom said without a body part."""
    return f"請問您說的「{source}」是在哪個部位呢？"


def companion_question(symptom: str | None, asked: int) -> str:
    """Backend-owned fixed accompanying-symptom question; it reads for the patient or a relative."""
    if asked:
        return COMPANION_FOLLOWUP_QUESTION
    if symptom:
        return f"除了「{symptom}」，最近還有沒有其他不舒服的地方？"
    return "最近還有沒有其他不舒服的地方？"


# Says why the chat stops, so a patient who only chatted does not read it as a dangerous finding.
# Android matches SAFETY_UNRESOLVED_MARKER in History.kt against it; no "119", which Android reads as urgent.
SAFETY_UNRESOLVED_REPLY = (
    "因為還沒能確認您是否有急迫症狀，系統不能繼續一般掛號導引。"
    "若身體不適，請洽醫療人員協助評估；若情況緊急，請立即尋求緊急醫療協助。"
)
# A patient-reported positive safety answer stops at once; it asserts no TTAS level.
SAFETY_POSITIVE_REPLY = (
    "您回報的狀況可能需要立即處理，線上問診無法判斷嚴重程度，因此不會繼續掛號導引。"
    "請立即前往急診；情況危急請撥打 119。"
)
SAFETY_RETRY_REPLY = "先回到剛剛的問題：以下狀況如果都沒有，回答「沒有」就可以。"
# Shown on the retry after which one more unclear answer stops the chat.
SAFETY_LAST_ATTEMPT_REPLY = "這題需要您回答「有」或「沒有」，系統才能繼續幫您找科別。"
SAFETY_SYMPTOM_ACK_LIMIT = 3
# Turns that do not count toward SAFETY_ANSWER_CAP; the hard turn cap still bounds them.
_UNCOUNTED_SAFETY_VALIDATIONS = frozenset({"request_explanation", "symptom_added"})
SAFETY_PENDING_INTENT = "safety_screen"
AVAILABILITY_PENDING_INTENT = "appointment_availability"
AVAILABILITY_QUESTION = "最近哪幾天、哪個時段比較方便看診？不方便的時間也可以告訴我；都可以的話直接說『都可以』。"
AVAILABILITY_SESSION_QUESTION = "那一天中，上午、下午還是晚上比較方便？都可以的話直接說『都可以』。"
AVAILABILITY_DATE_QUESTION = "那最近哪幾天比較方便？都可以的話直接說『都可以』。"
# Per dimension; the full question counts once for date and once for session.
AVAILABILITY_QUESTION_LIMIT = 2
AVAILABILITY_DIMENSIONS = ("date", "session")
UNRESOLVED_REPLY = "目前仍無法可靠釐清症狀，系統不會替你猜測科別；請改用手動選科或洽醫院掛號服務。"
# Only for provider failures, where sending the message again can help.
TECHNICAL_CLARIFICATION_REPLY = "系統剛剛沒有回應，請再傳一次剛才的訊息。"
# The provider answered but no planner question was usable: offer a way forward instead of a retry.
STUCK_OPEN_QUESTION = "可以再多說一點嗎？例如哪裡不舒服、從什麼時候開始、有多嚴重。"
# Asked once when the first message after 「我想修改」 changes nothing (例「我想要更改日期跟時段」).
REVISION_DETAIL_QUESTION = "請直接告訴我要改成什麼，例如「改成下週四下午」「改看皮膚科」，或補充新的症狀。"
SAFETY_EXPLANATION_FALLBACK = "這題是想確認您有沒有需要馬上去急診的狀況；如果都沒有，回答「沒有」就可以。"
_PROVIDER_FAILURES = frozenset({"provider_timeout", "provider_failure"})
FAMILY_MEDICINE_UNAVAILABLE_REPLY = "資訊不足，無法判斷科別。請改用手動選科或洽醫院掛號服務。"
DEPARTMENT_EVIDENCE_REPLY ="已記錄目前症狀，但正式科別資料仍不足以確認建議科別。您可以補充新的症狀資訊，或返回就診類型選擇「快速查詢」自行選科。"
_INTENT_PATTERN = re.compile(r"[a-z][a-z0-9_]{2,63}\Z")
_UNSAFE_QUESTION = re.compile(r"確診|診斷為|你患有|你得了|科|醫師|醫生|掛號|哪天有空|看診日期|看診時段|什麼時段方便")
_DURATION_QUESTION = re.compile(r"多久|幾天|幾週|幾個月|什麼時候開始")
_MULTI_QUESTION = re.compile(r"[?？].*[?？]|[;；]", re.DOTALL)
_EXPLANATION_PUNCTUATION = re.compile(r"[?？;；]")
_MULTI_DIMENSION_QUESTION = re.compile(
    r"(?:另外|並且|還有|以及).{0,12}(?:是否|有沒有|多久|何時|哪裡|什麼|如何|多少|幾次|幾天)|"
    r"(?:如何|多少|多久|幾次|幾天)[^？?]{0,40}[，、](?:是否|有沒有|何時|哪裡)"
)
_INTERNAL_STATE_TEXT = re.compile(
    r"已記錄本輪新資訊|上一個澄清(?:面向|問題)|目前仍需更多可靠|"
    r"科別仍不確定|科別.*(?:不確定|未收斂)|workflow|工作流程|"
    r"本輪症狀抽取|尚無\s*grounded|目前仍缺少可區分症狀|安全篩檢所列|確認是否有目前安全"
)
# Output-only guard for planner text: simplified-only characters absent from Taiwan Big5 (cp950).
_SIMPLIFIED_ONLY = re.compile(r"[这们说时为应发头无处认现过关门诊医难识边严态续疗该样实觉总给对还]")
_MEDICAL_FIELDS = {"symptom", "body_part", "duration", "severity", "onset", "accompanying_symptoms"}
_DETAIL_FIELDS = _MEDICAL_FIELDS - {"symptom"}
# Output-only guards for empty conversational filler, not patient medical NLP.
_GENERIC_INFORMATION_NEED = re.compile(r"更多(?:資訊|細節)|症狀細節|目前情況|剛才問題|上述問題|其他資訊")
_GENERIC_CLARIFICATION_QUESTION = re.compile(
    r"(?:請|可以|能否|可否)?(?:再)?(?:詳細)?(?:說明|補充|描述)(?:一下|目前情況)?(?:嗎)?[？?。]?|"
    r"(?:請|可以|能否|可否)?(?:再)?換個方式(?:說明|描述)(?:一下)?(?:嗎)?[？?。]?"
)


def requested_department_adopted(case: TriageCase) -> bool:
    """reason_about_departments adopted the patient's validated department request (例「我想要直接掛婦產科」).

    Like a Backend clue decision it completes the case once the safety screen is done, so
    the planner never adds questions (起因：修訂後 AI 改問「疼痛的程度」「下週二還是下週三」).
    """
    result = case.department_result
    requested = case.patient_input.requested_department_id
    return (case.conversation_state.department_status == "resolved" and result is not None
            and requested is not None and result.dept_id == requested)


def safety_screen_resolved(case: TriageCase) -> bool:
    return case.patient_input.red_flags_checked or bool(case.patient_input.red_flags)


def safety_symptom_acknowledgement(sources: list[str]) -> str:
    """Backend-owned line that echoes the symptoms a patient added instead of answering the screen."""
    quoted = "、".join(f"「{source}」" for source in sources)
    return f"已記下您說的{quoted}。繼續之前想先確認一件事，以下狀況如果都沒有，回答「沒有」就可以。"


def safety_unresolved_stop(case: TriageCase) -> bool:
    """The case ended because the emergency screen could not be answered reliably."""
    state = case.conversation_state
    return (
        state.clarification_status == "unresolved"
        and not state.safety_positive_reported
        and not safety_screen_resolved(case)
    )


@dataclass(frozen=True)
class ClarificationSuggestion:
    status: str
    question: str | None
    intent: str | None
    reason: str
    answered_intent: str | None = None
    answer_source_text: str | None = None
    answer_status: str | None = None
    answer_confidence: float | None = None
    information_need: str | None = None
    rejection_reason: str | None = None
    repair_attempted: bool = False
    reply_source: str = "AI_PLANNER"
    explanation: str | None = None
    # Validated official concepts of a clue question and the shared department it targets.
    clue_concepts: tuple[str, ...] = ()
    clue_dept_id: int | None = None


@dataclass(frozen=True)
class ClueQuestionSpec:
    """Backend-owned clue-mode contract for one planner call."""
    intent: str
    concepts: tuple[ClueOffer, ...]
    reask: bool
    per_question: int


def _clue_question_spec(case: TriageCase, safety_planning: bool) -> ClueQuestionSpec | None:
    """Clue mode re-asks a pending clue question, or asks the next one the decision allows.

    Before decide() runs (safety screen unresolved) clues are asked up to clue_question_limit;
    afterwards only ask_clue, or a tiebreak question that may exceed the limit, asks a new one.
    """
    if safety_planning:
        return None
    state = case.conversation_state
    index = clues.pending_clue_index(state)
    if index is not None:
        concepts = tuple(ClueOffer(concept=record.concept, dept_ids=record.dept_ids)
                         for record in state.clue_concepts_asked if record.question_index == index)
        return ClueQuestionSpec(clues.clue_intent(index), concepts, True, len(concepts)) if concepts else None
    settings = get_settings()
    kind = state.clue_decision.kind if state.clue_decision else None
    allowed = kind == "tiebreak" or (kind in {None, "ask_clue"}
                                     and state.clue_questions_asked < settings.clue_question_limit)
    if (allowed and state.pending_clarification_intent is None and state.department_status != "resolved"
            and state.department_clue_offer):
        return ClueQuestionSpec(clues.clue_intent(state.clue_questions_asked + 1),
                                tuple(state.department_clue_offer), False, settings.clue_concepts_per_question)
    return None


def settle_unanswered_clue_question(case: TriageCase) -> None:
    """Bound re-asking a clue question that got no accepted concept answer this turn.

    Runs before department reasoning, only on turns the Turn Interpreter completed. The first
    miss re-asks the same concepts once; the second marks its unanswered concepts uncertain and
    releases the pending question. Its question count is never changed. A patient declining
    further questions releases it at once so decide() can run this turn.
    """
    state = case.conversation_state
    index = clues.pending_clue_index(state)
    if index is None or state.last_dialogue_act == "request_explanation":
        return
    if state.clue_reask_count < 1 and state.last_dialogue_act != "decline_further_questions":
        state.clue_reask_count += 1
        logger.info("clue question case_id=%s index=%s reask=true", case.case_id, index)
        return
    for record in state.clue_concepts_asked:
        if record.question_index == index and record.answer == "unanswered":
            record.answer = "uncertain"
    state.pending_clarification_intent = None
    state.pending_clarification_question = None
    state.pending_clue_question_index = None
    state.clue_reask_count = 0
    state.next_information_needed = []
    logger.info("clue question case_id=%s index=%s reask_exhausted=true", case.case_id, index)


def validated_dialogue_act(
    dialogue_act: DialogueActInterpretation | None, user_sources: list[str],
) -> str | None:
    """Return the act only when it is confident and grounded in this turn's text."""
    if dialogue_act is None or dialogue_act.confidence < ACCEPT_THRESHOLD:
        return None
    source = dialogue_act.source_text.strip()
    if not source or not any(source in text for text in user_sources):
        return None
    return dialogue_act.act


def _pending_question(case: TriageCase) -> str | None:
    if case.conversation_state.pending_clarification_intent and case.conversation_state.pending_clarification_question:
        return case.conversation_state.pending_clarification_question
    return next(
        (item.content for item in reversed(case.history_records) if item.role == "assistant"),
        None,
    )


async def request_clarification(
    case: TriageCase,
    user_sources: list[str],
    *,
    focused_answer_status: str | None = None,
    dialogue_act: DialogueActInterpretation | None = None,
) -> ClarificationSuggestion | None:
    """Treat the provider's JSON as a proposal, never a workflow transition."""
    state = case.conversation_state
    safety_planning = state.clarification_status == "safety_check" and not safety_screen_resolved(case)
    act = validated_dialogue_act(dialogue_act, user_sources)
    clue = _clue_question_spec(case, safety_planning)
    required_intent = (
        SAFETY_PENDING_INTENT if safety_planning else
        clue.intent if clue is not None else
        state.pending_clarification_intent or state.department_next_question_intent
    )
    if required_intent in state.clarification_evidence:
        required_intent = None
    payload = {
        "conversation_history": [item.model_dump() for item in case.history_records[-24:]],
        "current_user_text": user_sources,
        "grounded_patient_evidence": case.patient_input.model_dump(),
        "semantic_extractions": [item.model_dump() for item in case.semantic_extractions[-20:]],
        "uncertainty_reasons": state.uncertainty_reasons,
        "next_information_needed": _patient_information_needs(state.next_information_needed),
        "asked_clarification_intents": state.asked_clarification_intents,
        "pending_clarification_intent": state.pending_clarification_intent,
        "pending_question": _pending_question(case) if state.pending_clarification_intent or safety_planning else None,
        "focused_pending_answer_status": focused_answer_status,
        "clarification_evidence": state.clarification_evidence,
        "department_status": state.department_status,
        "candidate_departments": [item.model_dump() for item in state.candidate_departments],
        "department_uncertainty_reason": state.department_uncertainty_reason,
        "department_next_information_needed": _patient_information_need(state.department_next_information_needed),
        "required_next_question_intent": required_intent,
        "safety_planning": safety_planning,
        "safety_screen_question": QUESTION_TEXTS[RED_FLAG_QUESTION_KEY] if safety_planning else None,
        "safety_answer_validation": state.safety_answer_validation,
        "dialogue_act": (dialogue_act or DialogueActInterpretation()).model_dump(),
    }
    if clue is not None:
        payload.update(
            clue_mode=True, clue_reask=clue.reask, clue_concepts_per_question=clue.per_question,
            clue_offer=[item.model_dump() for item in clue.concepts],
        )
    answer_contract = (
        "本輪 pending answer 已由 Turn Interpreter 與 semantic evidence 一併處理；你只規劃下一個問題，"
        "answered_intent、answer_source_text、answer_status、answer_confidence 必須全部輸出 null。"
    )
    prompt = (
        ("目前仍在安全澄清，不能改問科別問題。Backend 一律重問完整的 safety_screen_question，不採用你的 question，"
         "所以不要拆成單項追問；question 原樣填 safety_screen_question。使用者不確定、部分回答或要求解釋時，"
         "在 explanation 用 1～2 句白話說明清單中相關症狀的意思（例如「意識不清」指叫不醒、認不得人，"
         "不是頭昏，「突發」指突然出現），explanation 不得含問號或分號，不要把不知道、離題或部分否認當成整份陰性；不得新增醫療判級規則。"
         "intent 必須為 safety_screen；即使你認為已足夠，Backend 仍決定安全狀態。" if safety_planning else "") +
        "你是醫療問診的自然澄清問題規劃器，只提出目前最能降低不確定性的一個追問。"
        "question 與 explanation 必須使用台灣繁體中文，不得使用簡體字。"
        "不要按固定欄位順序詢問，也不要要求補滿看診日期或時段。"
        "已經有可靠回答的資訊不要重問。每個 intent 只對應一個資訊面向、一個聚焦問句、一次詢問要求；"
        "不得用連接詞在同一問句追加第二個臨床面向，即使整句只有一個問號也不可以；"
        "例如詢問程度時，不要在同一句再問疼痛或其他伴隨不適。"
        f"{answer_contract}"
        "partial 或 unclear 時維持同一 intent，提出一個更精確、不重複、只問同一面向的追問。"
        "dialogue_act.act=request_explanation 時，explanation 用 1～2 句白話說明使用者看不懂的內容"
        "（例如上一題提到的症狀是什麼意思），以及為什麼需要確認這項資訊；explanation 不得診斷，"
        "不得出現科、醫師、醫生、掛號等字，不得含問號或分號。question 仍只問同一 intent 的一個聚焦問題"
        "（安全篩檢時 intent 仍為 safety_screen）。此時 question 可在 explanation 之後原樣重問上一題，"
        "或改問同一面向更聚焦的一題；更早問過的其他問題仍不得重問。其他情況 explanation 一律輸出 null。"
        "claims_already_answered 時，先參照 accepted clarification_evidence，不得重問已回答 intent，"
        "也不得僅因使用者表示說過就新增證據。若仍缺資訊，明確指出那一個資訊。"
        "不要使用『剛才問題』『再補充細節』『換個方式』等沒有具體資訊的罐頭。"
        "department_status=ambiguous 時，優先問最能區分候選的一個自然問題；"
        "若有 required_next_question_intent，它只是 Backend/AI 間的 opaque correlation key：輸出的 intent 必須"
        "原樣 echo 並與它完全一致，但 question 仍然只問一個聚焦臨床面向。不得解析 key 名稱，也不得因 key"
        "看似包含多個英文概念，就要求 question 同時涵蓋所有概念。"
        "uncertainty_reasons 與 department_uncertainty_reason 是 Backend 診斷資訊，絕不可引用、改寫或當成患者追問焦點。"
        "next_information_needed 只在內容是患者可直接回答的具體資訊需求時使用；若為空，應依 required intent、"
        "既有對話和 grounded evidence 自然產生問題，不得把 snake_case intent 逐字翻譯成診斷。"
        "不得向患者顯示候選科別名稱。"
        "如果症狀資訊足夠，可建議 status=sufficient，但最終完成與否由 Backend 決定。"
        "目標是掛號科別導引，不是完成疾病診斷；當 department_status=resolved、"
        "grounded_patient_evidence.red_flags_checked=true、沒有 pending intent，且已有主要症狀與具體細節時，"
        "應建議 sufficient，不要為了找病因或補齊固定欄位繼續追問。"
        "不得診斷、建議科別或醫師、修改患者事實或控制 workflow。"
        "只輸出 JSON，欄位為 status (clarification_needed 或 sufficient), question, explanation (字串或 null), "
        "intent, reason, "
        "answered_intent, answer_source_text, answer_status, answer_confidence, information_needs。"
        "information_needs 是僅含一個患者能回答的具體資訊需求的字串陣列；不能寫安全狀態或工作流程。"
        "intent 用簡短英文 snake_case；沒有 pending 回答時，四個 answer 欄位用 null。\n"
        + (_clue_prompt(clue) if clue is not None else "")
        + (NO_CANDIDATE_PROMPT if _no_department_candidates(case, clue, safety_planning) else "") +
        f"資料：{json.dumps(payload, ensure_ascii=False)}"
    )
    explanation_required = act == "request_explanation"
    rejection = None
    initial_rejection = None
    # A proposal whose question passed but whose explanation did not.
    degraded: tuple[dict, int] | None = None
    for attempt in range(2):
        proposal = None
        raw = None
        try:
            raw = await complete_runtime_json(
                prompt, purpose="conversation_clarification" if attempt == 0 else "conversation_clarification_repair",
            )
        except Exception as exc:
            rejection = "provider_timeout" if isinstance(exc, TimeoutError) else "provider_failure"
            logger.warning("clarification failed case_id=%s reason=%s error_type=%s", case.case_id, rejection, type(exc).__name__)
            if not (safety_planning and state.safety_answer_validation == "request_explanation"):
                break
        else:
            try:
                proposal = json.loads(str(raw).strip())
            except (TypeError, ValueError):
                rejection = "invalid_json"
            else:
                # A restated explained question without a valid explanation stays a duplicate, so it never degrades.
                rejection = _question_rejection(case, proposal, required_intent, safety_planning, act, clue=clue)
                explanation = proposal.get("explanation") if isinstance(proposal, dict) else None
                if (rejection is None and explanation_required
                        and proposal["status"] == "clarification_needed"
                        and not _valid_explanation(explanation)):
                    rejection = (
                        "simplified_chinese_output"
                        if isinstance(explanation, str) and _SIMPLIFIED_ONLY.search(explanation)
                        else "missing_or_invalid_explanation"
                    )
                    degraded = (proposal, attempt)
        logger.info(
            "clarification proposal case_id=%s accepted=%s rejection_reason=%s repair_attempted=%s",
            case.case_id, rejection is None, rejection, bool(attempt),
        )
        if rejection is None:
            return _accepted_suggestion(proposal, attempt, initial_rejection,
                                        keep_explanation=explanation_required or safety_planning, clue=clue)
        if attempt == 0:
            initial_rejection = rejection
            repair_context = dict(payload, original_proposal=proposal if proposal is not None else str(raw),
                                  rejection_reason=rejection)
            prompt = (
                prompt + "\n這是唯一一次修復：依下列 validator rejection_reason 修正 proposal，"
                "只產生目前 required intent 的一個具體問句，不得新增患者事實或控制 state；"
                "question 與 explanation 必須使用台灣繁體中文，不得使用簡體字"
                "（rejection_reason=simplified_chinese_output 即表示含簡體字）；"
                "若 rejection_reason=missing_or_invalid_explanation，保留合格 question 並補上合格 explanation。"
                "若 rejection_reason=duplicate_question：只有 dialogue_act.act="
                "request_explanation 且附上合格 explanation 時，才可原樣重問上一題；否則改問同一面向、"
                "未曾問過的一個聚焦問題。\n"
                + json.dumps(repair_context, ensure_ascii=False)
            )
    if degraded is not None:
        # The question itself passed validation; ask it without an explanation rather than fail the turn.
        logger.info("clarification proposal case_id=%s accepted=true rejection_reason=explanation_degraded", case.case_id)
        return _accepted_suggestion(*degraded, "explanation_degraded", keep_explanation=False,
                                    repair_attempted=bool(attempt), clue=clue)
    if clue is not None and rejection not in {"provider_timeout", "provider_failure"}:
        # The provider answered but no proposal passed: list the validated official concepts verbatim.
        logger.info("clarification proposal case_id=%s accepted=true rejection_reason=%s reply_source=CLUE_TEMPLATE",
                    case.case_id, rejection)
        return _clue_template_suggestion(clue, rejection, attempt)
    return ClarificationSuggestion(
        "clarification_needed", None, None, "clarification_generation_failed",
        rejection_reason=rejection, repair_attempted=bool(attempt), reply_source="TECHNICAL_FALLBACK",
    )


NO_CANDIDATE_PROMPT = (
    "目前沒有任何候選科別：已接受的症狀還對不上官方症狀概念。請優先請患者把模糊的不適說得更具體，"
    "例如「頭腦昏沉」是比較像頭暈、天旋地轉，還是想睡、沒精神；或詢問還有沒有其他不舒服。"
    "不要用持續多久、嚴重程度、部位這類問題填補，它們不是必填。\n"
)


def _no_department_candidates(case: TriageCase, clue: ClueQuestionSpec | None, safety_planning: bool) -> bool:
    """True when department reasoning retrieved no official evidence and no clue question is due."""
    state = case.conversation_state
    return (not safety_planning and clue is None and state.department_status != "resolved"
            and not state.department_clue_candidates and not state.candidate_departments)


def _clue_prompt(clue: ClueQuestionSpec) -> str:
    return (
        "目前是線索題模式：clue_offer 是 Backend 依官方科別知識排序的可問症狀概念，dept_ids 只是不透明分組 ID。"
        + ("這是同一題線索題的重問或解釋：clue_concepts 必須正好是 clue_offer 的全部概念，並換一種說法詢問。"
           if clue.reask else
           f"從 clue_offer 選 1～{clue.per_question} 個概念，所選概念的 dept_ids 必須至少有一個共同值，"
           "優先選排序在前的概念。")
        + "question 必須逐字包含每個所選概念字串，可在概念後用括號附白話說明；可用頓號合併在同一問句，"
        "整題只能有一個問號，不得診斷、不得提及科別。"
        "question 要是完整的一句話，以「請問」開頭（例如「請問您有沒有…」），不要只列概念加「嗎」。"
        "若使用者在描述家人或他人的狀況（例如我兒子、我女兒、我媽），question 要問那個人"
        "（例如「請問她有沒有…」「請問您兒子有沒有…」），不得改問使用者本人。"
        "所選概念可能是症狀也可能是病名，問法要讓患者能回答「現在有沒有這個狀況，或以前曾被告知有」，"
        "不得使用「醫師」「醫生」「診斷」等字，不得暗示患者可能罹患該病。status 必須為 clarification_needed，"
        "intent 必須原樣等於 required_next_question_intent。另輸出 clue_concepts：所選概念字串陣列，"
        "每項逐字取自 clue_offer。information_needs 可為 null。\n"
    )


def _clue_rejection(proposal: dict, question: str, clue: ClueQuestionSpec) -> str | None:
    selected = proposal.get("clue_concepts")
    if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
        return "invalid_clue_concepts"
    if clue.reask:
        if len(selected) != len(set(selected)) or set(selected) != {item.concept for item in clue.concepts}:
            return "clue_concepts_mismatch"
    elif not clues.common_clue_departments(selected, clue.concepts, clue.per_question):
        return "invalid_clue_concepts"
    if clues.concepts_missing_from_question(question, selected):
        return "clue_concept_not_in_question"
    return None


def _clue_template_suggestion(clue: ClueQuestionSpec, rejection: str | None, attempt: int) -> ClarificationSuggestion:
    if clue.reask:
        concepts, dept_id = [item.concept for item in clue.concepts], None
    else:
        concepts, dept_id = clues.template_clue_selection(list(clue.concepts), clue.per_question)
    return ClarificationSuggestion(
        "clarification_needed", clues.clue_template_question(concepts, reask=clue.reask), clue.intent,
        "clue_template_fallback", rejection_reason=rejection, repair_attempted=bool(attempt),
        reply_source="CLUE_TEMPLATE", clue_concepts=tuple(concepts), clue_dept_id=dept_id,
    )


def _accepted_suggestion(proposal: dict, attempt: int, rejection_reason: str | None, *,
                         keep_explanation: bool, repair_attempted: bool | None = None,
                         clue: ClueQuestionSpec | None = None) -> ClarificationSuggestion:
    clarifying = proposal["status"] == "clarification_needed"
    needs = (proposal.get("information_needs") or []) if clarifying and clue is None else []
    selected = tuple(proposal["clue_concepts"]) if clarifying and clue is not None else ()
    common = (clues.common_clue_departments(list(selected), clue.concepts, clue.per_question)
              if selected and not clue.reask else set())
    return ClarificationSuggestion(
        proposal["status"],
        proposal.get("question", "").strip() if clarifying else None,
        proposal.get("intent") if clarifying else None,
        proposal["reason"].strip()[:240],
        information_need=needs[0].strip() if needs else None,
        rejection_reason=rejection_reason,
        repair_attempted=bool(attempt) if repair_attempted is None else repair_attempted,
        reply_source="AI_REPAIR" if attempt else "AI_PLANNER",
        # AI explanations are kept only for a validated request_explanation turn; otherwise dropped.
        explanation=(proposal["explanation"].strip()
                     if keep_explanation and clarifying and isinstance(proposal.get("explanation"), str) else None),
        clue_concepts=selected,
        clue_dept_id=min(common) if common else None,
    )


def _valid_explanation(value: object) -> bool:
    # Explanations may mention several symptoms, so the multi-dimension guard is not applied.
    return (
        isinstance(value, str) and 4 <= len(value.strip()) <= 150
        and not _UNSAFE_QUESTION.search(value) and not _INTERNAL_STATE_TEXT.search(value)
        and not _EXPLANATION_PUNCTUATION.search(value) and not _SIMPLIFIED_ONLY.search(value)
    )


def _asked_before(case: TriageCase, question: str) -> bool:
    """Displayed replies may prefix an explanation line; compare the question itself too."""
    question = question.strip()
    return any(
        item.role == "assistant" and (item.content == question or item.content.endswith("\n" + question))
        for item in case.history_records
    )


def _explained_question(case: TriageCase) -> str | None:
    """The question part of the last assistant reply before this turn (last line of an explained reply)."""
    last = next((item.content for item in reversed(case.history_records) if item.role == "assistant"), None)
    return last.rsplit("\n", 1)[-1].strip() if last else None


def _repeated_question(case: TriageCase, question: str, *, dialogue_act: str | None,
                       explanation: object, safety_planning: bool) -> str | None:
    """Shared duplicate rule: only a validly explained restatement of the explained question may repeat."""
    question = question.strip()
    if (dialogue_act == "request_explanation" and _valid_explanation(explanation)
            and question == _explained_question(case)):
        return None
    # The safety screen always restates the full question (continue_safety_check).
    if safety_planning:
        return None
    if _asked_before(case, question):
        return "duplicate_question"
    return None


def _with_explanation(suggestion: ClarificationSuggestion, question: str) -> str:
    return f"{suggestion.explanation}\n{question}" if suggestion.explanation else question


def _question_rejection(case: TriageCase, proposal: object, required_intent: str | None,
                        safety_planning: bool, dialogue_act: str | None = None, *,
                        clue: ClueQuestionSpec | None = None) -> str | None:
    if not isinstance(proposal, dict):
        return "invalid_schema"
    if set(proposal) & {"stage", "confirmed", "red_flags_checked", "department_result", "is_complete", "doctor_id", "schedule_id"}:
        return "workflow_authority"
    if not isinstance(proposal.get("status"), str) or proposal["status"] not in {"clarification_needed", "sufficient"}:
        return "invalid_status"
    if not isinstance(proposal.get("reason"), str) or not proposal["reason"].strip():
        return "invalid_reason"
    if proposal["status"] == "sufficient":
        if clue is not None:
            return "clue_question_required"
        if safety_planning:
            return "safety_still_pending"
        if required_intent is not None:
            return "unresolved_information_remaining"
        if case.conversation_state.department_status == "ambiguous":
            return "department_still_ambiguous"
        return None
    question, intent = proposal.get("question"), proposal.get("intent")
    if not isinstance(question, str) or not 4 <= len(question.strip()) <= 180:
        return "invalid_question_length"
    if not isinstance(intent, str) or not _INTENT_PATTERN.fullmatch(intent):
        return "invalid_intent"
    if intent in case.conversation_state.clarification_evidence:
        return "intent_already_answered"
    state = case.conversation_state
    if (intent in state.asked_clarification_intents
            and intent not in {state.pending_clarification_intent, state.department_next_question_intent}):
        return "intent_has_no_current_owner"
    if required_intent and intent != required_intent:
        return "required_intent_mismatch"
    if _SIMPLIFIED_ONLY.search(question):
        return "simplified_chinese_output"
    if _UNSAFE_QUESTION.search(question) or _INTERNAL_STATE_TEXT.search(question):
        return "unsafe_or_internal_question"
    # Rule B lets one clue question list several official concepts; it still needs a single question mark.
    if _MULTI_QUESTION.search(question) or (clue is None and _MULTI_DIMENSION_QUESTION.search(question)):
        return "multiple_information_needs"
    if clue is None:
        needs = proposal.get("information_needs")
        if not isinstance(needs, list) or len(needs) != 1 or not isinstance(needs[0], str) or not needs[0].strip():
            return "invalid_information_need"
        if _GENERIC_INFORMATION_NEED.fullmatch(needs[0].strip().strip("。？！? ")):
            return "generic_information_need"
        if _patient_information_need(needs[0]) is None:
            return "invalid_information_need"
    else:
        clue_reason = _clue_rejection(proposal, question, clue)
        if clue_reason:
            return clue_reason
    if _GENERIC_CLARIFICATION_QUESTION.fullmatch(question.strip()):
        return "generic_question"
    if re.search(r"剛才問題|上一個追問", question):
        return "generic_retry"
    repeated = _repeated_question(case, question, dialogue_act=dialogue_act,
                                  explanation=proposal.get("explanation"), safety_planning=safety_planning)
    if repeated:
        return repeated
    if not safety_planning and _asks_filled_duration(case, intent, question):
        return "already_collected_duration"
    return None


def clear_answered_pending(case: TriageCase) -> bool:
    """Reconcile stale workflow ownership only with an already accepted answer."""
    state = case.conversation_state
    if state.pending_clarification_intent in state.clarification_evidence:
        state.pending_clarification_intent = None
        state.pending_clarification_question = None
        state.next_information_needed = []
        return True
    return False


def validate_prior_answer_reference(
    value: object, dialogue_act: DialogueActInterpretation | None, pending_intent: str | None,
    prior_extractions: tuple[SemanticExtraction, ...], prior_user_sources: list[str],
) -> PriorAnswerReference | None:
    """Only the AI relates intent to meaning; Backend verifies a prior accepted source."""
    if (dialogue_act is None or dialogue_act.act != "claims_already_answered"
            or dialogue_act.confidence < ACCEPT_THRESHOLD
            or not pending_intent or pending_intent in {SAFETY_PENDING_INTENT, SAFETY_NET_PENDING_INTENT}):
        return None
    try:
        reference = PriorAnswerReference.model_validate(value)
    except (ValueError, TypeError):
        return None
    if (reference.answered_intent != pending_intent or reference.answer_confidence < ACCEPT_THRESHOLD
            or not reference.answer_source_text.strip()):
        return None
    if any(item.source_text == reference.answer_source_text for item in accepted_prior_answer_evidence(
        prior_extractions, prior_user_sources,
    )):
        return reference
    return None


def accepted_prior_answer_evidence(
    prior_extractions: tuple[SemanticExtraction, ...], prior_user_sources: list[str],
) -> list[SemanticExtraction]:
    accepted_items = []
    for item in prior_extractions:
        medical_assertion = item.field in {"symptom", "accompanying_symptoms"}
        if (item.field not in _MEDICAL_FIELDS
                or not item.source_text.strip()
                or not any(item.source_text in text for text in prior_user_sources)
                or type(item.confidence) not in {int, float} or not math.isfinite(item.confidence)
                or not ACCEPT_THRESHOLD <= item.confidence <= 1
                or item.semantic_status not in {"available", "partial", "unavailable"}
                or (item.semantic_status == "unavailable" and not (medical_assertion and item.assertion == "absent"))
                or (medical_assertion and item.assertion not in {"present", "absent"})
                or not ai_normalized_value_valid(item.field, item.normalized_value, item.semantic_status, item.source_text)):
            continue
        accepted_items.append(item)
    return accepted_items


def capture_prior_answer_reference(
    case: TriageCase, reference: PriorAnswerReference | None, dialogue_act: DialogueActInterpretation | None,
    *, prior_extractions: tuple[SemanticExtraction, ...], prior_user_sources: list[str],
) -> bool:
    state = case.conversation_state
    accepted_reference = None if state.clarification_status == "safety_check" else validate_prior_answer_reference(
        reference, dialogue_act, state.pending_clarification_intent, prior_extractions, prior_user_sources,
    )
    if accepted_reference is None:
        if reference is not None or (dialogue_act and dialogue_act.act == "claims_already_answered"):
            logger.info("prior answer case_id=%s intent=%s accepted=false reason=prior_answer_reference_invalid",
                        case.case_id, state.pending_clarification_intent)
        return False
    state.clarification_evidence[accepted_reference.answered_intent] = accepted_reference.answer_source_text
    clear_answered_pending(case)
    logger.info("prior answer case_id=%s intent=%s accepted=true reason=prior_answer_reference_accepted",
                case.case_id, accepted_reference.answered_intent)
    return True


def invalidate_corrected_department_context(case: TriageCase) -> None:
    """Release clinical decision ownership, without changing the safety lifecycle."""
    state = case.conversation_state
    case.department_result = None
    state.department_status = "unresolved"
    state.candidate_departments = []
    state.department_uncertainty_reason = None
    state.department_next_information_needed = None
    state.department_next_question_intent = None
    state.semantic_official_links = []
    state.department_clue_offer = []
    state.department_clue_candidates = []
    state.pending_clarification_intent = None
    state.pending_clarification_question = None
    state.next_information_needed = []
    case.confirmed = False
    state.confirmed = False


def capture_pending_answer(
    case: TriageCase,
    suggestion: ClarificationSuggestion | PendingAnswerInterpretation | None,
    user_sources: list[str],
) -> bool:
    """Persist only a grounded, confident answer before candidate recomputation."""
    if case.conversation_state.pending_clarification_intent == SAFETY_NET_PENDING_INTENT:
        return _capture_safety_net_choice(case, suggestion, user_sources)
    if case.conversation_state.pending_clarification_intent == pediatric.AGE_PENDING_INTENT:
        # Any reply releases it once; an unusable reply (「不知道」) leaves the patient an adult.
        if user_sources:
            case.conversation_state.patient_age_answer = "".join(user_sources)
            case.conversation_state.pending_clarification_intent = None
            case.conversation_state.pending_clarification_question = None
            case.conversation_state.next_information_needed = []
        return False
    if case.conversation_state.pending_clarification_intent in {BODY_PART_PENDING_INTENT, COMPANION_PENDING_INTENT}:
        # Any reply releases it; a body part or new symptom arrives as ordinary extracted evidence.
        if user_sources:
            case.conversation_state.pending_clarification_intent = None
            case.conversation_state.pending_clarification_question = None
            case.conversation_state.next_information_needed = []
        return False
    clue_index = clues.pending_clue_index(case.conversation_state)
    if clue_index is not None:
        return _capture_clue_answers(case, suggestion, user_sources, clue_index)
    if not suggestion or suggestion.answer_status != "answered" or suggestion.answer_confidence is None:
        return False
    state = case.conversation_state
    if not _grounded_clarification_answer(
        suggestion.answered_intent, suggestion.answer_source_text, suggestion.answer_status,
        suggestion.answer_confidence, state.pending_clarification_intent, user_sources,
    ) or suggestion.answer_confidence < ACCEPT_THRESHOLD:
        return False
    state.clarification_evidence[suggestion.answered_intent] = suggestion.answer_source_text or ""
    state.pending_clarification_intent = None
    state.pending_clarification_question = None
    state.next_information_needed = []
    return True


def _capture_safety_net_choice(
    case: TriageCase, suggestion: ClarificationSuggestion | PendingAnswerInterpretation | None,
    user_sources: list[str],
) -> bool:
    """Release the one-time safety-net question on any reply; only a grounded, confident
    Turn Interpreter choice selects family medicine, anything else continues clarifying."""
    state = case.conversation_state
    if not user_sources:
        return False
    chose = bool(
        suggestion is not None
        and getattr(suggestion, "answer_choice", None) == "family_medicine"
        and suggestion.answer_status == "answered"
        and suggestion.answer_confidence is not None
        and _grounded_clarification_answer(
            suggestion.answered_intent, suggestion.answer_source_text, suggestion.answer_status,
            suggestion.answer_confidence, SAFETY_NET_PENDING_INTENT, user_sources,
        )
        and suggestion.answer_confidence >= ACCEPT_THRESHOLD
    )
    state.safety_net_choice = "family_medicine" if chose else "continue"
    state.pending_clarification_intent = None
    state.pending_clarification_question = None
    state.next_information_needed = []
    logger.info("safety net case_id=%s choice=%s", case.case_id, state.safety_net_choice)
    return chose


def safety_net_family_medicine_reason(case: TriageCase, user_sources: list[str]) -> str | None:
    """Why this turn routes to family medicine: the patient's safety-net choice, declining further
    questions with no candidate department, or the hard turn cap.

    Only once the safety screen is complete and the department is still undecided; TTAS urgency
    is handled before this is reached.
    """
    state = case.conversation_state
    if (not user_sources or not safety_screen_resolved(case) or state.department_status == "resolved"
            or state.clarification_status == "unresolved"):
        return None
    if state.safety_net_choice == "family_medicine":
        return "safety_net_choice"
    # Reached only when decide() found no candidate department to settle on.
    if state.clue_user_declined:
        return "user_declined_no_candidates"
    # advance_conversation adds this turn's messages afterwards.
    if state.turn_count + len(user_sources) >= get_settings().hard_turn_cap:
        return "safety_net_turn_cap"
    return None


def adopt_safety_net_family_medicine(case: TriageCase, reason: str) -> None:
    """Existing family medicine path: configured dept_id with exactly one live Department row, else unavailable."""
    # Deferred: department_reasoning_service imports this module.
    from app.services import department_reasoning_service as reasoning
    state = case.conversation_state
    try:
        active = reasoning.fetch_active_departments()
    except Exception as exc:
        logger.warning("safety net departments unavailable case_id=%s error_type=%s", case.case_id, type(exc).__name__)
        active = []
    result = reasoning._live_department(active, get_settings().family_medicine_dept_id, 0.0, "目前症狀無法明確區分科別")
    state.clue_decision = ClueDecisionRecord(
        kind="family_medicine" if result else "family_medicine_unavailable", reason=reason,
        dept_id=result.dept_id if result else None,
    )
    logger.info("safety net case_id=%s reason=%s kind=%s", case.case_id, reason, state.clue_decision.kind)
    if result is not None:
        case.department_result = result
        state.department_status = "resolved"
        state.department_next_question_intent = None
        state.pending_clarification_intent = None
        state.pending_clarification_question = None
        state.pending_clue_question_index = None
        state.clue_reask_count = 0
        state.next_information_needed = []


def _capture_clue_answers(
    case: TriageCase, suggestion: ClarificationSuggestion | PendingAnswerInterpretation | None,
    user_sources: list[str], index: int,
) -> bool:
    """Record grounded per-concept answers; concepts left unanswered stay askable."""
    state = case.conversation_state
    if not suggestion or not _grounded_clarification_answer(
        suggestion.answered_intent, suggestion.answer_source_text, suggestion.answer_status,
        suggestion.answer_confidence, state.pending_clarification_intent, user_sources,
    ):
        return False
    asked = {record.concept for record in state.clue_concepts_asked if record.question_index == index}
    accepted = {}
    for item in getattr(suggestion, "concept_answers", None) or []:
        source = item.source_text.strip()
        if (item.concept in asked and item.concept not in accepted and source
                and any(source in text for text in user_sources)
                and math.isfinite(item.confidence) and ACCEPT_THRESHOLD <= item.confidence <= 1):
            accepted[item.concept] = (item.answer_assertion, source)
    if not accepted:
        return False
    for record in state.clue_concepts_asked:
        if record.question_index == index and record.concept in accepted:
            record.answer, record.answer_source_text = accepted[record.concept]
    state.clarification_evidence[suggestion.answered_intent] = suggestion.answer_source_text.strip()
    state.pending_clarification_intent = None
    state.pending_clarification_question = None
    state.pending_clue_question_index = None
    state.clue_reask_count = 0
    state.next_information_needed = []
    logger.info("clue answer case_id=%s index=%s accepted_concepts=%s", case.case_id, index, len(accepted))
    return True


def _record_clue_question(case: TriageCase, suggestion: ClarificationSuggestion) -> None:
    """Count a newly asked clue question and store its concepts as unanswered."""
    state = case.conversation_state
    index = state.clue_questions_asked + 1
    if suggestion.intent != clues.clue_intent(index) or suggestion.clue_dept_id is None:
        return
    offered = {item.concept: item.dept_ids for item in state.department_clue_offer}
    state.clue_concepts_asked = clues.append_clue_records(
        state.clue_concepts_asked, list(suggestion.clue_concepts), suggestion.clue_dept_id, index,
        dept_ids=offered,
    )
    state.clue_questions_asked = index
    state.pending_clue_question_index = index
    state.clue_reask_count = 0
    if state.clue_decision is not None and state.clue_decision.kind == "tiebreak":
        state.clue_tiebreak_used = True


def capture_safety_screen_answer(
    case: TriageCase,
    suggestion: PendingAnswerInterpretation | None,
    user_sources: list[str],
    *, dialogue_act: DialogueActInterpretation | None = None,
    current_extractions: list[SemanticExtraction] = (),
) -> bool:
    """Complete a negative safety screen from validated AI interpretation only."""
    state = case.conversation_state
    if state.clarification_status != "safety_check" or not user_sources:
        return False
    if (dialogue_act is not None and dialogue_act.act == "request_explanation"
            and dialogue_act.confidence >= ACCEPT_THRESHOLD):
        state.safety_answer_validation = "request_explanation"
        logger.info(
            "safety answer case_id=%s dialogue_act=request_explanation attempt_incremented=false "
            "reason=safety_request_explanation_no_attempt", case.case_id,
        )
        return False
    answered = bool(
        suggestion is not None and suggestion.answer_status in {"answered", "partial"}
        and suggestion.answer_confidence >= ACCEPT_THRESHOLD
    )
    if (not answered and validated_dialogue_act(dialogue_act, user_sources) in {"new_information", "correction"}
            and _safety_added_symptoms(current_extractions, user_sources)):
        # The patient kept describing symptoms: no attempt, but TTAS evidence still marks a positive.
        state.safety_answer_validation = "symptom_added"
        if _safety_positive_answer(case, suggestion, user_sources):
            state.safety_positive_reported = True
        logger.info(
            "safety answer case_id=%s dialogue_act=new_information attempt_incremented=false "
            "reason=safety_symptom_added_no_attempt", case.case_id,
        )
        return False
    state.safety_answer_attempts += 1
    logger.info("safety answer case_id=%s attempt_incremented=true", case.case_id)
    state.safety_answer_validation = (
        "missing_or_invalid_interpretation" if suggestion is None else
        "low_confidence" if suggestion.answer_confidence < ACCEPT_THRESHOLD else
        f"{suggestion.answer_status}_{suggestion.answer_assertion}"
    )
    if _safety_positive_answer(case, suggestion, user_sources):
        state.safety_positive_reported = True
        logger.info("safety answer case_id=%s positive_reported=true", case.case_id)
        if _patient_reported_positive(suggestion, user_sources):
            # The patient said yes to the screen: stop now instead of asking the remaining items.
            state.safety_answer_validation = "patient_reported_positive"
            _safety_unresolved(case)
            logger.info("safety answer accepted=false reason=patient_reported_positive attempt=%s",
                        state.safety_answer_attempts)
            return False
    if (
        case.conversation_state.clarification_status != "safety_check"
        or suggestion is None
        or suggestion.answer_status != "answered"
        or suggestion.answer_assertion != "absent"
        or suggestion.answer_confidence < ACCEPT_THRESHOLD
        or not _grounded_clarification_answer(
            suggestion.answered_intent,
            suggestion.answer_source_text,
            suggestion.answer_status,
            suggestion.answer_confidence,
            SAFETY_PENDING_INTENT,
            user_sources,
        )
    ):
        logger.info("safety answer accepted=false reason=%s attempt=%s", state.safety_answer_validation, state.safety_answer_attempts)
        return False
    if state.safety_positive_reported:
        # A later denial never turns a screen with an accepted positive answer negative;
        # urgency itself still comes only from TTAS.
        state.safety_answer_validation = "absent_after_positive"
        _safety_unresolved(case)
        logger.info("safety answer accepted=false reason=absent_after_positive attempt=%s", state.safety_answer_attempts)
        return False

    patient = case.patient_input
    state = case.conversation_state
    patient.red_flags = []
    patient.red_flags_checked = True
    patient.red_flags_status = "negative"
    if RED_FLAG_QUESTION_KEY not in patient.collected_fields:
        patient.collected_fields.append(RED_FLAG_QUESTION_KEY)
    if RED_FLAG_QUESTION_KEY not in state.consumed_fields:
        state.consumed_fields.append(RED_FLAG_QUESTION_KEY)
    state.field_statuses[RED_FLAG_QUESTION_KEY] = "unavailable"
    state.field_confidence[RED_FLAG_QUESTION_KEY] = suggestion.answer_confidence
    state.safety_answer_validation = "accepted_complete_negative"
    # Release safety ownership before department recomputation in this same turn.
    state.clarification_status = "collecting"
    state.pending_clarification_intent = None
    state.pending_clarification_question = None
    state.next_information_needed = []
    state.department_next_question_intent = None
    state.department_next_information_needed = None
    state.uncertainty_reasons = []
    state.last_question_key = None
    case.triage.next_question = None
    logger.info("safety answer accepted=true reason=complete_negative attempt=%s", state.safety_answer_attempts)
    return True


def _patient_reported_positive(
    suggestion: PendingAnswerInterpretation | None, user_sources: list[str],
) -> bool:
    """A grounded, confident present answer to the screen.

    A partial answer counts too: "有半邊無力，其他沒有" is a partial denial that still reports one item.
    """
    return bool(
        suggestion is not None and suggestion.answer_status in {"answered", "partial"}
        and suggestion.answer_assertion == "present"
        and suggestion.answer_confidence >= ACCEPT_THRESHOLD
        and _grounded_clarification_answer(
            suggestion.answered_intent, suggestion.answer_source_text, suggestion.answer_status,
            suggestion.answer_confidence, SAFETY_PENDING_INTENT, user_sources,
        )
    )


def _safety_positive_answer(
    case: TriageCase, suggestion: PendingAnswerInterpretation | None, user_sources: list[str],
) -> bool:
    """The patient's present answer, or validated TTAS evidence this turn reports as true."""
    if _patient_reported_positive(suggestion, user_sources):
        return True
    return any(
        item.semantic_status == "available" and item.value is True and item.confidence >= ACCEPT_THRESHOLD
        and any(item.source_text in text for text in user_sources)
        for item in case.ttas_evidence
    )


def _safety_added_symptoms(extractions: list[SemanticExtraction], user_sources: list[str]) -> list[str]:
    """Grounded medical spans from this turn, longest kept when one contains another."""
    spans: list[str] = []
    for item in extractions:
        source = item.source_text.strip()
        if (item.field in _MEDICAL_FIELDS and item.semantic_status in {"available", "partial"}
                and item.confidence >= ACCEPT_THRESHOLD and source
                and any(source in text for text in user_sources)
                and not any(source in kept for kept in spans)):
            spans = [kept for kept in spans if kept not in source] + [source]
    return spans[:SAFETY_SYMPTOM_ACK_LIMIT]


def _safety_unresolved(case: TriageCase) -> None:
    _unresolved(case)
    state = case.conversation_state
    state.uncertainty_reasons = ["safety_information_unresolved"]
    state.next_information_needed = []
    state.reply_source = "SAFETY_CONTROLLED_FALLBACK"


def continue_safety_check(
    case: TriageCase,
    suggestion: ClarificationSuggestion | None,
    *,
    user_sources: list[str] = (),
    current_extractions: list[SemanticExtraction] = (),
) -> None:
    """Bound safety workflow retries without interpreting patient language."""
    state = case.conversation_state
    state.pending_clarification_intent = None
    state.pending_clarification_question = None
    state.next_information_needed = []
    # Explanation requests do not count as attempts, so they stop only at the hard turn cap.
    if state.turn_count >= get_settings().hard_turn_cap or state.safety_answer_attempts >= SAFETY_ANSWER_CAP:
        _safety_unresolved(case)
        return
    # Always the full question: a focused single-item denial is partial and could never complete the screen.
    question = QUESTION_TEXTS[RED_FLAG_QUESTION_KEY]
    added = (
        _safety_added_symptoms(current_extractions, user_sources)
        if state.safety_answer_validation == "symptom_added" else []
    )
    last_chance = (
        state.safety_answer_validation not in _UNCOUNTED_SAFETY_VALIDATIONS
        and state.safety_answer_attempts == SAFETY_ANSWER_CAP - 1
    )
    if added:
        question = f"{safety_symptom_acknowledgement(added)}\n{question}"
        state.reply_source = "SAFETY_CONTROLLED_FALLBACK"
    elif suggestion is not None and suggestion.intent == SAFETY_PENDING_INTENT:
        state.reply_source = suggestion.reply_source
        if last_chance:
            question = f"{SAFETY_LAST_ATTEMPT_REPLY}\n{question}"
        if _valid_explanation(suggestion.explanation):
            question = f"{suggestion.explanation.strip()}\n{question}"
    elif state.safety_answer_validation == "request_explanation":
        question = f"{SAFETY_EXPLANATION_FALLBACK}\n{question}"
        state.reply_source = "SAFETY_CONTROLLED_FALLBACK"
    else:
        question = f"{SAFETY_LAST_ATTEMPT_REPLY if last_chance else SAFETY_RETRY_REPLY}\n{question}"
        state.reply_source = "SAFETY_CONTROLLED_FALLBACK"
    state.clarification_status = "safety_check"
    state.last_question_key = RED_FLAG_QUESTION_KEY
    state.is_complete = False
    case.triage.need_more_info = True
    case.triage.next_question = question
    case.triage.is_final = False


def advance_conversation(
    case: TriageCase,
    suggestion: ClarificationSuggestion | None,
    *,
    user_sources: list[str],
    current_extractions: list[SemanticExtraction],
    made_progress: bool = False,
    availability_changed: bool = False,
) -> None:
    """Apply only validated clarification proposals and Backend-owned state gates."""
    state = case.conversation_state
    state.reply_source = None
    state.question_rejection_reason = suggestion.rejection_reason if suggestion else "planner_unavailable"
    state.question_repair_attempted = suggestion.repair_attempted if suggestion else False
    state.next_information_needed = _patient_information_needs(state.next_information_needed)
    state.department_next_information_needed = _patient_information_need(
        state.department_next_information_needed
    )
    if state.clarification_status == "unresolved":
        _unresolved(case)
        return
    state.turn_count += len(user_sources)
    if state.clarification_status == "safety_check" and not safety_screen_resolved(case):
        continue_safety_check(case, suggestion, user_sources=user_sources, current_extractions=current_extractions)
        return

    clear_answered_pending(case)
    if user_sources:
        state.consecutive_no_progress_turns = (
            0 if made_progress else state.consecutive_no_progress_turns + len(user_sources)
        )

    grounded_symptom = bool(case.patient_input.symptom) and any(
        case.patient_input.symptom in item.content
        for item in case.history_records if item.role == "user"
    )
    grounded_detail = bool(state.clarification_evidence) or any(
        item.field in _DETAIL_FIELDS
        and item.semantic_status in {"available", "partial"}
        and item.confidence >= ACCEPT_THRESHOLD
        and item.source_text
        and any(item.source_text in message.content for message in case.history_records if message.role == "user")
        for item in case.semantic_extractions
    )
    low_confidence = any(
        item.field in _MEDICAL_FIELDS and item.confidence < ACCEPT_THRESHOLD
        for item in current_extractions
    ) or any(
        field in _MEDICAL_FIELDS
        and reason == "AI extraction confidence below threshold"
        and state.field_confidence.get(field, 1.0) < ACCEPT_THRESHOLD
        for field, reason in state.clarification_reasons.items()
    )
    safety_ready = grounded_symptom and grounded_detail and not low_confidence
    if safety_ready and not safety_screen_resolved(case):
        if state.turn_count >= get_settings().hard_turn_cap:
            _unresolved(case)
            return
        _enter_safety_check(case)
        return
    if _age_question_due(case):
        _ask_age_question(case)
        return

    decision = state.clue_decision.kind if state.clue_decision else None
    if decision == "family_medicine_unavailable":
        _family_medicine_unavailable(case)
        return
    # A Backend clue decision counts as sufficient; the safety and grounding gates still apply.
    clue_decided = decision in {"decide", "family_medicine"} and state.department_status == "resolved"
    # The safety-net route needs the completed safety screen (checked when adopted), not this turn's grounding.
    safety_net_routed = clue_decided and state.clue_decision.reason in SAFETY_NET_REASONS
    # So does the patient's validated department request: no clue or planner question follows it.
    backend_decided = clue_decided or requested_department_adopted(case)
    # An availability turn follows a medically sufficient turn; the gates below still apply.
    availability_turn = state.availability_status == "asked"
    sufficient = (
        (state.clarification_status == "safety_check" or backend_decided or availability_turn
         or (suggestion is not None and suggestion.status == "sufficient"))
        and (safety_ready or safety_net_routed)
        and state.pending_clarification_intent in {None, AVAILABILITY_PENDING_INTENT}
    )
    if sufficient:
        if safety_screen_resolved(case) and state.department_status == "resolved":
            _complete_or_ask_availability(case, availability_changed=availability_changed)
            return

    # A changed department or safety state takes the turn back from the availability question.
    _release_availability_pending(case)
    settings = get_settings()
    # Reached only when the safety-net family medicine route could not apply (e.g. safety screen open).
    if state.turn_count >= settings.hard_turn_cap:
        _unresolved(case)
        return
    if (state.body_part_question_source and state.department_status != "resolved"
            and safety_screen_resolved(case) and state.pending_clue_question_index is None
            and state.pending_clarification_intent is None):
        _ask_body_part_question(case, state.body_part_question_source)
        return
    if companion_question_due(case):
        _ask_companion_question(case)
        return
    # Stalled replies no longer end the case: consecutive_no_progress_turns never exceeds
    # turn_count, so a stalled case reaches this one-time question at the same turn.
    if (not state.safety_net_asked and (state.turn_count >= settings.safety_net_ask_turn
                                         or (state.turn_count >= settings.safety_net_no_candidate_turn
                                             and _knowledge_found_no_department(case)))
            and state.department_status != "resolved" and safety_screen_resolved(case)
            and state.pending_clue_question_index is None):
        _ask_safety_net_question(case)
        return

    if (suggestion is not None and suggestion.status == "sufficient"
            and safety_ready and safety_screen_resolved(case)
            and state.pending_clarification_intent is None and state.department_status != "resolved"):
        state.clarification_status = "clarifying"
        state.reply_source = "DEPARTMENT_EVIDENCE_UNRESOLVED"
        state.uncertainty_reasons = [state.department_uncertainty_reason or "正式科別證據不足"]
        state.next_information_needed = []
        state.is_complete = False
        case.triage.need_more_info = True
        case.triage.next_question = DEPARTMENT_EVIDENCE_REPLY
        case.triage.is_final = False
        return

    same_pending_intent = bool(
        suggestion is not None
        and state.pending_clarification_intent
        and suggestion.intent == state.pending_clarification_intent
    )
    valid_question = (
        suggestion is not None
        and suggestion.status == "clarification_needed"
        and bool(suggestion.question)
        and bool(suggestion.intent)
        and suggestion.intent not in state.clarification_evidence
        and (same_pending_intent or (
            state.pending_clarification_intent is None
            and (suggestion.intent not in state.asked_clarification_intents
                 or suggestion.intent == state.department_next_question_intent)
        ))
        and not _repeated_question(case, suggestion.question or "", dialogue_act=state.last_dialogue_act,
                                   explanation=suggestion.explanation, safety_planning=False)
        and not _asks_filled_duration(case, suggestion.intent, suggestion.question)
    )
    if valid_question:
        question = suggestion.question or ""
        if state.pending_clarification_intent is None:
            if suggestion.clue_concepts:
                _record_clue_question(case, suggestion)
            state.pending_clarification_intent = suggestion.intent
            if suggestion.intent not in state.asked_clarification_intents:
                state.asked_clarification_intents.append(suggestion.intent or "")
        state.pending_clarification_question = question
        reason = suggestion.reason
        # Information needs belong to this proposal, never to the previous owner.
        state.next_information_needed = [suggestion.information_need] if suggestion.information_need else []
        state.reply_source = suggestion.reply_source
        question = _with_explanation(suggestion, question)
    elif suggestion is not None and suggestion.rejection_reason not in _PROVIDER_FAILURES:
        _recover_stuck_turn(case, availability_changed=availability_changed)
        return
    else:
        question = TECHNICAL_CLARIFICATION_REPLY
        reason = "clarification_generation_failed"
        state.reply_source = "TECHNICAL_FALLBACK"
    if low_confidence:
        reason = "本輪症狀抽取信心不足，仍需釐清"
    elif not grounded_symptom:
        reason = "尚無 grounded 的主要症狀描述"
    elif not grounded_detail:
        reason = "目前仍缺少可區分症狀的細節"
    elif state.department_status != "resolved" and state.department_uncertainty_reason:
        reason = state.department_uncertainty_reason
    state.clarification_status = "clarifying"
    state.uncertainty_reasons = [reason]
    state.last_question_key = None
    state.is_complete = False
    case.triage.need_more_info = True
    case.triage.next_question = question
    case.triage.is_final = False
    case.triage.reasons.append(f"自然對話澄清中：{reason}")


def _complete_or_ask_availability(case: TriageCase, *, availability_changed: bool) -> None:
    state = case.conversation_state
    if not availability_settled(case, availability_changed=availability_changed):
        _ask_availability(case)
        return
    _release_availability_pending(case)
    state.clarification_status = "sufficient"
    state.uncertainty_reasons = []
    state.next_information_needed = []
    state.last_question_key = None
    state.is_complete = True
    case.triage.need_more_info = False
    case.triage.next_question = None
    case.triage.is_final = True
    case.triage.reasons.append("症狀資訊已通過 Backend 澄清完成條件；掛號偏好不是醫療完成門檻。")
    state.reply_source = "CONFIRMATION"


def _recover_stuck_turn(case: TriageCase, *, availability_changed: bool) -> None:
    """The provider answered but no question was usable (e.g. a repeat): never tell the patient
    to retry, since the planner would propose the same thing. Move forward with a fixed step."""
    state = case.conversation_state
    logger.info("clarification stuck case_id=%s rejection_reason=%s department_status=%s",
                case.case_id, state.question_rejection_reason, state.department_status)
    # The safety screen opens only once the symptom has details, so ask for them.
    if not safety_screen_resolved(case):
        _ask_stuck_open_question(case)
        return
    if state.department_status == "resolved":
        _complete_or_ask_availability(case, availability_changed=availability_changed)
        return
    # A patient who already chose to continue is not asked again; the hard turn cap bounds the loop.
    if state.safety_net_choice == "continue":
        _ask_stuck_open_question(case)
        return
    state.pending_clue_question_index = None
    _ask_safety_net_question(case)


def ask_revision_detail_question(case: TriageCase, user_sources: list[str]) -> None:
    """The patient asked to revise but said nothing to change yet: ask what to change, once.

    Replaces both a stale re-confirmation and the 「系統剛剛沒有回應」 reply for a turn the
    Turn Interpreter could not ground (例「我想要更改日期跟時段」 with no new time).
    """
    state = case.conversation_state
    state.turn_count += len(user_sources)
    state.clarification_status = "clarifying"
    state.uncertainty_reasons = ["病人要修改，但還沒說要改成什麼"]
    state.next_information_needed = []
    state.last_question_key = "revision_detail"
    state.reply_source = "REVISION_DETAIL_QUESTION"
    state.is_complete = False
    case.triage.need_more_info = True
    case.triage.next_question = REVISION_DETAIL_QUESTION
    case.triage.is_final = False
    case.triage.reasons.append("修訂後第一句沒有可更新的內容，改問要修改什麼。")


def _ask_stuck_open_question(case: TriageCase) -> None:
    """A pending question keeps its ownership; only a grounded answer releases it."""
    state = case.conversation_state
    state.clarification_status = "clarifying"
    state.uncertainty_reasons = [state.department_uncertainty_reason or "目前仍缺少可區分症狀的細節"]
    state.last_question_key = None
    state.reply_source = "STUCK_FALLBACK"
    state.is_complete = False
    case.triage.need_more_info = True
    case.triage.next_question = STUCK_OPEN_QUESTION
    case.triage.is_final = False
    case.triage.reasons.append("無可用的追問題目，改問固定的開放式問題。")


def has_accepted_time_preferences(case: TriageCase) -> bool:
    availability = case.availability
    return bool(availability.time_preferences or availability.preferred_dates
                or availability.preferred_days or availability.preferred_sessions)


def availability_dimension_has_data(case: TriageCase, dimension: str) -> bool:
    """Only accepted structured time data counts; relation does not matter."""
    availability = case.availability
    if dimension == "date":
        return bool(any(item.kind != "session" for item in availability.time_preferences)
                    or availability.preferred_dates or availability.preferred_days)
    return bool(any(item.session is not None for item in availability.time_preferences)
                or availability.preferred_sessions)


def _availability_dimension_done(case: TriageCase, dimension: str) -> bool:
    return (availability_dimension_has_data(case, dimension)
            or dimension in case.conversation_state.availability_dimension_outcomes)


def availability_settled(case: TriageCase, *, availability_changed: bool = False) -> bool:
    """Settle date and session separately: data, a grounded no-limit answer, or the per-dimension limit."""
    state = case.conversation_state
    if state.availability_status in {"answered", "unspecified"}:
        return True
    outcomes = state.availability_dimension_outcomes
    if AVAILABILITY_PENDING_INTENT in state.clarification_evidence and not availability_changed:
        # A grounded answer that produced no time data means no limit for the dimensions it was asked.
        for dimension in state.availability_pending_dimensions:
            if not availability_dimension_has_data(case, dimension):
                outcomes.setdefault(dimension, "unrestricted")
    attempts = {"date": state.availability_date_attempts, "session": state.availability_session_attempts}
    for dimension in AVAILABILITY_DIMENSIONS:
        if not _availability_dimension_done(case, dimension) and attempts[dimension] >= AVAILABILITY_QUESTION_LIMIT:
            outcomes[dimension] = "exhausted"
    if not all(_availability_dimension_done(case, dimension) for dimension in AVAILABILITY_DIMENSIONS):
        return False
    if (all(outcomes.get(dimension) == "exhausted" for dimension in AVAILABILITY_DIMENSIONS)
            and not has_accepted_time_preferences(case)):
        state.availability_status = "unspecified"
        logger.info("availability case_id=%s unspecified=true attempts=%s",
                    case.case_id, state.availability_question_attempts)
    else:
        state.availability_status = "answered"
    state.availability_pending_dimensions = []
    return True


def awaiting_availability_answer(case: TriageCase) -> bool:
    """The fixed availability question is still pending while department and safety stay settled."""
    state = case.conversation_state
    return bool(
        state.pending_clarification_intent == AVAILABILITY_PENDING_INTENT
        and state.department_status == "resolved"
        and safety_screen_resolved(case)
    )


def _release_availability_pending(case: TriageCase) -> None:
    state = case.conversation_state
    if state.pending_clarification_intent == AVAILABILITY_PENDING_INTENT:
        state.pending_clarification_intent = None
        state.pending_clarification_question = None
        state.next_information_needed = []


def _ask_availability(case: TriageCase) -> None:
    """Backend-owned fixed question for the missing dimensions; confirmation waits for it."""
    state = case.conversation_state
    missing = [dimension for dimension in AVAILABILITY_DIMENSIONS if not _availability_dimension_done(case, dimension)]
    if len(missing) == len(AVAILABILITY_DIMENSIONS):
        question = AVAILABILITY_QUESTION
    elif missing == ["date"]:
        question = AVAILABILITY_DATE_QUESTION
    else:
        question = AVAILABILITY_SESSION_QUESTION
    if "date" in missing:
        state.availability_date_attempts += 1
    if "session" in missing:
        state.availability_session_attempts += 1
    state.availability_pending_dimensions = missing
    # Each ask needs its own grounded answer; an earlier one belonged to the previous question.
    state.clarification_evidence.pop(AVAILABILITY_PENDING_INTENT, None)
    state.availability_status = "asked"
    state.availability_question_attempts += 1
    state.pending_clarification_intent = AVAILABILITY_PENDING_INTENT
    state.pending_clarification_question = question
    if AVAILABILITY_PENDING_INTENT not in state.asked_clarification_intents:
        state.asked_clarification_intents.append(AVAILABILITY_PENDING_INTENT)
    state.next_information_needed = []
    state.clarification_status = "clarifying"
    state.uncertainty_reasons = []
    state.last_question_key = None
    state.reply_source = "AVAILABILITY_QUESTION"
    state.is_complete = False
    case.triage.need_more_info = True
    case.triage.next_question = question
    case.triage.is_final = False
    case.triage.reasons.append("症狀資訊已完成，確認前詢問看診時間；時間不是醫療完成門檻。")


def _knowledge_found_no_department(case: TriageCase) -> bool:
    """Department reasoning ran and reached no candidate except family medicine (e.g. 「舌頭痛」).

    A failed reasoning turn (SQL or AI unavailable) also leaves no candidates, so it never counts.
    """
    state = case.conversation_state
    found = set(state.department_clue_candidates) | {item.dept_id for item in state.candidate_departments}
    if found:
        return not found - {get_settings().family_medicine_dept_id}
    return state.department_uncertainty_reason == NO_OFFICIAL_EVIDENCE_REASON


def _ask_safety_net_question(case: TriageCase) -> None:
    """Backend-owned fixed question, asked once; the Turn Interpreter judges the reply."""
    state = case.conversation_state
    state.safety_net_asked = True
    state.pending_clarification_intent = SAFETY_NET_PENDING_INTENT
    state.pending_clarification_question = SAFETY_NET_QUESTION
    if SAFETY_NET_PENDING_INTENT not in state.asked_clarification_intents:
        state.asked_clarification_intents.append(SAFETY_NET_PENDING_INTENT)
    state.next_information_needed = []
    state.clarification_status = "clarifying"
    state.uncertainty_reasons = [state.department_uncertainty_reason or "正式科別證據不足"]
    state.last_question_key = None
    state.reply_source = "SAFETY_NET_QUESTION"
    state.is_complete = False
    case.triage.need_more_info = True
    case.triage.next_question = SAFETY_NET_QUESTION
    case.triage.is_final = False
    case.triage.reasons.append("多輪後仍無法區分科別，詢問一次要繼續描述或先掛家庭醫學科。")


def _age_question_due(case: TriageCase) -> bool:
    """Once the safety screen is done, an age-less 「我兒子」 gets one age question; never with a chosen department."""
    state = case.conversation_state
    patient = case.patient_input
    return (
        safety_screen_resolved(case) and not state.patient_age_question_asked
        and state.pending_clarification_intent is None
        and patient.requested_department_id is None and not patient.requested_department_name
        and pediatric.patient_status(state) == "ask"
    )


def _ask_age_question(case: TriageCase) -> None:
    """Backend-owned fixed question; any reply releases it and the age is read by code."""
    state = case.conversation_state
    question = pediatric.age_question(state)
    state.patient_age_question_asked = True
    state.pending_clarification_intent = pediatric.AGE_PENDING_INTENT
    state.pending_clarification_question = question
    if pediatric.AGE_PENDING_INTENT not in state.asked_clarification_intents:
        state.asked_clarification_intents.append(pediatric.AGE_PENDING_INTENT)
    state.next_information_needed = []
    state.clarification_status = "clarifying"
    state.uncertainty_reasons = ["未滿 18 歲先看兒科，需確認就診者年齡"]
    state.last_question_key = None
    state.reply_source = "AGE_QUESTION"
    state.is_complete = False
    case.triage.need_more_info = True
    case.triage.next_question = question
    case.triage.is_final = False
    case.triage.reasons.append("使用者描述家人但未說年齡，先詢問一次年齡以判斷是否看兒科。")


def companion_question_due(case: TriageCase) -> bool:
    """decide() asked for the accompanying-symptom question and nothing else is pending.

    A decision kept from an earlier turn (department reasoning did not run) is not acted on again.
    """
    state = case.conversation_state
    decision = state.clue_decision
    return bool(
        decision is not None and decision.kind == "ask_companion"
        and state.companion_questions_asked < COMPANION_QUESTION_LIMIT
        and (state.companion_questions_asked == 0
             or not set(decision.scored_sources) <= set(state.companion_question_sources))
        and state.department_status != "resolved" and safety_screen_resolved(case)
        and state.pending_clarification_intent is None
    )


def _ask_companion_question(case: TriageCase) -> None:
    """Backend-owned fixed question; any reply releases it and new symptoms are re-scored."""
    state = case.conversation_state
    question = companion_question(case.patient_input.symptom, state.companion_questions_asked)
    state.companion_questions_asked += 1
    state.companion_question_sources = list(state.clue_decision.scored_sources) if state.clue_decision else []
    state.pending_clarification_intent = COMPANION_PENDING_INTENT
    state.pending_clarification_question = question
    if COMPANION_PENDING_INTENT not in state.asked_clarification_intents:
        state.asked_clarification_intents.append(COMPANION_PENDING_INTENT)
    state.next_information_needed = []
    state.clarification_status = "clarifying"
    state.uncertainty_reasons = ["只有一個候選科別，先確認有沒有其他不舒服"]
    state.last_question_key = None
    state.reply_source = "COMPANION_QUESTION"
    state.is_complete = False
    case.triage.need_more_info = True
    case.triage.next_question = question
    case.triage.is_final = False
    case.triage.reasons.append("症狀只對到一個科別，定科前先問一次有沒有其他不舒服。")


def _ask_body_part_question(case: TriageCase, source: str) -> None:
    """Backend-owned fixed question, asked once per symptom source; any reply releases it."""
    state = case.conversation_state
    question = body_part_question(source)
    state.body_part_questions_asked.append(source)
    state.body_part_question_source = None
    state.pending_clarification_intent = BODY_PART_PENDING_INTENT
    state.pending_clarification_question = question
    if BODY_PART_PENDING_INTENT not in state.asked_clarification_intents:
        state.asked_clarification_intents.append(BODY_PART_PENDING_INTENT)
    state.next_information_needed = []
    state.clarification_status = "clarifying"
    state.uncertainty_reasons = ["症狀未說明部位"]
    state.last_question_key = None
    state.reply_source = "BODY_PART_QUESTION"
    state.is_complete = False
    case.triage.need_more_info = True
    case.triage.next_question = question
    case.triage.is_final = False
    case.triage.reasons.append("症狀未說明部位，且只有這個症狀指向其他科別，先詢問部位。")


def _enter_safety_check(case: TriageCase) -> None:
    state = case.conversation_state
    if state.clarification_status != "safety_check":
        state.safety_answer_attempts = 0
        state.safety_answer_validation = None
    state.clarification_status = "safety_check"
    state.uncertainty_reasons = ["急迫症狀篩檢尚未完成"]
    state.next_information_needed = []
    state.pending_clarification_intent = None
    state.pending_clarification_question = None
    state.department_next_question_intent = None
    state.department_next_information_needed = None
    state.reply_source = "SAFETY_CONTROLLED_FALLBACK"
    state.is_complete = False
    mark_questions_asked(case, [RED_FLAG_QUESTION_KEY])
    case.triage.need_more_info = True
    case.triage.next_question = QUESTION_TEXTS[RED_FLAG_QUESTION_KEY]
    case.triage.is_final = False
    case.triage.reasons.append("症狀描述已足夠，仍須完成既有急迫症狀篩檢。")


def _asks_filled_duration(case: TriageCase, intent: str | None, question: str | None) -> bool:
    return bool(
        case.patient_input.duration
        and ((intent and "duration" in intent) or (question and _DURATION_QUESTION.search(question)))
    )


def _patient_information_needs(values: list[str]) -> list[str]:
    needs = [_patient_information_need(value) for value in values]
    return [need for need in needs if need]


def _patient_information_need(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    focus = value.strip().strip("。？！? ")
    if (
        not focus
        or len(focus) > 60
        or _UNSAFE_QUESTION.search(focus)
        or _INTERNAL_STATE_TEXT.search(focus)
        or _GENERIC_INFORMATION_NEED.fullmatch(focus)
    ):
        return None
    return focus


def _grounded_clarification_answer(
    intent: object, source_text: object, status: object, confidence: object,
    pending_intent: str | None, user_sources: list[str],
) -> bool:
    return bool(
        pending_intent
        and intent == pending_intent
        and isinstance(source_text, str)
        and source_text.strip()
        and any(source_text.strip() in source for source in user_sources)
        and isinstance(status, str)
        and status in {"answered", "partial", "unclear"}
        and isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(confidence)
        and 0 <= confidence <= 1
    )


def _family_medicine_unavailable(case: TriageCase) -> None:
    """Terminal when the configured family medicine id is not exactly one live department."""
    _unresolved(case)
    state = case.conversation_state
    state.reply_source = "FAMILY_MEDICINE_UNAVAILABLE"
    state.uncertainty_reasons = ["family_medicine_department_unavailable"]
    case.triage.reasons[-1] = "線索題無法區分科別，且設定的家醫科無法對應唯一正式科別；未虛構或改推其他科別。"


def _unresolved(case: TriageCase) -> None:
    state = case.conversation_state
    state.clarification_status = "unresolved"
    state.reply_source = "UNRESOLVED"
    state.uncertainty_reasons = ["多輪後仍無法可靠釐清症狀"]
    state.next_information_needed = []
    state.pending_clarification_intent = None
    state.pending_clarification_question = None
    state.is_complete = False
    state.last_question_key = None
    case.triage.need_more_info = True
    case.triage.next_question = None
    case.triage.is_final = False
    case.triage.reasons.append("已達自然澄清保護上限；未自動補值或選科。")
