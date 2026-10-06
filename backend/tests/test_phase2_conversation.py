from __future__ import annotations

import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas import TriageCase
from app.services import conversation_service, rag_triage_adapter
from app.services.conversation_service import SAFETY_PENDING_INTENT
from app.services.case_store import save_case
from app.services.rule_engine import QUESTION_TEXTS, RED_FLAG_QUESTION_KEY


chat_route = importlib.import_module("app.routes.chat")
recommend_route = importlib.import_module("app.routes.recommend")


def extraction(field, value, source, confidence=0.95, assertion=None):
    item = {
        "field": field,
        "normalized_value": value,
        "source_text": source,
        "confidence": confidence,
        "semantic_status": "available",
    }
    if assertion is not None or field in {"symptom", "accompanying_symptoms"}:
        item["assertion"] = assertion or "present"
    return item


class Phase2ConversationTest(unittest.TestCase):
    def post(
        self, message, extractions, plan, *, triage_case=None, confirmed=False,
        ttas_evidence=None, safety_assertion=None,
    ):
        if triage_case:
            save_case(TriageCase.model_validate(triage_case))
        settings = SimpleNamespace(cerebras_api_key="test-key", batch_triage_enabled=False)
        pending_answer = None
        if isinstance(plan, dict) and triage_case and plan.get("answered_intent"):
            pending_answer = {
                "answered_intent": plan.get("answered_intent"),
                "answer_status": plan.get("answer_status"),
                "answer_source_text": plan.get("answer_source_text"),
                "answer_confidence": plan.get("answer_confidence"),
            }
        if (
            triage_case
            and triage_case.get("conversation_state", {}).get("clarification_status") == "safety_check"
            and safety_assertion is not None
        ):
            pending_answer = {
                "answered_intent": SAFETY_PENDING_INTENT,
                "answer_status": "answered",
                "answer_source_text": message,
                "answer_confidence": 0.98,
                "answer_assertion": safety_assertion,
            }
        semantic = AsyncMock(return_value=json.dumps(
            {
                "semantic_extractions": extractions,
                "pending_answer": pending_answer,
                "ttas_evidence": ttas_evidence or [],
            },
            ensure_ascii=False,
        ))
        clarification = AsyncMock(
            side_effect=plan if isinstance(plan, Exception) else None,
            return_value=json.dumps(plan, ensure_ascii=False) if isinstance(plan, dict) else plan,
        )
        department = AsyncMock(return_value=None)
        with patch.object(chat_route, "get_settings", return_value=settings), patch.object(
            rag_triage_adapter, "get_settings", return_value=settings,
        ), patch.object(rag_triage_adapter, "complete_prompt", new=semantic), patch.object(
            conversation_service, "complete_runtime_json", new=clarification,
        ), patch.object(chat_route, "detect_department_result", new=department), patch.object(
            chat_route, "generate_triage_reply",
            new=AsyncMock(side_effect=lambda **kw: kw["fallback_reply"]),
        ):
            response = TestClient(app).post("/chat", json={
                **({"case_id": triage_case["case_id"]} if triage_case else {}),
                "message": message,
                "confirmed": confirmed,
                **({"triage_case": triage_case} if triage_case else {}),
            })
        self.assertEqual(response.status_code, 200)
        return response.json(), semantic, clarification, department

    def rich_first_turn(self, *, confirmed=False, triage_case=None):
        message = "我上禮拜從樓梯踩空，右腳腳背腫起來，最近走路越來越痛，晚上也會痛醒。"
        return self.post(
            message,
            [
                extraction("symptom", "腳背腫痛", "右腳腳背腫起來"),
                extraction("body_part", "右腳腳背", "右腳腳背"),
                extraction("duration", "1週", "上禮拜"),
                extraction("severity", "severe", "晚上也會痛醒"),
                extraction("onset", "從樓梯踩空", "從樓梯踩空"),
                extraction("accompanying_symptoms", ["走路越來越痛"], "走路越來越痛"),
            ],
            {"status": "sufficient", "question": None, "intent": None,
             "reason": "症狀、起因與影響已足夠"},
            confirmed=confirmed,
            triage_case=triage_case,
        )

    def test_dizziness_gets_contextual_question_not_checklist_body_part(self):
        question = "你說的頭暈比較像周圍在旋轉，還是快昏倒、眼前發黑？"
        result, semantic, clarification, department = self.post(
            "我最近一直頭暈",
            [extraction("symptom", "頭暈", "頭暈")],
            {"status": "clarification_needed", "question": question,
             "intent": "clarify_dizziness_type", "reason": "頭暈型態仍不清楚"},
        )
        semantic.assert_awaited_once()
        clarification.assert_awaited_once()
        department.assert_not_awaited()
        self.assertEqual(result["next_question"], question)
        self.assertEqual(result["conversation_state"]["asked_clarification_intents"], ["clarify_dizziness_type"])
        self.assertEqual(result["conversation_state"]["turn_count"], 1)
        self.assertEqual(result["triage_case"]["history_records"][-1]["content"], question)

    def test_rich_description_can_be_sufficient_without_preferences(self):
        result, _, clarification, department = self.rich_first_turn()
        clarification.assert_awaited_once()
        department.assert_not_awaited()
        self.assertTrue(result["needMoreInfo"])
        self.assertFalse(result["conversation_state"]["is_complete"])
        self.assertEqual(result["conversation_state"]["stage"], "collecting")
        self.assertEqual(result["conversation_state"]["clarification_status"], "safety_check")
        self.assertEqual(result["conversation_state"]["last_question_key"], "red_flags")
        self.assertEqual(result["conversation_state"]["question_attempts"]["red_flags"], 1)
        self.assertEqual(result["conversation_state"]["asked_clarification_intents"], [])
        self.assertEqual(result["next_question"], QUESTION_TEXTS[RED_FLAG_QUESTION_KEY])
        self.assertFalse(result["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertFalse(result["triage"]["is_final"])
        self.assertEqual(result["triage_case"]["patient_input"]["symptom"], "右腳腳背腫起來")
        self.assertEqual(result["triage_case"]["availability"]["preferred_days"], [])

    def test_negative_safety_answer_does_not_bypass_unresolved_department(self):
        first, _, _, _ = self.rich_first_turn()
        answer = "沒有胸痛，也沒有呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛"
        second, semantic, clarification, department = self.post(
            answer,
            [
                extraction("accompanying_symptoms", ["胸痛"], "沒有胸痛", assertion="absent"),
                extraction("accompanying_symptoms", ["呼吸困難"], "沒有呼吸困難", assertion="absent"),
            ],
            None, triage_case=first["triage_case"],
            safety_assertion="absent",
        )
        semantic.assert_awaited_once()
        department.assert_not_awaited()
        self.assertTrue(second["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(second["triage_case"]["patient_input"]["red_flags"], [])
        self.assertEqual(second["triage_case"]["patient_input"]["accompanying_symptoms"], ["走路越來越痛"])
        self.assertEqual(
            len(second["triage_case"]["semantic_extractions"]),
            len(first["triage_case"]["semantic_extractions"]) + 2,
        )
        self.assertIn({"role": "user", "content": answer}, second["triage_case"]["history_records"])
        self.assertFalse(second["conversation_state"]["is_complete"])
        self.assertTrue(second["needMoreInfo"])
        self.assertEqual(second["conversation_state"]["stage"], "collecting")
        self.assertEqual(second["conversation_state"]["department_status"], "unresolved")

    def test_eighth_symptom_turn_enters_safety_check_before_hard_cap(self):
        case = TriageCase(case_id="phase22-eighth-turn")
        case.conversation_state.turn_count = 7
        eighth, semantic, clarification, department = self.rich_first_turn(
            triage_case=case.model_dump(mode="json"),
        )
        semantic.assert_awaited_once()
        clarification.assert_awaited_once()
        department.assert_not_awaited()
        state = eighth["conversation_state"]
        self.assertEqual(state["turn_count"], 8)
        self.assertEqual(state["clarification_status"], "safety_check")
        self.assertFalse(state["is_complete"])
        self.assertEqual(eighth["next_question"], QUESTION_TEXTS[RED_FLAG_QUESTION_KEY])

        completed, semantic, clarification, department = self.post(
            "沒有胸痛、呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛",
            [extraction("symptom", "胸痛", "胸痛")], None,
            triage_case=eighth["triage_case"],
            safety_assertion="absent",
        )
        semantic.assert_awaited_once()
        department.assert_not_awaited()
        self.assertEqual(completed["conversation_state"]["turn_count"], 8)
        self.assertEqual(completed["conversation_state"]["clarification_status"], "unresolved")
        self.assertFalse(completed["conversation_state"]["is_complete"])
        self.assertTrue(completed["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(completed["triage_case"]["patient_input"]["red_flags"], [])

    def test_confirmation_cannot_skip_safety_question(self):
        result, _, _, department = self.rich_first_turn(confirmed=True)
        department.assert_not_awaited()
        self.assertEqual(result["conversation_state"]["stage"], "collecting")
        self.assertFalse(result["conversation_state"]["confirmed"])
        self.assertFalse(result["conversation_state"]["is_complete"])
        self.assertTrue(result["needMoreInfo"])

    def test_recommend_rejects_forged_completion_without_safety(self):
        first, _, _, _ = self.rich_first_turn()
        forged = first["triage_case"]
        forged["conversation_state"]["is_complete"] = True
        forged["conversation_state"]["confirmed"] = True
        forged["triage"]["need_more_info"] = False
        forged["confirmed"] = True
        with patch.object(recommend_route, "detect_department_result", new=AsyncMock()) as detector, patch.object(
            recommend_route, "recommend_appointments", new=AsyncMock(),
        ) as recommender:
            response = TestClient(app).post("/recommend", json={
                "triage_case": forged, "visit_type": "initial", "confirmed": True,
            })
        self.assertEqual(response.status_code, 400)
        self.assertIn("尚未完成", response.json()["detail"])
        detector.assert_not_awaited()
        recommender.assert_not_awaited()

    def test_positive_red_flag_keeps_urgent_path(self):
        result, _, clarification, department = self.post(
            "化學藥劑濺到我的眼睛", [extraction("symptom", "眼睛灼傷", "眼睛")], None,
            ttas_evidence=[{
                "field": "chemical_eye_injury",
                "value": True,
                "semantic_status": "available",
                "confidence": 0.95,
                "source_text": "化學藥劑濺到我的眼睛",
            }],
        )
        clarification.assert_not_awaited()
        department.assert_not_awaited()
        self.assertEqual(result["triage_case"]["ttas_result"]["level_candidate"], 2)
        self.assertEqual(result["triage"]["urgency_score"], None)
        self.assertTrue(result["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual(result["triage_case"]["patient_input"]["red_flags_status"], "positive_ttas")
        self.assertEqual(result["conversation_state"]["clarification_status"], "urgent")
        self.assertFalse(result["needMoreInfo"])
        self.assertIsNone(result["next_question"])

    def test_follow_up_answer_uses_history_and_does_not_repeat_intent(self):
        first, _, _, _ = self.post(
            "最近一直頭暈", [extraction("symptom", "頭暈", "頭暈")],
            {"status": "clarification_needed", "question": "比較像房子在轉，還是快昏倒？",
             "intent": "clarify_dizziness_type", "reason": "型態未明"},
        )
        second, _, clarification, department = self.post(
            "像房子在轉", [extraction("accompanying_symptoms", ["像房子在轉"], "像房子在轉")],
            {"status": "sufficient", "question": None, "intent": None,
             "reason": "頭暈型態已釐清", "answered_intent": "clarify_dizziness_type",
             "answer_source_text": "像房子在轉", "answer_status": "answered",
             "answer_confidence": 0.95},
            triage_case=first["triage_case"],
        )
        prompt = clarification.await_args.args[0]
        self.assertIn("比較像房子在轉，還是快昏倒？", prompt)
        self.assertIn("像房子在轉", prompt)
        self.assertEqual(second["conversation_state"]["clarification_evidence"]["clarify_dizziness_type"], "像房子在轉")
        self.assertEqual(second["conversation_state"]["asked_clarification_intents"], ["clarify_dizziness_type"])
        self.assertIsNone(second["conversation_state"]["pending_clarification_intent"])
        self.assertEqual(second["conversation_state"]["turn_count"], 2)
        self.assertTrue(second["needMoreInfo"])
        self.assertEqual(second["conversation_state"]["clarification_status"], "safety_check")
        self.assertIn("突發胸痛", second["next_question"])
        department.assert_not_awaited()

    def test_filled_duration_is_not_reasked(self):
        result, _, _, _ = self.post(
            "我頭暈，上禮拜開始", [
                extraction("symptom", "頭暈", "頭暈"),
                extraction("duration", "1週", "上禮拜"),
            ],
            {"status": "clarification_needed", "question": "這樣持續多久了？",
             "intent": "clarify_timeline", "reason": "持續時間不清楚"},
        )
        self.assertEqual(result["triage_case"]["patient_input"]["duration"], "1週")
        self.assertNotIn("duration", result["conversation_state"]["asked_clarification_intents"])
        self.assertNotEqual(result["next_question"], "這樣持續多久了？")

    def test_provider_failure_and_malformed_json_keep_facts_unchanged(self):
        for plan in (TimeoutError(), "{bad json"):
            with self.subTest(plan=type(plan).__name__):
                result, _, clarification, department = self.post(
                    "我頭暈", [extraction("symptom", "頭暈", "頭暈")], plan,
                )
                clarification.assert_awaited_once()
                department.assert_not_awaited()
                self.assertEqual(result["triage_case"]["patient_input"]["symptom"], "頭暈")
                self.assertIsNone(result["triage_case"]["patient_input"]["duration"])
                self.assertTrue(result["needMoreInfo"])
                self.assertIn("描述", result["next_question"])

    def test_ai_cannot_set_workflow_or_complete_blank_case(self):
        result, _, _, department = self.post(
            "我不太確定", [],
            {"status": "sufficient", "question": None, "intent": None,
             "reason": "足夠", "is_complete": True, "stage": "confirmed",
             "confirmed": True, "recommendation_generated": True},
        )
        state = result["conversation_state"]
        self.assertFalse(state["is_complete"])
        self.assertEqual(state["stage"], "collecting")
        self.assertFalse(state["confirmed"])
        self.assertFalse(result["triage_case"]["recommendation_generated"])
        department.assert_not_awaited()

    def test_vague_symptom_cannot_be_declared_sufficient(self):
        result, _, _, department = self.post(
            "肚子不舒服", [extraction("symptom", "肚子不舒服", "肚子不舒服")],
            {"status": "sufficient", "question": None, "intent": None,
             "reason": "資訊足夠", "is_complete": True},
        )
        self.assertTrue(result["needMoreInfo"])
        self.assertEqual(result["conversation_state"]["clarification_status"], "clarifying")
        self.assertFalse(result["conversation_state"]["is_complete"])
        department.assert_not_awaited()

    def test_low_confidence_and_pending_answer_block_sufficient(self):
        first, _, _, _ = self.post(
            "我頭暈", [extraction("symptom", "頭暈", "頭暈")],
            {"status": "clarification_needed", "question": "是旋轉感還是快昏倒？",
             "intent": "clarify_dizziness_type", "reason": "型態未明"},
        )
        second, _, _, department = self.post(
            "大概是吧", [extraction("severity", "mild", "大概是吧", confidence=0.2)],
            {"status": "sufficient", "question": None, "intent": None,
             "reason": "足夠", "answered_intent": "clarify_dizziness_type",
             "answer_source_text": "不在原文", "answer_status": "answered",
             "answer_confidence": 0.95},
            triage_case=first["triage_case"],
        )
        self.assertTrue(second["needMoreInfo"])
        self.assertEqual(second["conversation_state"]["pending_clarification_intent"], "clarify_dizziness_type")
        self.assertFalse(second["conversation_state"]["is_complete"])
        department.assert_not_awaited()

    def test_answered_clinical_pending_without_structured_evidence_fails_closed(self):
        first, _, _, _ = self.post(
            "腳背腫痛", [extraction("symptom", "腳背腫痛", "腳背腫痛")],
            {"status": "clarification_needed", "question": "現在還能承重走路嗎？",
             "intent": "fracture_assessment", "reason": "承重情況未明"},
        )
        answer = "還能走路，但是踩地會痛"
        second, semantic, _, department = self.post(
            answer, [],
            {"status": "sufficient", "question": None, "intent": None,
             "reason": "已回答", "answered_intent": "fracture_assessment",
             "answer_source_text": answer, "answer_status": "answered",
             "answer_confidence": 0.95},
            triage_case=first["triage_case"],
        )
        self.assertTrue(second["needMoreInfo"])
        self.assertEqual(second["conversation_state"]["pending_clarification_intent"], "fracture_assessment")
        self.assertEqual(second["conversation_state"]["clarification_evidence"], {})
        self.assertEqual(second["conversation_state"]["clarification_status"], "clarifying")
        self.assertEqual(second["triage_case"]["patient_input"]["symptom"], "腳背腫痛")
        self.assertEqual(len(second["triage_case"]["semantic_extractions"]), 1)
        self.assertEqual(semantic.await_count, 2)
        department.assert_not_awaited()

    def test_ungrounded_free_form_answer_cannot_clear_pending(self):
        first, _, _, _ = self.post(
            "腳背腫痛", [extraction("symptom", "腳背腫痛", "腳背腫痛")],
            {"status": "clarification_needed", "question": "現在還能承重走路嗎？",
             "intent": "fracture_assessment", "reason": "承重情況未明"},
        )
        second, _, _, department = self.post(
            "還能走路，但是踩地會痛", [],
            {"status": "sufficient", "question": None, "intent": None,
             "reason": "已回答", "answered_intent": "fracture_assessment",
             "answer_source_text": "完全不能走路", "answer_status": "answered",
             "answer_confidence": 0.95},
            triage_case=first["triage_case"],
        )
        self.assertEqual(second["conversation_state"]["pending_clarification_intent"], "fracture_assessment")
        self.assertEqual(second["conversation_state"]["clarification_evidence"], {})
        department.assert_not_awaited()

    def test_partial_answer_keeps_pending_but_allows_targeted_follow_up(self):
        first, _, _, _ = self.post(
            "腳背腫痛", [extraction("symptom", "腳背腫痛", "腳背腫痛")],
            {"status": "clarification_needed", "question": "現在還能承重走路嗎？",
             "intent": "fracture_assessment", "reason": "承重情況未明"},
        )
        follow_up = "走路時能把重量放在右腳上，還是只能用左腳支撐？"
        second, _, _, department = self.post(
            "還能走路", [],
            {"status": "clarification_needed", "question": follow_up,
             "intent": "fracture_assessment", "reason": "承重程度仍不清楚",
             "answered_intent": "fracture_assessment", "answer_source_text": "還能走路",
             "answer_status": "partial", "answer_confidence": 0.91},
            triage_case=first["triage_case"],
        )
        self.assertEqual(second["next_question"], follow_up)
        self.assertEqual(second["conversation_state"]["pending_clarification_intent"], "fracture_assessment")
        self.assertEqual(second["conversation_state"]["asked_clarification_intents"], ["fracture_assessment"])
        self.assertEqual(second["conversation_state"]["clarification_evidence"], {})
        department.assert_not_awaited()

    def test_unclear_answer_allows_safe_follow_up_but_rejects_duplicate_or_unsafe_question(self):
        first, _, _, _ = self.post(
            "腳背腫痛", [extraction("symptom", "腳背腫痛", "腳背腫痛")],
            {"status": "clarification_needed", "question": "現在還能承重走路嗎？",
             "intent": "fracture_assessment", "reason": "承重情況未明"},
        )
        answer = {
            "status": "clarification_needed", "intent": "fracture_assessment",
            "reason": "承重情況仍不清楚", "answered_intent": "fracture_assessment",
            "answer_source_text": "不太確定", "answer_status": "unclear",
            "answer_confidence": 0.8,
        }
        follow_up = "試著站立時，右腳可以承受身體重量嗎？"
        second, _, _, _ = self.post(
            "不太確定", [], {**answer, "question": follow_up},
            triage_case=first["triage_case"],
        )
        self.assertEqual(second["next_question"], follow_up)
        self.assertEqual(second["conversation_state"]["pending_clarification_intent"], "fracture_assessment")

        for question in ("現在還能承重走路嗎？", "你確診骨折了嗎？", "要掛哪個科別？"):
            with self.subTest(question=question):
                rejected, _, _, _ = self.post(
                    "不太確定", [], {**answer, "question": question},
                    triage_case=first["triage_case"],
                )
                self.assertNotEqual(rejected["next_question"], question)
                self.assertEqual(rejected["conversation_state"]["pending_clarification_intent"], "fracture_assessment")
                self.assertEqual(rejected["conversation_state"]["clarification_evidence"], {})

    def test_low_confidence_free_form_answer_keeps_pending(self):
        first, _, _, _ = self.post(
            "腳背腫痛", [extraction("symptom", "腳背腫痛", "腳背腫痛")],
            {"status": "clarification_needed", "question": "現在還能承重走路嗎？",
             "intent": "fracture_assessment", "reason": "承重情況未明"},
        )
        for confidence in (0.54, float("nan"), float("inf"), 1.1, True):
            with self.subTest(confidence=confidence):
                second, _, _, department = self.post(
                    "還能走路，但是踩地會痛", [],
                    {"status": "sufficient", "question": None, "intent": None,
                     "reason": "已回答", "answered_intent": "fracture_assessment",
                     "answer_source_text": "還能走路，但是踩地會痛",
                     "answer_status": "answered", "answer_confidence": confidence},
                    triage_case=first["triage_case"],
                )
                self.assertEqual(second["conversation_state"]["pending_clarification_intent"], "fracture_assessment")
                self.assertEqual(second["conversation_state"]["clarification_evidence"], {})
                department.assert_not_awaited()

    def test_wrong_answered_intent_cannot_clear_pending(self):
        first, _, _, _ = self.post(
            "腳背腫痛", [extraction("symptom", "腳背腫痛", "腳背腫痛")],
            {"status": "clarification_needed", "question": "現在還能承重走路嗎？",
             "intent": "fracture_assessment", "reason": "承重情況未明"},
        )
        second, _, _, department = self.post(
            "還能走路，但是踩地會痛", [],
            {"status": "sufficient", "question": None, "intent": None,
             "reason": "已回答", "answered_intent": "clarify_dizziness_type",
             "answer_source_text": "還能走路，但是踩地會痛",
             "answer_status": "answered", "answer_confidence": 0.95},
            triage_case=first["triage_case"],
        )
        self.assertEqual(second["conversation_state"]["pending_clarification_intent"], "fracture_assessment")
        self.assertEqual(second["conversation_state"]["clarification_evidence"], {})
        department.assert_not_awaited()

    def test_hard_cap_is_unresolved_without_guessing(self):
        case = None
        for turn in range(8):
            result, _, _, department = self.post(
                f"還不清楚{turn}", [], TimeoutError(), triage_case=case,
            )
            case = result["triage_case"]
            department.assert_not_awaited()
        state = result["conversation_state"]
        self.assertEqual(state["turn_count"], 8)
        self.assertEqual(state["clarification_status"], "unresolved")
        self.assertFalse(state["is_complete"])
        self.assertIsNone(result["next_question"])
        self.assertIn("不會替你猜測科別", result["reply"])
        self.assertEqual(result["triage_case"]["patient_input"]["symptom"], "")
        self.assertIsNone(result["department_result"])
        repeated, semantic, clarification, department = self.post(
            "還是不清楚", [], TimeoutError(), triage_case=case,
        )
        semantic.assert_not_awaited()
        clarification.assert_not_awaited()
        department.assert_not_awaited()
        self.assertEqual(repeated["conversation_state"]["turn_count"], 8)


if __name__ == "__main__":
    unittest.main()
