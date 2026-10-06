from __future__ import annotations

import json
import unittest
from uuid import uuid4
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.routes import chat as chat_route
from app.schemas import BatchAnswer, Message, TriageCase, VisitType
from app.services import batch_extraction_service, department_preference_service, rag_triage_adapter
from app.services.batch_extraction_service import extract_batch_answers
from app.services.case_store import get_case, save_case
from app.services.field_acceptance import has_duration_semantics, normalize_body_part
from app.services.rule_engine import apply_user_message, mark_questions_asked, missing_checklist_fields


DEPARTMENTS = [
    {"dept_id": 204, "parent_dept": "其他科", "child_dept": "皮膚科"},
    {"dept_id": 205, "parent_dept": "五官科", "child_dept": "耳科"},
]


class StrictKeyedAcceptanceTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _semantic_provider(*extractions: dict) -> AsyncMock:
        return AsyncMock(
            return_value=json.dumps(
                {"extractions": list(extractions)},
                ensure_ascii=False,
            )
        )

    async def _extract(
        self,
        key: str,
        answer: str,
        *,
        provider: AsyncMock | None = None,
    ) -> tuple[TriageCase, AsyncMock]:
        case = TriageCase(case_id=f"strict-{key}", visit_type=VisitType.INITIAL)
        semantic = provider or AsyncMock(
            side_effect=AssertionError("explicit deterministic/off-topic text must not call Cerebras")
        )
        with patch.object(
            chat_route,
            "runtime_ai_available",
            return_value=False,
        ), patch.object(
            department_preference_service,
            "fetch_active_departments",
            return_value=DEPARTMENTS,
        ), patch.object(
            batch_extraction_service,
            "runtime_ai_available",
            return_value=True,
        ), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=semantic,
        ):
            await extract_batch_answers(case, [BatchAnswer(key=key, answer=answer)])
        return case, semantic

    async def test_off_topic_department_and_availability_do_not_complete_symptom(self):
        provider = self._semantic_provider()
        case, provider = await self._extract(
            "symptom",
            "我要看皮膚科禮拜三上午",
            provider=provider,
        )

        self.assertEqual(case.patient_input.symptom, "")
        self.assertIn("symptom", missing_checklist_fields(case))
        self.assertEqual(case.availability.preferred_days, ["週三"])
        self.assertEqual(case.availability.preferred_sessions, ["上午"])
        self.assertEqual(case.patient_input.requested_department_id, 204)
        self.assertEqual(case.patient_input.requested_department_name, "皮膚科")
        self.assertIsNone(case.department_result)
        provider.assert_awaited_once()

    def test_chat_route_reasks_symptom_after_off_topic_keyed_answer(self):
        case_id = f"strict-route-{uuid4().hex}"
        provider = self._semantic_provider()
        with patch.object(
            chat_route,
            "runtime_ai_available",
            return_value=False,
        ), patch.object(
            department_preference_service,
            "fetch_active_departments",
            return_value=DEPARTMENTS,
        ), patch.object(
            batch_extraction_service,
            "runtime_ai_available",
            return_value=True,
        ), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            client = TestClient(app)
            client.post("/chat", json={"case_id": case_id, "visit_type": "initial"})
            response = client.post(
                "/chat",
                json={
                    "case_id": case_id,
                    "answers": [
                        {"key": "symptom", "answer": "我要看皮膚科禮拜三上午"}
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["question_batch"][0]["key"], "symptom")
        self.assertEqual(data["triage_case"]["patient_input"]["symptom"], "")
        self.assertEqual(data["triage_case"]["availability"]["preferred_days"], ["週三"])
        self.assertEqual(data["triage_case"]["availability"]["preferred_sessions"], ["上午"])
        provider.assert_awaited_once()

    async def test_department_registration_intent_does_not_complete_duration(self):
        provider = self._semantic_provider()
        case, provider = await self._extract(
            "duration",
            "我想看皮膚科直接幫我掛號",
            provider=provider,
        )

        self.assertIsNone(case.patient_input.duration)
        self.assertIn("duration", missing_checklist_fields(case))
        self.assertEqual(case.patient_input.requested_department_id, 204)
        provider.assert_awaited_once()

    async def test_exact_off_topic_duration_answer_is_seen_once_but_not_guessed(self):
        provider = self._semantic_provider()
        case, provider = await self._extract(
            "duration",
            "我想直接掛皮膚科",
            provider=provider,
        )

        provider.assert_awaited_once()
        self.assertIsNone(case.patient_input.duration)
        self.assertIn("duration", missing_checklist_fields(case))

    async def test_day_and_session_do_not_complete_severity(self):
        provider = self._semantic_provider()
        case, provider = await self._extract(
            "severity",
            "星期五上午",
            provider=provider,
        )

        self.assertIsNone(case.patient_input.severity)
        self.assertIn("severity", missing_checklist_fields(case))
        self.assertEqual(case.availability.preferred_days, ["週五"])
        self.assertEqual(case.availability.preferred_sessions, ["上午"])
        provider.assert_awaited_once()

    async def test_registration_intent_does_not_complete_body_part(self):
        provider = self._semantic_provider()
        case, provider = await self._extract(
            "body_part",
            "我想直接掛號",
            provider=provider,
        )

        self.assertIsNone(case.patient_input.body_part)
        self.assertIn("body_part", missing_checklist_fields(case))
        provider.assert_awaited_once()

    async def test_off_topic_red_flag_answer_keeps_safety_screen_open(self):
        case, provider = await self._extract("red_flags", "我說我想要看皮膚科禮拜三下午")

        self.assertFalse(case.patient_input.red_flags_checked)
        self.assertEqual(case.patient_input.red_flags_status, "not_checked")
        self.assertIn("red_flags", missing_checklist_fields(case))
        self.assertEqual(case.availability.preferred_days, [])
        self.assertEqual(case.availability.preferred_sessions, [])
        self.assertEqual(case.patient_input.requested_department_id, 204)
        provider.assert_not_awaited()

    async def test_duration_can_merge_out_of_order_without_completing_session(self):
        provider = self._semantic_provider()
        case, provider = await self._extract(
            "preferred_sessions",
            "三天",
            provider=provider,
        )

        self.assertEqual(case.patient_input.duration, "3天")
        self.assertEqual(case.availability.preferred_sessions, [])
        self.assertIn("preferred_sessions", missing_checklist_fields(case))
        provider.assert_awaited_once()

    async def test_valid_duration_is_accepted(self):
        case, provider = await self._extract("duration", "大概三天")

        self.assertEqual(case.patient_input.duration, "3天")
        self.assertNotIn("duration", missing_checklist_fields(case))
        provider.assert_not_awaited()

    async def test_natural_week_duration_forms_are_canonicalized_without_ai(self):
        clear_examples = (
            ("癢一個禮拜了", "1週"),
            ("癢一禮拜了", "1週"),
            ("我已經痛兩個禮拜", "2週"),
            ("我已經痛兩禮拜", "2週"),
            ("一個星期", "1週"),
            ("痛兩個星期了", "2週"),
            ("幾個禮拜了", "幾週"),
            ("好幾個禮拜了", "好幾週"),
        )
        for answer, expected in clear_examples:
            with self.subTest(answer=answer):
                case, provider = await self._extract("duration", answer)

                self.assertEqual(case.patient_input.duration, expected)
                self.assertNotIn("duration", missing_checklist_fields(case))
                provider.assert_not_awaited()

        weak_tone_examples = (
            ("大概一週左右", "1週"),
            ("這個症狀差不多三個禮拜了", "3週"),
        )
        for answer, expected in weak_tone_examples:
            with self.subTest(answer=answer):
                case, provider = await self._extract("duration", answer)

                self.assertEqual(case.patient_input.duration, expected)
                self.assertNotIn("duration", missing_checklist_fields(case))
                provider.assert_not_awaited()

    async def test_weekday_phrases_are_not_consumed_as_duration(self):
        examples = (
            ("禮拜三上午", ["週三"], ["上午"]),
            ("下禮拜三看診", ["週三"], []),
        )
        for answer, expected_days, expected_sessions in examples:
            with self.subTest(answer=answer):
                provider = self._semantic_provider()
                case, provider = await self._extract("duration", answer, provider=provider)

                self.assertIsNone(case.patient_input.duration)
                self.assertIn("duration", missing_checklist_fields(case))
                self.assertEqual(case.availability.preferred_days, expected_days)
                self.assertEqual(case.availability.preferred_sessions, expected_sessions)
                provider.assert_awaited_once()

    async def test_valid_functional_severity_is_accepted(self):
        case, provider = await self._extract("severity", "已經痛到影響睡覺")

        self.assertEqual(case.patient_input.severity, "severe")
        self.assertNotIn("severity", missing_checklist_fields(case))
        provider.assert_not_awaited()

    async def test_natural_severity_phrases_are_canonicalized_without_ai(self):
        examples = (
            ("沒有都可以正常作息", "mild"),
            ("沒有影響", "mild"),
            ("沒有影響生活", "mild"),
            ("沒有影響日常活動", "mild"),
            ("沒有到影響日常活動", "mild"),
            ("沒有明顯影響日常生活", "mild"),
            ("沒有到影響日常生活", "mild"),
            ("都可以正常生活", "mild"),
            ("可以正常作息", "mild"),
            ("不影響作息", "mild"),
            ("不影響日常生活", "mild"),
            ("不影響睡眠", "mild"),
            ("還能正常上班", "mild"),
            ("還能正常走路", "mild"),
            ("沒什麼影響", "mild"),
            ("沒有很嚴重", "mild"),
            ("還好，可以正常生活", "mild"),
            ("痛到睡不著", "severe"),
            ("已經影響睡眠", "severe"),
            ("沒辦法正常生活", "severe"),
            ("沒辦法上班", "severe"),
            ("不能正常走路", "severe"),
            ("嚴重影響生活", "severe"),
            ("已經無法正常活動", "severe"),
        )
        for answer, expected in examples:
            with self.subTest(answer=answer):
                case, provider = await self._extract("severity", answer)

                self.assertEqual(case.patient_input.severity, expected)
                self.assertEqual(
                    case.patient_input.severity_normalized.semantic_status,
                    "available",
                )
                self.assertNotIn("severity", missing_checklist_fields(case))
                provider.assert_not_awaited()

    async def test_bare_negative_does_not_guess_mild_severity(self):
        provider = self._semantic_provider()

        case, provider = await self._extract("severity", "沒有", provider=provider)

        self.assertIsNone(case.patient_input.severity)
        self.assertIn("severity", missing_checklist_fields(case))
        provider.assert_awaited_once()

    async def test_meta_reply_rechecks_previous_severity_answer(self):
        case = TriageCase(case_id="strict-meta-recheck", visit_type=VisitType.INITIAL)
        case.history_records.append(
            Message(role="user", content="[severity] 沒有都可以正常作息")
        )
        case.conversation_state.question_attempts["severity"] = 2
        case.conversation_state.field_statuses["severity"] = "unknown"
        provider = AsyncMock(side_effect=AssertionError("meta reply must not call Cerebras"))

        with patch.object(
            chat_route,
            "runtime_ai_available",
            return_value=False,
        ), patch.object(
            batch_extraction_service,
            "runtime_ai_available",
            return_value=True,
        ), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            await extract_batch_answers(
                case,
                [BatchAnswer(key="severity", answer="我剛剛已經回答了")],
            )

        self.assertEqual(case.patient_input.severity, "mild")
        self.assertEqual(case.conversation_state.field_statuses["severity"], "available")
        self.assertNotEqual(case.patient_input.severity, "我剛剛已經回答了")
        provider.assert_not_awaited()

    async def test_unresolved_meta_reply_does_not_consume_or_count_as_invalid_answer(self):
        case = TriageCase(case_id="strict-meta-unresolved", visit_type=VisitType.INITIAL)
        case.history_records.append(
            Message(role="user", content="[severity] 我不知道怎麼形容")
        )
        case.conversation_state.question_attempts["severity"] = 2
        case.conversation_state.field_statuses["severity"] = "unknown"
        case.conversation_state.consumed_fields.append("severity")
        provider = self._semantic_provider()

        with patch.object(
            chat_route,
            "runtime_ai_available",
            return_value=False,
        ), patch.object(
            batch_extraction_service,
            "runtime_ai_available",
            return_value=True,
        ), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            await extract_batch_answers(
                case,
                [BatchAnswer(key="severity", answer="我不是說了嗎")],
            )

        self.assertIsNone(case.patient_input.severity)
        self.assertNotIn("severity", case.conversation_state.consumed_fields)
        self.assertIn("severity", missing_checklist_fields(case))
        self.assertEqual(case.conversation_state.question_attempts["severity"], 1)
        provider.assert_awaited_once()

    async def test_meta_reply_replays_previous_compound_session_answer_semantically(self):
        case = TriageCase(case_id="strict-meta-session-replay", visit_type=VisitType.INITIAL)
        successful_reply = json.dumps(
            {
                "extractions": [
                    {
                        "field": "preferred_sessions",
                        "normalized_value": ["下午"],
                        "semantic_status": "partial",
                        "confidence": 0.9,
                        "source_text": "我不要上午和晚上",
                        "needs_clarification": False,
                        "follow_up_reason": None,
                    }
                ]
            },
            ensure_ascii=False,
        )
        provider = AsyncMock(side_effect=[RuntimeError("timeout"), successful_reply])

        with patch.object(
            batch_extraction_service,
            "runtime_ai_available",
            return_value=True,
        ), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            first_outcome = await extract_batch_answers(
                case,
                [BatchAnswer(key="preferred_sessions", answer="我不要上午和晚上")],
            )
            self.assertEqual(case.availability.preferred_sessions, [])
            self.assertTrue(first_outcome.ai_attempted)

            # The repeated question has been displayed, but the meta reply
            # must not consume that second clinical-answer opportunity.
            case.conversation_state.question_attempts["preferred_sessions"] = 2
            with self.assertLogs(batch_extraction_service.logger, level="INFO") as logs:
                replay_outcome = await extract_batch_answers(
                    case,
                    [BatchAnswer(key="preferred_sessions", answer="我剛剛說過了")],
                )

        self.assertEqual(provider.await_count, 2)
        self.assertTrue(replay_outcome.ai_attempted)
        self.assertEqual(case.availability.preferred_sessions, ["下午"])
        self.assertNotIn("preferred_sessions", missing_checklist_fields(case))
        self.assertEqual(case.conversation_state.question_attempts["preferred_sessions"], 1)
        user_history = [item.content for item in case.history_records if item.role == "user"]
        self.assertEqual(user_history.count("[preferred_sessions] 我不要上午和晚上"), 1)
        self.assertEqual(user_history.count("[preferred_sessions] 我剛剛說過了"), 1)
        log_output = "\n".join(logs.output)
        self.assertIn("[SEMANTIC_META_REPLAY]", log_output)
        self.assertIn("previous_answer_found=true", log_output)
        self.assertIn("already_satisfied=false", log_output)
        self.assertIn("semantic_replay=true", log_output)

    async def test_meta_reply_preserves_an_already_accepted_value_without_ai(self):
        case = TriageCase(case_id="strict-meta-already-satisfied", visit_type=VisitType.INITIAL)
        provider = AsyncMock(side_effect=AssertionError("accepted meta reply must not call Cerebras"))

        with patch.object(
            batch_extraction_service,
            "runtime_ai_available",
            return_value=True,
        ), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            await extract_batch_answers(
                case,
                [BatchAnswer(key="preferred_sessions", answer="下午")],
            )
            outcome = await extract_batch_answers(
                case,
                [BatchAnswer(key="preferred_sessions", answer="我剛剛說過了")],
            )

        self.assertEqual(case.availability.preferred_sessions, ["下午"])
        self.assertIn("preferred_sessions", outcome.accepted_fields)
        self.assertNotIn("preferred_sessions", missing_checklist_fields(case))
        provider.assert_not_awaited()

    async def test_meta_reply_preserves_unavailable_availability_without_ai(self):
        for field_name in ("preferred_days", "preferred_sessions"):
            with self.subTest(field=field_name):
                case = TriageCase(
                    case_id=f"strict-meta-unavailable-{field_name}",
                    visit_type=VisitType.INITIAL,
                )
                case.conversation_state.field_statuses[field_name] = "unavailable"
                provider = AsyncMock(
                    side_effect=AssertionError("completed unavailable field must not call Cerebras")
                )

                with patch.object(
                    batch_extraction_service,
                    "runtime_ai_available",
                    return_value=True,
                ), patch.object(
                    batch_extraction_service,
                    "complete_prompt",
                    new=provider,
                ):
                    with self.assertLogs(batch_extraction_service.logger, level="INFO") as logs:
                        outcome = await extract_batch_answers(
                            case,
                            [BatchAnswer(key=field_name, answer="我剛剛說過了")],
                        )

                value = getattr(case.availability, field_name)
                self.assertEqual(value, [])
                self.assertEqual(
                    case.conversation_state.field_statuses[field_name],
                    "unavailable",
                )
                self.assertIn(field_name, outcome.accepted_fields)
                self.assertNotIn(field_name, missing_checklist_fields(case))
                provider.assert_not_awaited()
                log_output = "\n".join(logs.output)
                self.assertIn("already_satisfied=true", log_output)
                self.assertIn("semantic_replay=false", log_output)

    async def test_meta_reply_without_previous_answer_stays_unresolved(self):
        case = TriageCase(case_id="strict-meta-no-previous", visit_type=VisitType.INITIAL)
        case.conversation_state.question_attempts["preferred_sessions"] = 2
        provider = AsyncMock(side_effect=AssertionError("meta text itself must not call Cerebras"))

        with patch.object(
            batch_extraction_service,
            "runtime_ai_available",
            return_value=True,
        ), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            with self.assertLogs(batch_extraction_service.logger, level="INFO") as logs:
                await extract_batch_answers(
                    case,
                    [BatchAnswer(key="preferred_sessions", answer="我剛剛說過了")],
                )

        self.assertEqual(case.availability.preferred_sessions, [])
        self.assertIn("preferred_sessions", missing_checklist_fields(case))
        self.assertNotIn("preferred_sessions", case.conversation_state.consumed_fields)
        self.assertEqual(case.conversation_state.question_attempts["preferred_sessions"], 1)
        provider.assert_not_awaited()
        log_output = "\n".join(logs.output)
        self.assertIn("previous_answer_found=false", log_output)
        self.assertIn("semantic_replay=false", log_output)

    def test_meta_reply_variants_are_recognized(self):
        variants = (
            "我剛剛說過了",
            "我剛才說過了",
            "我已經說過了",
            "我前面說過了",
            "剛剛不是說了嗎",
        )
        for text in variants:
            with self.subTest(text=text):
                self.assertTrue(batch_extraction_service._is_meta_reply(text))

    async def test_valid_symptom_is_accepted(self):
        case, provider = await self._extract("symptom", "手臂一直很癢")

        self.assertEqual(case.patient_input.symptom, "手臂一直很癢")
        self.assertEqual(case.patient_input.body_part, "手臂")
        self.assertNotIn("symptom", missing_checklist_fields(case))
        provider.assert_not_awaited()

    async def test_body_part_with_colloquial_discomfort_is_a_deterministic_symptom(self):
        examples = (
            "我感覺膝蓋怪怪的",
            "膝蓋不舒服",
            "膝蓋卡卡的",
            "膝蓋酸酸的",
            "手麻麻的",
            "胸口悶悶的",
            "肚子怪怪的",
            "耳朵怪怪的",
            "身上不太舒服",
            "關節不太對勁",
        )
        for answer in examples:
            with self.subTest(answer=answer):
                provider = self._semantic_provider()
                case, provider = await self._extract("symptom", answer, provider=provider)

                self.assertEqual(case.patient_input.symptom, answer)
                self.assertNotIn("symptom", missing_checklist_fields(case))
                self.assertLessEqual(provider.await_count, 1)

    async def test_gluteal_synonyms_have_distinct_canonical_body_parts(self):
        examples = (
            ("屁股", "臀部"),
            ("屁股痛", "臀部"),
            ("屁股疼", "臀部"),
            ("臀部", "臀部"),
            ("臀部痛", "臀部"),
            ("臀", "臀部"),
            ("臀側", "臀部"),
            ("左屁股", "左臀部"),
            ("右屁股", "右臀部"),
            ("左邊屁股", "左臀部"),
            ("右邊屁股", "右臀部"),
        )
        for answer, expected in examples:
            with self.subTest(answer=answer):
                case, provider = await self._extract("body_part", answer)

                self.assertEqual(case.patient_input.body_part, expected)
                self.assertNotIn("body_part", missing_checklist_fields(case))
                provider.assert_not_awaited()

    async def test_buttock_symptom_extracts_symptom_and_body_part_together(self):
        examples = (
            ("屁股痛", "臀部"),
            ("右邊屁股很痛", "右臀部"),
            ("臀部痛", "臀部"),
            ("我屁股怪怪的", "臀部"),
        )
        for answer, expected_body_part in examples:
            with self.subTest(answer=answer):
                case, provider = await self._extract("symptom", answer)

                self.assertEqual(case.patient_input.symptom, answer)
                self.assertEqual(case.patient_input.body_part, expected_body_part)
                self.assertNotIn("symptom", missing_checklist_fields(case))
                self.assertNotIn("body_part", missing_checklist_fields(case))
                self.assertIsNone(case.department_result)
                provider.assert_not_awaited()

    async def test_anus_tailbone_and_hip_are_not_normalized_as_buttock(self):
        examples = (
            ("肛門痛", "肛門"),
            ("排便時肛門會痛", "肛門"),
            ("肛門周圍痛", "肛門周圍"),
            ("尾椎痛", "尾椎"),
            ("髖部痛", "髖部"),
        )
        for answer, expected_body_part in examples:
            with self.subTest(answer=answer):
                provider = self._semantic_provider()
                case, provider = await self._extract("symptom", answer, provider=provider)

                self.assertEqual(case.patient_input.body_part, expected_body_part)
                self.assertNotEqual(case.patient_input.body_part, "臀部")
                self.assertLessEqual(provider.await_count, 1)

    def test_chat_skips_body_part_after_buttock_symptom(self):
        case_id = f"strict-buttock-route-{uuid4().hex}"
        provider = AsyncMock(
            side_effect=AssertionError("buttock extraction must not call Cerebras")
        )
        with patch.object(
            chat_route,
            "runtime_ai_available",
            return_value=False,
        ), patch.object(
            batch_extraction_service,
            "runtime_ai_available",
            return_value=True,
        ), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            client = TestClient(app)
            client.post("/chat", json={"case_id": case_id, "visit_type": "initial"})
            symptom_response = client.post(
                "/chat",
                json={
                    "case_id": case_id,
                    "answers": [{"key": "symptom", "answer": "屁股痛"}],
                },
            )
            stored_case = get_case(case_id)
            self.assertIsNotNone(stored_case)
            stored_case.patient_input.red_flags_checked = True
            stored_case.patient_input.red_flags_status = "negative"
            save_case(stored_case)
            red_flag_response = client.post(
                "/chat",
                json={
                    "case_id": case_id,
                    "answers": [{"key": "red_flags", "answer": "都沒有"}],
                },
            )

        self.assertEqual(symptom_response.status_code, 200)
        symptom_data = symptom_response.json()
        self.assertEqual(symptom_data["triage_case"]["patient_input"]["symptom"], "屁股痛")
        self.assertEqual(symptom_data["triage_case"]["patient_input"]["body_part"], "臀部")
        self.assertEqual(symptom_data["question_batch"][0]["key"], "red_flags")

        self.assertEqual(red_flag_response.status_code, 200)
        red_flag_data = red_flag_response.json()
        self.assertEqual(red_flag_data["question_batch"][0]["key"], "duration")
        self.assertIn(
            "body_part",
            red_flag_data["triage_case"]["conversation_state"]["consumed_fields"],
        )
        self.assertEqual(red_flag_data["triage_case"]["patient_input"]["body_part"], "臀部")
        provider.assert_not_awaited()

    async def test_department_or_day_request_remains_invalid_as_symptom(self):
        for answer in ("我要看骨科", "禮拜三上午"):
            with self.subTest(answer=answer):
                provider = self._semantic_provider()
                case, provider = await self._extract(
                    "symptom",
                    answer,
                    provider=provider,
                )

                self.assertEqual(case.patient_input.symptom, "")
                self.assertIn("symptom", missing_checklist_fields(case))
                provider.assert_awaited_once()

    async def test_excluded_weekday_complements_use_semantic_ai(self):
        examples = (
            "除了禮拜三都可以",
            "除了週三之外都行",
            "星期三不行，其他都可以",
        )
        expected = ["週一", "週二", "週四", "週五", "週六", "週日"]
        for answer in examples:
            with self.subTest(answer=answer):
                provider = self._semantic_provider(
                    {
                        "field": "preferred_days",
                        "normalized_value": expected,
                        "semantic_status": "partial",
                        "confidence": 0.9,
                        "source_text": answer,
                        "needs_clarification": False,
                    }
                )
                case, provider = await self._extract(
                    "preferred_days",
                    answer,
                    provider=provider,
                )

                self.assertEqual(case.availability.preferred_days, expected)
                self.assertEqual(case.availability.preferred_sessions, [])
                self.assertIn("preferred_sessions", missing_checklist_fields(case))
                provider.assert_awaited_once()

    async def test_any_session_is_scoped_to_session_question(self):
        case, provider = await self._extract("preferred_sessions", "都可以")

        self.assertEqual(case.availability.preferred_days, [])
        self.assertEqual(case.availability.preferred_sessions, ["上午", "下午", "夜間"])
        provider.assert_not_awaited()

    async def test_explicit_weekdays_do_not_broaden_to_every_day_or_session(self):
        case, provider = await self._extract("preferred_days", "週五週六都可以")

        self.assertEqual(case.availability.preferred_days, ["週五", "週六"])
        self.assertEqual(case.availability.preferred_sessions, [])
        provider.assert_not_awaited()

    async def test_explicit_day_and_session_any_semantics_can_complete_both(self):
        answer = "哪天都可以，上午下午晚上也都行"
        provider = self._semantic_provider(
            {
                "field": "preferred_sessions",
                "normalized_value": ["上午", "下午", "夜間"],
                "semantic_status": "available",
                "confidence": 0.9,
                "source_text": answer,
            }
        )
        case, provider = await self._extract(
            "preferred_days",
            answer,
            provider=provider,
        )

        self.assertEqual(
            case.availability.preferred_days,
            ["週一", "週二", "週三", "週四", "週五", "週六", "週日"],
        )
        self.assertEqual(case.availability.preferred_sessions, ["上午", "下午", "夜間"])
        provider.assert_awaited_once()

    async def test_symptom_time_at_night_does_not_become_appointment_session(self):
        case, provider = await self._extract(
            "severity",
            "有時候晚上會痠痛的睡不著",
        )

        self.assertEqual(case.patient_input.severity, "severe")
        self.assertEqual(case.availability.preferred_sessions, [])
        self.assertIn("preferred_sessions", missing_checklist_fields(case))
        provider.assert_not_awaited()

    def test_chat_waits_for_session_after_excluded_day_answer(self):
        case = TriageCase(
            case_id=f"strict-day-route-{uuid4().hex}",
            visit_type=VisitType.INITIAL,
        )
        case.patient_input.symptom = "膝蓋怪怪的"
        case.patient_input.body_part = "膝"
        case.patient_input.duration = "1個月"
        case.patient_input.severity = "severe"
        case.patient_input.red_flags_checked = True
        case.patient_input.red_flags_status = "negative"
        save_case(case)
        answer = "除了禮拜三都可以"
        expected = ["週一", "週二", "週四", "週五", "週六", "週日"]
        provider = self._semantic_provider(
            {
                "field": "preferred_days",
                "normalized_value": expected,
                "semantic_status": "partial",
                "confidence": 0.9,
                "source_text": answer,
                "needs_clarification": False,
            }
        )
        with patch.object(batch_extraction_service, "runtime_ai_available", return_value=True), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            response = TestClient(app).post(
                "/chat",
                json={
                    "case_id": case.case_id,
                    "answers": [{"key": "preferred_days", "answer": answer}],
                },
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["needMoreInfo"])
        self.assertEqual(data["question_batch"][0]["key"], "preferred_sessions")
        self.assertEqual(data["triage_case"]["availability"]["preferred_days"], expected)
        self.assertEqual(data["triage_case"]["availability"]["preferred_sessions"], [])
        self.assertIsNone(data["department_result"])
        provider.assert_awaited_once()

    async def test_weak_tone_half_year_duration_is_deterministic(self):
        answer = "大概半年了吧"
        provider = self._semantic_provider(
            {
                "field": "duration",
                "normalized_value": "6個月",
                "semantic_status": "available",
                "confidence": 0.95,
                "source_text": answer,
                "needs_clarification": False,
                "follow_up_reason": None,
            }
        )

        case, provider = await self._extract("duration", answer, provider=provider)

        provider.assert_not_awaited()
        self.assertEqual(case.patient_input.duration, "6個月")
        self.assertNotIn("duration", missing_checklist_fields(case))
        self.assertEqual(case.semantic_extractions[-1].source_text, answer)

    def test_chat_route_accepts_deterministic_half_year_and_advances(self):
        answer = "大概半年了吧"
        case = TriageCase(
            case_id=f"strict-half-year-route-{uuid4().hex}",
            visit_type=VisitType.INITIAL,
        )
        case.patient_input.symptom = "頭痛"
        case.patient_input.body_part = "頭"
        case.patient_input.red_flags_checked = True
        case.patient_input.red_flags_status = "negative"
        case.conversation_state.last_question_key = "duration"
        save_case(case)
        provider = self._semantic_provider(
            {
                "field": "duration",
                "normalized_value": "6個月",
                "semantic_status": "available",
                "confidence": 0.95,
                "source_text": answer,
                "needs_clarification": False,
                "follow_up_reason": None,
            }
        )

        with patch.object(
            chat_route,
            "runtime_ai_available",
            return_value=False,
        ), patch.object(
            batch_extraction_service,
            "runtime_ai_available",
            return_value=True,
        ), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            response = TestClient(app).post(
                "/chat",
                json={
                    "case_id": case.case_id,
                    "answers": [{"key": "duration", "answer": answer}],
                },
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        provider.assert_not_awaited()
        self.assertEqual(data["triage_case"]["patient_input"]["duration"], "6個月")
        self.assertEqual(data["conversation_state"]["field_statuses"]["duration"], "available")
        self.assertEqual(data["question_batch"][0]["key"], "severity")

    async def test_ambiguous_body_part_is_refined_once(self):
        answer = "好像是肚臍右下附近吧"
        provider = self._semantic_provider(
            {
                "field": "body_part",
                "normalized_value": "右下腹",
                "semantic_status": "available",
                "confidence": 0.93,
                "source_text": answer,
                "needs_clarification": False,
                "follow_up_reason": None,
            }
        )

        case, provider = await self._extract("body_part", answer, provider=provider)

        provider.assert_awaited_once()
        self.assertEqual(case.patient_input.body_part, "右下腹")
        self.assertNotIn("body_part", missing_checklist_fields(case))

    async def test_compound_severity_is_refined_instead_of_accepting_early_mild(self):
        answer = "白天還好，可是晚上常常痛醒"
        provider = self._semantic_provider(
            {
                "field": "severity",
                "normalized_value": "severe",
                "semantic_status": "available",
                "confidence": 0.94,
                "source_text": answer,
                "needs_clarification": False,
                "follow_up_reason": None,
            }
        )

        case, provider = await self._extract("severity", answer, provider=provider)

        provider.assert_awaited_once()
        self.assertEqual(case.patient_input.severity, "severe")
        self.assertNotIn("severity", missing_checklist_fields(case))

    async def test_weak_tone_excluded_day_answer_uses_semantic_ai(self):
        answer = "下禮拜除了禮拜三之外應該都可以"
        expected = ["週一", "週二", "週四", "週五", "週六", "週日"]
        provider = self._semantic_provider(
            {
                "field": "preferred_days",
                "normalized_value": expected,
                "semantic_status": "partial",
                "confidence": 0.91,
                "source_text": answer,
                "needs_clarification": False,
                "follow_up_reason": None,
            }
        )

        case, provider = await self._extract("preferred_days", answer, provider=provider)

        provider.assert_awaited_once()
        self.assertEqual(case.availability.preferred_days, expected)
        self.assertNotIn("preferred_days", missing_checklist_fields(case))

    async def test_natural_after_work_session_is_refined_to_night(self):
        answer = "白天要上班，六點下班以後才比較方便"
        provider = self._semantic_provider(
            {
                "field": "preferred_sessions",
                "normalized_value": ["夜間"],
                "semantic_status": "available",
                "confidence": 0.92,
                "source_text": answer,
                "needs_clarification": False,
                "follow_up_reason": None,
            }
        )

        case, provider = await self._extract(
            "preferred_sessions",
            answer,
            provider=provider,
        )

        provider.assert_awaited_once()
        self.assertEqual(case.availability.preferred_sessions, ["夜間"])
        self.assertNotIn("preferred_sessions", missing_checklist_fields(case))

    async def test_clear_keyed_answers_do_not_call_cerebras(self):
        examples = (
            ("duration", "3天", lambda case: case.patient_input.duration == "3天"),
            ("preferred_days", "週三", lambda case: case.availability.preferred_days == ["週三"]),
            ("preferred_sessions", "下午", lambda case: case.availability.preferred_sessions == ["下午"]),
            ("severity", "很嚴重", lambda case: case.patient_input.severity == "severe"),
        )
        for key, answer, accepted in examples:
            with self.subTest(key=key, answer=answer):
                case, provider = await self._extract(key, answer)

                self.assertTrue(accepted(case))
                provider.assert_not_awaited()

    async def test_uninterpreted_red_flags_fail_closed(self):
        case, provider = await self._extract("red_flags", "不知道有沒有")

        provider.assert_not_awaited()
        self.assertFalse(case.patient_input.red_flags_checked)
        self.assertEqual(case.patient_input.red_flags_status, "not_checked")
        self.assertIn("red_flags", missing_checklist_fields(case))

    async def test_repeated_uninterpreted_red_flag_answers_stay_open(self):
        examples = (
            ("不知道有沒有", "真的不知道"),
            ("可能吧", "可能吧"),
            ("還好", "還好"),
            ("說不準", "說不準"),
        )
        for first, second in examples:
            with self.subTest(first=first, second=second):
                case = TriageCase(
                    case_id=f"strict-red-flag-{first}",
                    visit_type=VisitType.INITIAL,
                )
                case.conversation_state.last_question_key = "red_flags"
                provider = AsyncMock(side_effect=AssertionError("red flags must never call Cerebras"))
                with patch.object(
                    batch_extraction_service,
                    "runtime_ai_available",
                    return_value=True,
                ), patch.object(
                    batch_extraction_service,
                    "complete_prompt",
                    new=provider,
                ):
                    await extract_batch_answers(
                        case,
                        [BatchAnswer(key="red_flags", answer=first)],
                    )
                    self.assertFalse(case.patient_input.red_flags_checked)
                    self.assertEqual(case.patient_input.red_flags_status, "not_checked")
                    case.conversation_state.last_question_key = "red_flags"
                    await extract_batch_answers(
                        case,
                        [BatchAnswer(key="red_flags", answer=second)],
                    )

                provider.assert_not_awaited()
                self.assertFalse(case.patient_input.red_flags_checked)
                self.assertEqual(case.patient_input.red_flags_status, "not_checked")
                self.assertIn("red_flags", missing_checklist_fields(case))

    async def test_keyed_red_flag_phrases_do_not_create_medical_state(self):
        examples = (
            "有胸痛",
            "可能有胸痛",
            "都沒有",
        )
        for second in examples:
            with self.subTest(second=second):
                case = TriageCase(
                    case_id=f"strict-red-flag-definitive-{second}",
                    visit_type=VisitType.INITIAL,
                )
                case.conversation_state.last_question_key = "red_flags"
                provider = AsyncMock(side_effect=AssertionError("red flags must never call Cerebras"))
                with patch.object(
                    batch_extraction_service,
                    "runtime_ai_available",
                    return_value=True,
                ), patch.object(
                    batch_extraction_service,
                    "complete_prompt",
                    new=provider,
                ):
                    await extract_batch_answers(
                        case,
                        [BatchAnswer(key="red_flags", answer="不知道有沒有")],
                    )
                    self.assertEqual(case.patient_input.red_flags_status, "not_checked")
                    case.conversation_state.last_question_key = "red_flags"
                    await extract_batch_answers(
                        case,
                        [BatchAnswer(key="red_flags", answer=second)],
                    )

                provider.assert_not_awaited()
                self.assertFalse(case.patient_input.red_flags_checked)
                self.assertEqual(case.patient_input.red_flags_status, "not_checked")
                self.assertEqual(case.patient_input.red_flags, [])

    def test_rule_engine_does_not_interpret_repeated_red_flag_text(self):
        case = TriageCase(case_id="strict-rule-red-flag", visit_type=VisitType.INITIAL)
        case.conversation_state.last_question_key = "red_flags"

        apply_user_message(case, "可能吧")
        self.assertFalse(case.patient_input.red_flags_checked)
        self.assertEqual(case.patient_input.red_flags_status, "not_checked")

        apply_user_message(case, "可能吧")
        self.assertFalse(case.patient_input.red_flags_checked)
        self.assertEqual(case.patient_input.red_flags_status, "not_checked")

    async def test_red_flag_phrases_require_ai_ttas_interpretation(self):
        examples = (
            "都沒有",
            "沒有以上症狀",
            "沒有胸痛也沒有呼吸困難",
            "無上述情形",
            "有胸痛",
            "有一點喘",
        )
        for answer in examples:
            with self.subTest(answer=answer):
                case, provider = await self._extract("red_flags", answer)

                provider.assert_not_awaited()
                self.assertFalse(case.patient_input.red_flags_checked)
                self.assertEqual(case.patient_input.red_flags_status, "not_checked")
                self.assertEqual(case.patient_input.red_flags, [])

    async def test_morning_wakeup_symptom_does_not_fill_duration_or_session(self):
        case = TriageCase(case_id="strict-morning-symptom", visit_type=VisitType.INITIAL)
        case.patient_input.body_part = "腹"
        case.patient_input.red_flags_checked = True
        case.patient_input.red_flags_status = "negative"
        case.conversation_state.last_question_key = "symptom"
        provider = AsyncMock(side_effect=AssertionError("clear symptom must not call Cerebras"))
        with patch.object(batch_extraction_service, "runtime_ai_available", return_value=True), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            await extract_batch_answers(
                case,
                [BatchAnswer(key="symptom", answer="我最近早上起床的時候會有點想吐")],
            )

        provider.assert_not_awaited()
        self.assertEqual(case.patient_input.symptom, "我最近早上起床的時候會有點想吐")
        self.assertIsNone(case.patient_input.duration)
        self.assertEqual(case.availability.preferred_sessions, [])
        self.assertEqual(missing_checklist_fields(case)[0], "duration")

    async def test_explicit_yesterday_onset_still_sets_duration(self):
        case, provider = await self._extract("symptom", "我從昨天開始就一直想吐")

        provider.assert_not_awaited()
        self.assertEqual(case.patient_input.duration, "1天")

    async def test_valid_daypart_and_anchored_duration_onsets(self):
        examples = (
            ("從早上開始", "從早上開始"),
            ("早上開始", "從早上開始"),
            ("晚上開始", "從晚上開始"),
            ("從下午開始", "從下午開始"),
            ("從昨天開始", "1天"),
            ("今天早上開始", "1天內"),
        )
        for answer, expected in examples:
            with self.subTest(answer=answer):
                case, provider = await self._extract("duration", answer)

                provider.assert_not_awaited()
                self.assertEqual(case.patient_input.duration, expected)
                self.assertTrue(has_duration_semantics(answer))
                self.assertNotIn("duration", missing_checklist_fields(case))

    async def test_wakeup_and_night_rising_phrases_are_not_duration(self):
        examples = (
            "早上起床",
            "今天早上起床",
            "昨天早上起床後開始頭暈",
            "我每天早上起床都會想吐",
            "早上起來會頭暈",
            "晚上起夜",
            "半夜起床",
        )
        for answer in examples:
            with self.subTest(answer=answer):
                provider = self._semantic_provider()
                case, provider = await self._extract("duration", answer, provider=provider)

                provider.assert_awaited_once()
                self.assertIsNone(case.patient_input.duration)
                self.assertFalse(has_duration_semantics(answer))
                self.assertIn("duration", missing_checklist_fields(case))

    async def test_weak_tone_mild_severity_is_deterministic(self):
        case, provider = await self._extract("severity", "算輕微吧")

        provider.assert_not_awaited()
        self.assertEqual(case.patient_input.severity, "mild")
        self.assertNotIn("severity", missing_checklist_fields(case))

    async def test_stomach_and_directional_abdomen_terms_are_canonicalized(self):
        examples = (
            ("胃痛", "腹"),
            ("胃這邊不舒服", "腹"),
            ("可能是腸胃吧", "腹"),
            ("右下腹痛", "右下腹"),
        )
        for answer, expected in examples:
            with self.subTest(answer=answer):
                provider = self._semantic_provider()
                case, provider = await self._extract("body_part", answer, provider=provider)

                self.assertLessEqual(provider.await_count, 1)
                self.assertEqual(case.patient_input.body_part, expected)
                self.assertEqual(normalize_body_part(answer), expected)

        ai_result = batch_extraction_service._parse_ai_extractions(
            json.dumps(
                {
                    "extractions": [
                        {
                            "field": "body_part",
                            "normalized_value": "腹",
                            "semantic_status": "available",
                            "confidence": 0.91,
                            "source_text": "可能是腸胃吧",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            {"body_part": "可能是腸胃吧"},
            ["body_part"],
        )
        self.assertEqual(ai_result[0].normalized_value, "腹")

    async def test_ai_stomach_value_is_applied_as_abdomen_canonical(self):
        answer = "不知道是不是胃部"
        provider = self._semantic_provider(
            {
                "field": "body_part",
                "normalized_value": "胃部",
                "semantic_status": "available",
                "confidence": 0.91,
                "source_text": answer,
            }
        )

        case, provider = await self._extract("body_part", answer, provider=provider)

        provider.assert_awaited_once()
        self.assertEqual(case.patient_input.body_part, "腹")

    async def test_compound_session_exclusions_and_subsets_use_semantic_ai(self):
        examples = (
            ("除了夜間都可以", ["上午", "下午"]),
            ("除了下午都可以", ["上午", "夜間"]),
            ("除了上午都可以", ["下午", "夜間"]),
            ("夜間不行，其他都可以", ["上午", "下午"]),
            ("下午不方便，其他時段都行", ["上午", "夜間"]),
            ("上午跟下午都可以", ["上午", "下午"]),
            ("我不想要夜間也不想上午", ["下午"]),
            ("不要夜間", ["上午", "下午"]),
            ("不要上午", ["下午", "夜間"]),
            ("上午不方便", ["下午", "夜間"]),
            ("晚上不要", ["上午", "下午"]),
            ("夜間不行", ["上午", "下午"]),
            ("下午沒空", ["上午", "夜間"]),
            ("下午可以，夜間不要", ["下午"]),
            ("上午可以，下午不方便", ["上午"]),
            ("下午跟夜間都可以", ["下午", "夜間"]),
        )
        for answer, expected in examples:
            with self.subTest(answer=answer):
                provider = self._semantic_provider(
                    {
                        "field": "preferred_sessions",
                        "normalized_value": expected,
                        "semantic_status": "partial",
                        "confidence": 0.9,
                        "source_text": answer,
                    }
                )
                case, provider = await self._extract(
                    "preferred_sessions",
                    answer,
                    provider=provider,
                )

                provider.assert_awaited_once()
                self.assertEqual(case.availability.preferred_sessions, expected)

    async def test_second_body_part_answer_is_parsed_before_attempt_fallback(self):
        case = TriageCase(case_id="strict-second-body-answer", visit_type=VisitType.INITIAL)
        case.conversation_state.last_question_key = "body_part"
        case.conversation_state.question_attempts["body_part"] = 2
        case.conversation_state.field_statuses["body_part"] = "unknown"
        provider = AsyncMock(side_effect=AssertionError("clear second answer must not call Cerebras"))
        with patch.object(batch_extraction_service, "runtime_ai_available", return_value=True), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            await extract_batch_answers(
                case,
                [BatchAnswer(key="body_part", answer="右下腹")],
            )

        provider.assert_not_awaited()
        self.assertEqual(case.patient_input.body_part, "右下腹")
        self.assertNotEqual(case.conversation_state.field_statuses["body_part"], "unknown")
        self.assertNotIn("body_part", missing_checklist_fields(case))

    async def test_second_unresolved_answer_falls_back_only_after_empty_ai_result(self):
        case = TriageCase(case_id="strict-second-body-empty", visit_type=VisitType.INITIAL)
        case.conversation_state.last_question_key = "body_part"
        case.conversation_state.question_attempts["body_part"] = 2
        case.conversation_state.field_statuses["body_part"] = "unknown"
        provider = self._semantic_provider()
        with patch.object(batch_extraction_service, "runtime_ai_available", return_value=True), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            outcome = await extract_batch_answers(
                case,
                [BatchAnswer(key="body_part", answer="完全說不上來")],
            )

        provider.assert_awaited_once()
        self.assertIsNone(case.patient_input.body_part)
        self.assertIn("body_part", case.conversation_state.consumed_fields)
        self.assertEqual(case.conversation_state.field_statuses["body_part"], "unknown")
        self.assertNotIn("body_part", missing_checklist_fields(case))
        self.assertNotIn("body_part", outcome.accepted_fields)

    def test_severity_prompt_contract_and_strict_ai_values(self):
        case = TriageCase(case_id="strict-severity-prompt", visit_type=VisitType.INITIAL)
        prompt = batch_extraction_service._build_batch_prompt(
            case,
            {"severity": "算輕微吧"},
            ["severity"],
        )
        legacy_prompt = rag_triage_adapter._build_symptom_collection_prompt(case)
        for value in ("mild", "moderate", "severe"):
            self.assertIn(value, prompt)
            self.assertIn(value, legacy_prompt)

        accepted = batch_extraction_service._parse_ai_extractions(
            json.dumps(
                {
                    "extractions": [
                        {
                            "field": "severity",
                            "normalized_value": "mild",
                            "semantic_status": "available",
                            "confidence": 0.9,
                            "source_text": "算輕微吧",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            {"severity": "算輕微吧"},
            ["severity"],
        )
        rejected = batch_extraction_service._parse_ai_extractions(
            json.dumps(
                {
                    "extractions": [
                        {
                            "field": "severity",
                            "normalized_value": "輕微",
                            "semantic_status": "available",
                            "confidence": 0.9,
                            "source_text": "算輕微吧",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            {"severity": "算輕微吧"},
            ["severity"],
        )

        self.assertEqual(accepted[0].normalized_value, "mild")
        self.assertEqual(rejected, [])

    async def test_ai_mild_severity_lands_and_noncanonical_chinese_is_rejected(self):
        answer = "我說不上來，但如果一定要選就算輕微"
        valid_provider = self._semantic_provider(
            {
                "field": "severity",
                "normalized_value": "mild",
                "semantic_status": "available",
                "confidence": 0.9,
                "source_text": answer,
            }
        )
        valid_case, valid_provider = await self._extract(
            "severity",
            answer,
            provider=valid_provider,
        )

        invalid_provider = self._semantic_provider(
            {
                "field": "severity",
                "normalized_value": "輕微",
                "semantic_status": "available",
                "confidence": 0.9,
                "source_text": answer,
            }
        )
        invalid_case, invalid_provider = await self._extract(
            "severity",
            answer,
            provider=invalid_provider,
        )

        valid_provider.assert_awaited_once()
        invalid_provider.assert_awaited_once()
        self.assertEqual(valid_case.patient_input.severity, "mild")
        self.assertIsNone(invalid_case.patient_input.severity)
        self.assertIn("severity", missing_checklist_fields(invalid_case))

    async def test_weak_body_part_tone_keeps_reliable_deterministic_value(self):
        provider = AsyncMock(
            return_value=json.dumps(
                {
                    "extractions": [
                        {
                            "field": "body_part",
                            "normalized_value": "右肩",
                            "semantic_status": "available",
                            "confidence": 0.91,
                            "source_text": "肩膀附近",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

        case, provider = await self._extract(
            "body_part",
            "好像在肩膀附近吧",
            provider=provider,
        )

        self.assertEqual(case.patient_input.body_part, "肩膀")
        provider.assert_not_awaited()

    async def test_ai_cannot_validate_an_off_topic_raw_value(self):
        provider = AsyncMock(
            return_value=json.dumps(
                {
                    "extractions": [
                        {
                            "field": "symptom",
                            "normalized_value": "我要掛號",
                            "semantic_status": "available",
                            "confidence": 0.99,
                            "source_text": "好像有一種不舒服但說不上來",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        case, provider = await self._extract(
            "symptom",
            "好像有一種不舒服但說不上來",
            provider=provider,
        )

        self.assertEqual(provider.await_count, 1)
        self.assertEqual(case.patient_input.symptom, "")
        self.assertIn("symptom", missing_checklist_fields(case))

    def test_retry_fallback_skips_as_unknown_without_polluting_patient_value(self):
        case = TriageCase(case_id="strict-retry", visit_type=VisitType.INITIAL)
        case.conversation_state.field_statuses["symptom"] = "unknown"
        case.conversation_state.field_confidence["symptom"] = 0.0
        mark_questions_asked(case, ["symptom", "symptom"])

        missing_checklist_fields(case)

        self.assertEqual(case.patient_input.symptom, "")
        self.assertEqual(case.conversation_state.field_statuses["symptom"], "unknown")
        self.assertIn("symptom", case.conversation_state.consumed_fields)

    def test_duplicate_department_name_keeps_preference_unresolved(self):
        case = TriageCase(case_id="strict-department-duplicate", visit_type=VisitType.INITIAL)
        duplicates = [
            {"dept_id": 204, "parent_dept": "其他科", "child_dept": "皮膚科"},
            {"dept_id": 304, "parent_dept": "特別門診", "child_dept": "皮膚科"},
        ]
        with patch.object(
            department_preference_service,
            "fetch_active_departments",
            return_value=duplicates,
        ):
            preference = department_preference_service.capture_department_preference(
                case,
                "我要看皮膚科",
            )

        self.assertIsNotNone(preference)
        self.assertEqual(case.patient_input.requested_department_name, "皮膚科")
        self.assertIsNone(case.patient_input.requested_department_id)
        self.assertIsNone(case.department_result)


if __name__ == "__main__":
    unittest.main()
