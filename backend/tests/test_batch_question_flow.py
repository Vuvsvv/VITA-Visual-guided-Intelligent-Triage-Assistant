from __future__ import annotations

import asyncio
from datetime import date, timedelta
from io import BytesIO
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from app.main import app
from app.routes import chat as chat_route
from app.routes import voice as voice_route
from app.schemas import (
    BatchAnswer,
    ConversationStage,
    DepartmentResult,
    TriageCase,
    TriageResult,
    VisitType,
)
from app.services import (
    ai_service,
    appointment_service,
    batch_extraction_service,
    department_preference_service,
)
from app.services.batch_extraction_service import extract_batch_answers
from app.services.batch_question_service import build_question_batch
from app.services.case_store import save_case
from app.services.rule_engine import missing_checklist_fields
from app.services.semantic_normalizer import (
    ALL_WEEKDAYS,
    _taipei_today,
    normalize_preferred_days,
    normalize_preferred_sessions,
)


def _settings(
    *,
    batch_enabled: bool = True,
    provider: str = "cerebras",
    cerebras_key: str = "test-cerebras-key",
) -> SimpleNamespace:
    return SimpleNamespace(
        batch_triage_enabled=batch_enabled,
        batch_extraction_provider=provider,
        cerebras_api_key=cerebras_key,
        cerebras_model="gpt-oss-120b",
        google_api_key="test-gemini-key",
        ai_provider="gemini",
        ai_timeout_seconds=0.05,
        llm_model="gemini-test",
    )


def _case_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


def _department() -> DepartmentResult:
    return DepartmentResult(dept_id=7, parentDept="外科系", childDept="一般骨科", confidence=1.0)


def _prescreen_new_route_cases(test_case: unittest.TestCase) -> None:
    original = chat_route.create_case

    def create_pre_screened_case(case_id=None):
        case = original(case_id)
        case.patient_input.red_flags_checked = True
        case.patient_input.red_flags_status = "negative"
        return case

    chat_route.create_case = create_pre_screened_case
    test_case.addCleanup(setattr, chat_route, "create_case", original)


class BatchQuestionFlowTest(unittest.TestCase):
    def setUp(self):
        _prescreen_new_route_cases(self)
        self.settings = _settings(cerebras_key="")
        self.settings_patch = patch("app.routes.chat.get_settings", return_value=self.settings)
        self.batch_settings_patch = patch.object(
            batch_extraction_service,
            "get_settings",
            return_value=self.settings,
        )
        self.department_detector = AsyncMock(return_value=_department())
        self.department_patch = patch(
            "app.routes.chat.detect_department_result",
            new=self.department_detector,
        )
        self.complete_patch = patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=AsyncMock(side_effect=AssertionError("clear keyed answers must not call AI")),
        )
        self.settings_patch.start()
        self.batch_settings_patch.start()
        self.department_patch.start()
        self.complete_patch.start()

    def tearDown(self):
        self.complete_patch.stop()
        self.department_patch.stop()
        self.batch_settings_patch.stop()
        self.settings_patch.stop()

    def test_initial_response_contains_only_the_true_next_question(self):
        response = TestClient(app).post(
            "/chat",
            json={"case_id": _case_id("initial"), "visit_type": "initial"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        keys = [item["key"] for item in data["question_batch"]]
        self.assertEqual(
            keys,
            ["symptom"],
        )
        self.assertEqual(len(keys), 1)
        self.assertEqual(data["triage_case"]["visit_type"], "initial")

    def test_followup_first_batch(self):
        response = TestClient(app).post(
            "/chat",
            json={"case_id": _case_id("followup"), "visit_type": "followup"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["triage_case"]["visit_type"], "followup")
        self.assertEqual([item["key"] for item in response.json()["question_batch"]], ["symptom"])

    def test_collected_fields_are_not_asked_and_only_next_gap_is_returned(self):
        case = TriageCase(case_id=_case_id("missing_two"), visit_type=VisitType.INITIAL)
        case.patient_input.symptom = "膝痛"
        case.patient_input.red_flags_checked = True
        case.patient_input.body_part = "膝蓋"
        case.patient_input.duration = "2週"
        case.patient_input.severity = "moderate"

        questions = build_question_batch(case, mark_asked=False)

        self.assertEqual([item.key for item in questions], ["preferred_days"])

    def test_batch_answers_update_multiple_fields_and_next_batch(self):
        case_id = _case_id("multi")
        TestClient(app).post("/chat", json={"case_id": case_id, "visit_type": "initial"})

        response = TestClient(app).post(
            "/chat",
            json={
                "case_id": case_id,
                "visit_type": "initial",
                "answers": [
                    {"key": "symptom", "answer": "膝蓋疼痛"},
                    {"key": "red_flags", "answer": "以上急迫症狀都沒有"},
                    {"key": "body_part", "answer": "右膝蓋"},
                    {"key": "duration", "answer": "持續2週"},
                    {"key": "severity", "answer": "中等，走路會痛"},
                    {"key": "preferred_days", "answer": "週一或週二"},
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        case = response.json()["triage_case"]
        self.assertEqual(case["patient_input"]["duration"], "2週")
        self.assertTrue(case["patient_input"]["red_flags_checked"])
        self.assertIn("週一", case["availability"]["preferred_days"])
        self.assertEqual(
            [item["key"] for item in response.json()["question_batch"]],
            ["preferred_sessions"],
        )

    def test_ai_unavailable_keyed_red_flag_answer_fails_closed(self):
        case = TriageCase(case_id=_case_id("negative_red"), visit_type=VisitType.INITIAL)
        asyncio.run(extract_batch_answers(
            case,
            [BatchAnswer(key="red_flags", answer="都沒有上述症狀")],
            semantic_first=False,
        ))

        self.assertFalse(case.patient_input.red_flags_checked)
        self.assertEqual(case.patient_input.red_flags, [])
        self.assertEqual(case.conversation_state.field_statuses["red_flags"], "unknown")

    def test_ambiguous_red_flag_answers_use_clarification_without_completion(self):
        for answer in ("有", "有一點", "好像有", "應該有", "還好", "不確定", "可能吧", "我不知道"):
            with self.subTest(answer=answer):
                case = TriageCase(case_id=_case_id("ambiguous_red"), visit_type=VisitType.INITIAL)
                asyncio.run(extract_batch_answers(
                    case,
                    [BatchAnswer(key="red_flags", answer=answer)],
                    semantic_first=False,
                ))
                self.assertFalse(case.patient_input.red_flags_checked)
                self.assertEqual(case.patient_input.red_flags, [])
                self.assertEqual(case.conversation_state.field_statuses["red_flags"], "unknown")

    def test_repeated_uninterpreted_red_flag_answers_remain_open(self):
        for first, second in (("有一點", "我不知道"), ("我不知道", "我就不知道啊")):
            with self.subTest(first=first, second=second):
                case = TriageCase(case_id=_case_id("uncertain_red"), visit_type=VisitType.INITIAL)
                for answer in (first, second):
                    asyncio.run(extract_batch_answers(
                        case,
                        [BatchAnswer(key="red_flags", answer=answer)],
                        semantic_first=False,
                    ))
                self.assertFalse(case.patient_input.red_flags_checked)
                self.assertEqual(case.patient_input.red_flags_status, "not_checked")

    def test_repeated_start_never_auto_completes_unasked_red_flag(self):
        case_id = _case_id("repeat_start")
        case = TriageCase(case_id=case_id, visit_type=VisitType.INITIAL)
        save_case(case)
        response = None
        for _ in range(3):
            response = TestClient(app).post(
                "/chat",
                json={"case_id": case_id, "visit_type": "initial"},
            )

        assert response is not None
        self.assertFalse(response.json()["triage_case"]["patient_input"]["red_flags_checked"])
        self.assertEqual([item["key"] for item in response.json()["question_batch"]], ["symptom"])

    def test_positive_keyed_free_text_cannot_bypass_ttas(self):
        case = TriageCase(case_id=_case_id("positive_red"), visit_type=VisitType.INITIAL)
        asyncio.run(extract_batch_answers(
            case,
            [BatchAnswer(key="red_flags", answer="有突發胸痛而且呼吸困難")],
            semantic_first=False,
        ))
        self.assertFalse(case.patient_input.red_flags_checked)
        self.assertEqual(case.patient_input.red_flags, [])

    def test_complete_batch_waits_for_confirmation_then_confirms(self):
        case_id = _case_id("complete")
        TestClient(app).post("/chat", json={"case_id": case_id, "visit_type": "initial"})
        response = TestClient(app).post(
            "/chat",
            json={
                "case_id": case_id,
                "answers": [
                    {"key": "symptom", "answer": "膝蓋疼痛"},
                    {"key": "red_flags", "answer": "以上都沒有"},
                    {"key": "body_part", "answer": "膝蓋"},
                    {"key": "duration", "answer": "2週"},
                    {"key": "severity", "answer": "中等程度"},
                    {"key": "preferred_days", "answer": "週一"},
                    {"key": "preferred_sessions", "answer": "上午"},
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["conversation_state"]["stage"], "waiting_confirmation")
        self.assertTrue(response.json()["conversation_state"]["is_complete"])

        confirmed = TestClient(app).post(
            "/chat",
            json={"case_id": case_id, "visit_type": "initial", "confirmed": True},
        )
        self.assertEqual(confirmed.status_code, 200)
        self.assertEqual(confirmed.json()["conversation_state"]["stage"], "recommending")
        self.assertTrue(confirmed.json()["conversation_state"]["confirmed"])

    def test_confirmation_does_not_rerun_department_detector(self):
        ear = DepartmentResult(dept_id=1333, parentDept="五官科", childDept="耳科", confidence=1.0)
        family = DepartmentResult(dept_id=1233, parentDept="內科", childDept="家庭醫學科", confidence=1.0)
        self.department_detector.side_effect = [ear, family]
        case_id = _case_id("stable_department")
        TestClient(app).post("/chat", json={"case_id": case_id, "visit_type": "initial"})
        completed = TestClient(app).post(
            "/chat",
            json={
                "case_id": case_id,
                "answers": [
                    {"key": "symptom", "answer": "耳朵疼痛"},
                    {"key": "red_flags", "answer": "都沒有"},
                    {"key": "body_part", "answer": "耳朵"},
                    {"key": "duration", "answer": "2天"},
                    {"key": "severity", "answer": "中等"},
                    {"key": "preferred_days", "answer": "週五"},
                    {"key": "preferred_sessions", "answer": "上午"},
                ],
            },
        )
        self.assertEqual(completed.json()["department_result"]["childDept"], "耳科")

        confirmed = TestClient(app).post("/chat", json={"case_id": case_id, "confirmed": True})

        self.assertEqual(self.department_detector.await_count, 1)
        self.assertEqual(confirmed.json()["department_result"]["childDept"], "耳科")
        self.assertEqual(confirmed.json()["department_result"]["dept_id"], 1333)
        self.assertEqual(confirmed.json()["triage_case"]["department_result"]["childDept"], "耳科")
        self.assertEqual(confirmed.json()["triage_case"]["department_result"]["dept_id"], 1333)

    def test_visit_type_conflict_is_rejected(self):
        case_id = _case_id("conflict")
        first = TestClient(app).post("/chat", json={"case_id": case_id, "visit_type": "initial"})
        second = TestClient(app).post("/chat", json={"case_id": case_id, "visit_type": "followup"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)

    def test_return_visit_skips_symptom_batch(self):
        response = TestClient(app).post(
            "/chat",
            json={"case_id": _case_id("return"), "visit_type": "return_visit"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["question_batch"], [])
        self.assertIn("/followup/recommend", response.json()["reply"])

    def test_old_single_message_path_remains_compatible_when_flag_off(self):
        with patch("app.routes.chat.get_settings", return_value=_settings(batch_enabled=False)):
            response = TestClient(app).post(
                "/chat",
                json={"case_id": _case_id("legacy"), "message": "我膝蓋痛兩天"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("triage_case", response.json())
        self.assertEqual(response.json()["question_batch"], [])


class BatchExtractionTest(unittest.IsolatedAsyncioTestCase):
    def test_any_day_phrases_are_narrow_and_explicit_days_win(self):
        self.assertEqual(normalize_preferred_days("哪天都可以").normalized_value, ALL_WEEKDAYS)
        self.assertEqual(normalize_preferred_days("日期沒差").normalized_value, ALL_WEEKDAYS)
        self.assertIsNone(normalize_preferred_days("都可以"))
        self.assertEqual(
            normalize_preferred_days("都可以", "preferred_days").normalized_value,
            ALL_WEEKDAYS,
        )
        self.assertEqual(
            normalize_preferred_days("週五週六都可以").normalized_value,
            ["週五", "週六"],
        )

    async def test_real_flow_keeps_mild_severity_and_only_non_negated_session(self):
        case = TriageCase(case_id=_case_id("severity_session_negation"), visit_type=VisitType.INITIAL)
        case.patient_input.symptom = "頭痛"
        case.patient_input.red_flags_checked = True
        case.patient_input.red_flags_status = "negative"
        case.patient_input.body_part = "頭"
        case.patient_input.duration = "3天"
        provider = AsyncMock(
            return_value=json.dumps(
                {
                    "extractions": [
                        {
                            "field": "preferred_sessions",
                            "normalized_value": ["下午"],
                            "semantic_status": "partial",
                            "confidence": 0.9,
                            "source_text": "我不想要夜間也不想上午",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

        with patch.object(batch_extraction_service, "runtime_ai_available", return_value=True), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            await extract_batch_answers(
                case,
                [BatchAnswer(key="severity", answer="沒有到影響日常活動")],
            )
            self.assertEqual(case.patient_input.severity, "mild")
            self.assertEqual(
                [item.key for item in build_question_batch(case, mark_asked=False)],
                ["preferred_days"],
            )

            await extract_batch_answers(
                case,
                [BatchAnswer(key="preferred_days", answer="週三或週四")],
            )
            self.assertEqual(case.availability.preferred_days, ["週三", "週四"])

            await extract_batch_answers(
                case,
                [BatchAnswer(key="preferred_sessions", answer="我不想要夜間也不想上午")],
            )

        self.assertEqual(case.availability.preferred_sessions, ["下午"])
        self.assertNotIn("上午", case.availability.preferred_sessions)
        self.assertNotIn("夜間", case.availability.preferred_sessions)
        self.assertEqual(missing_checklist_fields(case), [])
        provider.assert_awaited_once()

    async def test_availability_complements_complete_the_real_batch_flow(self):
        case = TriageCase(case_id=_case_id("availability_complements"), visit_type=VisitType.INITIAL)
        case.patient_input.symptom = "頭痛"
        case.patient_input.body_part = "頭"
        case.patient_input.duration = "3天"
        case.patient_input.severity = "mild"
        case.patient_input.red_flags_checked = True
        case.patient_input.red_flags_status = "negative"
        day_answer = "我下週三有事其他時間應該都可以"
        session_answer = "我不喜歡早上跟晚上"
        expected_days = ["週一", "週二", "週四", "週五", "週六", "週日"]
        provider = AsyncMock(
            side_effect=(
                json.dumps(
                    {
                        "extractions": [
                            {
                                "field": "preferred_days",
                                "normalized_value": expected_days,
                                "semantic_status": "partial",
                                "confidence": 0.9,
                                "source_text": day_answer,
                                "needs_clarification": False,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "extractions": [
                            {
                                "field": "preferred_sessions",
                                "normalized_value": ["下午"],
                                "semantic_status": "partial",
                                "confidence": 0.9,
                                "source_text": session_answer,
                                "needs_clarification": False,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
            )
        )

        with patch.object(batch_extraction_service, "runtime_ai_available", return_value=True), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            await extract_batch_answers(
                case,
                [BatchAnswer(key="preferred_days", answer=day_answer)],
            )
            self.assertEqual(case.availability.preferred_days, expected_days)
            self.assertEqual(
                [item.key for item in build_question_batch(case, mark_asked=False)],
                ["preferred_sessions"],
            )

            await extract_batch_answers(
                case,
                [BatchAnswer(key="preferred_sessions", answer=session_answer)],
            )

        self.assertEqual(provider.await_count, 2)
        self.assertEqual(case.availability.preferred_days, expected_days)
        self.assertEqual(case.availability.preferred_sessions, ["下午"])
        self.assertNotIn("週三", case.availability.preferred_days)
        self.assertNotIn("上午", case.availability.preferred_sessions)
        self.assertNotIn("夜間", case.availability.preferred_sessions)
        self.assertEqual(missing_checklist_fields(case), [])

    def test_relative_days_preserve_both_later_dates(self):
        weekday_names = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]
        today = _taipei_today()
        expected = [
            weekday_names[(today + timedelta(days=2)).weekday()],
            weekday_names[(today + timedelta(days=3)).weekday()],
        ]

        self.assertEqual(
            normalize_preferred_days("後天跟大後天都可以").normalized_value,
            expected,
        )

    def test_vague_week_tail_is_not_expanded_to_the_whole_week(self):
        result = normalize_preferred_days(
            "這禮拜後半幾天白天應該都有空",
            "preferred_days",
        )

        self.assertTrue(result is None or result.normalized_value != ALL_WEEKDAYS)

    async def test_day_answer_opportunistically_extracts_session_without_ai(self):
        case = TriageCase(case_id=_case_id("day_session"), visit_type=VisitType.INITIAL)
        provider = AsyncMock(side_effect=AssertionError("clear deterministic slots must not call AI"))
        with patch.object(batch_extraction_service, "complete_prompt", new=provider):
            await extract_batch_answers(
                case,
                [BatchAnswer(key="preferred_days", answer="星期五上午")],
            )

        self.assertEqual(case.availability.preferred_days, ["週五"])
        self.assertEqual(case.availability.preferred_sessions, ["上午"])
        self.assertNotIn("preferred_sessions", missing_checklist_fields(case))
        provider.assert_not_awaited()

    async def test_explicit_date_and_session_fill_both_slots_without_ai(self):
        case = TriageCase(case_id=_case_id("explicit_date_session"), visit_type=VisitType.INITIAL)
        provider = AsyncMock(side_effect=AssertionError("clear date/session must not call AI"))
        with patch("app.services.semantic_normalizer._taipei_today", return_value=date(2026, 9, 14)), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            await extract_batch_answers(
                case,
                [BatchAnswer(key="preferred_days", answer="9月20號的下午")],
            )

        self.assertEqual(case.availability.preferred_dates, ["2026-09-20"])
        self.assertEqual(case.availability.preferred_days, ["週日"])
        self.assertEqual(case.availability.preferred_sessions, ["下午"])
        self.assertNotIn("preferred_days", missing_checklist_fields(case))
        self.assertNotIn("preferred_sessions", missing_checklist_fields(case))
        provider.assert_not_awaited()

    async def test_symptom_answer_opportunistically_keeps_explicit_appointment_session(self):
        case = TriageCase(case_id=_case_id("early_session"), visit_type=VisitType.INITIAL)
        answer = "因為我的喉嚨最近感覺怪怪的然後我想要掛下午的診次"

        await extract_batch_answers(case, [BatchAnswer(key="symptom", answer=answer)])

        self.assertEqual(case.patient_input.symptom, answer)
        self.assertEqual(case.patient_input.body_part, "喉嚨")
        self.assertEqual(case.availability.preferred_sessions, ["下午"])
        self.assertNotIn("preferred_sessions", missing_checklist_fields(case))

    async def test_early_session_is_not_reasked_after_other_fields_are_completed(self):
        case = TriageCase(case_id=_case_id("early_session_complete"), visit_type=VisitType.INITIAL)
        case.patient_input.red_flags_checked = True
        case.patient_input.red_flags_status = "negative"
        await extract_batch_answers(
            case,
            [BatchAnswer(key="symptom", answer="喉嚨怪怪的，我想掛下午的診次")],
        )
        with patch("app.services.semantic_normalizer._taipei_today", return_value=date(2026, 9, 14)):
            await extract_batch_answers(
                case,
                [
                    BatchAnswer(key="red_flags", answer="以上急迫症狀都沒有"),
                    BatchAnswer(key="duration", answer="3天"),
                    BatchAnswer(key="severity", answer="輕微"),
                    BatchAnswer(key="preferred_days", answer="9月21日可以看診"),
                ],
            )

        self.assertEqual(missing_checklist_fields(case), [])
        self.assertNotIn(
            "preferred_sessions",
            [item.key for item in build_question_batch(case, mark_asked=False)],
        )

    async def test_symptom_answer_can_fill_body_duration_date_and_session_together(self):
        case = TriageCase(case_id=_case_id("early_multi_slot"), visit_type=VisitType.INITIAL)
        answer = "我腿痛三週，9月21日下午可以看診"
        with patch("app.services.semantic_normalizer._taipei_today", return_value=date(2026, 9, 14)):
            await extract_batch_answers(case, [BatchAnswer(key="symptom", answer=answer)])

        self.assertEqual(case.patient_input.symptom, answer)
        self.assertEqual(case.patient_input.body_part, "腿")
        self.assertEqual(case.patient_input.duration, "3週")
        self.assertEqual(case.availability.preferred_dates, ["2026-09-21"])
        self.assertEqual(case.availability.preferred_days, ["週一"])
        self.assertEqual(case.availability.preferred_sessions, ["下午"])

    async def test_symptom_onset_time_is_not_mistaken_for_appointment_session(self):
        for answer in (
            "昨天下午開始喉嚨痛",
            "下午開始頭痛",
            "下午比較嚴重",
            "我最近早上起床會想吐",
            "晚上會痛醒",
            "下午比較痛",
        ):
            with self.subTest(answer=answer):
                case = TriageCase(case_id=_case_id("onset_not_session"), visit_type=VisitType.INITIAL)
                await extract_batch_answers(case, [BatchAnswer(key="symptom", answer=answer)])
                self.assertEqual(case.availability.preferred_sessions, [])

    def test_explicit_session_preference_context_is_supported(self):
        examples = (
            "下午有空",
            "下午可以",
            "下午方便",
            "想掛下午",
            "想看下午",
            "想掛下午的診",
            "想掛下午的診次",
            "希望下午看",
            "下午門診",
            "下午時段",
            "上午有空",
            "想掛上午的診",
            "晚上看診",
            "夜間門診",
            "想掛夜診",
            "我下午比較方便看診",
            "掛號的話晚上可以",
        )
        for answer in examples:
            with self.subTest(answer=answer):
                result = normalize_preferred_sessions(answer, "symptom")
                self.assertIsNotNone(result)
                self.assertTrue(result.normalized_value)

    async def test_bare_session_is_accepted_when_session_is_current_question(self):
        case = TriageCase(case_id=_case_id("direct_session"), visit_type=VisitType.INITIAL)

        await extract_batch_answers(
            case,
            [BatchAnswer(key="preferred_sessions", answer="下午")],
        )

        self.assertEqual(case.availability.preferred_sessions, ["下午"])
        self.assertNotIn("preferred_sessions", missing_checklist_fields(case))

    async def test_sleep_problem_is_accepted_as_symptom(self):
        case = TriageCase(case_id=_case_id("sleep_symptom"), visit_type=VisitType.INITIAL)

        await extract_batch_answers(
            case,
            [BatchAnswer(key="symptom", answer="半夜睡不著")],
        )

        self.assertEqual(case.patient_input.symptom, "半夜睡不著")
        self.assertNotEqual(missing_checklist_fields(case)[0], "symptom")

    async def test_leg_symptom_opportunistically_fills_body_part(self):
        for answer, expected_body_part in (("腿痛", "腿"), ("大腿痛", "大腿")):
            with self.subTest(answer=answer):
                case = TriageCase(case_id=_case_id("leg_symptom"), visit_type=VisitType.INITIAL)

                await extract_batch_answers(
                    case,
                    [BatchAnswer(key="symptom", answer=answer)],
                )

                self.assertEqual(case.patient_input.symptom, answer)
                self.assertEqual(case.patient_input.body_part, expected_body_part)
                self.assertNotIn("body_part", missing_checklist_fields(case))

    async def test_explicit_thigh_refines_broad_leg_without_degrading(self):
        case = TriageCase(case_id=_case_id("specific_leg"), visit_type=VisitType.INITIAL)
        await extract_batch_answers(case, [BatchAnswer(key="symptom", answer="腿痛")])
        await extract_batch_answers(case, [BatchAnswer(key="body_part", answer="大腿")])

        self.assertEqual(case.patient_input.body_part, "大腿")
        self.assertNotIn("body_part", missing_checklist_fields(case))

        await extract_batch_answers(case, [BatchAnswer(key="body_part", answer="腿")])
        self.assertEqual(case.patient_input.body_part, "大腿")

    async def test_explicit_dates_persist_on_completed_case(self):
        for answer, expected_date, expected_day in (
            ("9月20號的下午", "2026-09-20", "週日"),
            ("9月21號下午", "2026-09-21", "週一"),
        ):
            with self.subTest(answer=answer):
                case = TriageCase(case_id=_case_id("complete_date"), visit_type=VisitType.INITIAL)
                case.patient_input.red_flags_checked = True
                case.patient_input.red_flags_status = "negative"
                with patch("app.services.semantic_normalizer._taipei_today", return_value=date(2026, 9, 14)):
                    await extract_batch_answers(
                        case,
                        [
                            BatchAnswer(key="symptom", answer="腿痛"),
                            BatchAnswer(key="red_flags", answer="以上都沒有"),
                            BatchAnswer(key="body_part", answer="大腿"),
                            BatchAnswer(key="duration", answer="3週"),
                            BatchAnswer(key="severity", answer="中等程度"),
                            BatchAnswer(key="preferred_days", answer=answer),
                        ],
                    )

                self.assertEqual(missing_checklist_fields(case), [])
                self.assertEqual(case.patient_input.body_part, "大腿")
                self.assertEqual(case.availability.preferred_dates, [expected_date])
                self.assertEqual(case.availability.preferred_days, [expected_day])
                self.assertEqual(case.availability.preferred_sessions, ["下午"])

    async def test_explicit_date_session_variants_are_supported(self):
        variants = ("9月20日下午", "9/20下午", "9月20號下午", "9月20日的下午", "20號下午")
        for answer in variants:
            with self.subTest(answer=answer):
                case = TriageCase(case_id=_case_id("date_variant"), visit_type=VisitType.INITIAL)
                with patch("app.services.semantic_normalizer._taipei_today", return_value=date(2026, 9, 14)):
                    await extract_batch_answers(
                        case,
                        [BatchAnswer(key="preferred_days", answer=answer)],
                    )
                self.assertEqual(case.availability.preferred_dates, ["2026-09-20"])
                self.assertEqual(case.availability.preferred_sessions, ["下午"])

    async def test_any_session_phrases_complete_without_ai(self):
        for answer in ("我隨便給你安排", "我都可以", "任何時段都行"):
            with self.subTest(answer=answer):
                case = TriageCase(case_id=_case_id("any_session"), visit_type=VisitType.INITIAL)
                provider = AsyncMock(side_effect=AssertionError("any-session must not call AI"))
                with patch.object(batch_extraction_service, "get_settings", return_value=_settings()), patch.object(
                    batch_extraction_service,
                    "complete_prompt",
                    new=provider,
                ):
                    outcome = await extract_batch_answers(
                        case,
                        [BatchAnswer(key="preferred_sessions", answer=answer)],
                    )

                self.assertEqual(case.availability.preferred_sessions, ["上午", "下午", "夜間"])
                self.assertNotIn("preferred_sessions", missing_checklist_fields(case))
                self.assertFalse(outcome.ai_attempted)
                provider.assert_not_awaited()

    async def test_approximate_duration_is_deterministic_and_keeps_relative_day(self):
        case = TriageCase(case_id=_case_id("relative_day"), visit_type=VisitType.INITIAL)
        provider = AsyncMock(
            return_value=json.dumps(
                {
                    "extractions": [
                        {
                            "field": "duration",
                            "normalized_value": "3天",
                            "semantic_status": "available",
                            "confidence": 0.95,
                            "source_text": "大概三天了我想要在後天看診",
                            "needs_clarification": False,
                            "follow_up_reason": None,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        with patch.object(batch_extraction_service, "get_settings", return_value=_settings()), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            outcome = await extract_batch_answers(
                case,
                [BatchAnswer(key="duration", answer="大概三天了我想要在後天看診")],
            )

        self.assertEqual(case.patient_input.duration, "3天")
        self.assertEqual(len(case.availability.preferred_days), 1)
        self.assertEqual(case.availability.preferred_sessions, [])
        self.assertFalse(outcome.ai_attempted)
        provider.assert_not_awaited()


class DepartmentPreferenceRevisionTest(unittest.TestCase):
    DEPARTMENTS = [
        {"dept_id": 1232, "parent_dept": "內科系", "child_dept": "一般內科"},
        {"dept_id": 1235, "parent_dept": "內科系", "child_dept": "神經內科"},
        {"dept_id": 1238, "parent_dept": "內科系", "child_dept": "胃腸肝膽科"},
        {"dept_id": 1298, "parent_dept": "外科系", "child_dept": "一般骨科"},
        {"dept_id": 1402, "parent_dept": "其他科", "child_dept": "皮膚科"},
        {"dept_id": 1501, "parent_dept": "其他科", "child_dept": "睡眠醫學中心"},
    ]

    def setUp(self):
        _prescreen_new_route_cases(self)
        self.settings = _settings(cerebras_key="")
        self.patches = [
            patch.object(chat_route, "get_settings", return_value=self.settings),
            patch.object(batch_extraction_service, "get_settings", return_value=self.settings),
            patch.object(
                batch_extraction_service,
                "complete_prompt",
                new=AsyncMock(side_effect=AssertionError("explicit revision must remain deterministic")),
            ),
            patch.object(
                chat_route,
                "generate_triage_reply",
                new=AsyncMock(side_effect=lambda **kwargs: kwargs["fallback_reply"]),
            ),
            patch.object(chat_route, "refine_case_with_ai", new=AsyncMock(return_value=None)),
            patch.object(
                department_preference_service,
                "fetch_active_departments",
                return_value=self.DEPARTMENTS,
            ),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    def _completed_sleep_case(self, prefix: str) -> TriageCase:
        case = TriageCase(case_id=_case_id(prefix), visit_type=VisitType.INITIAL)
        case.patient_input.symptom = "半夜睡不著"
        case.patient_input.body_part = "頭"
        case.patient_input.duration = "3週"
        case.patient_input.severity = "moderate"
        case.patient_input.red_flags_checked = True
        case.patient_input.red_flags_status = "negative"
        case.availability.preferred_dates = ["2026-09-20"]
        case.availability.preferred_days = ["週日"]
        case.availability.preferred_sessions = ["下午"]
        case.department_result = DepartmentResult(
            dept_id=1501,
            parentDept="其他科",
            childDept="睡眠醫學中心",
            confidence=0.9,
            reason=["原問診結果"],
        )
        case.triage.need_more_info = False
        case.triage.next_question = None
        case.conversation_state.is_complete = True
        case.conversation_state.stage = ConversationStage.WAITING_CONFIRMATION
        case.conversation_state.awaiting_confirmation = True
        save_case(case)
        return case

    def _revise(self, case: TriageCase, message: str):
        client = TestClient(app)
        begin = client.post("/chat", json={"case_id": case.case_id, "revision_requested": True})
        self.assertEqual(begin.status_code, 200)
        self.assertIn("科別", begin.json()["reply"])
        with patch("app.services.semantic_normalizer._taipei_today", return_value=date(2026, 9, 14)):
            response = client.post("/chat", json={"case_id": case.case_id, "message": message})
        self.assertEqual(response.status_code, 200)
        return response

    def test_direct_department_intent_is_acknowledged_without_becoming_symptom(self):
        examples = (
            ("我想要看一般骨科", "一般骨科"),
            ("我要掛一般骨科", "一般骨科"),
            ("我想掛骨科", "一般骨科"),
            ("我想看神經內科", "神經內科"),
            ("我要看一般內科", "一般內科"),
            ("我要掛腸胃科", "胃腸肝膽科"),
            ("想掛皮膚科", "皮膚科"),
        )
        for answer, expected in examples:
            with self.subTest(answer=answer):
                case_id = _case_id("direct_department")
                client = TestClient(app)
                client.post("/chat", json={"case_id": case_id, "visit_type": "initial"})
                response = client.post(
                    "/chat",
                    json={
                        "case_id": case_id,
                        "visit_type": "initial",
                        "answers": [{"key": "symptom", "answer": answer}],
                    },
                )

                data = response.json()
                patient = data["triage_case"]["patient_input"]
                self.assertEqual(response.status_code, 200)
                self.assertEqual(patient["symptom"], "")
                self.assertEqual(patient["requested_department_name"], expected)
                self.assertIsNotNone(patient["requested_department_id"])
                self.assertEqual(data["question_batch"][0]["key"], "symptom")
                self.assertIn(expected, data["question_batch"][0]["question"])
                self.assertIn(expected, data["reply"])

    def test_direct_department_intent_keeps_date_day_and_session_from_same_answer(self):
        examples = (
            (
                "我想要掛一般骨科9月22號的下午",
                "一般骨科",
                ["2026-09-22"],
                ["週二"],
                ["下午"],
            ),
            (
                "我想看神經內科9/23上午",
                "神經內科",
                ["2026-09-23"],
                ["週三"],
                ["上午"],
            ),
            (
                "我要掛一般內科，星期五下午方便",
                "一般內科",
                [],
                ["週五"],
                ["下午"],
            ),
        )
        for answer, department, dates, days, sessions in examples:
            with self.subTest(answer=answer):
                case_id = _case_id("direct_department_availability")
                client = TestClient(app)
                client.post("/chat", json={"case_id": case_id, "visit_type": "initial"})
                with patch("app.services.semantic_normalizer._taipei_today", return_value=date(2026, 9, 14)):
                    response = client.post(
                        "/chat",
                        json={
                            "case_id": case_id,
                            "visit_type": "initial",
                            "answers": [{"key": "symptom", "answer": answer}],
                        },
                    )

                data = response.json()
                patient = data["triage_case"]["patient_input"]
                availability = data["triage_case"]["availability"]
                self.assertEqual(response.status_code, 200)
                self.assertEqual(patient["requested_department_name"], department)
                self.assertIsNotNone(patient["requested_department_id"])
                self.assertEqual(patient["symptom"], "")
                self.assertEqual(availability["preferred_dates"], dates)
                self.assertEqual(availability["preferred_days"], days)
                self.assertEqual(availability["preferred_sessions"], sessions)
                self.assertEqual(data["question_batch"][0]["key"], "symptom")
                self.assertIn(department, data["reply"])

    def test_direct_department_availability_survives_to_final_confirmation(self):
        case_id = _case_id("direct_department_complete")
        client = TestClient(app)
        client.post("/chat", json={"case_id": case_id, "visit_type": "initial"})
        with patch("app.services.semantic_normalizer._taipei_today", return_value=date(2026, 9, 14)):
            first = client.post(
                "/chat",
                json={
                    "case_id": case_id,
                    "visit_type": "initial",
                    "answers": [
                        {"key": "symptom", "answer": "我想要掛一般骨科9月22號的下午"}
                    ],
                },
            )
        first_case = first.json()["triage_case"]
        self.assertEqual(first_case["availability"]["preferred_dates"], ["2026-09-22"])
        self.assertEqual(first_case["availability"]["preferred_sessions"], ["下午"])

        selected = DepartmentResult(
            dept_id=1298,
            parentDept="外科系",
            childDept="一般骨科",
            confidence=0.9,
            reason=["使用者指定科別"],
        )
        with patch.object(chat_route, "detect_department_result", new=AsyncMock(return_value=selected)):
            completed = client.post(
                "/chat",
                json={
                    "case_id": case_id,
                    "visit_type": "initial",
                    "answers": [
                        {"key": "symptom", "answer": "腳背腫脹沒有急迫症狀"},
                        {"key": "red_flags", "answer": "以上急迫症狀都沒有"},
                        {"key": "duration", "answer": "三週前"},
                        {"key": "severity", "answer": "普通"},
                    ],
                },
            )

        data = completed.json()
        completed_case = TriageCase.model_validate(data["triage_case"])
        self.assertEqual(completed.status_code, 200)
        self.assertEqual(completed_case.patient_input.requested_department_name, "一般骨科")
        self.assertEqual(completed_case.availability.preferred_dates, ["2026-09-22"])
        self.assertEqual(completed_case.availability.preferred_sessions, ["下午"])
        self.assertEqual(missing_checklist_fields(completed_case), [])
        self.assertEqual(data["question_batch"], [])
        self.assertIn("已記錄您希望看一般骨科", data["reply"])
        self.assertIn("就醫時間偏好：9 月 22 日下午。", data["reply"])

    def test_revision_overwrites_department_date_and_session_without_duration_contamination(self):
        response = self._revise(
            self._completed_sleep_case("revision_multi"),
            "我想要看一般內科9月21號的上午",
        )

        data = response.json()
        case = data["triage_case"]
        patient = case["patient_input"]
        self.assertEqual(patient["requested_department_name"], "一般內科")
        self.assertEqual(patient["requested_department_id"], 1232)
        self.assertEqual(patient["duration"], "3週")
        self.assertEqual(case["availability"]["preferred_dates"], ["2026-09-21"])
        self.assertEqual(case["availability"]["preferred_days"], ["週一"])
        self.assertEqual(case["availability"]["preferred_sessions"], ["上午"])
        self.assertEqual(data["department_result"]["dept_id"], 1232)
        self.assertIn("使用者明確指定", data["department_result"]["reason"][0])
        self.assertEqual(data["conversation_state"]["department_status"], "resolved")
        self.assertFalse(data["needMoreInfo"])
        self.assertNotIn("持續時間「9月」", data["reply"])
        self.assertNotIn("較符合一般內科", data["reply"])

    def test_revision_only_replaces_explicitly_provided_preferences(self):
        time_response = self._revise(
            self._completed_sleep_case("revision_time"),
            "改成9月21日上午",
        )
        time_case = time_response.json()["triage_case"]
        self.assertEqual(time_case["availability"]["preferred_dates"], ["2026-09-21"])
        self.assertEqual(time_case["availability"]["preferred_sessions"], ["上午"])
        self.assertEqual(time_case["patient_input"]["duration"], "3週")

        department_response = self._revise(
            self._completed_sleep_case("revision_department"),
            "改看一般骨科",
        )
        department_case = department_response.json()["triage_case"]
        self.assertEqual(department_case["patient_input"]["requested_department_name"], "一般骨科")
        self.assertEqual(department_case["availability"]["preferred_dates"], ["2026-09-20"])
        self.assertEqual(department_case["availability"]["preferred_sessions"], ["下午"])

        combined_response = self._revise(
            self._completed_sleep_case("revision_combined"),
            "我想改成神經內科9/23上午",
        )
        combined_case = combined_response.json()["triage_case"]
        self.assertEqual(combined_case["patient_input"]["requested_department_name"], "神經內科")
        self.assertEqual(combined_case["availability"]["preferred_dates"], ["2026-09-23"])
        self.assertEqual(combined_case["availability"]["preferred_sessions"], ["上午"])

    def test_calendar_month_does_not_replace_real_duration(self):
        for message in ("9月21號上午", "9/21上午"):
            with self.subTest(message=message):
                response = self._revise(self._completed_sleep_case("duration_guard"), message)
                patient = response.json()["triage_case"]["patient_input"]
                self.assertEqual(patient["duration"], "3週")

    def test_symptom_revision_invalidates_the_old_system_department(self):
        response = self._revise(
            self._completed_sleep_case("revision_symptom"),
            "我想把症狀改成頭暈",
        )

        data = response.json()
        self.assertIn("頭暈", data["triage_case"]["patient_input"]["symptom"])
        self.assertIsNone(data["department_result"])
        self.assertTrue(data["needMoreInfo"])
        self.assertNotEqual(data["conversation_state"]["stage"], ConversationStage.WAITING_CONFIRMATION.value)


class DepartmentRecoveryAndExtractionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _prescreen_new_route_cases(self)
        self.settings = _settings(cerebras_key="")
        self.settings_patches = [
            patch.object(chat_route, "get_settings", return_value=self.settings),
            patch.object(batch_extraction_service, "get_settings", return_value=self.settings),
            patch.object(
                batch_extraction_service,
                "complete_prompt",
                new=AsyncMock(side_effect=AssertionError("exact case must remain deterministic")),
            ),
            patch.object(
                chat_route,
                "generate_triage_reply",
                new=AsyncMock(side_effect=lambda **kwargs: kwargs["fallback_reply"]),
            ),
        ]
        for item in self.settings_patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.settings_patches):
            item.stop()

    def test_exact_headache_case_resolves_general_internal_medicine_after_ai_failure(self):
        departments = [
            {"dept_id": 1232, "parent_dept": "內科", "child_dept": "一般內科"},
            {"dept_id": 1233, "parent_dept": "家庭醫學部", "child_dept": "家庭醫學科"},
        ]
        ai_detector = AsyncMock(return_value=None)
        with patch.object(appointment_service, "fetch_active_departments", return_value=departments), patch.object(
            appointment_service,
            "detect_department_with_project_smart_adapter",
            new=ai_detector,
        ):
            case_id = _case_id("headache")
            response = TestClient(app).post(
                "/chat",
                json={
                    "case_id": case_id,
                    "visit_type": "initial",
                    "answers": [
                        {"key": "symptom", "answer": "我的頭有點痛"},
                        {"key": "red_flags", "answer": "都沒有"},
                        {"key": "duration", "answer": "大概三天了我想要在後天看診"},
                        {"key": "preferred_sessions", "answer": "我都可以"},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        patient = response.json()["triage_case"]["patient_input"]
        self.assertEqual(patient["body_part"], "頭")
        self.assertEqual(patient["duration"], "3天")
        self.assertEqual(patient["severity"], "mild")
        self.assertEqual(patient["red_flags_status"], "negative")
        self.assertEqual(response.json()["department_result"]["dept_id"], 1232)
        self.assertEqual(response.json()["department_result"]["childDept"], "一般內科")
        self.assertEqual(ai_detector.await_count, 1)

    def test_unresolved_department_clarifies_once_then_stops_retrying_without_422(self):
        departments = [{"dept_id": 1400, "parent_dept": "其他科", "child_dept": "特殊門診"}]
        ai_detector = AsyncMock(return_value=None)
        with patch.object(appointment_service, "fetch_active_departments", return_value=departments), patch.object(
            appointment_service,
            "detect_department_with_project_smart_adapter",
            new=ai_detector,
        ):
            case_id = _case_id("dept_unresolved")
            first = TestClient(app).post(
                "/chat",
                json={
                    "case_id": case_id,
                    "visit_type": "initial",
                    "answers": [
                        {"key": "symptom", "answer": "不明原因不舒服"},
                        {"key": "red_flags", "answer": "都沒有"},
                        {"key": "body_part", "answer": "全身"},
                        {"key": "duration", "answer": "3天"},
                        {"key": "severity", "answer": "中等"},
                        {"key": "preferred_days", "answer": "週五"},
                        {"key": "preferred_sessions", "answer": "上午"},
                    ],
                },
            )
            second = TestClient(app).post(
                "/chat",
                json={
                    "case_id": case_id,
                    "answers": [{"key": "department_clarification", "answer": "沒有其他明顯症狀"}],
                },
            )
            third = TestClient(app).post(
                "/chat",
                json={"case_id": case_id, "message": "現在怎麼回事"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["question_batch"][0]["key"], "department_clarification")
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["conversation_state"]["field_statuses"]["department"], "unresolved_final")
        self.assertEqual(third.status_code, 200)
        self.assertIn("沒有替你套用預設科別", third.json()["reply"])
        self.assertEqual(ai_detector.await_count, 2)

    async def test_symptom_answer_opportunistically_extracts_duration_and_body_part(self):
        case = TriageCase(case_id=_case_id("symptom_multi"), visit_type=VisitType.INITIAL)
        provider = AsyncMock(side_effect=AssertionError("clear deterministic slots must not call AI"))
        with patch.object(batch_extraction_service, "complete_prompt", new=provider):
            await extract_batch_answers(
                case,
                [BatchAnswer(key="symptom", answer="我頭暈兩天，主要是頭部")],
            )

        self.assertIn("頭暈", case.patient_input.symptom)
        self.assertEqual(case.patient_input.duration, "2天")
        self.assertIn("頭", case.patient_input.body_part or "")
        provider.assert_not_awaited()

    async def test_opportunistic_extraction_does_not_replace_existing_explicit_value(self):
        case = TriageCase(case_id=_case_id("no_replace"), visit_type=VisitType.INITIAL)
        case.availability.preferred_sessions = ["上午"]

        await extract_batch_answers(
            case,
            [BatchAnswer(key="preferred_days", answer="星期五下午")],
        )

        self.assertEqual(case.availability.preferred_days, ["週五"])
        self.assertEqual(case.availability.preferred_sessions, ["上午"])

    async def test_cerebras_extraction_success_and_single_call(self):
        case = TriageCase(case_id=_case_id("ai_success"), visit_type=VisitType.INITIAL)
        provider = AsyncMock(
            return_value=json.dumps(
                {
                    "extractions": [
                        {
                            "field": "body_part",
                            "normalized_value": "肩膀",
                            "semantic_status": "available",
                            "confidence": 0.91,
                            "source_text": "不知道是不是肩膀附近",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        with patch.object(batch_extraction_service, "get_settings", return_value=_settings()), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            outcome = await extract_batch_answers(
                case,
                [BatchAnswer(key="body_part", answer="不知道是不是肩膀附近")],
            )

        self.assertEqual(provider.await_count, 1)
        self.assertIsInstance(provider.await_args.args[0], str)
        self.assertEqual(case.patient_input.body_part, "肩膀")
        self.assertEqual(outcome.ai_fields, ["body_part"])

    async def test_timeout_402_429_and_malformed_json_fall_back(self):
        failures = [
            asyncio.TimeoutError(),
            _ProviderStatusError(402),
            _ProviderStatusError(429),
            None,
        ]
        for failure in failures:
            with self.subTest(failure=failure):
                case = TriageCase(case_id=_case_id("fallback"), visit_type=VisitType.INITIAL)
                provider = AsyncMock(
                    side_effect=failure if failure is not None else None,
                    return_value="not-json" if failure is None else None,
                )
                with patch.object(batch_extraction_service, "get_settings", return_value=_settings()), patch.object(
                    batch_extraction_service,
                    "complete_prompt",
                    new=provider,
                ):
                    outcome = await extract_batch_answers(
                        case,
                        [BatchAnswer(key="body_part", answer="不知道是不是肩膀附近")],
                    )

                self.assertEqual(provider.await_count, 1)
                self.assertIsNotNone(outcome.fallback_reason)
                self.assertIn("body_part", outcome.unresolved_fields)

    async def test_ai_flow_state_keys_reject_the_entire_semantic_envelope(self):
        case = TriageCase(case_id=_case_id("ai_gate"), visit_type=VisitType.INITIAL)
        provider = AsyncMock(
            return_value=json.dumps(
                {
                    "is_complete": True,
                    "stage": "recommending",
                    "red_flags_checked": True,
                    "extractions": [
                        {
                            "field": "body_part",
                            "normalized_value": "右肩",
                            "semantic_status": "available",
                            "confidence": 0.9,
                        },
                        {
                            "field": "red_flags",
                            "normalized_value": [],
                            "semantic_status": "unavailable",
                            "confidence": 1.0,
                        }
                    ],
                }
            )
        )
        with patch.object(batch_extraction_service, "get_settings", return_value=_settings()), patch.object(
            batch_extraction_service,
            "complete_prompt",
            new=provider,
        ):
            outcome = await extract_batch_answers(
                case,
                [
                    BatchAnswer(key="red_flags", answer="不確定"),
                    BatchAnswer(key="body_part", answer="不知道是不是肩膀附近"),
                ],
            )

        self.assertFalse(case.patient_input.red_flags_checked)
        self.assertFalse(case.conversation_state.is_complete)
        self.assertEqual(case.conversation_state.stage, ConversationStage.COLLECTING)
        self.assertTrue(outcome.ai_attempted)
        self.assertEqual(outcome.fallback_reason, "ValueError")
        self.assertIsNone(case.patient_input.body_part)

    async def test_cerebras_adapter_uses_async_client(self):
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))]
        )
        create = AsyncMock(return_value=completion)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        settings = _settings()
        with patch.object(ai_service, "get_settings", return_value=settings), patch.object(
            ai_service,
            "initialize_ai",
            return_value=None,
        ), patch.object(ai_service, "_cerebras_client", client):
            result = await ai_service.complete_prompt("extract", provider="cerebras")

        self.assertEqual(result, '{"ok":true}')
        create.assert_awaited_once()
        self.assertEqual(create.await_args.kwargs["model"], "gpt-oss-120b")


class VisitTypeRecommendationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_fetch = appointment_service.fetch_available_slots
        self.original_fetch_departments = appointment_service.fetch_active_departments
        self.original_score = appointment_service.score_doctor_specialties

        async def neutral_score(*_args, **_kwargs):
            return {}

        appointment_service.score_doctor_specialties = neutral_score
        appointment_service.fetch_active_departments = lambda: [
            {"dept_id": 7, "parent_dept": "外科系", "child_dept": "一般骨科"}
        ]

    def tearDown(self):
        appointment_service.fetch_available_slots = self.original_fetch
        appointment_service.fetch_active_departments = self.original_fetch_departments
        appointment_service.score_doctor_specialties = self.original_score

    async def test_initial_and_followup_are_exact_and_case_is_source_of_truth(self):
        rows = [_slot("初診醫師", "初診"), _slot("複診醫師", "複診")]
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: rows

        for visit_type, expected in [(VisitType.INITIAL, "初診醫師"), (VisitType.FOLLOWUP, "複診醫師")]:
            with self.subTest(visit_type=visit_type):
                case = _confirmed_case(visit_type)
                save_case(case)
                response = TestClient(app).post("/recommend", json={"case_id": case.case_id})
                self.assertEqual(response.status_code, 200)
                doctors = [item["doctor"] for item in response.json()["recommendations"]["specialty_first"]]
                self.assertEqual(doctors, [expected])

    async def test_return_visit_maps_to_followup_schedule(self):
        self.assertEqual(appointment_service.schedule_visit_type_for(VisitType.RETURN_VISIT), "複診")
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: [
            _slot("初診醫師", "初診"),
            _slot("回診可掛醫師", "複診"),
        ]
        case = _confirmed_case(VisitType.RETURN_VISIT)
        save_case(case)

        response = TestClient(app).post("/recommend", json={"case_id": case.case_id})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["doctor"] for item in response.json()["recommendations"]["specialty_first"]],
            ["回診可掛醫師"],
        )

    async def test_no_compatible_or_blank_visit_type_never_falls_back(self):
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: [
            _slot("只有複診", "複診"),
            _slot("空白類型", ""),
        ]
        case = _confirmed_case(VisitType.INITIAL)
        save_case(case)

        response = TestClient(app).post("/recommend", json={"case_id": case.case_id})

        self.assertEqual(response.status_code, 503)

    async def test_no_matching_availability_returns_503_instead_of_open_rows(self):
        row = _slot("初診醫師", "初診")
        row.update({"date": "2026-07-13", "session": "上午"})
        appointment_service.fetch_available_slots = lambda *_args, **_kwargs: [row]
        case = _confirmed_case(VisitType.INITIAL)
        case.availability.preferred_days = ["週二"]
        case.availability.preferred_sessions = ["晚上"]
        case.availability.can_take_leave = False
        save_case(case)

        response = TestClient(app).post("/recommend", json={"case_id": case.case_id})

        self.assertEqual(response.status_code, 503)

    async def test_recommend_request_conflict_is_rejected(self):
        case = _confirmed_case(VisitType.INITIAL)
        save_case(case)
        response = TestClient(app).post(
            "/recommend",
            json={"case_id": case.case_id, "visit_type": "followup"},
        )
        self.assertEqual(response.status_code, 409)


class VoiceCompatibilityTest(unittest.IsolatedAsyncioTestCase):
    async def test_voice_gateway_connection_failure_is_observable_and_user_readable(self):
        settings = SimpleNamespace(
            voice_enabled=True,
            voice_gateway_url="http://localhost:8000",
            voice_gateway_key="test-key",
            voice_default_lang="taiwanese",
            voice_use_fixed_cache=False,
        )
        gateway_error = voice_route.VoiceGatewayError(
            "語音服務目前無法連線，請確認 Voice Gateway 已啟動。",
            upstream_status="connection_error",
            exception_type="ConnectionError",
            exception_message="connection refused",
        )

        def fail(*_args, **_kwargs):
            raise gateway_error

        client = SimpleNamespace(transcribe=fail, synthesize_to_base64=fail)
        with patch.object(voice_route, "get_settings", return_value=settings), patch.object(
            voice_route,
            "_get_client",
            return_value=client,
        ), self.assertLogs("app.routes.voice", level="INFO") as logs:
            asr = await voice_route.voice_asr(
                UploadFile(filename="voice.wav", file=BytesIO(b"RIFF")),
                "taiwanese",
            )
            tts = await voice_route.voice_tts({"text": "測試語音", "lang": "taiwanese"})

        self.assertTrue(asr["asr_failed"])
        self.assertIn("Voice Gateway", asr["error"])
        self.assertTrue(tts["tts_failed"])
        self.assertIn("Voice Gateway", tts["error"])
        combined = "\n".join(logs.output)
        self.assertIn("[VOICE] operation=asr backend_route=/voice/asr", combined)
        self.assertIn("[VOICE] operation=tts backend_route=/voice/tts", combined)
        self.assertIn("upstream_provider=taiwanese_asr", combined)
        self.assertIn("upstream_provider=taiwanese_tts", combined)
        self.assertIn("upstream_status=connection_error", combined)
        self.assertIn("exception_type=ConnectionError", combined)

    async def test_voice_chat_passes_optional_visit_type_to_shared_chat(self):
        case = TriageCase(case_id=_case_id("voice"), visit_type=VisitType.INITIAL)
        triage_result = TriageResult(
            case_id=case.case_id,
            triage_case=case,
            conversation_state=case.conversation_state,
            triage=case.triage,
            reply="請繼續回答",
        )
        chat_handler = AsyncMock(return_value=triage_result)
        client = SimpleNamespace(
            transcribe=lambda *_args, **_kwargs: "我膝蓋痛",
            synthesize_to_base64=lambda *_args, **_kwargs: SimpleNamespace(
                audio_base64="dGVzdA==",
                audio_format="wav",
            ),
        )
        settings = SimpleNamespace(
            voice_enabled=True,
            voice_gateway_url="https://voice.invalid",
            voice_gateway_key="test-key",
            voice_default_lang="chinese",
            voice_use_fixed_cache=False,
        )
        upload = UploadFile(filename="voice.wav", file=BytesIO(b"RIFF"))
        with patch.object(voice_route, "get_settings", return_value=settings), patch.object(
            voice_route,
            "_get_client",
            return_value=client,
        ), patch.object(voice_route, "chat_handler", new=chat_handler):
            response = await voice_route.voice_chat(
                file=upload,
                case_id=None,
                visit_type=VisitType.INITIAL,
                lang="chinese",
                confirmed=False,
            )

        self.assertEqual(response.case_id, case.case_id)
        forwarded = chat_handler.await_args.args[0]
        self.assertEqual(forwarded.visit_type, VisitType.INITIAL)


class _ProviderStatusError(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__(f"status={status_code}")
        self.status_code = status_code


def _slot(doctor: str, visit_type: str) -> dict:
    return {
        "parent_dept": "外科系",
        "child_dept": "一般骨科",
        "doctor_id": doctor,
        "doctor": doctor,
        "schedule_id": f"s-{doctor}",
        "date": "2026-08-24",
        "session": "上午",
        "slot": "3201診",
        "room": "3201診",
        "specialty_tags": "",
        "status": "open",
        "source": "db",
        "visit_type": visit_type,
        "dept_id": 7,
    }


def _confirmed_case(visit_type: VisitType) -> TriageCase:
    case = TriageCase(case_id=_case_id("recommend"), visit_type=visit_type)
    case.patient_input.symptom = "膝痛"
    case.patient_input.body_part = "膝蓋"
    case.patient_input.duration = "2週"
    case.patient_input.severity = "moderate"
    case.patient_input.red_flags_checked = True
    case.availability.preferred_days = ["週一"]
    case.availability.preferred_sessions = ["上午"]
    case.triage.need_more_info = False
    case.triage.is_final = True
    case.conversation_state.is_complete = True
    case.conversation_state.stage = ConversationStage.RECOMMENDING
    case.conversation_state.confirmed = True
    case.confirmed = True
    case.department_result = _department()
    return case


if __name__ == "__main__":
    unittest.main()
