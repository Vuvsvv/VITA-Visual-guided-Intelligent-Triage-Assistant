from __future__ import annotations

import random
import unittest
from unittest.mock import AsyncMock, patch

from app.schemas import TriageCase
from app.services import ai_reply_generator
from app.services.batch_question_service import build_question_batch
from app.services.question_specs import (
    QUESTION_SPECS,
    QUESTION_SPEC_BY_FIELD,
    QUESTION_SPEC_BY_ID,
    select_question_variant,
)


class QuestionSpecTest(unittest.IsolatedAsyncioTestCase):
    def test_seven_questions_each_have_exactly_five_variants(self):
        self.assertEqual(len(QUESTION_SPECS), 7)
        self.assertEqual(
            {spec.state_field for spec in QUESTION_SPECS},
            {"symptom", "red_flags", "body_part", "duration", "severity", "preferred_days", "preferred_sessions"},
        )
        for spec in QUESTION_SPECS:
            with self.subTest(question_id=spec.question_id):
                self.assertEqual(len(spec.variants), 5)
                self.assertEqual(QUESTION_SPEC_BY_ID[spec.question_id].state_field, spec.state_field)
                self.assertIs(QUESTION_SPEC_BY_FIELD[spec.state_field], spec)

    def test_every_severity_variant_asks_for_the_same_three_level_scale(self):
        variants = QUESTION_SPEC_BY_ID["severity"].variants

        self.assertEqual(len(variants), 5)
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertIn("輕微", variant)
                self.assertTrue(any(term in variant for term in ("普通", "中等")))
                self.assertTrue(any(term in variant for term in ("嚴重", "很痛")))

    def test_seeded_selector_is_reproducible(self):
        first_rng = random.Random(42)
        second_rng = random.Random(42)
        first = [select_question_variant("duration", first_rng) for _ in range(8)]
        second = [select_question_variant("duration", second_rng) for _ in range(8)]
        self.assertEqual(first, second)

    def test_batch_keeps_state_key_and_exposes_separate_question_id(self):
        case = TriageCase(case_id="question-contract")
        case.patient_input.symptom = "頭暈"
        questions = build_question_batch(case, mark_asked=False)
        red = next(item for item in questions if item.key == "red_flags")

        self.assertEqual(red.key, "red_flags")
        self.assertEqual(red.state_field, "red_flags")
        self.assertEqual(red.question_id, "emergency_symptoms")
        self.assertIn(red.question, QUESTION_SPEC_BY_ID["emergency_symptoms"].variants)

    def test_every_red_flag_variant_preserves_six_safety_concepts(self):
        required = ("突發胸痛", "嚴重呼吸困難", "意識不清", "大量出血", "半邊無力", "劇烈頭痛")
        for variant in QUESTION_SPEC_BY_ID["emergency_symptoms"].variants:
            with self.subTest(variant=variant):
                for term in required:
                    self.assertIn(term, variant)

    async def test_question_variant_never_calls_ai_provider_or_changes_case(self):
        case = TriageCase(case_id="no-question-ai")
        before = case.model_copy(deep=True)
        selected = select_question_variant("duration", random.Random(7))
        with patch.object(ai_reply_generator, "complete_prompt", new_callable=AsyncMock) as provider:
            reply = await ai_reply_generator.generate_triage_reply(
                case=case,
                next_question=selected,
                fallback_reply=selected,
            )

        self.assertEqual(reply, selected)
        self.assertEqual(case, before)
        provider.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
