from __future__ import annotations

import asyncio
import unittest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import TriageCase
from app.services import appointment_service, project_smart_department_adapter, rag_triage_adapter
from app.services.appointment_service import detect_department_result
from app.services.project_smart_department_adapter import (
    detect_department_with_project_smart_adapter,
    keyword_department_hints,
)
from app.services.rule_engine import apply_user_message, evaluate_urgency


class ProjectSmartDepartmentAdapterTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_adapter_settings = project_smart_department_adapter.get_settings
        self.original_adapter_complete = project_smart_department_adapter.complete_prompt
        self.original_rag_settings = rag_triage_adapter.get_settings
        self.original_rag_complete = rag_triage_adapter.complete_prompt
        self.original_fetch_departments = appointment_service.fetch_active_departments

    def tearDown(self):
        project_smart_department_adapter.get_settings = self.original_adapter_settings
        project_smart_department_adapter.complete_prompt = self.original_adapter_complete
        rag_triage_adapter.get_settings = self.original_rag_settings
        rag_triage_adapter.complete_prompt = self.original_rag_complete
        appointment_service.fetch_active_departments = self.original_fetch_departments

    async def test_adapter_skips_without_cerebras_api_key(self):
        class FakeSettings:
            cerebras_api_key = ""

        project_smart_department_adapter.get_settings = lambda: FakeSettings()
        case = _case_from_message("膝蓋走路疼痛2週")

        result = await detect_department_with_project_smart_adapter(case, _departments())

        self.assertIsNone(result)

    async def test_adapter_ai_exception_falls_back_to_none(self):
        class FakeSettings:
            cerebras_api_key = "test-key"

        async def fail_complete(_: str) -> str:
            raise RuntimeError("simulated ai failure")

        project_smart_department_adapter.get_settings = lambda: FakeSettings()
        project_smart_department_adapter.complete_prompt = fail_complete
        case = _case_from_message("膝蓋走路疼痛2週")

        result = await detect_department_with_project_smart_adapter(case, _departments())

        self.assertIsNone(result)

    async def test_adapter_ai_timeout_falls_back_to_none(self):
        class FakeSettings:
            cerebras_api_key = "test-key"

        async def timeout_complete(_: str) -> str:
            raise asyncio.TimeoutError("simulated timeout")

        project_smart_department_adapter.get_settings = lambda: FakeSettings()
        project_smart_department_adapter.complete_prompt = timeout_complete
        case = _case_from_message("皮膚發癢、紅疹2週")

        result = await detect_department_with_project_smart_adapter(case, _departments())

        self.assertIsNone(result)

    async def test_adapter_ai_success_returns_backend_department_result(self):
        class FakeSettings:
            cerebras_api_key = "test-key"

        async def fake_complete(_: str) -> str:
            return """
            {
              "dept_id": 101,
              "confidence": 0.86,
              "reason": ["紅疹與發癢符合皮膚科評估", "關鍵字命中皮膚相關症狀"]
            }
            """

        project_smart_department_adapter.get_settings = lambda: FakeSettings()
        project_smart_department_adapter.complete_prompt = fake_complete
        case = _case_from_message("皮膚發癢、紅疹2週")

        result = await detect_department_with_project_smart_adapter(case, _departments())

        self.assertIsNotNone(result)
        self.assertEqual(result.childDept, "皮膚科")
        self.assertEqual(result.parentDept, "其他科")
        self.assertGreaterEqual(result.confidence, 0.86)
        self.assertTrue(any("project-smart" in reason for reason in result.reason))

    async def test_adapter_rejects_department_id_not_in_candidates(self):
        class FakeSettings:
            cerebras_api_key = "test-key"

        async def fake_complete(_: str) -> str:
            return """
            {
              "dept_id": 999999,
              "confidence": 0.99,
              "reason": ["故意不一致"]
            }
            """

        project_smart_department_adapter.get_settings = lambda: FakeSettings()
        project_smart_department_adapter.complete_prompt = fake_complete

        result = await detect_department_with_project_smart_adapter(
            _case_from_message("皮膚發癢"),
            _departments(),
        )

        self.assertIsNone(result)

    async def test_general_orthopedic_prompt_prefers_schedulable_general_leaf(self):
        class FakeSettings:
            cerebras_api_key = "test-key"

        captured = {}

        async def fake_complete(prompt: str) -> str:
            captured["prompt"] = prompt
            return '{"dept_id":1298,"confidence":0.91,"reason":["一般膝關節症狀"]}'

        project_smart_department_adapter.get_settings = lambda: FakeSettings()
        project_smart_department_adapter.complete_prompt = fake_complete
        departments = [
            {"dept_id": 1298, "parent_dept": "外科系", "child_dept": "一般骨科"},
            {"dept_id": 1302, "parent_dept": "外科系", "child_dept": "骨科"},
            {"dept_id": 1299, "parent_dept": "婦幼", "child_dept": "兒童骨科"},
        ]

        result = await detect_department_with_project_smart_adapter(
            _case_from_message("膝蓋走路疼痛，爬樓梯更明顯"),
            departments,
        )

        self.assertEqual(result.dept_id, 1298)
        self.assertEqual(result.childDept, "一般骨科")
        self.assertIn("優先選正式候選中的「一般骨科」", captured["prompt"])

    async def test_supported_orthopedic_subspecialty_id_is_not_overridden(self):
        class FakeSettings:
            cerebras_api_key = "test-key"

        async def fake_complete(_: str) -> str:
            return '{"dept_id":1299,"confidence":0.93,"reason":["兒童骨科症狀"]}'

        project_smart_department_adapter.get_settings = lambda: FakeSettings()
        project_smart_department_adapter.complete_prompt = fake_complete
        departments = [
            {"dept_id": 1298, "parent_dept": "外科系", "child_dept": "一般骨科"},
            {"dept_id": 1302, "parent_dept": "外科系", "child_dept": "骨科"},
            {"dept_id": 1299, "parent_dept": "婦幼", "child_dept": "兒童骨科"},
        ]

        result = await detect_department_with_project_smart_adapter(
            _case_from_message("10歲兒童手臂骨折疼痛"),
            departments,
        )

        self.assertEqual(result.dept_id, 1299)
        self.assertEqual(result.childDept, "兒童骨科")

    async def test_detect_department_result_falls_back_after_ai_failure(self):
        class FakeSettings:
            cerebras_api_key = "test-key"

        async def fail_complete(_: str) -> str:
            raise RuntimeError("simulated ai failure")

        appointment_service.fetch_active_departments = _departments
        project_smart_department_adapter.get_settings = lambda: FakeSettings()
        project_smart_department_adapter.complete_prompt = fail_complete
        rag_triage_adapter.get_settings = lambda: FakeSettings()
        rag_triage_adapter.complete_prompt = fail_complete
        case = _case_from_message("膝蓋走路疼痛2週，爬樓梯吃力")

        result = await detect_department_result(case)

        self.assertIn(result.childDept, {"一般骨科", "骨科"})
        self.assertGreater(result.confidence, 0)

    async def test_chat_integration_preserves_response_contract_and_gate(self):
        class FakeSettings:
            cerebras_api_key = "test-key"

        async def fake_complete(_: str) -> str:
            return """
            {
              "dept_id": 101,
              "confidence": 0.88,
              "reason": ["紅疹與發癢符合皮膚科評估", "關鍵字命中皮膚相關症狀"]
            }
            """

        appointment_service.fetch_active_departments = _departments
        project_smart_department_adapter.get_settings = lambda: FakeSettings()
        project_smart_department_adapter.complete_prompt = fake_complete

        client = TestClient(app)
        response = client.post(
            "/chat",
            json={
                "visit_type": "initial",
                "message": (
                    "皮膚紅疹發癢2週，中等程度，慢慢變嚴重，"
                    "沒有胸痛呼吸困難意識不清大量出血，週一上午可以看診"
                )
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["conversation_state"]["stage"], "collecting")
        self.assertIsNone(data["department_result"])
        self.assertEqual(data["conversation_state"]["department_status"], "unresolved")
        self.assertFalse(data["conversation_state"]["confirmed"])

        recommend_response = client.post("/recommend", json={"case_id": data["case_id"]})
        self.assertEqual(recommend_response.status_code, 400)
        self.assertIn("尚未完成", recommend_response.json()["detail"])

    def test_keyword_hints_cover_required_symptoms(self):
        skin_hints = keyword_department_hints(_case_from_message("皮膚發癢、紅疹"))
        knee_hints = keyword_department_hints(_case_from_message("膝蓋走路疼痛"))
        chest_case = _case_from_message("胸痛、呼吸困難")
        chest_hints = keyword_department_hints(chest_case)
        chest_case.triage = evaluate_urgency(chest_case)

        self.assertIn("皮膚科", {hint.childDept for hint in skin_hints})
        self.assertTrue({"骨科", "復健醫學"} & {hint.childDept for hint in knee_hints})
        self.assertTrue({"心臟內科", "胸腔內科"} & {hint.childDept for hint in chest_hints})
        self.assertIsNone(chest_case.triage.urgency_level)
        self.assertIsNone(chest_case.triage.urgency_score)
        self.assertFalse(chest_case.triage.warning_required)
        self.assertEqual(chest_case.ttas_result.status, "insufficient_information")

    def test_keyword_hints_ignore_negated_red_flags(self):
        hints = keyword_department_hints(
            _case_from_message("頭暈一天，沒有胸痛、呼吸困難、意識不清、大量出血")
        )

        hinted_departments = {hint.childDept for hint in hints}
        self.assertIn("一般內科", hinted_departments)
        self.assertNotIn("心臟內科", hinted_departments)
        self.assertNotIn("胸腔內科", hinted_departments)

    async def test_requested_general_internal_medicine_short_circuits_adapter(self):
        case = _case_from_message("頭暈一天，想看一般內科，沒有胸痛呼吸困難")

        result = await detect_department_with_project_smart_adapter(case, _departments())

        self.assertIsNotNone(result)
        self.assertEqual(result.childDept, "一般內科")
        self.assertTrue(any("想看 一般內科" in reason for reason in result.reason))


def _case_from_message(message: str) -> TriageCase:
    case = TriageCase(case_id="case_adapter_test")
    apply_user_message(case, message)
    return case


def _departments() -> list[dict]:
    return [
        {"dept_id": 101, "parent_dept": "其他科", "child_dept": "皮膚科"},
        {"dept_id": 102, "parent_dept": "外科系", "child_dept": "一般骨科"},
        {"dept_id": 103, "parent_dept": "外科系", "child_dept": "骨科"},
        {"dept_id": 104, "parent_dept": "其他科", "child_dept": "復健醫學"},
        {"dept_id": 105, "parent_dept": "一般內科", "child_dept": "心臟內科"},
        {"dept_id": 106, "parent_dept": "一般內科", "child_dept": "胸腔內科"},
        {"dept_id": 107, "parent_dept": "一般內科", "child_dept": "一般內科"},
        {"dept_id": 108, "parent_dept": "一般內科", "child_dept": "家庭醫學科(一般門診/戒菸)"},
    ]


if __name__ == "__main__":
    unittest.main()
