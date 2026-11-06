"""「我想看婦產科」 with no symptom: acknowledge the department, then ask the fixed symptom question."""
import json
from unittest.mock import patch

from app.services import department_preference_service as preference
from app.services.rule_engine import QUESTION_TEXTS
from test_phase7_natural_safety import runtime, evidence

ACTIVE = [{"dept_id": 1251, "parent_dept": "婦女醫學部", "child_dept": "婦產科"}]


def _interpretation(text, extractions=()):
    return json.dumps(dict(
        semantic_extractions=list(extractions), pending_answer=None, ttas_evidence=[], time_preferences=[],
        prior_answer_reference=None,
        department_preference=dict(dept_id=1251, department_name="婦產科", source_text="婦產科", confidence=0.99),
        dialogue_act=dict(act="new_information", source_text=text, confidence=0.99)))


def test_department_only_gets_acknowledgement_and_fixed_symptom_question(runtime):
    client, interpreter, planner, _ = runtime
    text = "我想看婦產科"
    interpreter.return_value = _interpretation(text)
    with patch.object(preference, "fetch_active_departments", return_value=ACTIVE):
        result = client.post("/chat", json=dict(visit_type="initial", message=text)).json()
    state = result["conversation_state"]
    assert result["reply"] == (
        "收到，已記錄您希望看婦產科。這是您指定的科別偏好；"
        f"為了協助確認是否適合，仍需要了解症狀。{QUESTION_TEXTS['symptom']}"
    )
    assert state["reply_source"] == "REQUESTED_DEPARTMENT_SYMPTOM"
    assert state["last_question_key"] == "symptom"
    planner.assert_not_awaited()


def test_department_with_symptom_keeps_normal_flow(runtime):
    client, interpreter, _, _ = runtime
    text = "我想看婦產科因為經痛很嚴重"
    interpreter.return_value = _interpretation(text, [
        evidence("symptom", "經痛", "經痛"), evidence("severity", "很嚴重", "很嚴重")])
    with patch.object(preference, "fetch_active_departments", return_value=ACTIVE):
        result = client.post("/chat", json=dict(visit_type="initial", message=text)).json()
    assert result["conversation_state"]["reply_source"] != "REQUESTED_DEPARTMENT_SYMPTOM"
    assert "已記錄您希望看" not in result["reply"]
