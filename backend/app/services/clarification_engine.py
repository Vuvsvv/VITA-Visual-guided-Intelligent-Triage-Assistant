from __future__ import annotations


CLARIFICATION_PROMPTS = {
        "symptom": "想確認一下，您最主要想處理的不舒服症狀是什麼？",
        "red_flags": "想再確認一下，你目前有沒有剛才提到的任何一項急迫症狀？如果有，請告訴我是胸痛、嚴重呼吸困難、意識不清、大量出血、半邊無力或劇烈頭痛；如果都沒有，請回答『都沒有』。",
        "body_part": "想確認一下，症狀主要在身體哪個部位？",
        "duration": "想確認一下，這個症狀大約持續多久了？例如幾天、幾週或幾個月。",
        "severity": "想確認一下，您的疼痛或不舒服是輕微、普通，還是嚴重到影響睡眠或日常活動？",
        "preferred_days": "想確認一下，您可就醫的日期是平日、週末、每天都可以，還是目前不確定？",
        "preferred_sessions": "想確認一下，您比較方便的時段是上午、下午、夜間，還是全天都可以？",
        "can_take_leave": "想確認一下，如果需要較早看診，您是否方便請假？",
}


def clarification_prompt(field: str, semantic_status: str | None = None) -> str:
    if semantic_status == "unavailable" and field == "preferred_days":
        return "了解您最近可能不方便。想確認是否完全沒有可就醫日期，還是只有部分日期可以？"
    return CLARIFICATION_PROMPTS.get(field, "想再確認一下剛剛那個回答，可以請您用更明確的方式說明嗎？")
