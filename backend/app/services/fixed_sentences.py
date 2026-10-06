"""TTS fixed sentences derived from the controlled question registry."""

from app.services.clarification_engine import CLARIFICATION_PROMPTS
from app.services.question_specs import QUESTION_SPECS

# ── 後端會回的追問句（做法B 用來比對 reply_text）──
# key = audio_id, value = 文字（跟後端常數一模一樣）
_AUDIO_KEY_BY_FIELD = {
    "symptom": "ask_symptom",
    "red_flags": "ask_red_flag",
    "body_part": "ask_body_part",
    "duration": "ask_duration",
    "severity": "ask_severity",
    "preferred_days": "ask_preferred_days",
    "preferred_sessions": "ask_preferred_sessions",
}

BACKEND_FIXED_SENTENCES = {
    (base if index == 1 else f"{base}_v{index}"): text
    for spec in QUESTION_SPECS
    for base in [_AUDIO_KEY_BY_FIELD[spec.state_field]]
    for index, text in enumerate(spec.variants, start=1)
}
BACKEND_FIXED_SENTENCES.update(
    {f"clarify_{field.removesuffix('s') if field == 'red_flags' else field}": text for field, text in CLARIFICATION_PROMPTS.items()}
)
BACKEND_FIXED_SENTENCES["clarify_red_flag"] = CLARIFICATION_PROMPTS["red_flags"]
BACKEND_FIXED_SENTENCES["confirm_department"] = "已確認分診結果，接下來為您推薦掛號方案。"

# ── 引導語（app 主動說，後端不回，但要生成給前端）──
GUIDANCE_SENTENCES = {
    "greeting": "您好，請問哪裡不舒服？我來幫您掛號。",
    "querying": "好的，正在為您查詢，請稍候。",       # 墊場音，蓋住後端處理時間
    "listening": "請說，我在聽。",
    "not_understood": "抱歉，沒聽清楚，可以再說一次嗎？",
}

# 全部合起來（生成腳本用）
ALL_FIXED_SENTENCES = {**BACKEND_FIXED_SENTENCES, **GUIDANCE_SENTENCES}
