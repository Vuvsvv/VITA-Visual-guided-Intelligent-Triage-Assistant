from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ENV_FILE,
        env_file_encoding="utf-8-sig",
        extra="ignore",
    )

    db_driver: str = Field(default="ODBC Driver 17 for SQL Server", alias="DB_DRIVER")
    db_server: str = Field(default="medichain-server.database.windows.net", alias="DB_SERVER")
    db_name: str = Field(default="MediChainDB", alias="DB_NAME")
    db_user: str = Field(default="medichain_admin", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")
    db_trust_server_certificate: bool = Field(default=False, alias="DB_TRUST_SERVER_CERTIFICATE")

    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    llm_model: str = Field(default="gemini-3.5-flash", alias="LLM_MODEL")
    ai_provider: Literal["gemini", "cerebras"] = Field(default="cerebras", alias="AI_PROVIDER")
    cerebras_api_key: str = Field(default="", alias="CEREBRAS_API_KEY")
    cerebras_model: Literal["gpt-oss-120b"] = Field(default="gpt-oss-120b", alias="CEREBRAS_MODEL")
    ai_timeout_seconds: float = Field(default=8.0, alias="AI_TIMEOUT_SECONDS")
    # true＝所有 Cerebras 呼叫都用 temperature=0＋固定 seed（同一句話盡量得到同一個答案）；
    # false＝只有醫師專長評分固定，其他呼叫用供應商預設溫度
    ai_temperature_zero: bool = Field(default=True, alias="AI_TEMPERATURE_ZERO")
    ai_reply_generation_enabled: bool = Field(default=False, alias="AI_REPLY_GENERATION_ENABLED")
    ai_reply_timeout_seconds: float = Field(default=20.0, alias="AI_REPLY_TIMEOUT_SECONDS")
    ai_doctor_scoring_enabled: bool = Field(default=False, alias="AI_DOCTOR_SCORING_ENABLED")
    doctor_scoring_batch_size: int = Field(default=10, ge=1, le=25, alias="DOCTOR_SCORING_BATCH_SIZE")
    doctor_scoring_max_candidates: int = Field(default=40, ge=1, le=100, alias="DOCTOR_SCORING_MAX_CANDIDATES")
    doctor_scoring_max_batches: int = Field(default=4, ge=1, le=10, alias="DOCTOR_SCORING_MAX_BATCHES")
    doctor_scoring_total_timeout_seconds: float = Field(
        default=12.0,
        gt=0,
        le=60,
        alias="DOCTOR_SCORING_TOTAL_TIMEOUT_SECONDS",
    )
    # 線索題收斂（docs/修改計畫.md 第 9 節「線索題收斂」）
    clue_question_limit: int = Field(default=5, ge=1, le=10, alias="CLUE_QUESTION_LIMIT")
    # 兩科以上候選時，問滿這麼多題線索題才定科（AI 已收斂也要等）
    clue_min_questions: int = Field(default=3, ge=0, le=10, alias="CLUE_MIN_QUESTIONS")
    clue_concepts_per_question: int = Field(default=3, ge=1, le=5, alias="CLUE_CONCEPTS_PER_QUESTION")
    clue_planner_concept_limit: int = Field(default=10, ge=1, le=20, alias="CLUE_PLANNER_CONCEPT_LIMIT")
    # Canonical KB/DB id, validated against live Department rows like resolve_department_names.
    family_medicine_dept_id: int = Field(default=1233, ge=1, alias="FAMILY_MEDICINE_DEPT_ID")
    family_medicine_fallback_message: str = Field(
        default="目前的症狀無法明確區分科別，系統建議先由家庭醫學科初步評估。",
        alias="FAMILY_MEDICINE_FALLBACK_MESSAGE",
    )
    # 安全網（docs/修改計畫.md Phase 2「防止 AI 無限聊」）：UX 防呆，不是醫療規則；以使用者訊息輪數計
    safety_net_ask_turn: int = Field(default=8, ge=1, alias="SAFETY_NET_ASK_TURN")
    # 除了家醫科一個候選科都沒有時（例「舌頭痛」「屁股疼」知識庫對不到），提早到這一輪問同一題
    safety_net_no_candidate_turn: int = Field(default=3, ge=1, alias="SAFETY_NET_NO_CANDIDATE_TURN")
    hard_turn_cap: int = Field(default=20, ge=1, alias="HARD_TURN_CAP")
    # 語意比對前先挑症狀詞用的本地模型（app/services/concept_shortlist.py）；設成空字串＝停用，改送整份清單
    concept_embedding_model: str = Field(
        default="jinaai/jina-embeddings-v2-base-zh", alias="CONCEPT_EMBEDDING_MODEL",
    )
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    deploy_mode: str = Field(default="local", alias="DEPLOY_MODE")
    disable_local_embedding: bool = Field(default=False, alias="DISABLE_LOCAL_EMBEDDING")

    # 語音 gateway 設定
    voice_enabled: bool = Field(default=False, alias="VOICE_ENABLED")
    voice_gateway_url: str = Field(default="http://localhost:8000", alias="VOICE_GATEWAY_URL")
    voice_gateway_key: str = Field(default="", alias="VOICE_GATEWAY_KEY")
    voice_gateway_is_ngrok: bool = Field(default=False, alias="VOICE_GATEWAY_IS_NGROK")
    voice_default_lang: str = Field(default="taiwanese", alias="VOICE_DEFAULT_LANG")
    voice_timeout: float = Field(default=60.0, alias="VOICE_TIMEOUT")
    # 固定句緩存開關：true=固定句回 audio_id 不合成（需 Android 有本地檔）
    #                false=全部即時合成（還沒放語音檔時用這個）
    voice_use_fixed_cache: bool = Field(default=False, alias="VOICE_USE_FIXED_CACHE")

def get_settings():
    return Settings()
