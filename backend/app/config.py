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
    batch_triage_enabled: bool = Field(default=False, alias="BATCH_TRIAGE_ENABLED")
    batch_extraction_provider: Literal["gemini", "cerebras"] = Field(
        default="cerebras",
        alias="BATCH_EXTRACTION_PROVIDER",
    )
    ai_timeout_seconds: float = Field(default=8.0, alias="AI_TIMEOUT_SECONDS")
    ai_reply_generation_enabled: bool = Field(default=False, alias="AI_REPLY_GENERATION_ENABLED")
    ai_reply_timeout_seconds: float = Field(default=20.0, alias="AI_REPLY_TIMEOUT_SECONDS")
    ai_doctor_scoring_enabled: bool = Field(default=False, alias="AI_DOCTOR_SCORING_ENABLED")
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
