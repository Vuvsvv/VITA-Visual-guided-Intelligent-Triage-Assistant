"""
語音聊天端點 (/voice/chat)
==========================
把語音接進現有的問診流程（A2 方案）。

流程：
  Android 傳音訊 → ASR 轉文字 → 呼叫現有 chat() 邏輯
                → 拿回覆文字 → TTS 轉語音 → 回傳給 Android

關鍵：完全重用現有的 chat() 函式，不複製問診邏輯。
現有 /chat /recommend /generate_script 完全不動。

容錯：任何語音環節失敗，至少回傳文字，讓 Android 降級用系統 TTS。
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.config import get_settings
from app.schemas import ChatRequest, VisitType, VoiceChatResponse
from app.routes.chat import chat as chat_handler
from app.services.voice_client import VoiceGatewayClient, VoiceGatewayError
from app.services.fixed_sentences import BACKEND_FIXED_SENTENCES
from app.services.tts_cache import tts_cache
from app.services.voice_perf import VoiceTrace, trace_for, run_gateway, safe_cache_key

# 反查表：文字 → audio_id（做法B 用）
_FIXED_LOOKUP = {text.strip(): sid for sid, text in BACKEND_FIXED_SENTENCES.items()}

router = APIRouter(prefix="/voice", tags=["voice"])
logger = logging.getLogger(__name__)


def _voice_log(
    operation: str,
    backend_route: str,
    provider: str | None,
    *,
    status: str,
    error: VoiceGatewayError | None = None,
) -> None:
    logger.info(
        "[VOICE] operation=%s backend_route=%s upstream_provider=%s upstream_status=%s "
        "exception_type=%s exception_message=%s",
        operation,
        backend_route,
        provider or "unknown",
        status,
        error.exception_type if error else "none",
        error.exception_message if error else "none",
    )


def _voice_config_error() -> str | None:
    settings = get_settings()
    if not settings.voice_enabled:
        return "語音功能未啟用（請設定 VOICE_ENABLED=true）"
    if not settings.voice_gateway_url.strip():
        return "VOICE_GATEWAY_URL 未設定"
    if not settings.voice_gateway_key.strip():
        return "VOICE_GATEWAY_KEY 未設定，請填入 Voice Gateway 的 X-API-Key"
    return None


def _get_client() -> VoiceGatewayClient:
    """依設定建立 gateway client"""
    settings = get_settings()
    return VoiceGatewayClient(
        base_url=settings.voice_gateway_url,
        api_key=settings.voice_gateway_key,
        timeout=settings.voice_timeout,
        is_ngrok=settings.voice_gateway_is_ngrok,
    )


@router.post("/asr")
async def voice_asr(
    file: UploadFile = File(..., description="長輩說話的音訊檔（wav）"),
    lang: Optional[str] = Form(None, description="taiwanese（台語）或 chinese（國語）"),
):
    """
    單純語音辨識：音訊進 → 文字出。

    Android 上傳實際麥克風錄音；只回傳可編輯草稿，不推進問診。
    """
    trace = trace_for("asr")
    settings = get_settings()
    config_error = _voice_config_error()
    if config_error:
        trace.outcome = "voice_disabled"
        return {"text": "", "lang": lang or settings.voice_default_lang, "asr_failed": True, "error": config_error}

    use_lang = lang or settings.voice_default_lang
    if use_lang not in ("taiwanese", "chinese"):
        trace.outcome = "invalid_language"
        return {"text": "", "lang": use_lang, "asr_failed": True, "error": f"不支援的語言: {use_lang}"}

    try:
        provider = VoiceGatewayClient._ASR_SERVICE.get(use_lang)
        audio_bytes = await file.read()
        user_text = await run_gateway(trace, _get_client().transcribe,
            audio_bytes, filename=file.filename or "audio.wav", lang=use_lang
        )
        trace.outcome = "ok" if user_text.strip() else "empty_transcript"
        _voice_log("asr", "/voice/asr", provider, status="ok" if user_text.strip() else "empty")
        return {"text": user_text, "lang": use_lang, "asr_failed": not bool(user_text.strip()), "error": None}
    except VoiceGatewayError as e:
        trace.outcome = "asr_failed"
        _voice_log(
            "asr",
            "/voice/asr",
            VoiceGatewayClient._ASR_SERVICE.get(use_lang),
            status=e.upstream_status,
            error=e,
        )
        return {"text": "", "lang": use_lang, "asr_failed": True, "error": str(e)}


@router.post("/chat", response_model=VoiceChatResponse)
async def voice_chat(
    file: UploadFile = File(..., description="長輩說話的音訊檔（wav）"),
    case_id: Optional[str] = Form(None, description="對話 case_id，多輪要帶回來"),
    visit_type: Optional[VisitType] = Form(None, description="首次建立 case 時的 initial/followup/return_visit"),
    lang: Optional[str] = Form(None, description="taiwanese（台語）或 chinese（國語）"),
    confirmed: bool = Form(False, description="是否確認分診結果"),
):
    """
    語音版問診：音訊進 → 文字+語音出

    Android 錄音後呼叫這個端點，內部串接 ASR → chat → TTS。
    """
    settings = get_settings()

    # 1. 語音功能開關（Render 上可關掉，本機開）
    config_error = _voice_config_error()
    if config_error:
        raise HTTPException(503, config_error)

    # 決定語言（沒傳就用預設）
    use_lang = lang or settings.voice_default_lang
    if use_lang not in ("taiwanese", "chinese"):
        raise HTTPException(400, f"不支援的語言: {use_lang}")

    client = _get_client()

    # 2. ASR：音訊 → 文字
    try:
        audio_bytes = await file.read()
        user_text = await run_gateway(VoiceTrace("asr"), client.transcribe,
            audio_bytes, filename=file.filename or "audio.wav", lang=use_lang
        )
    except VoiceGatewayError as e:
        logger.warning("Legacy ASR failed error_type=%s", type(e).__name__)
        raise HTTPException(400, "聽不清楚，請再說一次")

    if not user_text.strip():
        raise HTTPException(400, "聽不清楚，請再說一次")

    logger.info("Legacy ASR completed lang=%s", use_lang)

    # 3. 呼叫現有 chat 邏輯（重用，不複製）
    chat_req = ChatRequest(
        case_id=case_id,
        message=user_text,
        confirmed=confirmed,
        visit_type=visit_type,
    )
    triage_result = await chat_handler(chat_req)

    reply_text = triage_result.reply or ""

    # 4. 判斷是否固定句（做法B）
    #    固定句 → 不合成，回 audio_id，Android 播本地檔（零延遲、省後端時間）
    #    動態句 → 即時合成
    reply_audio_base64 = ""
    reply_audio_id = ""
    audio_format = "wav"
    tts_failed = False

    # 只有開啟固定句緩存時才走做法B（需 Android 有本地檔）
    audio_id = _FIXED_LOOKUP.get(reply_text.strip()) if settings.voice_use_fixed_cache else None
    if audio_id:
        # 固定句：不合成，只回 id
        reply_audio_id = audio_id
    elif reply_text.strip():
        # 動態句（或緩存關閉時的固定句）：即時合成
        try:
            tts_result = await run_gateway(VoiceTrace("tts"), client.synthesize_to_base64, reply_text, lang=use_lang)
            reply_audio_base64 = tts_result.audio_base64
            audio_format = tts_result.audio_format
        except VoiceGatewayError as e:
            logger.warning("Legacy TTS failed error_type=%s", type(e).__name__)
            tts_failed = True  # Android 收到後改用系統 TTS 念 reply_text

    # 5. 組回應
    return VoiceChatResponse(
        case_id=triage_result.case_id,
        user_text=user_text,
        reply_text=reply_text,
        reply_audio_base64=reply_audio_base64,
        reply_audio_id=reply_audio_id,
        audio_format=audio_format,
        needMoreInfo=triage_result.needMoreInfo,
        stage=triage_result.conversation_state.stage.value
        if triage_result.conversation_state
        else "",
        department_result=triage_result.department_result,
        tts_failed=tts_failed,
    )


@router.post("/tts")
async def voice_tts(payload: dict):
    """
    文字轉語音（給前端「播放」按鈕用）
    請求 JSON: {text, lang, speed, session_id?}。session_id 是 Android 產生的 UUID。
    回應 JSON: {audio_base64, audio_format, tts_failed, error}
    """
    trace = trace_for("tts")
    settings = get_settings()
    config_error = _voice_config_error()
    if config_error:
        trace.outcome = "voice_disabled"
        return {"audio_base64": "", "audio_format": "wav", "tts_failed": True,
                "error": config_error}

    text = (payload.get("text") or "").strip()
    lang = payload.get("lang") or settings.voice_default_lang
    if lang not in ("taiwanese", "chinese"):
        lang = settings.voice_default_lang
    if not text:
        trace.outcome = "empty_text"
        return {"audio_base64": "", "audio_format": "wav", "tts_failed": True,
                "error": "文字為空"}

    session_id = _session_id(payload.get("session_id"))
    trace.session_id = session_id
    try:
        speed = float(payload.get("speed", 1.0))
        if not 0.25 <= speed <= 4.0:
            raise ValueError()
    except (ValueError, TypeError):
        raise HTTPException(400, "speed 必須介於 0.25 與 4.0")

    trace.cache_key = safe_cache_key(session_id or trace.request_id, text, lang, speed)
    try:
        client = _get_client()
        service_type = VoiceGatewayClient.tts_service_type(lang)
        logger.info("voice/tts request lang=%s service_type=%s text_len=%s", lang, service_type, len(text))
        async def generate():
            # Use bounded TTS workers; ASR has separate executor slots.
            return await run_gateway(trace, client.synthesize_to_base64, text, lang=lang, speed=speed)

        if session_id:
            tts_result = await tts_cache.get(session_id, text, lang, speed, generate, trace=trace)
        else:
            trace.set_cache("uncached")
            tts_result = await generate()
        logger.info(
            "voice/tts success lang=%s service_type=%s audio_format=%s audio_b64_len=%s",
            lang,
            service_type,
            tts_result.audio_format,
            len(tts_result.audio_base64),
        )
        _voice_log("tts", "/voice/tts", service_type, status="ok")
        return {"audio_base64": tts_result.audio_base64, "audio_format": tts_result.audio_format,
                "tts_failed": False, "error": None}
    except VoiceGatewayError as e:
        trace.outcome = "tts_failed"
        _voice_log(
            "tts",
            "/voice/tts",
            VoiceGatewayClient.tts_service_type(lang),
            status=e.upstream_status,
            error=e,
        )
        return {"audio_base64": "", "audio_format": "wav", "tts_failed": True,
                "error": str(e)}


def _session_id(value) -> str | None:
    if value is None:
        return None  # Existing clients remain supported without session caching.
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(400, "session_id 必須是 UUID")


@router.post("/tts/cleanup")
async def cleanup_tts(payload: dict):
    session_id = _session_id(payload.get("session_id"))
    if session_id is None:
        raise HTTPException(400, "session_id 不可為空")
    # Cleanup works even when voice is disabled or the gateway is unavailable.
    tts_cache.prune()
    tts_cache.cleanup(session_id)
    return {"session_id": session_id, "cleared": True}


@router.get("/health")
async def voice_health():
    """檢查語音 gateway 是否可用"""
    settings = get_settings()
    config_error = _voice_config_error()
    if config_error:
        return {
            "voice_enabled": settings.voice_enabled,
            "gateway": "not_configured",
            "error": config_error,
        }
    try:
        client = _get_client()
        gateway_status = client.health()
        return {"voice_enabled": True, "gateway": gateway_status}
    except VoiceGatewayError as e:
        return {"voice_enabled": True, "gateway": "unreachable", "error": str(e)}
