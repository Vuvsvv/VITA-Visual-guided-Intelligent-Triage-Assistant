import asyncio
from typing import Any
import logging
import time

from app.config import get_settings
from app.services.chat_perf import record_ai_call, safe_exception_status

RUNTIME_AI_PROVIDER = "cerebras"
RUNTIME_AI_MODEL = "gpt-oss-120b"

_initialized = False
_llama_settings: Any = None
_google_client: Any = None
_cerebras_client: Any = None
_cerebras_initialized = False
logger = logging.getLogger(__name__)

def initialize_ai(provider: str | None = None) -> None:
    settings = get_settings()
    selected_provider = str(provider or RUNTIME_AI_PROVIDER).strip().lower()
    if selected_provider == "cerebras":
        _initialize_cerebras(settings)
        return
    if selected_provider != "gemini":
        logger.warning("Unsupported AI provider: %s", selected_provider)
        return

    global _google_client, _initialized, _llama_settings
    if _initialized:
        return
    if not settings.google_api_key:
        logger.info("AI initialization skipped: GOOGLE_API_KEY is not configured")
        return

    if _is_render_free_mode(settings):
        try:
            from google import genai
        except ImportError as exc:
            logger.warning("Lightweight Gemini client is unavailable: %s", exc)
            return

        _google_client = genai.Client(api_key=settings.google_api_key)
        _initialized = True
        logger.info("AI initialized in lightweight Gemini mode")
        return

    try:
        from llama_index.core import Settings as LlamaSettings
        from llama_index.llms.google_genai import GoogleGenAI

        LlamaSettings.llm = GoogleGenAI(
            model=settings.llm_model,
            api_key=settings.google_api_key,
        )
        if not settings.disable_local_embedding:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            LlamaSettings.embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model)
        _llama_settings = LlamaSettings
    except ImportError as exc:
        logger.warning("Full AI stack is unavailable; falling back to lightweight Gemini if possible: %s", exc)
        try:
            from google import genai
        except ImportError as google_exc:
            logger.warning("Lightweight Gemini client is unavailable: %s", google_exc)
            return
        _google_client = genai.Client(api_key=settings.google_api_key)
    _initialized = True

async def complete_prompt(
    prompt: str,
    *,
    provider: str | None = None,
    json_mode: bool = False,
) -> str:
    settings = get_settings()
    selected_provider = str(provider or RUNTIME_AI_PROVIDER).strip().lower()
    initialize_ai(selected_provider)
    if selected_provider == "cerebras":
        return await _complete_cerebras(prompt, settings, json_mode=json_mode)
    if selected_provider != "gemini":
        raise ValueError(f"Unsupported AI provider: {selected_provider}")
    if _llama_settings is not None and getattr(_llama_settings, "llm", None) is not None:
        provider_started_at = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                _llama_settings.llm.acomplete(prompt),
                timeout=settings.ai_timeout_seconds,
            )
        except BaseException as exc:
            record_ai_call(
                latency_ms=(time.perf_counter() - provider_started_at) * 1000,
                success=False,
                provider="llama_index_google_genai",
                model=settings.llm_model,
                error_type=type(exc).__name__,
                status=safe_exception_status(exc),
            )
            raise
        raw_reply = str(response).strip()
        record_ai_call(
            latency_ms=(time.perf_counter() - provider_started_at) * 1000,
            success=True,
            provider="llama_index_google_genai",
            model=settings.llm_model,
            raw_reply=raw_reply,
        )
        return raw_reply
    if _google_client is not None:
        provider_started_at = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                _google_client.aio.models.generate_content(
                    model=settings.llm_model,
                    contents=prompt,
                ),
                timeout=settings.ai_timeout_seconds,
            )
        except BaseException as exc:
            record_ai_call(
                latency_ms=(time.perf_counter() - provider_started_at) * 1000,
                success=False,
                provider="google_genai",
                model=settings.llm_model,
                error_type=type(exc).__name__,
                status=safe_exception_status(exc),
            )
            raise
        raw_reply = str(getattr(response, "text", response)).strip()
        record_ai_call(
            latency_ms=(time.perf_counter() - provider_started_at) * 1000,
            success=True,
            provider="google_genai",
            model=settings.llm_model,
            raw_reply=raw_reply,
        )
        return raw_reply
    raise RuntimeError("AI is not initialized")


def runtime_ai_available(settings: Any | None = None) -> bool:
    active_settings = settings or get_settings()
    return bool(str(getattr(active_settings, "cerebras_api_key", "") or "").strip())


async def complete_runtime_json(prompt: str, *, purpose: str) -> str:
    settings = get_settings()
    started_at = time.perf_counter()
    logger.info(
        "[AI] purpose=%s provider=%s model=%s called=true success=pending",
        purpose,
        RUNTIME_AI_PROVIDER,
        RUNTIME_AI_MODEL,
    )
    try:
        result = await complete_prompt(
            prompt,
            provider=RUNTIME_AI_PROVIDER,
            json_mode=True,
        )
    except BaseException as exc:
        logger.warning(
            "[AI] purpose=%s provider=%s model=%s called=true success=false "
            "latency_ms=%.1f fallback_reason=%s",
            purpose,
            RUNTIME_AI_PROVIDER,
            RUNTIME_AI_MODEL,
            (time.perf_counter() - started_at) * 1000,
            type(exc).__name__,
        )
        raise
    logger.info(
        "[AI] purpose=%s provider=%s model=%s called=true success=true "
        "latency_ms=%.1f fallback_reason=null",
        purpose,
        RUNTIME_AI_PROVIDER,
        RUNTIME_AI_MODEL,
        (time.perf_counter() - started_at) * 1000,
    )
    return result


def _initialize_cerebras(settings: Any) -> None:
    global _cerebras_client, _cerebras_initialized
    if _cerebras_initialized:
        return
    if not getattr(settings, "cerebras_api_key", ""):
        logger.info("Cerebras initialization skipped: CEREBRAS_API_KEY is not configured")
        return
    try:
        from cerebras.cloud.sdk import AsyncCerebras
    except ImportError as exc:
        logger.warning("Cerebras SDK is unavailable: %s", exc)
        return
    _cerebras_client = AsyncCerebras(
        api_key=settings.cerebras_api_key,
        timeout=settings.ai_timeout_seconds,
        max_retries=0,
    )
    _cerebras_initialized = True
    logger.info("AI initialized in Cerebras async mode model=%s", RUNTIME_AI_MODEL)


async def _complete_cerebras(prompt: str, settings: Any, *, json_mode: bool = False) -> str:
    if _cerebras_client is None:
        raise RuntimeError("Cerebras AI is not initialized")
    provider_started_at = time.perf_counter()
    try:
        request: dict[str, Any] = {
            "model": RUNTIME_AI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        response = await asyncio.wait_for(
            _cerebras_client.chat.completions.create(**request),
            timeout=settings.ai_timeout_seconds,
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("Cerebras response has no choices")
        message = getattr(choices[0], "message", None)
        raw_reply = str(getattr(message, "content", "") or "").strip()
        if not raw_reply:
            raise ValueError("Cerebras response content is empty")
    except BaseException as exc:
        record_ai_call(
            latency_ms=(time.perf_counter() - provider_started_at) * 1000,
            success=False,
            provider="cerebras",
            model=RUNTIME_AI_MODEL,
            error_type=type(exc).__name__,
            status=safe_exception_status(exc),
        )
        raise
    record_ai_call(
        latency_ms=(time.perf_counter() - provider_started_at) * 1000,
        success=True,
        provider="cerebras",
        model=RUNTIME_AI_MODEL,
        raw_reply=raw_reply,
    )
    return raw_reply


def _is_render_free_mode(settings: Any) -> bool:
    return (
        settings.deploy_mode.lower() == "render_free"
        or settings.disable_local_embedding
    )
