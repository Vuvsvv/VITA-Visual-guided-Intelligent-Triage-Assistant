from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import asyncio
from contextlib import suppress

from app.routes.chat import router as chat_router
from app.routes.recommend import router as recommend_router
from app.routes.generate_script import router as generate_script_router
from app.routes.voice import router as voice_router
from app.routes.followup import router as followup_router
from app.routes.reference import router as reference_router
from app.routes.schedules import router as schedules_router
from app.db import DatabaseUnavailableError
from app.services.ai_service import initialize_ai
from app.services.tts_cache import tts_cache
from app.services.voice_perf import VoicePerfMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI 智慧醫療掛號導引系統",
    description="根據使用者症狀與偏好，提供多輪問答、推薦方案與 Android 導引腳本。",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(VoicePerfMiddleware)

app.include_router(chat_router)
app.include_router(recommend_router)
app.include_router(generate_script_router)
app.include_router(voice_router)
app.include_router(followup_router)
app.include_router(reference_router)
app.include_router(schedules_router)


@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable_handler(
    _request: Request,
    _exc: DatabaseUnavailableError,
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "正式資料目前無法查詢，請稍後重試。"},
    )


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
def on_startup() -> None:
    try:
        initialize_ai()
    except Exception as exc:
        logger.warning("AI initialization failed during startup; API fallback remains available: %s", exc)


@app.on_event("startup")
async def start_tts_cleanup() -> None:
    async def sweep():
        while True:
            await asyncio.sleep(60)
            tts_cache.prune()
    app.state.tts_sweeper = asyncio.create_task(sweep())


@app.on_event("shutdown")
async def stop_tts_cleanup() -> None:
    app.state.tts_sweeper.cancel()
    with suppress(asyncio.CancelledError):
        await app.state.tts_sweeper
    for session_id in list(tts_cache.sessions):
        tts_cache.cleanup(session_id)
