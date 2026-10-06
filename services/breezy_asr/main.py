"""
台語語音識別服務 (Breeze-ASR-26)
================================
台語語音 → 中文字（直接輸出，不需額外翻譯）
模型: MediaTek-Research/Breeze-ASR-26
基於 Whisper 微調，專為台語語音優化
端口: 8001

v2.3 變更：
  - 推論改到 ThreadPoolExecutor 執行，不再阻塞 event loop（原本連 / health check
    都會被推論卡住，網關 5 秒的 health probe 會把推論中的服務誤判成 unhealthy）
v2.2 變更：
  - 修正預熱邏輯：改用 3 秒噪音並帶上 output_scores，走與真實請求相同的路徑
v2.1 變更：
  - 新增 confidence 欄位（token log-prob → exp 換算為 0–1）
  - 啟動時預熱模型，消除首次請求的冷啟動延遲
  - 推論改為直接呼叫 model.generate（pipeline 無法輸出 scores）
"""
import asyncio
import os
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
import torch
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="台語語音識別服務 (Breeze-ASR-26)",
    description="台語語音 → 中文字",
    version="2.3.0",
)

MODEL_ID = "MediaTek-Research/Breeze-ASR-26"

processor = None
model = None
device = None
torch_dtype = None

# GPU bound，單執行緒同時也是序列化 GPU 存取的閘門
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")


class TranscriptionResponse(BaseModel):
    text: str
    audio_duration: float
    confidence: Optional[float] = None
    success: bool
    error: Optional[str] = None


def initialize_model():
    global processor, model, device, torch_dtype
    try:
        logger.info(f"正在載入模型: {MODEL_ID} ...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        logger.info(f"裝置: {device}, 精度: {torch_dtype}")

        processor = WhisperProcessor.from_pretrained(MODEL_ID)
        model = WhisperForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        logger.info("模型載入成功！")

        logger.info("開始預熱 ...")
        # 用真實語音預熱：合成噪音會觸發 Whisper 的 no_speech 判定而提早結束，
        # 生成的 token 太少，走不到真實請求會經過的 kernel。
        warm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warmup.wav")
        if os.path.exists(warm_path):
            warm_audio, _ = librosa.load(warm_path, sr=16000)
        else:
            logger.warning("找不到 warmup.wav，改用合成噪音（預熱效果有限）")
            warm_audio = (np.random.randn(16000 * 3) * 0.01).astype(np.float32)
        warm_feat = processor(
            warm_audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device).to(torch_dtype)
        with torch.no_grad():
            model.generate(
                warm_feat,
                return_dict_in_generate=True,
                output_scores=True,
            )
        logger.info("預熱完成，服務就緒")

    except Exception as e:
        logger.error(f"模型載入失敗: {e}")
        raise


def compute_confidence(sequences, scores) -> Optional[float]:
    """
    從 generate 輸出的 scores 計算整句信心度。

    compute_transition_scores 回傳每個 token 被選中的 log probability，
    取平均後 exp 還原為 0–1 的機率值，代表模型對整句的平均把握程度。
    """
    try:
        transition = model.compute_transition_scores(
            sequences, scores, normalize_logits=True
        )
        row = transition[0]
        valid = row[torch.isfinite(row)]
        if valid.numel() == 0:
            return None
        conf = float(torch.exp(valid.mean().float()))
        return max(0.0, min(1.0, conf))
    except Exception as e:
        logger.warning(f"信心度計算失敗，回傳 None: {e}")
        return None


@app.on_event("startup")
async def startup():
    initialize_model()


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "台語語音識別 (Breeze-ASR-26)",
        "model": MODEL_ID,
        "ready": model is not None,
    }


def run_transcription(temp_path: str) -> Tuple[str, float, Optional[float]]:
    """同步推論：librosa.load 與 model.generate 都是阻塞呼叫，只能在 threadpool 裡跑。"""
    audio_array, sr = librosa.load(temp_path, sr=16000)
    audio_duration = len(audio_array) / sr
    logger.info(f"音訊: {audio_duration:.1f}s")

    input_features = processor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device).to(torch_dtype)

    with torch.no_grad():
        gen = model.generate(
            input_features,
            return_dict_in_generate=True,
            output_scores=True,
        )

    text = processor.batch_decode(
        gen.sequences, skip_special_tokens=True
    )[0].strip()

    confidence = compute_confidence(gen.sequences, gen.scores)
    return text, audio_duration, confidence


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    audio_file: UploadFile = File(..., description="台語音訊檔案"),
):
    """台語語音 → 中文字，附帶信心度"""
    temp_path = None
    try:
        if model is None:
            raise HTTPException(503, "模型尚未載入完成")

        contents = await audio_file.read()
        if len(contents) == 0:
            raise HTTPException(400, "檔案為空")

        suffix = os.path.splitext(audio_file.filename or ".wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            temp_path = tmp.name

        loop = asyncio.get_running_loop()
        text, audio_duration, confidence = await loop.run_in_executor(
            _executor, run_transcription, temp_path
        )

        conf_str = f"{confidence:.3f}" if confidence is not None else "N/A"
        logger.info(f"識別結果: {text} (信心度: {conf_str}, 檔案: {audio_file.filename})")

        return TranscriptionResponse(
            text=text,
            audio_duration=audio_duration,
            confidence=confidence,
            success=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        # 內部失敗要回 5xx，不能回 200 + success:false —— 網關與呼叫端才有辦法
        # 用 status code 做重試 / 告警（見審查報告 C18）
        logger.exception("識別失敗")
        return JSONResponse(
            status_code=500,
            content=TranscriptionResponse(
                text="", audio_duration=0, confidence=None, success=False, error=str(e)
            ).model_dump(),
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)