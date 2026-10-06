"""Chatterbox 台語 TTS 引擎（HTTP）

取代原本 GPT-SoVITS api_v2 的位置：
    main.py（:8003，可在 Docker）  ──POST /tts──▶  本引擎（:9881，宿主機 GPU）

必須在 `cb` conda 環境、有 GPU、能存取 chatterbox-finetuning 目錄的機器上執行：

    conda activate cb
    CB_ROOT=~/chatterbox-finetuning python services/taiwanese_tts/engine/cb_server.py

⚠️ 本引擎沒有任何認證。預設只綁 127.0.0.1；台語 TTS 改用 Docker 跑時必須設
   CB_HOST=0.0.0.0 讓容器連得到，此時務必用防火牆擋掉 9881（sudo ufw deny 9881/tcp）。

環境變數（括號內為預設值）：
    CB_ROOT        chatterbox-finetuning 目錄（~/chatterbox-finetuning）
    CB_ADAPTER     LoRA 權重，相對於 CB_ROOT（chatterbox_output_sandhi/new_lang_adapter，即模型 B）
    CB_REF         參考音，相對於 CB_ROOT（speaker_reference/ref3.wav）
    CB_SANDHI      輸入是否先做變調轉換（1；模型 B 必須為 1，模型 A 設 0）
    CB_CANDIDATES  每段生成幾次、挑長度居中者（1；設 3 可降低開頭重複，但耗時約 3 倍）
    CB_SEED        起始 seed（2）
    CB_HOST / CB_PORT  綁定位址（127.0.0.1 / 9881）
"""
import asyncio
import io
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

ENGINE_DIR = Path(__file__).resolve().parent
CB_ROOT = Path(os.path.expanduser(os.getenv("CB_ROOT", "~/chatterbox-finetuning"))).resolve()

# 引擎目錄優先：使用本 repo 版控的 sandhi.py / cb_infer.py，而不是 CB_ROOT 裡的副本
sys.path[:0] = [str(ENGINE_DIR), str(CB_ROOT)]
os.chdir(CB_ROOT)   # 工具包的 TrainConfig 使用相對路徑（./pretrained_models）

import soundfile as sf                          # noqa: E402
from fastapi import FastAPI, HTTPException      # noqa: E402
from fastapi.responses import Response          # noqa: E402
from pydantic import BaseModel                  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cb_engine")

ADAPTER = os.getenv("CB_ADAPTER", "chatterbox_output_sandhi/new_lang_adapter")
REF = os.getenv("CB_REF", "speaker_reference/ref3.wav")
SANDHI = os.getenv("CB_SANDHI", "1") == "1"
CANDIDATES = max(1, int(os.getenv("CB_CANDIDATES", "1")))
SEED = int(os.getenv("CB_SEED", "2"))

_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")

app = FastAPI(title="Chatterbox 台語引擎", version="1.0.0")
_tts = None
_lock = asyncio.Lock()   # 模型推論不是執行緒安全的，一次只跑一個請求


class TTSRequest(BaseModel):
    text: str                    # 本調數字調台羅
    seed: Optional[int] = None


@app.on_event("startup")
async def startup():
    global _tts
    from cb_infer import TaigiTTS
    logger.info(f"CB_ROOT={CB_ROOT}  adapter={ADAPTER}  ref={REF}  "
                f"sandhi={SANDHI}  candidates={CANDIDATES}  seed={SEED}")
    _tts = await asyncio.to_thread(TaigiTTS, ADAPTER, REF, SANDHI)
    logger.info(f"✅ 模型載入完成（取樣率 {_tts.sr}Hz）")


def _synthesize(text: str, seed: int):
    if CANDIDATES == 1:
        return _tts(text, seed=seed)
    # 開頭重複會讓長度變長、截斷會變短：取長度居中者，兩種異常都會被排除
    cands = [_tts(text, seed=seed + i) for i in range(CANDIDATES)]
    med = sorted(len(c) for c in cands)[len(cands) // 2]
    return min(cands, key=lambda c: abs(len(c) - med))


@app.post("/tts")
async def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "text 不可為空")
    if _CJK.search(text):
        raise HTTPException(400, "引擎只接受台羅數字調，輸入含有漢字")
    if _tts is None:
        raise HTTPException(503, "模型尚未載入完成")
    seed = SEED if req.seed is None else req.seed
    async with _lock:
        wav = await asyncio.to_thread(_synthesize, text, seed)
    buf = io.BytesIO()
    sf.write(buf, wav, _tts.sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Chatterbox 台語引擎（模型 B）",
        "ready": _tts is not None,
        "adapter": ADAPTER,
        "ref": REF,
        "sandhi": SANDHI,
        "candidates": CANDIDATES,
        "seed": SEED,
        "sample_rate": _tts.sr if _tts else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("CB_HOST", "127.0.0.1"),
                port=int(os.getenv("CB_PORT", "9881")))
