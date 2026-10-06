"""
中文文字轉語音服務 (BreezyVoice)
================================
FP16（預設開）+ torch.compile（選用）+ 長句切割逐段合成
模型: MediaTek-Research/BreezyVoice-300M
端口: 8004

v4.3.0：修正 FP16 未生效 —— 舊版對包裝類別呼叫 .half() 不會生效，
        三個子模組實際一直是 FP32。改為逐一轉換並驗證。
        可用 FP16_MODULES 指定範圍（預設 llm,flow，不含聲碼器）。

v4.2.0：說話者注音改為啟動時預算，省下每次請求約 1.4 秒。

v4.1.0：新增 M4A(AAC) 輸出並設為預設。
        BreezyVoice 本來就是 22050Hz，不需降採樣，僅做壓縮。
        體積約降至 WAV 的 1/7，縮短傳輸與播放前的等待。
        回傳新增 audio_format 欄位 —— 後端靠它決定副檔名，
        少了這個欄位，AAC 資料會被存成 .wav 而播不出來。
        要回到未壓縮輸出：TTS_OUTPUT_FORMAT=wav

v4.0.0：本檔原名 main_v2.py。舊的 3.0.0 實作（main.py）已刪除，
        兩份並存造成每個修正都要做兩遍，而且部署的一直是沒優化的那份。

註：
  - 切句由 MAX_CHUNK_LEN 控制，逐段合成後串接（不是平行，GPU 只有一條 worker）
  - cudnn benchmark 預設關閉：TTS 的輸入長度每次都不同，autotune 幫助有限甚至變慢；
    需要時設 USE_CUDNN_BENCHMARK=1
  - speed 是否生效取決於底層 CosyVoice 版本，啟動時會探測，可在 GET / 查 speed_supported
"""

import os
import sys
import io
import re
import base64
import logging
import asyncio
import time
import tempfile
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="中文文字轉語音服務 (BreezyVoice)",
    description="中文文字 → 台灣國語語音（語音克隆）",
    version="4.5.0",
)

# ── 全域變數 ─────────────────────────────────────────────────────────────────
cosyvoice            = None
bopomofo_converter   = None
get_bopomofo_rare_fn = None
torchaudio           = None
prompt_speech_16k    = None          # 預先載好，避免每次 I/O
speaker_bopomo_cache = None          # 說話者注音，啟動時算一次（v4.2.0）

BREEZY_REPO          = os.getenv("BREEZY_REPO_PATH", os.path.expanduser("~/BreezyVoice"))
MODEL_PATH           = os.getenv("BREEZY_MODEL_PATH", "MediaTek-Research/BreezyVoice-300M")
DEFAULT_SPEAKER_AUDIO= os.getenv("DEFAULT_SPEAKER_AUDIO", "")
DEFAULT_SPEAKER_TEXT = os.getenv(
    "DEFAULT_SPEAKER_TEXT",
    "在密碼學中，加密是將明文資訊改變為難以讀取的密文內容，使之不可讀的方法。"
    "只有擁有解密方法的對象，經由解密過程才能將密文還原為正常可讀的內容。"
)

SAMPLE_RATE   = 22050
MAX_CHUNK_LEN = int(os.getenv("MAX_CHUNK_LEN", "30"))  # 切句最大字數
# 半精度有兩種做法：
#   autocast（預設）── 不改模型權重，PyTorch 在每個運算當下決定精度，
#                      並自動處理型別轉換，不會有 Float/Half 混用的錯誤。
#   直接轉權重      ── 速度與省記憶體幅度較大，但輸入端（參考音訊、
#                      中間張量）仍是 FP32，實測會拋
#                      "mat1 and mat2 must have the same dtype"。
#                      要用得先把所有輸入一併轉換，牽涉 CosyVoice 內部多處。
USE_AUTOCAST  = os.getenv("USE_AUTOCAST", "1") == "1"
USE_FP16      = os.getenv("USE_FP16", "0") == "1"
# 要轉半精度的子模組。預設不含 hift：HiFi-GAN 聲碼器對數值範圍敏感，
# FP16 有機率產生雜音或爆音。確認音質無虞後可設 "llm,flow,hift"。
FP16_MODULES  = [m.strip() for m in os.getenv("FP16_MODULES", "llm,flow").split(",") if m.strip()]
USE_COMPILE   = os.getenv("USE_COMPILE", "0") == "1"   # 首次慢 30s，預設關
# TTS 每次的輸入長度都不一樣，cudnn autotune 反而可能變慢，所以預設關閉
USE_CUDNN_BENCHMARK = os.getenv("USE_CUDNN_BENCHMARK", "0") == "1"

# ── 輸出格式設定（v4.1.0 新增）────────────────────────────
# 本模型輸出已是 22050Hz，不需降採樣，僅做 AAC 壓縮。
# 啟動後先跑一次推論。首次推論要配置顯存並編譯運算核心，實測比穩定後
# 慢一個量級；預熱把這筆一次性成本挪到沒人等待的啟動階段。
WARMUP_ENABLED = os.getenv("TTS_WARMUP", "1") == "1"
WARMUP_TEXT = os.getenv("TTS_WARMUP_TEXT", "你好")

DEFAULT_FORMAT = os.getenv("TTS_OUTPUT_FORMAT", "m4a").lower()
AAC_BITRATE    = os.getenv("AAC_BITRATE", "64k")   # 語音用 64k 已充裕

_executor = ThreadPoolExecutor(max_workers=1)  # GPU bound，單執行緒避免競爭
_speed_supported: Optional[bool] = None        # 由 initialize_model() 探測


# ── Pydantic ─────────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text:  str
    speed: float = 1.0
    # v4.1.0：預設值由 TTS_OUTPUT_FORMAT 決定（預設 m4a）
    format: str = DEFAULT_FORMAT      # "wav" 或 "m4a"


class TTSResponse(BaseModel):
    success:        bool
    original_text:  str
    audio_base64:   Optional[str]   = None
    # v4.1.0 新增：後端與前端靠這個欄位決定副檔名，缺了會播不出來
    audio_format:   str             = "wav"
    audio_duration: Optional[float] = None
    sample_rate:    int             = SAMPLE_RATE
    elapsed_ms:     Optional[int]   = None
    error:          Optional[str]   = None


# ── 句子切割 ─────────────────────────────────────────────────────────────────
def split_text(text: str, max_len: int = MAX_CHUNK_LEN) -> List[str]:
    """
    依中文標點切句，每段不超過 max_len 字。
    短句合批以減少 overhead，超長句強制在逗頓號後再切。
    """
    parts = re.split(r'(?<=[。！？；\n])', text)
    chunks, buf = [], ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) > max_len:
            sub_parts = re.split(r'(?<=[，、])', part)
            for sp in sub_parts:
                sp = sp.strip()
                if not sp:
                    continue
                if len(buf) + len(sp) <= max_len:
                    buf += sp
                else:
                    if buf:
                        chunks.append(buf)
                    buf = sp
        else:
            if len(buf) + len(part) <= max_len:
                buf += part
            else:
                if buf:
                    chunks.append(buf)
                buf = part

    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


# ── 模型初始化 ────────────────────────────────────────────────────────────────
def initialize_model():
    global cosyvoice, bopomofo_converter, get_bopomofo_rare_fn
    global torchaudio, DEFAULT_SPEAKER_AUDIO, prompt_speech_16k

    t0 = time.time()
    logger.info("⏳ 正在載入 BreezyVoice 模型...")

    if not os.path.exists(BREEZY_REPO):
        raise RuntimeError(
            f"找不到 BreezyVoice repo: {BREEZY_REPO}\n"
            f"請設定 BREEZY_REPO_PATH 指向你 clone 的位置"
        )

    if BREEZY_REPO not in sys.path:
        sys.path.insert(0, BREEZY_REPO)
    matcha = os.path.join(BREEZY_REPO, "third_party", "Matcha-TTS")
    if os.path.exists(matcha) and matcha not in sys.path:
        sys.path.insert(0, matcha)

    original_cwd = os.getcwd()
    os.chdir(BREEZY_REPO)
    try:
        import torch
        import torchaudio as _ta
        torchaudio = _ta

        from single_inference import CustomCosyVoice, get_bopomofo_rare
        from g2pw import G2PWConverter

        cosyvoice            = CustomCosyVoice(MODEL_PATH)
        bopomofo_converter   = G2PWConverter()
        get_bopomofo_rare_fn = get_bopomofo_rare

        # ── 優化 1：FP16（速度 +30~50%，記憶體減半）─────────────────────────
        # 舊版寫 cosyvoice.model.half()，但 CosyVoiceModel 是包裝類別而非 nn.Module，
        # 那行不會報錯也不會生效 —— log 印「已啟用」，/debug 卻顯示三個子模組
        # 全是 float32，等於一直在跑全精度。故改為逐一轉換子模組並驗證結果。
        if USE_FP16 and torch.cuda.is_available():
            for name in FP16_MODULES:
                sub = getattr(cosyvoice.model, name, None)
                if sub is None:
                    logger.warning(f"FP16：找不到子模組 {name}，略過")
                    continue
                if not hasattr(sub, "half"):
                    logger.warning(f"FP16：{name} 不支援 half()，略過")
                    continue
                try:
                    setattr(cosyvoice.model, name, sub.half())
                except Exception as e:
                    logger.warning(f"FP16：{name} 轉換失敗，維持 FP32：{e}")

            # 轉完必須驗證：只看有沒有拋例外會重蹈舊版覆轍
            states = []
            for name in ("llm", "flow", "hift"):
                sub = getattr(cosyvoice.model, name, None)
                if sub is None:
                    continue
                try:
                    dt = str(next(sub.parameters()).dtype).replace("torch.", "")
                except Exception:
                    dt = "unknown"
                states.append(f"{name}={dt}")
            logger.info(f"✅ 精度狀態：{' | '.join(states)}")

        # ── 優化 2：cudnn benchmark（固定輸入尺寸時 +5~15%）─────────────────
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = USE_CUDNN_BENCHMARK
            logger.info(
                "✅ cudnn benchmark 已啟用"
                if USE_CUDNN_BENCHMARK
                else "⏭️  cudnn benchmark 未啟用（設 USE_CUDNN_BENCHMARK=1 可開啟）"
            )

        # ── 優化 3：torch.compile（選用，首次慢 30s，之後 +20~40%）─────────
        if USE_COMPILE and torch.cuda.is_available():
            try:
                cosyvoice.model = torch.compile(
                    cosyvoice.model, mode="reduce-overhead"
                )
                logger.info("✅ torch.compile 已啟用（首次推論會預熱）")
            except Exception as e:
                logger.warning(f"torch.compile 啟用失敗：{e}")

    finally:
        os.chdir(original_cwd)

    # 預設說話者音訊 → 預先載到記憶體（省掉每次 I/O）
    if not DEFAULT_SPEAKER_AUDIO:
        example = os.path.join(BREEZY_REPO, "data", "example.wav")
        if os.path.exists(example):
            DEFAULT_SPEAKER_AUDIO = example
        else:
            logger.warning("找不到 data/example.wav，請在請求時上傳參考音訊")

    if DEFAULT_SPEAKER_AUDIO:
        from cosyvoice.utils.file_utils import load_wav
        prompt_speech_16k = load_wav(DEFAULT_SPEAKER_AUDIO, 16000)
        logger.info(f"✅ 說話者音訊已預載：{DEFAULT_SPEAKER_AUDIO}")

    # ── 說話者注音預先轉換（v4.2.0）─────────────────────────────────
    # DEFAULT_SPEAKER_TEXT 有 60 多字且永遠不變，原本每次請求都重跑一次
    # G2PW 注音轉換，實測佔掉每次請求約 1.4 秒。啟動時算一次即可。
    global speaker_bopomo_cache
    if cosyvoice is not None:
        _ts = time.time()
        _norm = cosyvoice.frontend.text_normalize_new(DEFAULT_SPEAKER_TEXT, split=False)
        speaker_bopomo_cache = get_bopomofo_rare_fn(_norm, bopomofo_converter)
        logger.info(f"✅ 說話者注音已預算（{time.time()-_ts:.2f}s，之後每次請求省下這段）")

    # 這個 CosyVoice 版本支不支援 speed？不支援的話要明確報錯，不能默默忽略（C19）
    global _speed_supported
    _speed_supported = _detect_speed_support()
    logger.info(f"speed 參數支援: {_speed_supported}")

    elapsed = time.time() - t0
    logger.info(f"🚀 BreezyVoice 載入完成（{elapsed:.1f} 秒）")
    logger.info(f"輸出格式: {DEFAULT_FORMAT}（{SAMPLE_RATE}Hz, {AAC_BITRATE}）")


def _detect_speed_support() -> bool:
    """
    檢查底層 CosyVoice 的推論函式吃不吃 speed 參數。
    只認「簽章裡明確有 speed」。**kwargs 不算 —— 那種情況參數照收但可能被丟掉，
    正是 C19 要消滅的「靜默吃掉」行為。
    """
    import inspect
    try:
        sig = inspect.signature(cosyvoice.inference_zero_shot_no_normalize)
    except (TypeError, ValueError):
        return False
    return "speed" in sig.parameters


def speed_supported() -> bool:
    return bool(_speed_supported)


# ── 核心合成 ──────────────────────────────────────────────────────────────────
def _synth_chunk(text_bopomo: str, speaker_bopomo: str, speed: float = 1.0) -> np.ndarray:
    import torch

    kwargs = {"speed": speed} if (_speed_supported and speed != 1.0) else {}

    if USE_AUTOCAST and torch.cuda.is_available():
        # autocast 只在安全的運算（矩陣乘法、卷積）使用半精度，
        # 正規化與 softmax 等數值敏感處自動維持 FP32，並在邊界自動轉型，
        # 因此不需要事先把參考音訊或中間張量轉成 Half。
        with torch.autocast("cuda", dtype=torch.float16):
            output = cosyvoice.inference_zero_shot_no_normalize(
                text_bopomo, speaker_bopomo, prompt_speech_16k, **kwargs
            )
    else:
        output = cosyvoice.inference_zero_shot_no_normalize(
            text_bopomo, speaker_bopomo, prompt_speech_16k, **kwargs
        )

    speech = output["tts_speech"]
    # autocast 下輸出可能是 half，轉回 float32 再進 numpy，
    # 否則後續 np.clip 與 int16 轉換的精度會受影響
    if speech.dtype != torch.float32:
        speech = speech.float()
    waveform = speech.numpy()
    return waveform.squeeze() if waveform.ndim > 1 else waveform


def synthesize(text: str, speed: float = 1.0):
    """長句切割 → 逐句合成 → 合併，回傳 (waveform, sample_rate, elapsed_ms)"""
    t0 = time.time()

    # 說話者注音：啟動時已算好，直接取用（v4.2.0）
    # 舊版在此重跑 normalize + bopomofo，每次請求多花約 1.4 秒
    if speaker_bopomo_cache is not None:
        speaker_bopomo = speaker_bopomo_cache
    else:
        logger.warning("說話者注音快取未建立，改為即時轉換（會較慢）")
        _norm = cosyvoice.frontend.text_normalize_new(DEFAULT_SPEAKER_TEXT, split=False)
        speaker_bopomo = get_bopomofo_rare_fn(_norm, bopomofo_converter)

    chunks = split_text(text)
    if not chunks:
        # split_text 現在真的會切了（見 C4），純標點 / 空白的輸入會切成空 list
        raise ValueError("文字切句後為空，請確認輸入不是只有空白或標點")
    logger.info(f"切成 {len(chunks)} 句")

    parts = []
    for i, chunk in enumerate(chunks):
        t_a = time.time()
        content_norm = cosyvoice.frontend.text_normalize_new(chunk, split=False)
        t_b = time.time()
        content_bopomo = get_bopomofo_rare_fn(content_norm, bopomofo_converter)
        t_c = time.time()
        waveform = _synth_chunk(content_bopomo, speaker_bopomo, speed)
        t_d = time.time()
        logger.info(f"normalize {t_b-t_a:.2f}s | bopomofo {t_c-t_b:.2f}s | inference {t_d-t_c:.2f}s")
        parts.append(waveform)

    combined   = np.concatenate(parts) if len(parts) > 1 else parts[0]
    elapsed_ms = int((time.time() - t0) * 1000)
    # 舊版這行把「音訊長度」寫成「總計」，與實際耗時混淆，故拆開標示
    logger.info(
        f"✅ 耗時 {elapsed_ms} ms ｜ 產出音訊 {len(combined)/SAMPLE_RATE:.2f}s"
    )
    return combined, SAMPLE_RATE, elapsed_ms


def waveform_to_wav_bytes(waveform: np.ndarray, sr: int) -> bytes:
    import scipy.io.wavfile as wav
    # 先 clip：超過 ±1.0 的樣本乘上 32767 之後會 int16 溢位翻號，聽起來就是爆音
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wav.write(buf, sr, pcm)
    buf.seek(0)
    return buf.read()


def waveform_to_m4a_bytes(waveform: np.ndarray, sr: int) -> bytes:
    """轉成 M4A(AAC)。本模型已是 22050Hz，維持原取樣率只做壓縮。

    ffmpeg 讀不了 pipe 進來的 m4a 輸出（需要 seek 寫 moov box），
    所以走暫存檔而非 stdout。
    """
    wav_bytes = waveform_to_wav_bytes(waveform, sr)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        t.write(wav_bytes)
        wav_path = t.name

    m4a_path = str(Path(wav_path).with_suffix(".m4a"))
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", wav_path,
                "-ar", str(sr), "-ac", "1",
                "-c:a", "aac", "-b:a", AAC_BITRATE,
                m4a_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg 轉檔失敗 (exit {result.returncode}): {result.stderr[:300]}"
            )
        with open(m4a_path, "rb") as f:
            return f.read()
    finally:
        for p in (wav_path, m4a_path):
            if os.path.exists(p):
                os.remove(p)


def _warmup():
    """啟動後跑一次短句推論，把首次呼叫的初始化成本吃掉。"""
    if not WARMUP_ENABLED or cosyvoice is None or prompt_speech_16k is None:
        return
    try:
        t0 = time.time()
        synthesize(WARMUP_TEXT, 1.0)
        logger.info(f"🔥 預熱完成（{time.time()-t0:.1f} 秒），首位使用者不必等待初始化")
    except Exception as e:
        # 預熱失敗不影響服務可用性，只是第一次請求會慢
        logger.warning(f"預熱失敗（不影響服務）：{e}")


# ── FastAPI 事件 ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, initialize_model)
    await loop.run_in_executor(_executor, _warmup)


def _check_speed(speed: float) -> None:
    """speed 不支援時要明講，不要收下參數然後靜默忽略（C19）。"""
    if speed != 1.0 and not speed_supported():
        raise HTTPException(
            400,
            "此 CosyVoice 版本不支援 speed 參數，請用 speed=1.0（可查 GET / 的 speed_supported）",
        )


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status":        "running",
        "service":       "中文文字轉語音 (BreezyVoice)",
        "model":         MODEL_PATH,
        "ready":         cosyvoice is not None,
        "autocast":      USE_AUTOCAST,
        "fp16":          USE_FP16,
        "fp16_modules":  FP16_MODULES,
        "warmup":        WARMUP_ENABLED,
        "compile":       USE_COMPILE,
        "cudnn_benchmark": USE_CUDNN_BENCHMARK,
        "max_chunk_len": MAX_CHUNK_LEN,
        "speed_supported": speed_supported(),
        "output_format": DEFAULT_FORMAT,
        "aac_bitrate":   AAC_BITRATE,
    }


@app.get("/debug")
async def debug():
    import onnxruntime

    info = {"onnx_providers": onnxruntime.get_available_providers(),
            "python":         sys.executable,      # ← 服務用的直譯器
            "onnx_file":      onnxruntime.__file__, # ← import 到哪個套件
        }
    for name in ("llm", "flow", "hift"):
        sub = getattr(cosyvoice.model, name, None)
        if sub is None:
            info[name] = "not found"
            continue
        try:
            p = next(sub.parameters())
            info[name] = {"device": str(p.device), "dtype": str(p.dtype)}
        except Exception as e:
            info[name] = f"error: {e}"
    return info


@app.post("/api/synthesize", response_model=TTSResponse)
async def api_synthesize_json(request: TTSRequest):
    """JSON body → base64 音訊（與 gateway 相容）"""
    if cosyvoice is None:
        raise HTTPException(503, "模型尚未載入，請稍候")
    if not request.text.strip():
        raise HTTPException(400, "text 不可為空")
    if prompt_speech_16k is None:
        raise HTTPException(400, "找不到預設說話者音訊")

    fmt = request.format.lower()
    if fmt not in ("wav", "m4a"):
        raise HTTPException(400, "format 只支援 wav 或 m4a")

    _check_speed(request.speed)

    try:
        loop = asyncio.get_event_loop()
        waveform, sr, elapsed_ms = await loop.run_in_executor(
            _executor, synthesize, request.text, request.speed
        )

        _t_enc = time.time()
        audio_bytes = (waveform_to_m4a_bytes(waveform, sr) if fmt == "m4a"
                       else waveform_to_wav_bytes(waveform, sr))
        logger.info(
            f"├ 編碼 {time.time()-_t_enc:.2f}s ｜ {fmt} "
            f"{len(audio_bytes)//1024}KB @ {sr}Hz"
        )

        return TTSResponse(
            success        = True,
            original_text  = request.text,
            audio_base64   = base64.b64encode(audio_bytes).decode(),
            audio_format   = fmt,
            audio_duration = len(waveform) / sr,
            sample_rate    = sr,
            elapsed_ms     = elapsed_ms,
        )
    except HTTPException:
        raise
    except Exception as e:
        # 內部失敗要回 5xx，不能回 200 + success:false（見審查報告 C18）
        logger.exception("合成失敗")
        return JSONResponse(
            status_code=500,
            content=TTSResponse(
                success=False, original_text=request.text, error=str(e)
            ).model_dump(),
        )


@app.post("/api/synthesize-stream")
async def api_synthesize_stream(
    text:  str   = Form(...),
    speed: float = Form(1.0),
):
    """Form body → 直接回傳 WAV 串流

    這個端點維持 WAV：它是給人工測試用的（curl 下載後直接播），
    壓縮反而增加除錯時的不確定性。
    """
    if cosyvoice is None:
        raise HTTPException(503, "模型尚未載入")
    if not text.strip():
        raise HTTPException(400, "text 不可為空")
    if prompt_speech_16k is None:
        raise HTTPException(400, "找不到預設說話者音訊")

    _check_speed(speed)

    try:
        loop = asyncio.get_event_loop()
        waveform, sr, _ = await loop.run_in_executor(
            _executor, synthesize, text, speed
        )
        wav_bytes = waveform_to_wav_bytes(waveform, sr)
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=breezyvoice.wav"},
        )
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004, workers=1)
