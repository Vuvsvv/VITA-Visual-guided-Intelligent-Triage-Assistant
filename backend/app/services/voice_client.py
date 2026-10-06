"""
語音 Gateway 客戶端
==================
呼叫語音 gateway 的統一入口（POST /api/process）。

這是一個獨立的工具類別，不依賴後端任何現有邏輯。
後端需要語音時 import 來用即可。

gateway 提供四個服務（用 service_type 區分）：
  taiwanese_asr  台語語音 → 中文字
  chinese_asr    國語語音 → 中文字
  taiwanese_tts  中文字 → 台語語音
  chinese_tts    中文字 → 國語語音

用法：
    client = VoiceGatewayClient(base_url="https://xxx.ngrok-free.dev",
                                api_key="your_key")

    # 語音轉文字（lang 決定台語或國語）
    text = client.transcribe(audio_bytes, filename="a.wav", lang="taiwanese")

    # 文字轉語音（回傳 wav bytes）
    wav = client.synthesize("你好", lang="chinese")
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Literal

import requests

logger = logging.getLogger(__name__)

Lang = Literal["taiwanese", "chinese"]


class VoiceGatewayError(Exception):
    """gateway 呼叫失敗時拋出"""

    def __init__(
        self,
        message: str,
        *,
        upstream_status: str = "error",
        exception_type: str | None = None,
        exception_message: str | None = None,
    ):
        super().__init__(message)
        self.upstream_status = upstream_status
        self.exception_type = exception_type or type(self).__name__
        self.exception_message = exception_message or message


@dataclass(frozen=True)
class TtsGatewayResult:
    audio_base64: str
    audio_format: str = "wav"


class VoiceGatewayClient:
    """語音 gateway 客戶端"""

    # lang → service_type 對照
    _ASR_SERVICE = {
        "taiwanese": "taiwanese_asr",
        "chinese": "chinese_asr",
    }
    _TTS_SERVICE = {
        "taiwanese": "taiwanese_tts",
        "chinese": "chinese_tts",
    }

    @classmethod
    def tts_service_type(cls, lang: str) -> str | None:
        return cls._TTS_SERVICE.get(lang)

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        is_ngrok: bool = True,
    ):
        """
        Args:
            base_url: gateway 網址（例如 https://xxx.ngrok-free.dev，結尾不用斜線）
            api_key: gateway 的 X-API-Key
            timeout: 請求逾時秒數（TTS 在 CPU 上較慢，預設給寬一點）
            is_ngrok: 走 ngrok 時自動帶上跳過警告頁的 header
        """
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/api/process"
        self.timeout = timeout

        self.headers = {"X-API-Key": api_key}
        if is_ngrok:
            self.headers["ngrok-skip-browser-warning"] = "true"

    # ── 語音轉文字（ASR）──
    def transcribe(
        self,
        audio: bytes,
        filename: str = "audio.wav",
        lang: Lang = "taiwanese",
    ) -> str:
        """
        語音 → 中文字

        Args:
            audio: 音訊檔的 bytes
            filename: 檔名（gateway 用來判斷格式）
            lang: "taiwanese"（台語）或 "chinese"（國語）

        Returns:
            辨識出的中文字串

        Raises:
            VoiceGatewayError: 呼叫失敗或 gateway 回報錯誤
        """
        service_type = self._ASR_SERVICE.get(lang)
        if not service_type:
            raise VoiceGatewayError(f"不支援的語言: {lang}")

        try:
            resp = requests.post(
                self.endpoint,
                headers=self.headers,
                files={"file": (filename, audio, "audio/wav")},
                data={"service_type": service_type},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise VoiceGatewayError(
                f"語音辨識服務目前無法連線（{type(e).__name__}），請稍後再試或確認 Voice Gateway 已啟動。",
                upstream_status="connection_error",
                exception_type=type(e).__name__,
                exception_message=str(e),
            ) from e

        payload = self._parse(resp, service_type)
        text = payload.get("data", {}).get("text", "")
        if not text:
            logger.warning("ASR 回傳空字串 service_type=%s", service_type)
        return text.strip()

    # ── 文字轉語音（TTS）──
    def synthesize(
        self,
        text: str,
        lang: Lang = "chinese",
        speed: float = 1.0,
    ) -> bytes:
        """
        中文字 → 語音

        Args:
            text: 要合成的中文字
            lang: "taiwanese"（台語）或 "chinese"（國語）
            speed: 語速

        Returns:
            wav 音訊的 bytes（可直接寫檔或回傳給 Android 播放）

        Raises:
            VoiceGatewayError: 呼叫失敗或 gateway 回報錯誤
        """
        result = self.synthesize_to_base64(text, lang=lang, speed=speed)
        try:
            return base64.b64decode(result.audio_base64)
        except Exception as e:
            raise VoiceGatewayError(f"音訊 base64 解碼失敗: {e}") from e

    # ── 文字轉語音，回傳 base64 字串（給 Android 直接用）──
    def synthesize_base64(
        self,
        text: str,
        lang: Lang = "chinese",
        speed: float = 1.0,
    ) -> str:
        """
        同 synthesize，但直接回傳 base64 字串。
        給 /voice/chat 端點回傳給 Android 時用，省去再編碼一次。
        """
        return self.synthesize_to_base64(text, lang=lang, speed=speed).audio_base64

    def synthesize_to_base64(
        self,
        text: str,
        lang: Lang = "chinese",
        speed: float = 1.0,
    ) -> TtsGatewayResult:
        service_type = self._TTS_SERVICE.get(lang)
        if not service_type:
            raise VoiceGatewayError(f"不支援的語言: {lang}")
        if not text.strip():
            raise VoiceGatewayError("text 不可為空")

        data = {
            "service_type": service_type,
            "text_input": text,
            "language": lang,
        }
        if speed != 1.0:
            import json
            data["extra_params"] = json.dumps({"speed": speed})

        logger.info(
            "TTS gateway request service_type=%s lang=%s text_len=%s",
            service_type,
            lang,
            len(text),
        )
        try:
            resp = requests.post(
                self.endpoint,
                headers=self.headers,
                data=data,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            logger.warning("TTS gateway connection failed service_type=%s error_type=%s", service_type, type(e).__name__)
            raise VoiceGatewayError(
                f"語音播放服務目前無法連線（{type(e).__name__}），請稍後再試或確認 Voice Gateway 已啟動。",
                upstream_status="connection_error",
                exception_type=type(e).__name__,
                exception_message=str(e),
            ) from e

        payload = self._parse(resp, service_type)
        data = payload.get("data", {}) or {}
        logger.info(
            "TTS gateway response service_type=%s success=%s",
            service_type,
            payload.get("success") is True,
        )
        if not isinstance(data, dict):
            raise VoiceGatewayError(f"TTS 回傳 data 格式錯誤 service_type={service_type}: {type(data).__name__}")
        if data.get("success") is False and data.get("error"):
            raise VoiceGatewayError(
                f"TTS gateway 服務失敗 service_type={service_type}"
            )

        raw_audio_b64 = data.get("audio_base64") or data.get("wav_base64") or data.get("base64") or ""
        audio_b64, data_url_format = _strip_data_url_prefix(raw_audio_b64)
        audio_format = _normalize_audio_format(
            data_url_format or data.get("audio_format") or data.get("format") or data.get("mime_type") or "wav"
        )
        if not audio_b64:
            raise VoiceGatewayError(
                f"TTS 未回傳音訊 service_type={service_type}，"
                "gateway response is missing audio"
            )
        return TtsGatewayResult(audio_base64=str(audio_b64), audio_format=audio_format)

    # ── 健康檢查 ──
    def health(self) -> dict:
        """
        檢查 gateway 和各服務狀態。
        回傳 gateway 的 /health JSON。
        """
        try:
            resp = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise VoiceGatewayError(
                f"Voice Gateway 健康檢查無法連線（{type(e).__name__}）。",
                upstream_status="connection_error",
                exception_type=type(e).__name__,
                exception_message=str(e),
            ) from e

    # ── 內部：解析 gateway 回應 ──
    def _parse(self, resp: requests.Response, service_type: str) -> dict:
        """檢查 HTTP 狀態 + gateway 的 success 欄位"""
        if resp.status_code != 200:
            logger.warning("gateway HTTP error service_type=%s status=%s", service_type, resp.status_code)
            raise VoiceGatewayError(
                f"gateway HTTP {resp.status_code} "
                f"(service_type={service_type})",
                upstream_status=str(resp.status_code),
                exception_type="HTTPError",
                exception_message=f"upstream returned HTTP {resp.status_code}",
            )

        try:
            payload = resp.json()
        except ValueError as e:
            raise VoiceGatewayError(
                "gateway 回應非 JSON"
            ) from e

        if not payload.get("success", False):
            logger.warning(
                "gateway returned failure service_type=%s",
                service_type,
            )
            raise VoiceGatewayError(
                f"gateway 回報失敗 (service_type={service_type})"
            )

        return payload


def _normalize_audio_format(value: object) -> str:
    text = str(value or "wav").strip().lower()
    text = text.split(";")[0]
    if "/" in text:
        text = text.split("/")[-1]
    if text.startswith("."):
        text = text[1:]
    if text in {"wave", "x-wav"}:
        return "wav"
    if text in {"mpeg", "mpga", "mpg"}:
        return "mp3"
    return text or "wav"


def _strip_data_url_prefix(value: object) -> tuple[str, str | None]:
    text = str(value or "").strip()
    if not text.startswith("data:"):
        return text, None
    header, sep, payload = text.partition(",")
    if not sep:
        return text, None
    audio_format = None
    if ";base64" in header:
        mime_type = header.removeprefix("data:").split(";")[0]
        audio_format = _normalize_audio_format(mime_type)
    return payload.strip(), audio_format


# ── 單獨測試用 ──
# 直接執行這個檔案就能測 client 能不能連上你的 gateway：
#   python voice_client.py
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # ↓↓↓ 改成你的 ngrok 網址和 key ↓↓↓
    GATEWAY_URL = "https://你的網址.ngrok-free.dev"
    API_KEY = "你的key"

    print("=" * 50)
    print("語音 Gateway Client 連線測試")
    print("=" * 50)

    client = VoiceGatewayClient(base_url=GATEWAY_URL, api_key=API_KEY)

    # 1. 健康檢查
    print("\n[1] 健康檢查...")
    try:
        h = client.health()
        print(f"  gateway: {h.get('gateway')}")
        for name, status in h.get("services", {}).items():
            mark = "✅" if status == "healthy" else "❌"
            print(f"  {mark} {name}: {status}")
    except VoiceGatewayError as e:
        print(f"  ❌ 失敗: {e}")
        sys.exit(1)

    # 2. TTS 測試（國語）
    print("\n[2] 國語 TTS 測試...")
    try:
        wav = client.synthesize("你好，這是測試", lang="chinese")
        with open("test_chinese_tts.wav", "wb") as f:
            f.write(wav)
        print(f"  ✅ 國語語音已存成 test_chinese_tts.wav ({len(wav)} bytes)")
    except VoiceGatewayError as e:
        print(f"  ❌ 失敗: {e}")

    # 3. TTS 測試（台語）
    print("\n[3] 台語 TTS 測試...")
    try:
        wav = client.synthesize("你好", lang="taiwanese")
        with open("test_taiwanese_tts.wav", "wb") as f:
            f.write(wav)
        print(f"  ✅ 台語語音已存成 test_taiwanese_tts.wav ({len(wav)} bytes)")
    except VoiceGatewayError as e:
        print(f"  ❌ 失敗: {e}")

    # 4. ASR 測試（如果有測試音檔）
    print("\n[4] ASR 測試（需要 test_input.wav）...")
    import os
    if os.path.exists("test_input.wav"):
        try:
            with open("test_input.wav", "rb") as f:
                audio = f.read()
            text = client.transcribe(audio, lang="taiwanese")
            print("  ✅ 辨識完成（結果內容不寫入 log）")
        except VoiceGatewayError as e:
            print(f"  ❌ 失敗: {e}")
    else:
        print("  （跳過，沒有 test_input.wav）")

    print("\n測試完成。")
