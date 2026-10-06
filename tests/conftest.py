"""測試共用設定：載入一個乾淨的 gateway 實例（可注入設定檔與環境變數）。"""
import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE_CONFIG = {
    "services": {
        "taiwanese_asr": {
            "name": "台語語音識別",
            "description": "台語語音 → 中文字",
            "version": "2.0.0",
            "enabled": True,
            "endpoint": "http://asr.test",
            "health_check": "/",
            "timeout": 30,
            "input_type": "audio",
            "output_type": "text",
            "forward_path": "/api/transcribe",
            "file_field": "audio_file",
            "models": ["MediaTek-Research/Breeze-ASR-26"],
        },
        "taiwanese_tts": {
            "name": "台語文字轉語音",
            "description": "中文字 → 台語語音",
            "version": "5.0.0",
            "enabled": True,
            "endpoint": "http://tts.test",
            "health_check": "/",
            "timeout": 30,
            "input_type": "text",
            "output_type": "audio",
            "forward_path": "/api/synthesize",
            "file_field": None,
            "models": ["gemini-2.0-flash", "Chatterbox"],
        },
        "chinese_tts": {
            "name": "中文文字轉語音",
            "description": "中文字 → 台灣國語語音",
            "version": "4.0.0",
            "enabled": False,
            "endpoint": "http://ctts.test",
            "health_check": "/",
            "timeout": 30,
            "input_type": "text",
            "output_type": "audio",
            "forward_path": "/api/synthesize",
            "file_field": None,
            "models": ["MediaTek-Research/BreezyVoice"],
        },
    },
    "global": {"gateway_port": 8000},
}

GATEWAY_ENV_KEYS = (
    "GATEWAY_CONFIG",
    "API_KEY",
    "API_KEYS",
    "ALLOWED_ORIGINS",
    "RATE_LIMIT",
    "RATE_WINDOW",
    "MAX_FILE_SIZE_MB",
    "TRUSTED_PROXY_HOPS",
    "RATE_LIMIT_CLEANUP_INTERVAL",
)


@pytest.fixture
def make_gateway(tmp_path, monkeypatch):
    """回傳一個工廠：make_gateway(config=..., env=...) → 重新載入的 gateway module。"""

    def _make(config=None, env=None):
        cfg = copy.deepcopy(BASE_CONFIG) if config is None else config
        cfg_path = tmp_path / "services_config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

        for key in GATEWAY_ENV_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("GATEWAY_CONFIG", str(cfg_path))
        monkeypatch.setenv("API_KEY", "sk-test-key")
        monkeypatch.setenv("ALLOWED_ORIGINS", "https://app.example.com")
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, str(v))

        sys.modules.pop("gateway", None)
        sys.modules.pop("security", None)
        import gateway  # noqa: WPS433 - 需要在環境變數設定後才 import

        return gateway

    return _make


@pytest.fixture
def wav_bytes():
    """一個帶有合法 RIFF magic bytes 的最小 WAV。"""
    return b"RIFF" + b"\x00" * 4 + b"WAVEfmt " + b"\x00" * 64
