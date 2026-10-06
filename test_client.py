"""
AI 模型服務網關 - 測試客戶端
後端呼叫範例：只需改 service_type 就能切換不同模型
"""

import requests
import json
import sys
import os
import base64

GATEWAY = os.getenv("GATEWAY_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


def print_header(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def check_gateway():
    """確認網關是否運行"""
    try:
        r = requests.get(f"{GATEWAY}/", timeout=5)
        data = r.json()
        print(f"✓ 網關運行中")
        print(f"  可用服務: {data['available_services']}")
        return True
    except Exception as e:
        print(f"✗ 無法連線到網關 ({GATEWAY}): {e}")
        return False


def list_services():
    """列出所有服務狀態"""
    print_header("所有服務狀態")
    r = requests.get(f"{GATEWAY}/services", headers=HEADERS)
    for svc in r.json():
        icon = "🟢" if svc["status"] == "healthy" else "🔴" if svc["status"] == "unhealthy" else "⚪"
        print(f"  {icon} {svc['service_type']}")
        print(f"     名稱: {svc['name']}")
        print(f"     模型: {', '.join(svc['models'])}")
        print(f"     輸入→輸出: {svc['input_type']} → {svc['output_type']}")
        print(f"     狀態: {svc['status']}")
        print()


def test_asr(service_type: str, audio_path: str, language: str = None):
    """測試語音識別服務"""
    print_header(f"語音識別: {service_type}")

    if not os.path.exists(audio_path):
        print(f"  ✗ 檔案不存在: {audio_path}")
        return None

    with open(audio_path, "rb") as f:
        data = {"service_type": service_type}
        if language:
            data["language"] = language

        r = requests.post(
            f"{GATEWAY}/api/process",
            headers=HEADERS,
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            data=data
        )

    result = r.json()

    if result["success"]:
        print(f"  ✓ 成功！ ({result['processing_time_ms']}ms)")
        # ASR 回應欄位：text / audio_duration / confidence（台語）或 language（中文）
        print(f"  辨識結果: {result['data']['text']}")
        print(f"  完整回應:")
        print(json.dumps(result["data"], ensure_ascii=False, indent=4))
    else:
        # 失敗時 HTTP status code 已經帶了語意（400/404/413/502/503/504）
        print(f"  ✗ 失敗 (HTTP {r.status_code}): {result['error']}")

    return result


def test_tts(text: str, speed: float = 1.0, save_path: str = "output.wav"):
    """測試台語TTS服務"""
    print_header("台語 TTS: 中文 → 台語語音")
    print(f"  輸入文字: {text}")
    print(f"  語速: {speed}")

    r = requests.post(
        f"{GATEWAY}/api/process",
        headers=HEADERS,
        data={
            "service_type": "taiwanese_tts",
            "text_input": text,
            "extra_params": json.dumps({"speed": speed})
        }
    )

    result = r.json()

    if result["success"]:
        data = result["data"]
        print(f"  ✓ 成功！ ({result['processing_time_ms']}ms)")
        # v5.0.0：tailo_romanization 是送進模型的本調台羅；taigi_hanji 已 deprecated（一律空字串）
        print(f"  台羅: {data.get('tailo_romanization', 'N/A')}")
        print(f"  音訊時長: {data.get('audio_duration', 0):.2f} 秒")

        # 儲存音訊
        if data.get("audio_base64"):
            wav_bytes = base64.b64decode(data["audio_base64"])
            with open(save_path, "wb") as f:
                f.write(wav_bytes)
            print(f"  音訊已儲存: {save_path}")
    else:
        print(f"  ✗ 失敗 (HTTP {r.status_code}): {result['error']}")

    return result


def backend_example():
    """展示後端整合程式碼"""
    print_header("後端整合範例")

    code = '''
# ─── 你的後端只需要這些程式碼 ───

import os
import requests
import json
import base64

GATEWAY = "http://localhost:8000"
HEADERS = {"X-API-Key": os.environ["API_KEY"]}   # 缺這個會 401


def call_gateway(service_type: str, **kwargs):
    """統一的網關呼叫函式

    成功回 HTTP 200；失敗時 status code 已經帶了語意
    （400 輸入錯誤 / 404 未知服務 / 413 檔案過大 / 502、504 下游異常 / 503 未啟用），
    body 一律是同一個信封：{success, service_type, timestamp, data, error}
    """
    data = {"service_type": service_type, **kwargs}
    files = {}

    # 如果有音訊檔案
    if "audio_path" in kwargs:
        path = kwargs.pop("audio_path")
        data.pop("audio_path")
        with open(path, "rb") as f:
            files["file"] = f
            resp = requests.post(f"{GATEWAY}/api/process",
                                 headers=HEADERS, files=files, data=data)
    else:
        resp = requests.post(f"{GATEWAY}/api/process",
                             headers=HEADERS, data=data)

    if resp.status_code != 200:
        # 5xx 才值得重試，4xx 重試幾次都一樣。
        # 422 之類的框架錯誤沒有 error 欄位，所以用 .get() 取。
        body = resp.json()
        raise RuntimeError(
            f"HTTP {resp.status_code}: {body.get('error') or body.get('detail') or body}"
        )
    return resp.json()


# ─── 台語語音 → 文字（Breeze-ASR-26 直接輸出中文字）───
# 回應欄位：text / audio_duration / confidence
result = call_gateway("taiwanese_asr", audio_path="taiwanese.wav")
print(result["data"]["text"])                 # 中文字
print(result["data"]["confidence"])           # 0–1 的信心度


# ─── 中文語音 → 文字 ───
print(result["data"]["text"])                 # 中文文字


# ─── 中文文字 → 台語語音 ───
result = call_gateway("taiwanese_tts",
                       text_input="你好，今天天氣真好",
                       extra_params=json.dumps({"speed": 1.0}))

print(result["data"]["tailo_romanization"])   # 本調台羅（中間產物）
# 註：v5.0.0 起 taigi_hanji 已 deprecated（一律空字串），請改用 tailo_romanization

# 儲存音訊
wav_bytes = base64.b64decode(result["data"]["audio_base64"])
with open("taiwanese_output.wav", "wb") as f:
    f.write(wav_bytes)
# → 就可以播放台語語音了！


# ─── 完整的雙向對話流程 ───

# 1. 使用者說台語 → 辨識成中文
asr = call_gateway("taiwanese_asr", audio_path="user_speech.wav")
chinese_text = asr["data"]["text"]

# 2. 後端處理中文（你的業務邏輯）
reply = f"你說的是：{chinese_text}，沒問題！"

# 3. 回覆轉成台語語音
tts = call_gateway("taiwanese_tts", text_input=reply)
# → 播放 tts["data"]["audio_base64"]
'''

    print(code)


def main():
    print_header("AI 模型服務網關 - 測試客戶端")

    if not check_gateway():
        print("\n請先啟動服務: ./start_all.sh")
        sys.exit(1)

    print("\n選擇操作:")
    print("  1. 查看所有服務狀態")
    print("  2. 台語語音識別 (taiwanese_asr)")
    print("  4. 台語文字轉語音 (taiwanese_tts)")
    print("  5. 查看後端整合範例")
    print()

    choice = input("請選擇 (1-5): ").strip()

    if choice == "1":
        list_services()
    elif choice == "2":
        path = input("音訊檔案路徑: ").strip()
        test_asr("taiwanese_asr", path)
    elif choice == "3":
        path = input("音訊檔案路徑: ").strip()
        lang = input("語言 (預設 zh): ").strip() or "zh"
    elif choice == "4":
        text = input("輸入中文文字: ").strip()
        speed = float(input("語速 (預設 1.0): ").strip() or "1.0")
        test_tts(text, speed=speed)
    elif choice == "5":
        backend_example()
    else:
        print("無效選擇")


if __name__ == "__main__":
    main()
