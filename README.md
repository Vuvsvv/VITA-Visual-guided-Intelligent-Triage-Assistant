# AI 模型服務網關

統一管理多個 AI 模型的 API 網關。後端只需呼叫一個端點，透過 `service_type` 參數決定使用哪個模型。

## 架構

```
                          ┌──────────────────────┐
                          │  API Gateway (:8000)  │
     後端 ── POST ──────▶ │  依 service_type 路由  │
     service_type=xxx     └──┬─────────┬─────────┬┘
                             │         │         │
          ┌──────────────────┘         │         └──────────────────┐
          ▼                            ▼                            ▼
 ┌──────────────────┐  ┌─────────────────────┐  ┌──────────────────┐
 │ 台語 ASR (:8001) │  │ 台語 TTS (:8003)    │  │ 中文 TTS (:8004) │
 │ Breeze-ASR-26    │  │ Gemini(台羅) +      │  │ BreezyVoice      │
 │                  │  │ Chatterbox          │  │                  │
 │ 台語語音→中文字  │  │ 中文字→台語語音     │  │ 中文字→國語語音  │
 └──────────────────┘  └──────────┬──────────┘  └──────────────────┘
                                  │ HTTP
                                  ▼
                     宿主機 Chatterbox 引擎 (:9881)
```

上面這四個方框（網關 + 三個服務）**可以整組用本機跑，也可以整組用 Docker 跑** ——
`./start_all.sh` 走本機，`docker compose up -d` 走 Docker，選一種即可。

最底下的 **Chatterbox 引擎一律留在宿主機**（吃 GPU、需要微調權重與工具包目錄），兩種方式都一樣。

## 服務總覽

| service_type | 輸入 | 輸出 | 模型 | Port |
|---|---|---|---|---|
| `taiwanese_asr` | 台語音訊 | 中文字 | MediaTek-Research/Breeze-ASR-26 | 8001 |
| `taiwanese_tts` | 中文字 | 台語語音 + 台羅 | Gemini + Chatterbox（LoRA 微調，模型 B） | 8003 |
| `chinese_tts` | 中文字 | 台灣國語語音 | MediaTek-Research/BreezyVoice-300M | 8004 |

> **v5.0.0 起**，台語 TTS 的底層從 GPT-SoVITS 換成 **Chatterbox**（英文基底 + LoRA 微調，模型 B）。
> 這個服務本身仍只是一個 HTTP client（不載入任何模型、requirements 裡沒有 torch），
> 真正吃 GPU 的 Chatterbox 引擎（`services/taiwanese_tts/engine/cb_server.py`）留在宿主機跑。
> 完整變更見下方〈v5.0.0 變更紀錄〉。

> **中文 ASR 已移除。** 前端的國語辨識改用 Android 系統內建的語音輸入，
> 延遲遠低於自架 Whisper，且不佔用顯示記憶體。台語辨識沒有堪用的現成方案，
> 因此保留自建服務 —— 資源集中在真正需要突破的地方。

## v5.0.0 變更紀錄（台語 TTS：GPT-SoVITS → Chatterbox）

### 為什麼換

GPT-SoVITS 是中國開發者的專案，不符合比賽「不得使用來源自中國或中資之開源模型」的規定，
改用 Resemble AI（美國／加拿大）釋出、MIT 授權的 **Chatterbox**。

> ⚠️ **合規仍待確認**：Chatterbox 的語音解碼模組（S3Gen）架構衍生自阿里巴巴的 CosyVoice，
> 執行時也依賴 S3Tokenizer 套件，是否符合規定需由主辦單位判斷。
> 引擎以 HTTP 與本服務隔離，萬一需要更換，只要換掉引擎、本服務不用動。

### 使用的模型

| 項目 | 內容 |
|---|---|
| 基礎模型 | Chatterbox TTS **英文基底版**（`t3_cfg`，0.5B，MIT） |
| 微調方式 | LoRA（rank 256）只訓練 T3（文字→語音 token）；S3Gen 與語者編碼器凍結 |
| 詞彙表 | 擴充為 2,454 個字元單位 |
| 訓練資料 | TAT-TTS F1，12,593 句（約 10 小時；已剔除台羅欄位夾帶漢字的 104 句） |
| 輸入格式 | 經**連讀變調轉換**的台羅數字調（呼叫端送本調，引擎內部轉換） |
| 權重位置 | `~/chatterbox-finetuning/chatterbox_output_sandhi/new_lang_adapter` |

### ⚠️ 對呼叫端的影響（破壞性變更）

`/api/synthesize` 端點與欄位名稱不變，網關不用改；但**欄位內容**有變：

| 項目 | v4.x | v5.0.0 |
|---|---|---|
| `tailo_romanization` | 台語漢字（名實不符，曾標為 deprecated） | **本調台羅數字調**（恢復名實相符） |
| `taigi_hanji` | 台語漢字 | **一律空字串**（deprecated，v6.0.0 移除） |
| `sample_rate`（wav） | 32000 | **24000**（m4a 仍為 22050） |
| `speed` | 有效 | **忽略**（Chatterbox 不支援變速） |
| 專科名念法 | 保留中文（華語念法） | **台語讀音**（例：神經內科 sin5-king1 lai7-kho1） |
| `skip_gemini: true` 的 `text` | 台語漢字 | **本調台羅**（含漢字回 400） |
| `/api/convert` 回傳 | `taigi_raw`、`taigi_final`、`dict_*` | `tailo_raw`、`tailo_final`（字典啟用時才有 `dict_*`） |

**前端或後端若有顯示 `taigi_hanji`，改版後會拿到空字串，請改用 `tailo_romanization`。**

### 架構變更

- 底層引擎：GPT-SoVITS api_v2（:9880）→ **Chatterbox 引擎 `engine/cb_server.py`（:9881）**
- 新增 `services/taiwanese_tts/engine/`：`cb_server.py`（HTTP 引擎）、`cb_infer.py`（與訓練一致的推論設定）、`sandhi.py`（變調轉換）
- 前處理改為「中文 → 本調台羅」；變調由引擎處理，本服務只送本調

### 前處理變更

- **Gemini 指令**：`prompts/taigi_hanji.txt` → `prompts/taigi_tailo.txt`。原本 9 條規則語意保留，第 3 條「專科名保留中文」改為「專科名用台語讀音」（Chatterbox 看不懂漢字）。舊檔保留未刪，已不再使用。
- **台羅正規化**（新增）：全形標點 → 半形、調號 → 數字調（Gemini 偶爾不照指示）、去除引號括號。
- **漢字檢查**（新增）：Gemini 輸出夾帶漢字時附提示重試一次，仍失敗回 500；`skip_gemini` 的輸入含漢字回 400；引擎本身也拒收漢字。
- **本地字典預設停用**：`taigi_dict.json` 的 598 筆值都是台語漢字，Chatterbox 用不到。重新啟用時只接受 100% 覆蓋（移除 v4.x 的 0.75 寬容門檻）。
- **快取**改存台羅；啟動時自動丟棄含漢字的舊快取；`taigi_cache.json` 已重置。
- **替換表**改為「台羅 → 台羅、整詞比對」（避免 `ho7` 誤中 `kua3-ho7`）。原本 8 筆漢字條目（批價→納錢等）的意圖已移到 prompt 第 4 條。
- **拆句**改以台羅字元數計（上限 70），依「句號 → 逗號 → 空白」切，不會切斷詞，相鄰短句自動合併。
- **預熱句**改為台羅 `li2 ho2.`（直接送引擎、不經 Gemini）。

### 修改的檔案

| 檔案 | 變更 |
|---|---|
| `services/taiwanese_tts/main.py` | v4.8.0 → v5.0.0：引擎呼叫、台羅前處理、正規化、漢字檢查、拆句、健康檢查 |
| `services/taiwanese_tts/engine/` | **新增**：`cb_server.py`、`cb_infer.py`、`sandhi.py` |
| `services/taiwanese_tts/prompts/taigi_tailo.txt` | **新增**：輸出本調台羅的 Gemini 指令 |
| `services/taiwanese_tts/data/word_replace.json` | 改為台羅格式（目前無條目） |
| `services/taiwanese_tts/data/taigi_cache.json` | 重置（舊內容是台語漢字） |
| `services/taiwanese_tts/Dockerfile` | `GSV_API` → `CB_API=http://host.docker.internal:9881` |
| `services/taiwanese_tts/README.md` | 改寫為 Chatterbox 版（含引擎環境需求與推論設定原因） |
| `services_config.yaml`、`_docker.yaml`、`_vps.yaml` | 描述、版本 5.0.0、模型名稱 |
| `docker-compose.yml` | `GSV_API` → `CB_API` |
| `start_all.sh` | 改提示啟動 Chatterbox 引擎 |
| `deploy.sh` | 防火牆加擋 9881（9880 保留） |
| `.env.example` | `GSV_*` → `CB_*`（含引擎變數） |
| `gateway.py`、`test_client.py` | 說明文字；範例改印 `tailo_romanization` |
| `tests/` | GSV 專屬測試改為 Chatterbox 版；新增 4 個台羅前處理測試 |
| `README.md` | 本節，以及架構、部署、欄位、網路暴露面等章節 |

引擎在宿主機需要的環境（工具包、底模、權重、必要修正）見 `services/taiwanese_tts/README.md`。

## 快速開始

先準備 `.env`（兩種方式都需要）：

```bash
cp .env.example .env
nano .env   # 填入 GEMINI_API_KEY 和 API_KEY
```

然後**選一種**啟動方式。

### 方式 A：本機跑（WSL / conda）

不需要 Docker，Chatterbox 引擎綁 loopback 就好，也不用設防火牆。

```bash
# 終端機 1 —— Chatterbox 引擎（吃 GPU）
conda activate cb
cd <專案目錄>
CB_HOST=127.0.0.1 CB_ROOT=~/chatterbox-finetuning \
  python services/taiwanese_tts/engine/cb_server.py
# 等 curl -s http://127.0.0.1:9881/ 回 "ready": true

# 終端機 2 —— 網關 + 三個服務
conda activate breezy
cd <專案目錄>
./start_all.sh
```

`start_all.sh` 會依序起 8001 → 8003 → 8004 → 8000，
並且**等每個服務的 `/` 真的回 200 才往下走**（模型載入要幾分鐘是正常的）。
按 `Ctrl+C` 會一起關掉全部。

### 方式 B：Docker

Chatterbox 引擎必須改綁 `0.0.0.0`，因為對容器來說宿主機算「外部」，
綁 `127.0.0.1` 的話 `host.docker.internal` 連不進來。
**改了就一定要用防火牆擋住 9881**（見下方「網路暴露面」）。

```bash
# 終端機 1 —— Chatterbox 引擎
conda activate cb
cd <專案目錄>
CB_HOST=0.0.0.0 CB_ROOT=~/chatterbox-finetuning \
  python services/taiwanese_tts/engine/cb_server.py

# 只需做一次
sudo ufw deny 9881/tcp

# 終端機 2
sudo docker compose up -d
```

### 確認

```bash
curl http://localhost:8000/health     # 三個服務都要是 healthy
```

API 文件：http://localhost:8000/docs

## 後端整合

後端只需要這些程式碼，同一個端點切換三種服務：

```python
import os, requests, base64, json

GATEWAY = "http://localhost:8000"
HEADERS = {"X-API-Key": os.environ["API_KEY"]}   # 缺這個會 401

# ── 台語語音 → 中文字（Breeze-ASR-26）──
with open("taiwanese.wav", "rb") as f:
    r = requests.post(f"{GATEWAY}/api/process",
        headers=HEADERS,
        files={"file": f},
        data={"service_type": "taiwanese_asr"})
print(r.json()["data"]["text"])

# ── 中文字 → 台語語音（Gemini + Chatterbox）──
# 註：v5.0.0 起 speed 會被忽略（Chatterbox 不支援變速）
r = requests.post(f"{GATEWAY}/api/process",
    headers=HEADERS,
    data={
        "service_type": "taiwanese_tts",
        "text_input": "頭痛要看醫生",
    })
result = r.json()["data"]
print(result["tailo_romanization"])           # 本調台羅（中間產物）
# 預設輸出 m4a（AAC）；副檔名請依 result["audio_format"] 決定，不要寫死
audio = base64.b64decode(result["audio_base64"])
with open(f"taiwanese.{result['audio_format']}", "wb") as f:
    f.write(audio)

# ── 中文字 → 台灣國語語音（BreezyVoice）──
r = requests.post(f"{GATEWAY}/api/process",
    headers=HEADERS,
    data={
        "service_type": "chinese_tts",
        "text_input": "你好，今天天氣真好"
    })
result = r.json()["data"]
audio = base64.b64decode(result["audio_base64"])
with open(f"mandarin.{result['audio_format']}", "wb") as f:
    f.write(audio)
```

### 回應格式與狀態碼

body 一律是同一個信封 `{success, service_type, timestamp, data, error, processing_time_ms}`，
失敗時 **HTTP status code 也會帶語意**，呼叫端可以直接用來決定要不要重試：

| status | 意思 | 該不該重試 |
|---|---|---|
| 200 | 成功 | — |
| 400 | 輸入有問題（缺 text_input、檔案格式不對…） | 否 |
| 401 / 403 | API Key 無效／權限不足 | 否 |
| 404 | 未知的 service_type | 否 |
| 413 | 檔案超過大小上限 | 否 |
| 429 | 超過速率限制（看 `Retry-After`） | 稍後再試 |
| 503 | 該服務未啟用 | 否 |
| 502 / 504 | 下游服務異常／逾時 | 可以 |

判斷失敗請看 **status code**，不要只看 body 的 `success`：
FastAPI 的參數驗證錯誤（422）回的是 `{"detail": ...}`，沒有 `success` 欄位。

```python
if resp.status_code != 200:
    body = resp.json()
    raise RuntimeError(body.get("error") or body.get("detail") or body)
```

### 已棄用欄位

`taiwanese_tts` 回應中的 **`taigi_hanji` 自 v5.0.0 起已 deprecated、一律為空字串**：
前處理改為直接產生台羅（Chatterbox 看不懂漢字），不再有台語漢字這個中間產物。
欄位只為了相容舊呼叫端而保留，預計於 **v6.0.0 移除**。

反過來，**`tailo_romanization` 在 v5.0.0 恢復名實相符**，裝的是送進模型的本調台羅數字調。
（v4.x 時它裝的是台語漢字，文件曾建議改用 `taigi_hanji` —— 這個建議在 v5.0.0 已經反轉。）
新的呼叫端請一律使用 **`tailo_romanization`**。

## 網路暴露面

**只有網關的 `:8000` 應該對外。** 8001／8003／8004 那三個微服務**沒有任何 API Key 檢查** ——
它們的設計前提是「只有網關會來找」，所以對外開放等於讓人繞過認證直接呼叫模型，
其中 `:8003` 背後接的是 Gemini，會直接花掉你的額度。

| Port | 是什麼 | 對外 |
|---|---|---|
| 8000 | API 網關（有 API Key 認證） | ✅ 開放 |
| 8001, 8003, 8004 | 三個微服務（**無認證**） | ❌ 只綁 `127.0.0.1` |
| 9881 | Chatterbox 引擎（**無認證**） | ❌ 防火牆擋掉 |

- **Docker**：`docker-compose.yml` 已把三個微服務綁在 `127.0.0.1`。
  要除錯就在宿主機上 `curl http://127.0.0.1:8003/`，或開 SSH tunnel。
- **VPS**：`deploy.sh` 產生的 systemd unit 全部綁 `127.0.0.1`，並額外用 ufw 明確擋掉這些 port。
- **Chatterbox 引擎**：Docker 路徑因為容器要連得到，必須用 `CB_HOST=0.0.0.0` 啟動 ——
  所以**一定要自己用防火牆擋住 9881**，否則等於把模型直接放到網路上：

  ```bash
  sudo ufw deny 9881/tcp        # Linux / WSL
  ```

  只在本機（非 Docker）跑台語 TTS 的話，`CB_HOST=127.0.0.1`（預設值）就夠，不需要這條規則。
  `deploy.sh` 的防火牆清單同時擋 9880（舊 GPT-SoVITS）與 9881。

## API Key 與權限

- `API_KEY=<key>` — 單一金鑰，預設具備 `*`（含 admin）權限。
- `API_KEYS=<key>:<名稱>[:<權限>]` — 多組金鑰，逗號分隔。
  第三段省略時只有 `process` 權限，**打不了 `POST /api/reload-config`**。
  例：`API_KEYS=sk-a:後端A,sk-b:後端B,sk-ops:維運:*`
- 速率限制以「金鑰的 hash」分桶，同名的兩把 key 不會互相吃配額；
  未認證請求以來源 IP 分桶（在 Nginx 後面時要設 `TRUSTED_PROXY_HOPS=1`，
  否則所有匿名請求都會被算成同一個 `127.0.0.1`）。

## 台語 TTS 與 Chatterbox 引擎怎麼分工

```
Docker 容器（台語 TTS :8003）              宿主機
──────────────────────────────────       ──────────────────────
中文字 → 快取／Gemini → 本調台羅
        → 正規化 → 詞彙替換 → 拆句（≤70 字元）
        → POST host.docker.internal:9881/tts  ─▶  Chatterbox 引擎
                                                   （變調轉換 → GPU 推論）
        ◀── WAV（24kHz）────────────────────────
        → 串接 → base64 WAV / M4A
```

### 中文轉台羅的機制

Gemini 呼叫一次約 3.5 秒，佔掉整個請求七成以上的時間，所以前面有一層快取：

| 層 | 命中條件 | 耗時 |
|---|---|---|
| 快取 | 同一句先前轉換過 | ~0 秒 |
| Gemini | 快取沒中（結果會存進快取） | ~3.5 秒 |

Gemini 的輸出會先正規化（全形標點 → 半形、調號 → 數字調），再檢查**不能含任何漢字**；
夾帶漢字時附上提示重試一次，仍失敗就回 500 —— 漢字送進 Chatterbox 一定念不出來。

**本地字典（`data/taigi_dict.json`）v5.0.0 起預設停用**：598 筆的值都是台語漢字，Chatterbox 用不到。
要重新啟用，先把值全部換成本調台羅，再設 `TAIGI_DICT_ENABLED=1`；
此時只接受 100% 覆蓋（v4.x 的 0.75 寬容門檻已移除 —— 留下的漢字 GPT-SoVITS 會念，Chatterbox 不會）。

除錯時可以只看轉換結果、不合成：

```bash
curl -X POST http://localhost:8003/api/convert \
  -H "Content-Type: application/json" \
  -d '{"text":"我最近常常頭暈"}'
# 回傳 tailo_raw / tailo_final / segments
```

改過 prompt 之後，舊快取的轉換結果會與新邏輯不一致，記得清掉：

```bash
curl -X DELETE http://localhost:8003/api/cache
```

> Docker 的 compose 把 `data/` 掛成唯讀，容器內快取寫不回檔案（服務照常運作，只是重啟後快取歸零）。
> 這是 v5.0.0 之前就有的設定；要持久化快取，請把該 volume 的 `:ro` 拿掉。

### 可獨立調整的地方

改完存檔即生效，不用重啟（compose 已把前三個掛成 volume）：

1. `services/taiwanese_tts/prompts/taigi_tailo.txt` — Gemini 轉換指令（中文 → 本調台羅）
2. `services/taiwanese_tts/data/word_replace.json` — 念不好的詞替換表（台羅 → 台羅、整詞比對，在轉換之後套用）
3. `services/taiwanese_tts/data/taigi_dict.json` — 本地字典（v5.0.0 預設停用，見上方說明）
4. `.env` 的 `MAX_SEG_LEN` — 拆句長度上限（台羅字元數；實測 80 內穩定，預設 70）

### 音訊輸出

兩個 TTS 預設輸出 **M4A（AAC 64kbps、22.05kHz 單聲道）**，
體積約為未壓縮 WAV 的七分之一（台語短句 148KB → 21KB），
Android `MediaPlayer` 原生支援。回應的 `audio_format` 會如實回報格式，
**呼叫端請依它決定副檔名，不要寫死 `.wav`**。

需要未壓縮輸出時（例如要把音檔餵回 ASR 驗證，`soundfile` 讀不了 AAC）：

```bash
TTS_OUTPUT_FORMAT=wav    # 兩個 TTS 皆適用
TTS_OUTPUT_SR=24000      # 台語：回到 Chatterbox 原始取樣率
AAC_BITRATE=96k          # 覺得音質不夠可調高
```

### 啟動預熱

首次推論要配置顯存並初始化運算環境，實測比穩定後慢一個量級
（GPT-SoVITS 時期實測：台語首句 17.7 秒 vs 穩定後 0.9 秒）。兩個 TTS 都會在啟動後自動送一次短句
（台語 TTS 送的是台羅 `li2 ho2.`，直接給引擎、不經 Gemini），
把這筆一次性成本挪到沒人等待的時候。設 `TTS_WARMUP=0` 可關閉。

## 新增服務

只需兩步，不改程式碼：

**步驟 1：** 在 `services_config.yaml` 加設定（欄位會用 Pydantic 驗證，打錯字會在載入時就報錯）

**步驟 2：** 啟動對應微服務

熱重載：`POST /api/reload-config`（需要 admin 權限；驗證失敗時會保留原設定）

每個服務的 `endpoint` 都可以用環境變數 `<SERVICE_TYPE>_ENDPOINT` 覆蓋，
例如 `CHINESE_TTS_ENDPOINT=http://127.0.0.1:8004`。

## 測試

不需要 GPU、也不需要下游服務真的存在：

```bash
pip install pytest respx httpx fastapi pyyaml python-multipart
pytest tests -q
```

## 專案結構

```
├── gateway.py                      # API 網關
├── security.py                     # 安全中間件（認證 / 權限 / 速率限制）
├── services_config.yaml            # 本機環境設定
├── services_config_docker.yaml     # Docker 環境設定（容器名稱）
├── services_config_vps.yaml        # 裸機 VPS 設定（deploy.sh 用，全部 127.0.0.1）
├── docker-compose.yml              # Docker 編排（網關 + 3 個服務）
├── start_all.sh                    # 一鍵啟動腳本
├── deploy.sh                       # VPS 部署腳本
├── manage.sh                       # 日常管理工具
├── test_client.py                  # 互動式手動測試 / 後端整合範例
├── tests/                          # 不需 GPU 的自動化測試
└── services/
    ├── breezy_asr/                 # service_type: taiwanese_asr  (:8001)
    ├── taiwanese_tts/              # service_type: taiwanese_tts  (:8003)
    │   └── engine/                 # Chatterbox 引擎（宿主機 GPU，:9881）
    └── breezy_tts/                 # service_type: chinese_tts    (:8004)
```

每個服務都只有一個 `main.py`，三條啟動路徑（Docker / `start_all.sh` / systemd）
指向的模組由 `tests/test_deployment.py` 驗證，不會再出現「部署的是舊版」的情況。

> 目錄名稱和 `service_type` 目前對不起來（`breezy_asr` ↔ `taiwanese_asr`、
> `breezy_tts` ↔ `chinese_tts`），這是 `deploy.sh` 曾經寫錯模組路徑的原因。
> 後續建議統一用 `service_type` 當唯一命名。

### 中文 TTS 的版本

`services/breezy_tts/main.py` 是 **4.0.0**（FP16 + 長句切割），
原本叫 `main_v2.py`；舊的 3.0.0 實作已刪除。
`services_config*.yaml` 的 `chinese_tts.version` 必須與它自報的版本一致 ——
對不上時 `tests/test_deployment.py` 會擋下來。
