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

> **v5.0.0 起**
> 這個服務本身仍只是一個 HTTP client（不載入任何模型、requirements 裡沒有 torch），
> 真正吃 GPU 的 Chatterbox 引擎（`services/taiwanese_tts/engine/cb_server.py`）留在宿主機跑。


> **中文 ASR 已移除。** 前端的國語辨識改用 Android 系統內建的語音輸入，
> 延遲遠低於自架 Whisper，且不佔用顯示記憶體。台語辨識沒有堪用的現成方案，
> 因此保留自建服務 —— 資源集中在真正需要突破的地方。



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

- **Gemini 指令**：`prompts/taigi_hanji.txt` → `prompts/taigi_tailo.txt`。原本 9 條規則語意保留，第 3 條「專科名保留中文」改為「專科名用台語讀音」舊檔保留未刪，已不再使用。
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

---

## 後續更新紀錄（2026-11）：臺語與中文 TTS

> 本節接續前述 v5.0.0 文件，整理近期**本地端實作、實際測試結果與待驗收事項**。這些內容是開發／實驗紀錄；**不代表全部變更已提交至 GitHub 分支，亦不代表服務版本號已同步更新**。請以部署環境實際程式和設定為準。

### 一、更新重點總覽

| 項目 | 後續改動或測試結果 | 狀態 |
|---|---|---|
| 臺語 TTS 模型 | 沿用 Chatterbox Model B、T3 LoRA 微調及本調臺羅輸入；S3Gen 等解碼元件非主要微調目標 | 本地端已運作 |
| Gemini 翻譯 | 從先前 Gemini 3.8 Flash 測試，改以 `gemini-4-argon`、`thinking=low` 進行動態臺羅轉換實測 | 已實測，仍可能因翻譯／校驗失敗重試 |
| 臺羅輸出品質 | Gemini JSON Schema Structured Output、格式／已知錯讀檢查、失敗重試；加入醫療詞彙校驗規則 | 已有本地端實測，仍需人工校對語意與讀音 |
| 固定話術加速 | 對完全一致的固定句使用已審核內容或預先生成的音檔；AI 動態回答仍走即時翻譯及 TTS | 已測試固定句命中跳過 Gemini；錄音驗收持續中 |
| 臺語速度 | 使用長度／複雜度分組測試，分開記錄翻譯、TTS、音檔秒數與 RTF | 已量測，不應承諾所有句子同速 |
| 中文 TTS | BreezyVoice-300M：autocast、長句切割、音量標準化及無效波形排查 | 量測與修復進行中 |
| 中臺音量一致性 | 參照臺語音檔 -14.89 LUFS，對中文完整波形合併後進行響度標準化 | 已在中文 API 實測 |
| 中文固定問句錄音 | 23 句批次生成、無聲音檔稽核、異常句重錄 | 最新檢查 19 GOOD、4 SUSPECT、0 BAD；未全數驗收 |

### 二、臺語 TTS：Chatterbox Model B 與微調策略

- **底模與訓練**：以 Chatterbox 英文基底模型進行臺語適配，重點微調 T3，使用 LoRA（本文件先前紀錄 rank 256）；保留 S3Gen 等非主要微調模組的原始權重。
- **資料集**：採用 TAT-TTS-F1，前述版本紀錄為 **12,593 句**，文字與音檔配對，輸入統一使用**本調臺羅數字調**。推論仍沿用 Model B 的訓練設定。
- **模型輸入**：中文先翻成臺語本調臺羅，**不要先在呼叫端套用變調**；`engine/sandhi.py` 在 Chatterbox 推論前依目前實作的規則處理部分詞內變調（`sandhi=True`），避免雙重變調。
- **說話者參考音**：推論階段使用固定 Reference Audio，作為聲音條件而非本次微調目標；相同參考音無法保證不同句子完全沒有聲線漂移。
- **多語料實驗觀察**：曾試過混用兩份語料，發現相同參考音下偶爾有聲線改變的情況；因此目前以單一 F1 資料的 Model B 作為較穩定的工作版本。聲線漂移原因尚未以對照實驗完全確認，不能直接歸因於第二份語料。
- **架構分工**：`:8003` 負責中文→臺羅、校驗、快取、拆段與音檔封裝；`:9881` 的 `cb_server.py`／`cb_infer.py` 負責 Sandhi 與 GPU 語音生成。請勿混淆 `services/taiwanese_tts/engine/cb_infer.py` 與另外的 `~/chatterbox-finetuning/cb_infer.py`。

#### 動態中文 → 臺語語音流程

```text
AI／後端動態中文回答
    ↓
完全一致的固定話術／已審核結果（若適用）
    ├─ 命中：使用已確認的臺羅或預錄音檔，避免重複 Gemini 翻譯
    └─ 未命中：Gemini 4 Argon（JSON Schema Structured Output）
                     ↓
               本調臺羅數字調
                     ↓
       格式＋醫療詞彙／已知錯讀校驗
                     ├─ 不通過：修正提示後有限次重試；仍失敗則停止
                     └─ 通過：正規化、快取、拆段
                                      ↓
                           Chatterbox :9881
                           Sandhi → T3 → S3Gen
                                      ↓
                           WAV／M4A 臺語語音
```

> **安全限制**：JSON Schema 與臺羅 Guard 只能阻擋部分格式及已知錯讀問題，**不能保證臺語語意、症狀否定、嚴重程度或醫療建議百分之百正確**。固定急診警示與問診內容應先人工校對、試聽後再正式使用。

#### 臺羅校驗與效能測試

- 本地端使用 `medical_tailo_guard.py` 及醫療詞彙校驗來攔截部分已知問題，例如「呼吸」的錯誤讀音、入聲韻尾聲調異常、輸出混入漢字或 Markdown。詞彙庫應**用來校驗／回饋重試**，不可無條件替換不確定的語意。
- 測試動態句時，可用 `DELETE http://127.0.0.1:8003/api/cache` 排除舊翻譯快取的干擾；此操作會清除共用快取，請只在測試環境或確認不影響其他請求時使用。
- 使用 `skip_gemini=true` 時，`text` 應是**已確認的本調數字調臺羅**，可以單獨量測 TTS 或使用人工審核輸入；中文不要搭配此旗標。
- 在一次 `gemini-4-argon`、`thinking=low` 的 12 次長度／複雜度測試中，六組各 2 次皆成功；測得下列**組別平均值**：

| 組別 | Gemini 翻譯平均 | 完整流程平均（含 TTS） |
|---|---:|---:|
| 短句／簡單 | 6.375 秒 | 10.591 秒 |
| 短句／複雜 | 4.655 秒 | 10.798 秒 |
| 中句／簡單 | 5.527 秒 | 17.282 秒 |
| 中句／複雜 | 7.208 秒 | 19.878 秒 |
| 長句／簡單 | 7.038 秒 | 29.739 秒 |
| 長句／複雜 | 10.116 秒 | 36.459 秒 |

- 上述數字是**小樣本、特定輸入與當時環境**的量測，不保證其他句子、網路或負載下能重現。長句的總等待時間通常也受到輸出語音長度與切段數量影響。
- 有一組長複雜句的臺語 TTS 實測：**26.292 秒合成 30.180 秒音檔，RTF 約 0.871**。此 RTF 僅衡量速度，**不代表發音正確率**。
- T3 推論曾測試關閉部分 `output_attentions` 輸出以減少額外負擔；嘗試把注意力實作切成 SDPA 時，與 `AlignmentStreamAnalyzer` 的注意力權重需求衝突。**不能宣稱 SDPA 已成功啟用**，也不要未經驗證就停用對齊校驗。

### 三、中文 TTS：BreezyVoice-300M 調整

主要檔案：`services/breezy_tts/main.py`，服務埠號 `8004`。

#### 推論設定與速度基準

最近一次 `GET /` 回報的設定：`autocast=true`、`fp16=false`、`fp16_modules=["llm","flow"]`、`warmup=true`、`compile=false`、`max_chunk_len=30`、`speed_supported=false`。**Autocast 開啟不等於所有模型子模組已永久轉 FP16**。

| 測試文字 | 中文字數 | TTS 耗時 | 音檔長度 | RTF |
|---|---:|---:|---:|---:|
| 短句 | 23 | 12.041 秒 | 6.188 秒 | 1.946 |
| 中句 | 46 | 18.087 秒 | 13.305 秒 | 1.359 |
| 長句 | 135 | 46.079 秒 | 37.373 秒 | 1.233 |

- 上述為**調整音量前的速度基準**。不同長度的 RTF 會變動，尚不能斷言速度瓶頸在某單一模組。
- 推論流程包含文字切段、正規化、注音轉換、`inference_zero_shot_no_normalize()`、音訊串接及 M4A 編碼；需要分段記錄實際耗時，才可判斷瓶頸位置。
- 目前不應直接宣稱 FP16 或 `torch.compile` 必定帶來固定百分比的加速；同一張 16GB GPU 還承載其他語音服務，需考慮顯存與穩定性。

#### 中文音量對齊臺語

- 使用先前已確認的臺語樣本作為參考，量得 **-14.89 LUFS**，並以 FFmpeg `loudnorm` 對**合併後的整段中文波形**進行響度標準化，而非每個子句個別拉音量。
- 響度對照：中文原始樣本 **-28.85 LUFS** → 單獨測試後 **-16.59 LUFS**；實際 `8004` API 的短句 M4A 量得 **-15.77 LUFS**、True Peak **-0.75 dBTP**，與參考臺語相差約 **0.88 LU**。
- 設定目標為 -14.89 LUFS，並非所有輸出都必定精確達標；AAC 編碼後 True Peak 亦可能略有變動。不要為了追求完全相同而直接加大增益造成破音。
- 標準化應在波形**有效且非靜音**的前提下執行；`loudnorm` 無法修復模型本身產生的壞音訊。

#### 無聲、NaN／Inf 與無效音訊防護

批次錄音期間，部分中文合成雖回 HTTP 200，實際卻是近乎無聲或大段缺音。日誌曾出現：

```text
RuntimeWarning: invalid value encountered in cast
measured_I=-inf, measured_TP=-inf, offset=inf
```

這表示有些波形包含無效數值或沒有可量測的有效音量。**不能只以 HTTP 200、音檔時長或 `zh_status=ok` 判斷可用**。

- 本地曾準備波形 Guard 補丁，目的是在 `_synth_chunk()` 後檢查 `NaN`／`Inf`、無聲波形並進行有限次重試；此類修補是否已成功安裝，必須以**正在執行的 `main.py` 內容及實際 8004 日誌**確認，不能只看修補腳本的檔名或執行退出碼。
- 對整段音訊的最終輸出也應檢查有效性；失敗時應回傳明確錯誤，避免把損壞的錄音寫入正式音檔庫。
- 若持續出現無效波形，可獨立測試 `USE_AUTOCAST=0` 等數值穩定性選項，但要注意顯存需求，且**一次只變更一個條件**。


### 四、目前仍需確認／後續工作

1. **臺語翻譯品質**：Gemini 產生的本調臺羅仍會出現拼音、聲調、詞彙或醫療語意偏差，校驗器只能涵蓋部分已知錯誤；需持續建立已審核測試集。
2. **微調成效評估**：比較 Chatterbox 微調前後、單語料與雙語料的同句發音正確率、聲線一致性與自然度。**RTF 是速度指標，不能單獨證明音質變好**。
3. **環境與版本同步**：本章描述的是最近本地端實測；上方仍保留舊版本說明（例如 Gemini 耗時約 3.5 秒、中文 TTS v4.0.0 的描述），請在正式發佈前依實際 Git commit、啟動日誌及 `/` 版本資訊更新。
4. **資安與部署**：延續原本原則，只對外開放需要認證的 Gateway；8001／8003／8004／9881 保持內網或 loopback，並妥善管理 Gemini API Key。
