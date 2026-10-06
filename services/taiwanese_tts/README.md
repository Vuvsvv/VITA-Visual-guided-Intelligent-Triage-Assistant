# 台語 TTS 服務 (v5 — Chatterbox)

中文字 → 台羅數字調(Gemini) → 詞彙替換 → 拆句 → Chatterbox 引擎 → 語音

**port 8003，service_type 仍是 `taiwanese_tts`**，`/api/synthesize` 端點不變，gateway 不用改。
但回應欄位的**內容**有變（見〈API〉與根目錄 README 的〈v5.0.0 變更紀錄〉）。

## 架構

```
main.py（:8003，HTTP client，可在 Docker）
   │  POST /tts  {"text": "本調台羅"}
   ▼
engine/cb_server.py（:9881，宿主機 GPU，cb conda 環境）
   └─ cb_infer.TaigiTTS：變調轉換（sandhi.py）→ Chatterbox 推論 → WAV 24kHz
```

本服務不載入模型；引擎獨立成 HTTP 服務，所以要換模型時只動引擎。

## 啟動順序（兩層，缺一不可）

### 1. Chatterbox 引擎（cb conda 環境）

```bash
conda activate cb
CB_HOST=127.0.0.1 CB_ROOT=~/chatterbox-finetuning \
  python services/taiwanese_tts/engine/cb_server.py          # port 9881
# 本服務改用 Docker 跑時要改成 CB_HOST=0.0.0.0，並用防火牆擋掉 9881：sudo ufw deny 9881/tcp（引擎沒有認證）

curl -s http://127.0.0.1:9881/ | python3 -m json.tool          # "ready": true 才算載入完成
```

### 2. 本服務

```bash
export GEMINI_API_KEY=你的key
export GEMINI_MODEL=gemini-2.5-flash      # 2.0-flash 免費額度曾歸零，建議固定寫進 .env
python services/taiwanese_tts/main.py     # port 8003
```

## 引擎的環境需求（在 `~/chatterbox-finetuning`，不在本專案內）

| 需要 | 位置／說明 |
|---|---|
| 微調工具包 | `gokhaneraslan/chatterbox-finetuning`，clone 到 `~/chatterbox-finetuning` |
| 底模（英文版 Chatterbox） | `pretrained_models/`（工具包 `setup.py` 下載：`t3_cfg`、`s3gen`、`ve`、`tokenizer.json`） |
| 模型 B 權重 | `chatterbox_output_sandhi/new_lang_adapter` |
| 參考音 | `speaker_reference/ref3.wav` |
| `cb` conda 環境 | Python 3.11、torch **+cu128**（RTX 5060 Ti 是 Blackwell sm_120，必須 cu128）、peft、`chatterbox-tts`（以 `--no-deps` 安裝） |

**重裝工具包或環境時必須重做：**

1. **重複偵測閾值 2 → 5**（原始碼註解寫 3 次、實作只檢查 2 次；台語連續兩個相同語音 token 很正常，會被誤判而提早結束）
   `src/chatterbox_/models/t3/inference/alignment_stream_analyzer.py` 約第 157 行：
   ```python
   len(self.generated_tokens) >= 5 and
   len(set(self.generated_tokens[-5:])) == 1
   ```
2. **確認 torch 仍是 +cu128**：用 pip 裝其他套件時可能把 torch 換掉，裝完一律檢查
   `python -c "import torch; print(torch.__version__); torch.randn(9,9).cuda()"`
3. （只有重新訓練才需要）`src/preprocess_ljspeech.py` 第 46 行的 `torchaudio.load` 改用 `soundfile`（FFmpeg 8.x 與 torchcodec 不相容）

## 推論設定為什麼這樣（`engine/cb_infer.py`，請勿任意更改）

每一條都是實測發現「推論和訓練不一致」、修正後品質才明顯改善的：

| 設定 | 原因 |
|---|---|
| 底模用**英文版** T3（`t3_cfg`），不是多語版 | 工具包訓練用英文版；LoRA 修正量套到多語版上會失準（修正後「好很多」） |
| **不加語言標記** | 英文版本來就沒有；訓練時也沒有 |
| 參考音 prompt 長度 **75 token** | = 訓練的 `prompt_duration` 3 秒；用預設長度會「開頭重複第一個字」 |
| 句首大寫保留預設 | 訓練時也有；拿掉反而念得很不穩 |
| 輸入先做**變調轉換**（`sandhi.py`） | 模型 B 的訓練資料就是變調台羅；呼叫端送本調即可，引擎自動轉 |
| attention 用 `eager` | `sdpa` 不支援推論時需要的 `output_attentions` |
| 停用 perth 浮水印 | 部分環境載入失敗；不影響音質 |

## 可調整的地方

改完存檔即生效、不用重啟：

1. `prompts/taigi_tailo.txt` — Gemini 轉換指令（中文 → 本調台羅）
2. `data/word_replace.json` — 念不好的詞替換表，**台羅 → 台羅、整詞比對**：
   ```json
   { "本調台羅A": "本調台羅B" }
   ```
   適用情境：某個詞一直念不好（通常是音節在訓練語料裡沒出現過），換成語料裡有的同義說法。

環境變數：

```bash
# 本服務
MAX_SEG_LEN=70        # 拆句上限（台羅字元數；實測 80 內穩定，超過句尾聲調會失準）
SEG_GAP_SEC=0.25      # 段落間靜音長度
CB_API=http://127.0.0.1:9881
CB_TIMEOUT=280

# 引擎（啟動 cb_server.py 時帶上）
CB_ADAPTER=chatterbox_output_sandhi/new_lang_adapter   # 換成模型 A 時要同時設 CB_SANDHI=0
CB_REF=speaker_reference/ref3.wav
CB_CANDIDATES=1       # 設 3：每段生成 3 次取長度居中者，降低開頭重複，但耗時約 3 倍
CB_SEED=2
```

## API

### POST /api/synthesize

```json
{ "text": "你頭很痛嗎？建議去看神經內科", "format": "m4a" }
```

| 回應欄位 | 內容 |
|---|---|
| `tailo_romanization` | 送進模型的**本調台羅**（v5.0.0 起名實相符） |
| `taigi_hanji` | **一律空字串**（deprecated，v6.0.0 移除） |
| `segments` | 實際拆成的段落 |
| `audio_base64` / `audio_format` | 音訊與格式（預設 m4a） |
| `sample_rate` | wav 為 24000；m4a 為 22050 |

- `speed`：Chatterbox 不支援變速，欄位保留但**會被忽略**。
- `skip_gemini: true`：直接送**本調台羅**、跳過 Gemini；含漢字會回 400。

### POST /api/convert

只轉文字不合成，用來檢查 Gemini 轉出的台羅：

```bash
curl -X POST http://localhost:8003/api/convert \
  -H "Content-Type: application/json" \
  -d '{"text":"你頭很痛嗎？建議去看神經內科，請先去掛號"}'
# 回傳 tailo_raw / tailo_final / segments
```

### 引擎 API（:9881）

- `POST /tts`，body `{"text": "本調台羅", "seed": 選填}` → `audio/wav`（含漢字回 400；模型未載入回 503）
- `GET /` → `{"ready": true, ...}`

## 已知限制

| 限制 | 現況／處理 |
|---|---|
| 句長 | 超過約 80 字元句尾聲調失準 → 自動拆段（≤70） |
| 開頭偶爾重複第一個字 | 參考音長度修正後已大幅減少；仍出現可設 `CB_CANDIDATES=3` |
| 詞彙覆蓋 | TAT 是文學／日常語料，**醫療詞多半沒出現過**；音節學過的能拼但不一定準，音節完全沒出現過的一定念錯 → 用 `word_replace.json` 換說法 |
| 專科名念法 | v4.x 保留中文（華語念），v5.0.0 改為**台語讀音**（例：sin5-king1 lai7-kho1） |
| 變調 | 只處理詞內變調；跨詞變調、仔前變調、三疊字由模型自行處理 |
| Gemini 拼寫 | 台羅可能拼錯（例如漏聲調），會直接影響發音 → 用 `/api/convert` 檢查 |

## 版本沿革

| | v3 | v4 | v5（目前） |
|---|---|---|---|
| 模型 | VITS（mms-tts-nan） | GPT-SoVITS | **Chatterbox + LoRA（模型 B）** |
| 輸入 | 台羅 | 台語漢字 | **本調台羅（引擎內轉變調）** |
| 取樣率 | 16k | 32k | **24k** |
| 主要問題 | 男聲底模遷移女聲產生電音 | 中國來源，不符比賽規則 | 醫療詞覆蓋不足 |
