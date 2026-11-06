# 台語 ASR 服務改動說明

**日期**：2026-08-17
**影響範圍**：`docker-compose.yml`、`services/breezy_asr/main.py`
**版本**：v2.0.0 → v2.1.0

---

## 摘要

本次改動解決兩個問題並新增一項功能：

| 項目 | 問題 | 結果 |
|------|------|------|
| GPU 啟用 | 容器未取得 GPU，推論在 CPU 上執行 | 推論時間降至 0.5 秒 |
| 冷啟動預熱 | 首次請求需等待 11.86 秒 | 首次請求即為 0.5 秒 |
| 信心度輸出 | 無法判斷辨識結果是否可信 | API 回傳 0–1 信心度數值 |

---

## 一、Docker GPU 設定

### 問題描述

`taiwanese-asr` 與 `chinese-asr` 兩個容器啟動時未宣告 GPU 資源，導致 PyTorch 在容器內偵測不到 CUDA 裝置，所有推論退回 CPU 執行。

診斷方式：

```bash
docker inspect taiwanese-asr --format "{{json .HostConfig.DeviceRequests}}"
# 輸出 null 表示未配置 GPU
```

容器內驗證：

```
RuntimeError: Found no NVIDIA driver on your system.
```

### 改動內容

檔案：`docker-compose.yml`

```yaml
  taiwanese-asr:
    build:
      context: ./services/breezy_asr
    container_name: taiwanese-asr
    ports:
      - "8001:8001"
    volumes:
      - hf_cache:/root/.cache/huggingface
    deploy:
      resources:
        limits:
          memory: 6G
        reservations:              # ← 新增
          devices:                 # ← 新增
            - driver: nvidia       # ← 新增
              count: all           # ← 新增
              capabilities: [gpu]  # ← 新增
    restart: unless-stopped
```

注意 `reservations` 與 `limits` 為同層級，縮排錯誤會造成以下驗證失敗：

```
validating docker-compose.yml: services.taiwanese-asr.deploy.resources.limits
additional properties 'devices', 'reservations' not allowed
```

### 驗證結果

```bash
docker exec taiwanese-asr python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"
# True (12, 0)
```

`(12, 0)` 為 sm_120，代表容器內的 PyTorch 2.12.1+cu130 完整支援 RTX 5060 Ti 的 Blackwell 架構。

VRAM 佔用：約 3.3 GB（總佔用 4466 MiB 扣除桌面環境約 1.1 GB）。

---

## 二、模型預熱

### 問題描述

首次推論需要 JIT 編譯 CUDA kernel，實測耗時 11.86 秒，第二次之後穩定在 0.5 秒。此延遲會影響服務重啟後的第一位使用者。

### 改動內容

檔案：`services/breezy_asr/main.py`，於 `initialize_model()` 結尾新增：

```python
logger.info("開始預熱 ...")
warm_audio = np.zeros(16000, dtype=np.float32)   # 1 秒靜音
warm_feat = processor(
    warm_audio, sampling_rate=16000, return_tensors="pt"
).input_features.to(device).to(torch_dtype)
with torch.no_grad():
    model.generate(warm_feat, max_new_tokens=8)
logger.info("預熱完成，服務就緒")
```

服務啟動時間會因此增加約 12 秒，但首次請求不再有延遲。日誌出現「預熱完成，服務就緒」後即可接受請求。

---

## 三、信心度輸出

### 設計目的

語音辨識結果會直接影響後續的症狀判斷與科別推薦，錯誤的辨識文字會導致整條流程失準。信心度提供一個量化指標，讓系統能在辨識不可靠時請使用者重新錄音，而非帶著錯誤資料繼續執行。

### 為何無法沿用 pipeline

原本使用的 `AutomaticSpeechRecognitionPipeline` 不會輸出 token 層級的 log probability。取得信心度必須改為直接呼叫 `model.generate()` 並開啟 `output_scores=True`，再透過 `compute_transition_scores()` 換算。

### 改動內容

**回應模型新增欄位**

```python
class TranscriptionResponse(BaseModel):
    text: str
    audio_duration: float
    confidence: Optional[float] = None   # ← 新增，0–1，None 代表無法計算
    success: bool
    error: Optional[str] = None
```

**推論方式改寫**

```python
# 改動前
result = asr_pipeline(audio_array, return_timestamps=False)
text = result["text"].strip()

# 改動後
input_features = processor(
    audio_array, sampling_rate=16000, return_tensors="pt"
).input_features.to(device).to(torch_dtype)

with torch.no_grad():
    gen = model.generate(
        input_features,
        return_dict_in_generate=True,
        output_scores=True,
    )

text = processor.batch_decode(gen.sequences, skip_special_tokens=True)[0].strip()
confidence = compute_confidence(gen.sequences, gen.scores)
```

**信心度計算函式**

```python
def compute_confidence(sequences, scores) -> Optional[float]:
    try:
        transition = model.compute_transition_scores(
            sequences, scores, normalize_logits=True
        )
        row = transition[0]
        valid = row[torch.isfinite(row)]        # 濾掉 -inf（padding token）
        if valid.numel() == 0:
            return None
        conf = float(torch.exp(valid.mean().float()))
        return max(0.0, min(1.0, conf))
    except Exception as e:
        logger.warning(f"信心度計算失敗，回傳 None: {e}")
        return None
```

計算原理：`compute_transition_scores` 回傳每個 token 被選中時的 log probability，取平均後以指數還原為 0–1 的機率值，代表模型對整句辨識結果的平均把握程度。

**全域變數調整**

`processor` 與 `model` 原為 `initialize_model()` 的區域變數，因信心度計算需要使用，改為全域變數：

```python
processor = None
model = None
device = None
torch_dtype = None
```

### API 回應格式變化

改動前：

```json
{"text":"地上要看醫生","audio_duration":2.38,"success":true,"error":null}
```

改動後：

```json
{"text":"地上要看醫生","audio_duration":2.38,"confidence":0.73,"success":true,"error":null}
```

新增欄位不影響既有客戶端。Android 端的 data class 未宣告 `confidence` 時會自動忽略，不需同步修改。

---

## 已知限制

**長音訊截斷**

移除 pipeline 後失去 `chunk_length_s=30` 的自動分段機制，超過 30 秒的音訊會被截斷。掛號語音多為短句，目前評估影響有限。若未來需支援長音訊，需自行實作分段邏輯。

**門檻值尚未設定**

目前僅輸出信心度數值，未實作門檻判斷。建議先蒐集真實錄音的信心度分佈，再依據實測數據決定門檻，避免憑直覺設定造成過度攔截或攔截不足。

**信心度不保證正確性**

信心度反映模型的確信程度，而非實際正確率。模型可能對錯誤結果也給出高信心度。使用前應確認錯誤樣本的信心度分佈，判斷此指標在本場景下是否具備區辨力。

---

## 部署步驟

```bash
cd ~/ai-model-gateway_bigvgan

# 備份
cp docker-compose.yml docker-compose.yml.bak
cp services/breezy_asr/main.py services/breezy_asr/main.py.bak

# 驗證 compose 語法
docker compose config > /dev/null && echo "語法正確"

# 重建服務
docker compose up -d --build taiwanese-asr

# 確認預熱完成
docker logs -f taiwanese-asr
```

## 驗證方式

```bash
for i in 1 2 3; do
  echo "=== 第 $i 次 ==="
  curl -s -w "\n總耗時：%{time_total}s\n" \
    -F "audio_file=@/home/tku/test_tw.wav" \
    http://localhost:8001/api/transcribe
done
```

預期結果：三次皆約 0.5 秒，回應含 `confidence` 欄位。

---

## 效能對照

測試檔案：`test_tw.wav`，長度 2.38 秒

| 階段 | 首次請求 | 後續請求 | RTF |
|------|---------|---------|-----|
| CPU（改動前） | — | 未實測 | — |
| GPU（未預熱） | 11.86 s | 0.49–0.51 s | 0.21x |
| GPU（含預熱） | 待驗證 | 待驗證 | 待驗證 |

---

## 後續項目

以下項目尚未執行，依建議優先順序排列：

1. **蒐集信心度分佈** — 以多筆真實錄音實測，決定門檻值
2. **確認辨識準確率** — `test_tw.wav` 辨識為「地上要看醫生」，需確認原始內容是否相符
3. **chinese-asr 加入 GPU** — VRAM 尚有約 11.8 GB 餘裕，設定方式同上
4. **Android 端改用 Opus 上傳** — 可縮短上傳時間約 3–4 倍