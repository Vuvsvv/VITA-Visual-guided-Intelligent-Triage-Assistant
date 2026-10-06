
# 導底看哪科－智慧看診指南

本專案整合 Android 應用程式與 Python FastAPI 後端，提供症狀問答、科別判定、門診推薦、快速查詢、語音互動，以及臺北榮民總醫院 App 的掛號與取消導引。以下說明依目前程式碼整理；實際班表由 SQL Server 提供，語音辨識與合成由外部 gateway 提供。

## 系統架構

```text
Android（Kotlin / Jetpack Compose）
  ├─ 問診、需求確認、醫師選擇、快速查詢、歷史紀錄
  ├─ HTTP API → FastAPI
  │                ├─ 問診狀態、語意抽取、科別推論與急迫性初步篩檢
  │                ├─ SQL Server：正式科別、醫師與班表
  │                ├─ Cerebras / Gemini：可選 AI 語意處理
  │                └─ 外部 voice gateway：ASR / TTS
  └─ AccessibilityService / Overlay → 醫院 App
```

| 目錄 | 用途 |
| --- | --- |
| `android/` | Android 介面、API 呼叫、語音播放、掛號與取消導引 |
| `backend/app/routes/` | HTTP 路由 |
| `backend/app/services/` | 問診、語意、推薦、班表過濾、TTAS、導引腳本與語音服務 |
| `backend/app/db.py` | SQL Server 資料查詢與正式班表重新驗證 |
| `backend/app/schemas.py` | 請求、回應與案件資料模型 |
| `backend/knowledge/` | 科別知識、來源與版本化 TTAS 規則 |
| `backend/tests/` | 後端測試 |
| `scripts/` | Windows PowerShell 安裝、啟動與測試腳本 |
| `docs/` | 科別盤點與篩檢驗證文件 |

## 主要運作流程

### 症狀問診與推薦

1. Android 呼叫 `POST /chat` 建立案件，後續使用相同 `case_id` 延續問診。就診類型包含 `initial`、`followup`、`return_visit`；快速查詢使用獨立流程。
2. 後端蒐集症狀、部位、持續時間、嚴重度、危險徵象與日期／時段偏好。規則引擎配合可用的 AI 語意抽取處理回答，資訊不完整時追問或澄清。
3. 科別推論使用正式科別資料與科別知識。無法收斂到適當科別時，回傳進一步澄清或手動選科提示。
4. TTAS evaluator 只使用通過驗證的證據與可信年齡資料，執行已啟用規則。證據不足或未命中時回傳 `insufficient_information`，不預設為第四或第五級。此功能是初步篩檢，並非完整臨床分級。
5. 使用者確認需求後，呼叫 `POST /recommend`。後端檢查案件狀態、確認與篩檢條件，查詢正式可掛號班表，產生「專長優先」與「時間優先」推薦。
6. 選定推薦後呼叫 `POST /generate_script`，重新驗證班表，再產生開啟醫院 App、選擇科別、日期、醫師並進入個人資料頁的腳本。
7. Android 透過無障礙服務與浮動提示執行導引。正式掛號結果以醫院 App 顯示為準。

案件階段為 `collecting` → `waiting_confirmation` → `recommending` → `script_ready` → `done`。後端保留案件與推薦資料，並檢查案件 ID、就診類型與推薦關聯。

### 快速查詢

- `GET /reference/departments`、`GET /reference/doctors` 提供正式科別與醫師清單。
- `POST /followup/recommend` 依指定科別、醫師與時間偏好查詢回診方案。
- `GET /schedules/search` 依 `dept_id`、`date`、`period` 查詢班表；時段為 `morning`、`afternoon`、`evening`。
- 快速查詢選定的班表也須經 `/generate_script` 重新驗證，才可進入導引。
- Android 以 SharedPreferences 保存歷史紀錄，並提供取消掛號導引流程。

### 語音

Android 錄音後可呼叫 `/voice/asr` 取得辨識文字，或呼叫 `/voice/chat` 串接辨識與問診。`/voice/tts` 提供語音合成；啟用固定句快取時，可回傳對應 Android 本地音檔的 `audio_id`。

TTS session 快取保存在記憶體，閒置期限為 30 分鐘；`/voice/tts/cleanup` 可清除會話。後端每 60 秒清理過期快取。語音 gateway 的服務實作不包含在本專案內。

## API

| 方法 | 路徑 | 用途 |
| --- | --- | --- |
| GET | `/health` | API 程序健康狀態 |
| POST | `/chat` | 多輪問診、澄清與確認 |
| POST | `/recommend` | 產生門診推薦 |
| POST | `/followup/recommend` | 回診推薦 |
| GET | `/reference/departments` | 正式科別清單 |
| GET | `/reference/doctors` | 指定科別的醫師清單 |
| GET | `/schedules/search` | 指定科別、日期、時段的班表 |
| POST | `/generate_script` | 驗證班表並產生 Android 導引腳本 |
| POST | `/voice/asr` | 上傳音訊進行語音辨識 |
| POST | `/voice/chat` | 語音問診 |
| POST | `/voice/tts` | 語音合成或固定句音檔識別 |
| POST | `/voice/tts/cleanup` | 清除 TTS session |
| GET | `/voice/health` | 語音功能狀態 |

啟動後可於 `http://localhost:8080/docs` 查看完整 OpenAPI 請求與回應格式。`/health` 回傳 `{"status":"ok"}`，不代表 SQL、AI 或語音 gateway 已可用。

## 本機安裝與啟動

### 環境需求

- Python 3.11（`.env.example` 記錄版本為 3.11.9）。
- SQL Server 或 Azure SQL 連線與 Microsoft ODBC Driver 17 for SQL Server。
- Android SDK：compile SDK 36.1、target SDK 36、min SDK 24；Gradle daemon 設定使用 JDK 21。
- 若需 AI 語意功能，提供對應供應商 API key；若需語音，提供可連線的 voice gateway。

在專案根目錄執行：

```powershell
.\scripts\setup.ps1
Copy-Item .env.example backend\.env
```

編輯 `backend/.env` 後啟動：

```powershell
.\scripts\run-backend.ps1 -Port 8080
```

腳本預設 port 是 8000，但 Android 預設 API 位址使用 8080，因此上例明確指定 8080。亦可在 `backend/` 使用：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

`backend/main.py` 的直接啟動入口同樣使用 8080。請保留單一 worker，因目前案件與語音快取未跨程序共享。

### 設定

設定由 `backend/app/config.py` 載入 `backend/.env`；根目錄 `.env.example` 是設定範本。`PYTHON_VERSION` 是環境版本提示，不會自動安裝 Python。

| 設定 | 用途／預設 |
| --- | --- |
| `DB_DRIVER`, `DB_SERVER`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | 正式 SQL 連線；請填入實際環境 |
| `DB_TRUST_SERVER_CERTIFICATE` | 預設 false；本機 SQL 主機另有信任憑證處理 |
| `AI_PROVIDER` | 預設 `cerebras`，也支援 `gemini` |
| `CEREBRAS_API_KEY`, `CEREBRAS_MODEL` | Cerebras 認證；模型預設 `gpt-oss-120b` |
| `GOOGLE_API_KEY`, `LLM_MODEL` | Gemini 認證；模型預設 `gemini-3.5-flash` |
| `AI_TIMEOUT_SECONDS` | 預設 8 秒 |
| `BATCH_TRIAGE_ENABLED`, `BATCH_EXTRACTION_PROVIDER` | 批次問答預設關閉；抽取供應商預設 `cerebras` |
| `AI_REPLY_GENERATION_ENABLED`, `AI_DOCTOR_SCORING_ENABLED` | AI 回覆生成與醫師評分預設關閉 |
| `DEPLOY_MODE`, `DISABLE_LOCAL_EMBEDDING`, `EMBEDDING_MODEL` | 部署與本地 embedding 設定；範本停用本地 embedding |
| `VOICE_ENABLED` | 預設 false |
| `VOICE_GATEWAY_URL`, `VOICE_GATEWAY_KEY` | 外部語音服務位址與認證 |
| `VOICE_GATEWAY_IS_NGROK` | gateway 通道設定，預設 false |
| `VOICE_DEFAULT_LANG`, `VOICE_TIMEOUT` | 預設 `taiwanese`、60 秒 |
| `VOICE_USE_FIXED_CACHE` | 預設 false；啟用前須確認 Android 本地音檔對應 |

`requirements.txt` 是基本執行依賴；`requirements-dev.txt` 加入測試依賴。需要本地 RAG／embedding 時再安裝 `requirements-ai.txt`，其中包含較大的模型依賴。

SQL 查詢使用 `DepartmentCategory`、`Department`、`Doctor`、`Schedule` 等正式資料表。資料庫不可用時，相關 API 回傳 503；空班表與資料來源故障分開處理。此專案沒有隨附建立與填入正式資料庫的完整腳本。

### Android 設定

用 Android Studio 開啟 `android/`，設定 SDK 後同步 Gradle。可在 `android/local.properties` 加入：

```properties
API_BASE_URL=http://10.0.2.2:8080
```

此為 Android 模擬器存取電腦 localhost 的位址。實機測試須改為實機可連線的電腦區網 IP 或部署 URL；修改後重新建置。

掛號導引依賴醫院 App `tw.com.bicom.VGHTPE`，並須依介面提示啟用無障礙服務、浮動視窗及錄音等必要權限。導引依醫院 App 的畫面文字與節點尋找目標，介面改版時需要重新驗證。

`android/app/src/main/assets/teacher_profiles.json` 提供本地醫師介紹；正式科別、醫師選項與可掛號班表透過後端查詢。

## 驗證與測試

在根目錄執行既有測試腳本：

```powershell
.\scripts\test-backend.ps1
.\scripts\test-android.ps1
.\scripts\test-all.ps1
```

後端腳本執行既有指定測試及健康／OpenAPI 契約檢查；Android 腳本執行 `testDebugUnitTest` 與 `assembleDebug`。裝置測試位於 `android/app/src/androidTest/`，需要另行準備實機或模擬器。測試通過仍需用實際 SQL、AI、gateway 與醫院 App 驗證完整整合流程。

## 目前運作限制

- 案件與推薦保存在程序記憶體，案件閒置期限為兩小時；重新啟動會遺失，且多 worker 不共享。正式多實例部署需改用共用儲存。
- AI 初始化失敗時後端會記錄警告並保留可用的 fallback，但正式資料查詢仍依賴 SQL；不能把 API 啟動成功視為完整服務可用。
- 目前 CORS 允許所有來源；部署時須依實際存取範圍調整，並自行配置認證與連線保護。
- TTAS 規則僅涵蓋知識包內已啟用的情境；相關依據與驗證說明見 `backend/knowledge/ttas/README.md`、`docs/PHASE5_TTAS_VALIDATION.md`。
- 個人資料填寫、醫院端提交與最終結果仍由醫院 App 處理；後端生成導引腳本不等於已完成掛號。
