#!/bin/bash
set -e

echo "╔══════════════════════════════════════════╗"
echo "║   AI 模型服務網關 - 啟動所有服務          ║"
echo "╚══════════════════════════════════════════╝"
echo ""

if [ -f .env ]; then
    # 用 `set -a; . ./.env` 而不是 export $(cat .env | xargs)：
    # 後者遇到含空格或引號的值（例如 DEFAULT_SPEAKER_TEXT）會整個爆掉
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
    echo "✓ 已載入 .env"
fi

mkdir -p .pids

cleanup() {
    echo ""
    echo "正在停止所有服務..."
    for pidfile in .pids/*.pid; do
        [ -f "$pidfile" ] || continue
        pid=$(cat "$pidfile")
        name=$(basename "$pidfile" .pid)
        kill -0 "$pid" 2>/dev/null && kill "$pid" && echo "  ✓ 已停止 $name"
        rm -f "$pidfile"
    done
    exit 0
}
trap cleanup SIGINT SIGTERM

# 等某個服務的 / 端點回 200（模型載入可能要好幾分鐘，固定 sleep 5 根本不夠）
wait_ready() {
    local name=$1 url=$2 timeout=${3:-600} waited=0
    printf "  等待 %s 就緒" "$name"
    while [ $waited -lt "$timeout" ]; do
        if curl -sf -m 3 "$url" > /dev/null 2>&1; then
            echo " ✓ (${waited}s)"
            return 0
        fi
        printf "."
        sleep 3
        waited=$((waited + 3))
    done
    echo " ⚠️ 逾時（${timeout}s），繼續往下走"
    return 0
}

echo "[1/5] 台語 ASR — Breeze-ASR-26 (port 8001)..."
python services/breezy_asr/main.py &
echo $! > .pids/taiwanese_asr.pid



echo "[3/5] 台語 TTS — Gemini + Chatterbox (port 8003)..."
echo "  ℹ️  本服務只是 HTTP client，需要 Chatterbox 引擎已經在 ${CB_API:-http://127.0.0.1:9881} 跑著："
echo "     conda activate cb && CB_HOST=127.0.0.1 CB_ROOT=~/chatterbox-finetuning python services/taiwanese_tts/engine/cb_server.py"
python services/taiwanese_tts/main.py &
echo $! > .pids/taiwanese_tts.pid

echo "[4/5] 中文 TTS — BreezyVoice (port 8004)..."
python services/breezy_tts/main.py &
echo $! > .pids/chinese_tts.pid

echo ""
echo "等待微服務載入模型..."
wait_ready "台語 ASR"  "http://127.0.0.1:8001/"

wait_ready "台語 TTS"  "http://127.0.0.1:8003/" 60
wait_ready "中文 TTS"  "http://127.0.0.1:8004/"

echo ""
echo "[5/5] API 網關 (port 8000)..."
python gateway.py &
echo $! > .pids/gateway.pid

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  所有服務已啟動！                         ║"
echo "║                                          ║"
echo "║  網關入口:  http://localhost:8000         ║"
echo "║  API 文件:  http://localhost:8000/docs    ║"
echo "║                                          ║"
echo "║  台語 ASR:  :8001  (Breeze-ASR-26)        ║"
echo "║  台語 TTS:  :8003  (Gemini+Chatterbox)    ║"
echo "║  中文 TTS:  :8004  (BreezyVoice)          ║"
echo "║                                          ║"
echo "║  按 Ctrl+C 停止所有服務                   ║"
echo "╚══════════════════════════════════════════╝"
echo ""
wait
