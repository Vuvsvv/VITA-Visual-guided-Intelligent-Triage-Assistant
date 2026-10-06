#!/bin/bash
# ============================================
# AI 模型服務 - 生產環境一鍵部署腳本
# 適用於 Ubuntu 22.04 / 24.04 VPS
# ============================================

set -e

# ── 顏色 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  AI 模型服務 - 生產環境部署                    ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 檢查是否為 root ──
if [ "$EUID" -ne 0 ]; then
    error "請用 sudo 執行此腳本: sudo bash deploy.sh"
fi

# ── 收集設定 ──
read -p "你的域名（例如 api.example.com）: " DOMAIN
read -p "你的 Email（用於 SSL 憑證）: " EMAIL
read -p "你的 Gemini API Key: " GEMINI_KEY
read -p "設定一組 API Key（後端呼叫用）: " API_KEY

if [ -z "$DOMAIN" ] || [ -z "$EMAIL" ] || [ -z "$GEMINI_KEY" ] || [ -z "$API_KEY" ]; then
    error "所有欄位都是必填的"
fi

APP_DIR="/opt/ai-gateway"
APP_USER="aigateway"

info "開始部署到 ${DOMAIN}..."
echo ""

# ============================================
# 步驟 1: 系統更新 + 安裝依賴
# ============================================
info "[1/7] 安裝系統依賴..."

apt-get update -qq
apt-get install -y -qq \
    python3 python3-pip python3-venv \
    nginx certbot python3-certbot-nginx \
    ffmpeg libsndfile1 \
    ufw git curl > /dev/null

info "系統依賴安裝完成"

# ============================================
# 步驟 2: 建立應用使用者和目錄
# ============================================
info "[2/7] 建立應用目錄..."

# 建立專用使用者（如果不存在）
id -u $APP_USER &>/dev/null || useradd -r -m -s /bin/bash $APP_USER

# 建立目錄結構
mkdir -p $APP_DIR
mkdir -p /var/log/ai-gateway

# 複製檔案（假設當前目錄有所有檔案）
# ⚠️ 用 VPS 版設定檔：services_config.yaml / _docker.yaml 裡的 host.docker.internal
#    在沒有 Docker 的機器上不會解析（見審查報告 P4）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp -r "$SCRIPT_DIR"/{gateway.py,security.py,requirements_gateway.txt} $APP_DIR/
cp "$SCRIPT_DIR"/services_config_vps.yaml $APP_DIR/services_config.yaml
cp -r "$SCRIPT_DIR"/services $APP_DIR/

# 建立 .env
cat > $APP_DIR/.env << EOF
GEMINI_API_KEY=${GEMINI_KEY}
API_KEY=${API_KEY}
ALLOWED_ORIGINS=https://${DOMAIN}
RATE_LIMIT=60
RATE_WINDOW=60
GATEWAY_CONFIG=${APP_DIR}/services_config.yaml
# Nginx 就在前面一層，讓網關用 X-Forwarded-For 的最外層那一跳做速率限制分桶
TRUSTED_PROXY_HOPS=1
FORWARDED_ALLOW_IPS=127.0.0.1
EOF

chmod 600 $APP_DIR/.env
chown -R $APP_USER:$APP_USER $APP_DIR
chown -R $APP_USER:$APP_USER /var/log/ai-gateway

info "應用目錄建立完成: $APP_DIR"

# ============================================
# 步驟 3: Python 虛擬環境 + 安裝依賴
# ============================================
info "[3/7] 安裝 Python 依賴（這可能需要幾分鐘）..."

sudo -u $APP_USER python3 -m venv $APP_DIR/venv

# 網關依賴
sudo -u $APP_USER $APP_DIR/venv/bin/pip install -q --upgrade pip
sudo -u $APP_USER $APP_DIR/venv/bin/pip install -q -r $APP_DIR/requirements_gateway.txt

# 各服務依賴
for svc_dir in $APP_DIR/services/*/; do
    if [ -f "$svc_dir/requirements.txt" ]; then
        svc_name=$(basename $svc_dir)
        info "  安裝 ${svc_name} 的依賴..."
        sudo -u $APP_USER $APP_DIR/venv/bin/pip install -q -r "$svc_dir/requirements.txt"
    fi
done

info "Python 依賴安裝完成"

# ============================================
# 步驟 4: 建立 systemd 服務
# ============================================
info "[4/7] 建立 systemd 服務..."

# ── 台語 ASR ──
cat > /etc/systemd/system/ai-taiwanese-asr.service << EOF
[Unit]
Description=Taiwanese ASR Service
After=network.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/uvicorn services.breezy_asr.main:app --host 127.0.0.1 --port 8001
Restart=always
RestartSec=5
StandardOutput=append:/var/log/ai-gateway/taiwanese-asr.log
StandardError=append:/var/log/ai-gateway/taiwanese-asr.log

[Install]
WantedBy=multi-user.target
EOF

# ── 中文 ASR ──
[Unit]
Description=Chinese ASR Service
After=network.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# ── 台語 TTS ──
cat > /etc/systemd/system/ai-taiwanese-tts.service << EOF
[Unit]
Description=Taiwanese TTS Service
After=network.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/uvicorn services.taiwanese_tts.main:app --host 127.0.0.1 --port 8003
Restart=always
RestartSec=5
StandardOutput=append:/var/log/ai-gateway/taiwanese-tts.log
StandardError=append:/var/log/ai-gateway/taiwanese-tts.log

[Install]
WantedBy=multi-user.target
EOF

# ── 中文 TTS ──
# 設定檔把 chinese_tts 標成 enabled，就一定要有對應的 unit，否則 /health 永遠 unhealthy
cat > /etc/systemd/system/ai-chinese-tts.service << EOF
[Unit]
Description=Chinese TTS Service (BreezyVoice)
After=network.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/uvicorn services.breezy_tts.main:app --host 127.0.0.1 --port 8004
Restart=always
RestartSec=5
StandardOutput=append:/var/log/ai-gateway/chinese-tts.log
StandardError=append:/var/log/ai-gateway/chinese-tts.log

[Install]
WantedBy=multi-user.target
EOF

# ── API 網關 ──
cat > /etc/systemd/system/ai-gateway.service << EOF
[Unit]
Description=AI Model Gateway

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$APP_DIR/venv/bin/uvicorn gateway:app --host 127.0.0.1 --port 8000 --proxy-headers --forwarded-allow-ips 127.0.0.1
Restart=always
RestartSec=5
StandardOutput=append:/var/log/ai-gateway/gateway.log
StandardError=append:/var/log/ai-gateway/gateway.log

[Install]
WantedBy=multi-user.target
EOF

# 重新載入 + 啟用
systemctl daemon-reload

info "systemd 服務建立完成"

# ============================================
# 步驟 5: 設定 Nginx 反向代理
# ============================================
info "[5/7] 設定 Nginx..."

cat > /etc/nginx/sites-available/ai-gateway << 'NGINX'
server {
    listen 80;
    server_name DOMAIN_PLACEHOLDER;

    # 檔案上傳大小限制 —— 必須和網關的 MAX_FILE_SIZE_MB（預設 50）一致，
    # 兩邊不同步的話會出現「Nginx 放行但網關回 413」或反過來的怪現象
    client_max_body_size 50M;

    # 安全 headers
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # 隱藏 Nginx 版本
    server_tokens off;

    # 封鎖常見攻擊路徑
    location ~ /\.(git|env|htaccess) {
        deny all;
        return 404;
    }

    # API 端點
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 超時設定（ASR/TTS 可能需要較長時間）
        # ⚠️ 必須 >= services_config_vps.yaml 裡最大的 timeout（目前 TTS 是 300s），
        #    否則長文合成會在 Nginx 這一層先 504
        proxy_connect_timeout 10s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;

        # 禁止緩存 API 回應
        proxy_no_cache 1;
        proxy_cache_bypass 1;
    }

    # 封鎖直接訪問微服務端口
    location ~ ^/(8001|8002|8003|8004) {
        deny all;
    }
}
NGINX

# 替換域名
sed -i "s/DOMAIN_PLACEHOLDER/${DOMAIN}/g" /etc/nginx/sites-available/ai-gateway

# 啟用站點
ln -sf /etc/nginx/sites-available/ai-gateway /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# 測試 Nginx 設定
nginx -t || error "Nginx 設定有誤"
systemctl restart nginx

info "Nginx 設定完成"

# ============================================
# 步驟 6: 防火牆
# ============================================
info "[6/7] 設定防火牆..."

# ⚠️ 不做 `ufw --force reset` —— 那會清掉這台機器上既有的所有規則，
#    在別人的 VPS 上非常危險。這裡只「加上」需要的規則。
ufw allow ssh          # SSH
ufw allow 80/tcp       # HTTP（SSL 驗證需要）
ufw allow 443/tcp      # HTTPS

# 明確擋掉內部服務的 port。
# 8000 只給本機的 Nginx 反向代理；8001-8004 的微服務「完全沒有 API Key 檢查」，
# 對外開放等於讓人繞過網關直接呼叫模型；9881 是 Chatterbox 台語引擎（9880 是舊的 GPT-SoVITS，一併擋著）。
# ufw 預設放行 loopback，所以本機之間（Nginx→網關、網關→微服務）不受影響。
for port in 8000 8001 8002 8003 8004 9880 9881; do
    ufw deny ${port}/tcp > /dev/null
done

if ufw status | grep -q "Status: active"; then
    info "ufw 已在執行，只加上必要規則（不動既有設定）"
else
    ufw default deny incoming
    ufw default allow outgoing
    ufw --force enable
fi

info "防火牆設定完成（對外只開 SSH + HTTP + HTTPS；8000-8004、9880、9881 已明確擋下）"

# ============================================
# 步驟 7: SSL 憑證（Let's Encrypt）
# ============================================
info "[7/7] 取得 SSL 憑證..."

certbot --nginx \
    -d $DOMAIN \
    --email $EMAIL \
    --agree-tos \
    --non-interactive \
    --redirect

# 自動續約
systemctl enable certbot.timer

info "SSL 憑證取得成功"

# ============================================
# 啟動所有服務
# ============================================
info "啟動所有服務..."

systemctl start ai-taiwanese-asr
systemctl start ai-taiwanese-tts
systemctl start ai-chinese-tts
sleep 5  # 等微服務啟動（模型載入可能要好幾分鐘，網關會顯示 unhealthy 直到就緒）
systemctl start ai-gateway

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  部署完成！                                          ║"
echo "║                                                      ║"
echo "║  API 入口:  https://${DOMAIN}                        "
echo "║  API 文件:  https://${DOMAIN}/docs                   "
echo "║  健康檢查:  https://${DOMAIN}/health                 "
echo "║                                                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
# ⚠️ 刻意不把 API Key 印出來：終端機 screen buffer / CI log 會留下它。
echo "API Key 已寫入 ${APP_DIR}/.env（chmod 600），需要時請用："
echo "  sudo grep '^API_KEY=' ${APP_DIR}/.env"
echo ""
echo "後端呼叫方式:"
echo ""
echo "  curl -X POST https://${DOMAIN}/api/process \\"
echo "    -H \"X-API-Key: \$YOUR_API_KEY\" \\"
echo "    -F 'file=@audio.wav'"
echo ""
echo "常用管理命令:"
echo "  查看狀態:   systemctl status ai-gateway"
echo "  查看日誌:   tail -f /var/log/ai-gateway/gateway.log"
echo ""
