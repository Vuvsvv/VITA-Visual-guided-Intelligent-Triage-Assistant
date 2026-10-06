#!/bin/bash
# ============================================
# AI 模型服務 - 日常管理工具
# 用法: sudo bash manage.sh [命令]
# ============================================

SERVICES="ai-taiwanese-asr ai-taiwanese-tts ai-chinese-tts ai-gateway"
LOG_DIR="/var/log/ai-gateway"

case "$1" in

  status)
    echo "服務狀態:"
    echo "────────────────────────────────────"
    for svc in $SERVICES; do
        status=$(systemctl is-active $svc 2>/dev/null)
        if [ "$status" = "active" ]; then
            echo "  🟢 $svc"
        else
            echo "  🔴 $svc ($status)"
        fi
    done
    echo ""
    echo "Nginx: $(systemctl is-active nginx)"
    echo "SSL 到期: $(certbot certificates 2>/dev/null | grep 'Expiry' | head -1)"
    ;;

  start)
    echo "啟動所有服務..."
    systemctl start ai-taiwanese-asr ai-taiwanese-tts ai-chinese-tts
    sleep 5
    systemctl start ai-gateway
    echo "完成"
    ;;

  stop)
    echo "停止所有服務..."
    systemctl stop $SERVICES
    echo "完成"
    ;;

  restart)
    echo "重啟所有服務..."
    systemctl restart ai-taiwanese-asr ai-taiwanese-tts ai-chinese-tts
    sleep 5
    systemctl restart ai-gateway
    echo "完成"
    ;;

  logs)
    svc=${2:-gateway}
    echo "顯示 ${svc} 日誌 (Ctrl+C 退出)..."
    tail -f $LOG_DIR/${svc}.log
    ;;

  logs-all)
    echo "顯示所有日誌 (Ctrl+C 退出)..."
    tail -f $LOG_DIR/*.log
    ;;

  test)
    DOMAIN=$(grep server_name /etc/nginx/sites-available/ai-gateway 2>/dev/null | awk '{print $2}' | tr -d ';')
    if [ -z "$DOMAIN" ]; then
        DOMAIN="localhost:8000"
        PROTO="http"
    else
        PROTO="https"
    fi

    echo "測試 API ($PROTO://$DOMAIN)..."
    echo ""
    echo "健康檢查:"
    curl -s "$PROTO://$DOMAIN/health" | python3 -m json.tool 2>/dev/null || echo "失敗"
    echo ""
    echo "服務列表:"
    curl -s "$PROTO://$DOMAIN/services" | python3 -m json.tool 2>/dev/null || echo "失敗"
    ;;

  update-key)
    read -p "新的 API Key: " NEW_KEY
    sed -i "s/^API_KEY=.*/API_KEY=${NEW_KEY}/" /opt/ai-gateway/.env
    systemctl restart ai-gateway
    echo "API Key 已更新並重啟網關"
    ;;

  renew-ssl)
    certbot renew --quiet
    systemctl reload nginx
    echo "SSL 憑證已更新"
    ;;

  *)
    echo "AI 模型服務管理工具"
    echo ""
    echo "用法: sudo bash manage.sh [命令]"
    echo ""
    echo "  status       查看所有服務狀態"
    echo "  start        啟動所有服務"
    echo "  stop         停止所有服務"
    echo "  restart      重啟所有服務"
    echo "  logs [名稱]  查看日誌 (gateway/taiwanese-asr/taiwanese-tts/chinese-tts)"
    echo "  logs-all     查看所有日誌"
    echo "  test         測試 API 是否正常"
    echo "  update-key   更新 API Key"
    echo "  renew-ssl    更新 SSL 憑證"
    ;;
esac
