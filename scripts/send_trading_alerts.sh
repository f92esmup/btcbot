#!/bin/bash
# send_trading_alerts.sh - Enviar alertas de trading a servicios externos
#
# Este script lee los resultados del trading y envía alertas a servicios como
# Slack, Telegram o email cuando se realizan operaciones importantes.

set -e  # Exit immediately if a command exits with a non-zero status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuración
RESULTS_DIR="$PROJECT_ROOT/tmp/results"
LATEST_RESULTS_DIR=$(find "$RESULTS_DIR" -maxdepth 1 -type d | sort -r | head -n 1)
TRADING_RESULT_FILE="$LATEST_RESULTS_DIR/latest_trading_result.json"
TRADES_FILE="$LATEST_RESULTS_DIR/trades.csv"

# Webhook URLs (reemplaza con tus propias URLs)
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-""}"
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-""}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-""}"

# Verificar que existe el archivo de resultados
if [ ! -f "$TRADING_RESULT_FILE" ]; then
    echo "❌ Error: No se encontró el archivo de resultados en $TRADING_RESULT_FILE"
    exit 1
fi

# Leer el último resultado de trading
echo "📊 Leyendo resultados de trading de $TRADING_RESULT_FILE"
TIMESTAMP=$(jq -r '.timestamp' "$TRADING_RESULT_FILE")
PRICE=$(jq -r '.price' "$TRADING_RESULT_FILE")
POSITION=$(jq -r '.position' "$TRADING_RESULT_FILE")
BALANCE=$(jq -r '.balance' "$TRADING_RESULT_FILE")
PORTFOLIO_VALUE=$(jq -r '.portfolio_value' "$TRADING_RESULT_FILE")
LAST_ACTION=$(jq -r '.last_action' "$TRADING_RESULT_FILE")

# Formatear mensaje
MESSAGE="🤖 *Bitcoin Trading Bot - Alerta de Trading*\n"
MESSAGE+="📅 *Fecha:* $TIMESTAMP\n"
MESSAGE+="💰 *Precio BTC:* \$${PRICE}\n"
MESSAGE+="🔄 *Acción:* $LAST_ACTION\n"
MESSAGE+="📊 *Posición:* ${POSITION}\n"
MESSAGE+="💵 *Balance:* \$${BALANCE}\n"
MESSAGE+="📈 *Valor portafolio:* \$${PORTFOLIO_VALUE}\n"

# Solo enviar alertas si hubo una operación (no HOLD)
if [ "$LAST_ACTION" != "HOLD" ]; then
    echo "🔔 Operación detectada: $LAST_ACTION - Enviando alertas..."
    
    # Enviar alerta a Slack si está configurado
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        echo "📤 Enviando alerta a Slack..."
        
        # Preparar payload para Slack
        SLACK_PAYLOAD=$(cat <<EOF
{
  "text": "Bitcoin Trading Bot - Alerta",
  "blocks": [
    {
      "type": "header",
      "text": {
        "type": "plain_text",
        "text": "Bitcoin Trading Bot - Alerta de Trading",
        "emoji": true
      }
    },
    {
      "type": "section",
      "fields": [
        {
          "type": "mrkdwn",
          "text": "*Fecha:*\n$TIMESTAMP"
        },
        {
          "type": "mrkdwn",
          "text": "*Precio BTC:*\n\$${PRICE}"
        },
        {
          "type": "mrkdwn",
          "text": "*Acción:*\n$LAST_ACTION"
        },
        {
          "type": "mrkdwn",
          "text": "*Posición:*\n${POSITION}"
        },
        {
          "type": "mrkdwn",
          "text": "*Balance:*\n\$${BALANCE}"
        },
        {
          "type": "mrkdwn",
          "text": "*Valor portafolio:*\n\$${PORTFOLIO_VALUE}"
        }
      ]
    }
  ]
}
EOF
)
        
        # Enviar a Slack
        curl -s -X POST -H "Content-type: application/json" --data "$SLACK_PAYLOAD" "$SLACK_WEBHOOK_URL"
        echo "✅ Alerta enviada a Slack"
    fi
    
    # Enviar alerta a Telegram si está configurado
    if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
        echo "📤 Enviando alerta a Telegram..."
        
        # Formatear mensaje para Telegram (soporta MarkdownV2)
        TELEGRAM_MESSAGE="🤖 *Bitcoin Trading Bot \- Alerta*\n"
        TELEGRAM_MESSAGE+="\n📅 *Fecha:* $TIMESTAMP"
        TELEGRAM_MESSAGE+="\n💰 *Precio BTC:* \$${PRICE}"
        TELEGRAM_MESSAGE+="\n🔄 *Acción:* $LAST_ACTION"
        TELEGRAM_MESSAGE+="\n📊 *Posición:* ${POSITION}"
        TELEGRAM_MESSAGE+="\n💵 *Balance:* \$${BALANCE}"
        TELEGRAM_MESSAGE+="\n📈 *Valor portafolio:* \$${PORTFOLIO_VALUE}"
        
        # Enviar a Telegram
        curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
            -d "chat_id=$TELEGRAM_CHAT_ID" \
            -d "text=$TELEGRAM_MESSAGE" \
            -d "parse_mode=MarkdownV2"
        echo "✅ Alerta enviada a Telegram"
    fi
    
    # Enviar por email usando GCP SendGrid si está configurado
    # Esta parte requeriría configuración adicional y APIs de GCP
    
    echo "✅ Todas las alertas enviadas"
else
    echo "ℹ️ Última acción es HOLD, no se envían alertas"
fi

echo "✅ Script completado"
