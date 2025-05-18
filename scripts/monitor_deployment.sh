#!/bin/bash
# monitor_deployment.sh - Monitoreo continuo del Bitcoin Trading Bot en GCP
#
# Este script proporciona un monitoreo continuo del estado del despliegue
# y las métricas de rendimiento del bot de trading.

set -e  # Exit immediately if a command exits with a non-zero status

echo "🔍 Iniciando monitoreo continuo del Bitcoin Trading Bot..."
echo "=================================================="

# Get configuration from terraform.tfvars
cd "$(dirname "$0")/../terraform"
PROJECT_ID=$(grep 'project_id' terraform.tfvars | cut -d'"' -f2)
REGION=$(grep 'region' terraform.tfvars | cut -d'"' -f2)
cd ..

# Función para obtener el último pipeline ejecutado
get_latest_pipeline() {
    gcloud ai pipeline-jobs list --region="$REGION" --project="$PROJECT_ID" \
        --filter="display_name~btc-trading-bot" \
        --format="value(name)" --limit=1 2>/dev/null | head -1
}

# Función para obtener el estado del pipeline
get_pipeline_state() {
    local pipeline_id=$1
    gcloud ai pipeline-jobs describe "$pipeline_id" --region="$REGION" --project="$PROJECT_ID" \
        --format="value(state)" 2>/dev/null
}

# Función para buscar métricas en el último backtest
get_backtest_metrics() {
    local metrics_bucket="${PROJECT_ID}-btc-artifacts"
    local latest_metrics_file=$(gsutil ls -l "gs://$metrics_bucket/metrics/" 2>/dev/null | sort -k 2 -r | head -1 | awk '{print $3}')
    
    if [ -n "$latest_metrics_file" ]; then
        echo "📊 Últimas métricas de backtest ($latest_metrics_file):"
        gsutil cat "$latest_metrics_file" 2>/dev/null | grep -E "total_profit|sharpe_ratio|max_drawdown|win_rate" || echo "No se encontraron métricas específicas."
    else
        echo "❌ No se encontraron archivos de métricas."
    fi
}

# Función para mostrar el uso de recursos
get_resource_usage() {
    echo "💻 Uso de recursos en GCP:"
    
    # Storage usage
    echo "  Almacenamiento:"
    for bucket in "${PROJECT_ID}-btc-raw-data" "${PROJECT_ID}-btc-processed-data" "${PROJECT_ID}-btc-models" "${PROJECT_ID}-btc-artifacts" "${PROJECT_ID}-btc-tensorboard"; do
        size=$(gsutil du -s "gs://$bucket" 2>/dev/null | awk '{print $1}')
        if [ -n "$size" ]; then
            # Convert to MB or GB for readability
            if [ "$size" -gt 1048576 ]; then
                size_gb=$(echo "scale=2; $size/1048576" | bc)
                echo "    $bucket: ${size_gb}GB"
            else
                size_mb=$(echo "scale=2; $size/1024" | bc)
                echo "    $bucket: ${size_mb}MB"
            fi
        else
            echo "    $bucket: N/A"
        fi
    done
}

# Monitoreo continuo
interval=${1:-60}  # Intervalo de monitoreo en segundos (por defecto 60)
echo "⏱️ Intervalo de monitoreo: $interval segundos (Ctrl+C para detener)"
echo ""

while true; do
    clear
    echo "🤖 Bitcoin Trading Bot - Panel de Monitoreo"
    echo "=================================================="
    echo "📅 Fecha: $(date)"
    echo "🔍 Proyecto: $PROJECT_ID ($REGION)"
    echo ""
    
    # Verificar el último pipeline
    latest_pipeline=$(get_latest_pipeline)
    if [ -n "$latest_pipeline" ]; then
        state=$(get_pipeline_state "$latest_pipeline")
        echo "🔄 Último pipeline: $latest_pipeline"
        echo "📊 Estado: $state"
        
        # Si el pipeline está en ejecución, mostrar más detalles
        if [ "$state" = "PIPELINE_STATE_RUNNING" ]; then
            runtime=$(gcloud ai pipeline-jobs describe "$latest_pipeline" --region="$REGION" --project="$PROJECT_ID" \
                --format="value(createTime)" 2>/dev/null)
            echo "⏱️ En ejecución desde: $runtime"
        fi
    else
        echo "❌ No se encontraron pipelines."
    fi
    echo ""
    
    # Mostrar métricas de backtest
    get_backtest_metrics
    echo ""
    
    # Mostrar uso de recursos
    get_resource_usage
    echo ""
    
    echo "=================================================="
    echo "Actualizando en $interval segundos... (Ctrl+C para detener)"
    sleep "$interval"
done
