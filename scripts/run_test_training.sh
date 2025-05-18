#!/bin/bash
# run_test_training.sh - Script para ejecutar un entrenamiento de prueba
#
# Este script compila y ejecuta un pipeline de Vertex AI con parámetros reducidos
# para verificar rápidamente que todo funciona correctamente.

set -e  # Salir inmediatamente si un comando falla

echo "🚀 Ejecutando pipeline de entrenamiento de prueba para BTC Trading Bot..."
echo "---------------------------------------------------------------------"

# Parse command line arguments
EXECUTE_LOCALLY=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --local) EXECUTE_LOCALLY=true; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
done

# Establecer valores predeterminados
PROJECT_ID=$(grep 'project_id' ../terraform/terraform.tfvars | cut -d'"' -f2 || echo "local-project")
REGION="europe-southwest1"  # Región predeterminada según configuración de gcloud
SERVICE_ACCOUNT="btc-trading-bot-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Verificar que Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 no está instalado. Por favor instala Python3."
    exit 1
fi

# Cambiar al directorio raíz del proyecto
cd "$(dirname "$0")/.."
echo "📁 Cambiado al directorio: $(pwd)"

# Instalar dependencias si es necesario
echo "📦 Verificando e instalando dependencias necesarias..."
pip3 install -r requirements.txt

# Compilar el pipeline con configuración para prueba
echo "🔧 Compilando pipeline para pruebas..."
python3 scripts/compile_safe_pipeline.py --output-file pipeline_test.json --project-id $PROJECT_ID

# Obtener los nombres de los buckets desde terraform
ARTIFACTS_BUCKET="${PROJECT_ID}-btc-artifacts"

# Parámetros para el entrenamiento de prueba
echo "⚙️ Configurando parámetros de prueba..."
PARAMETERS="project_id=${PROJECT_ID},\
region=${REGION},\
gcs_bucket=${ARTIFACTS_BUCKET},\
download_start_date=2022-01-01,\
download_end_date=2022-01-31,\
training_steps=5000,\
n_backtest_episodes=1"

# Ejecutar el pipeline, ya sea localmente o en Vertex AI
if [ "$EXECUTE_LOCALLY" = true ]; then
    echo "🏠 Ejecutando entrenamiento localmente para pruebas..."
    
    # Crear directorios temporales para simular GCS
    mkdir -p tmp/data/raw tmp/data/processed/train tmp/data/processed/test tmp/models tmp/metrics tmp/plots
    
    # Ejecutar componentes localmente para pruebas
    echo "📊 Ejecutando componente de adquisición de datos..."
    python3 -m src.components.run_data_acquisition \
        --project_id ${PROJECT_ID} \
        --symbol "BTCUSDT" \
        --interval "1h" \
        --start_date "2022-01-01" \
        --end_date "2022-01-31" \
        --gcs_bucket "${ARTIFACTS_BUCKET}" \
        --api_key "test-api-key" \
        --api_secret "test-api-secret" \
        --data_output_path "./tmp/data/raw/metadata.json"
    
    echo "🔄 Ejecutando componente de preprocesamiento..."
    python3 -m src.components.run_preprocessing \
        --project_id ${PROJECT_ID} \
        --input_data_uri "./tmp/data/raw" \
        --output_gcs_bucket "${ARTIFACTS_BUCKET}" \
        --train_test_split_date "2022-01-15" \
        --sequence_length 100 \
        --train_dataset_output_path "./tmp/data/processed/train/metadata.json" \
        --test_dataset_output_path "./tmp/data/processed/test/metadata.json"
    
    echo "🧠 Ejecutando componente de entrenamiento del agente..."
    ENV_PARAMS='{"project_id":"'${PROJECT_ID}'","gcs_processed_data_uri":"./tmp/data/processed/train","initial_balance_usd":10000.0,"max_position_btc":1.0,"commission_rate":0.0004,"max_leverage":20,"random_episode_start":true,"episode_steps":1000,"slippage_model":"atr_based","slippage_factor":0.05}'
    TRANSFORMER_PARAMS='{"n_heads":4,"n_layers":2,"d_model":64}'
    
    python3 -m src.components.run_train_agent \
        --project-id ${PROJECT_ID} \
        --train-data-uri "./tmp/data/processed/train" \
        --env-params "$ENV_PARAMS" \
        --transformer-params "$TRANSFORMER_PARAMS" \
        --training-steps 5000 \
        --output-model-uri "./tmp/models" \
        --model-output-path "./tmp/models/metadata.json" \
        --metrics-output-path "./tmp/metrics/training_metrics.json"
    
    echo "🧪 Ejecutando componente de backtesting..."
    python3 -m src.components.run_backtest_agent \
        --project-id ${PROJECT_ID} \
        --model-uri "./tmp/models" \
        --test-data-uri "./tmp/data/processed/test" \
        --env-params "$ENV_PARAMS" \
        --n-episodes 1 \
        --metrics-output-path "./tmp/metrics/backtest_metrics.json" \
        --plots-output-path "./tmp/plots"
    
    echo "✅ Prueba local completada!"
else
    # Verificar que gcloud está instalado
    if ! command -v gcloud &> /dev/null; then
        echo "❌ Error: Google Cloud SDK (gcloud) no está instalado. Por favor instálalo o ejecuta con --local."
        exit 1
    fi

    # Ejecutar el pipeline de prueba en Vertex AI
    echo "🚀 Ejecutando pipeline de prueba en Vertex AI..."
    gcloud ai pipeline-jobs create \
      --pipeline-file pipeline_test.json \
      --display-name "btc-trading-bot-test-run-$(date +%Y%m%d-%H%M%S)" \
      --project ${PROJECT_ID} \
      --region ${REGION} \
      --parameter-values=${PARAMETERS} \
      --service-account=${SERVICE_ACCOUNT}

    echo "✅ Pipeline de prueba lanzado correctamente!"
    echo "📊 Puedes monitorizar el progreso en la consola de GCP:"
    echo "    https://console.cloud.google.com/vertex-ai/pipelines/runs?project=${PROJECT_ID}&region=${REGION}"
fi
