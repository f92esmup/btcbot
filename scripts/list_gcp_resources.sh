#!/bin/bash
# list_gcp_resources.sh - Listar todos los recursos del Bitcoin Trading Bot en GCP
#
# Este script muestra una lista de todos los recursos desplegados para el Bitcoin Trading Bot
# en Google Cloud Platform.

set -e  # Exit immediately if a command exits with a non-zero status

echo "📋 Listando recursos del Bitcoin Trading Bot en GCP..."
echo "=================================================="

# Get configuration from terraform.tfvars
cd "$(dirname "$0")/../terraform"
PROJECT_ID=$(grep 'project_id' terraform.tfvars | cut -d'"' -f2)
REGION=$(grep 'region' terraform.tfvars | cut -d'"' -f2)
cd ..

echo "🔍 Proyecto: $PROJECT_ID"
echo "🔍 Región: $REGION"
echo ""

# Listar buckets de almacenamiento
echo "📦 Buckets de Storage:"
echo "--------------------------------------------------"
gsutil ls -p "$PROJECT_ID" | grep "$PROJECT_ID" || echo "No se encontraron buckets."
echo ""

# Listar contenido de los buckets relevantes
echo "📦 Contenido de buckets relevantes:"
echo "--------------------------------------------------"
for bucket in "${PROJECT_ID}-btc-models" "${PROJECT_ID}-btc-artifacts"; do
    if gsutil ls -b "gs://$bucket" &>/dev/null; then
        echo "Contenido de gs://$bucket:"
        gsutil ls -r "gs://$bucket/**" | head -n 20
        count=$(gsutil ls -r "gs://$bucket/**" | wc -l)
        if [ "$count" -gt 20 ]; then
            echo "... y $(($count - 20)) archivos más."
        fi
        echo ""
    fi
done

# Listar imágenes en Artifact Registry
echo "🐳 Imágenes en Artifact Registry:"
echo "--------------------------------------------------"
if gcloud artifacts repositories describe "btc-trading-bot" --project="$PROJECT_ID" --location="$REGION" &>/dev/null; then
    gcloud artifacts docker images list "${REGION}-docker.pkg.dev/${PROJECT_ID}/btc-trading-bot" \
        --project="$PROJECT_ID" --format="table(name, tags, createTime)" || echo "No se encontraron imágenes."
else
    echo "Repositorio btc-trading-bot no encontrado."
fi
echo ""

# Listar pipelines de Vertex AI
echo "🔄 Pipelines de Vertex AI:"
echo "--------------------------------------------------"
gcloud ai pipeline-jobs list --region="$REGION" --project="$PROJECT_ID" \
    --filter="display_name~btc-trading-bot" \
    --format="table(name, display_name, state, create_time)" || echo "No se encontraron pipelines."
echo ""

# Listar Tensorboards
echo "📊 Tensorboards en Vertex AI:"
echo "--------------------------------------------------"
gcloud ai tensorboards list --region="$REGION" --project="$PROJECT_ID" \
    --format="table(name, display_name, create_time)" || echo "No se encontraron tensorboards."
echo ""

# Listar triggers de Cloud Build
echo "🔄 Triggers de Cloud Build:"
echo "--------------------------------------------------"
gcloud builds triggers list --region="$REGION" --project="$PROJECT_ID" \
    --format="table(name, github.name, github.push.branch, filename)" || echo "No se encontraron triggers."
echo ""

# Listar versiones de secretos
echo "🔒 Versiones de secretos:"
echo "--------------------------------------------------"
for secret in binance-api-key binance-api-secret; do
    if gcloud secrets describe "$secret" --project="$PROJECT_ID" &>/dev/null; then
        echo "Versiones de $secret:"
        gcloud secrets versions list "$secret" --project="$PROJECT_ID" \
            --format="table(name, state, createTime)" || echo "No se encontraron versiones."
    else
        echo "Secreto $secret no encontrado."
    fi
    echo ""
done

echo "✅ Listado de recursos completado!"
echo "=================================================="
