#!/bin/bash
# check_and_run_test.sh - Verifica el estado de los recursos y ejecuta el entrenamiento de prueba
#
# Este script verifica que los recursos necesarios ya existen y ejecuta el entrenamiento de prueba

set -e  # Salir inmediatamente si un comando falla

echo "🔍 Verificando recursos y ejecutando entrenamiento de prueba..."
echo "---------------------------------------------------------------------"

# Establecer valores predeterminados
PROJECT_ID="btcbot276299"
REGION="europe-southwest1"
SERVICE_ACCOUNT="btc-trading-bot-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Verificar la conexión a GCP
echo "🔄 Verificando autenticación en GCP..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" > /dev/null; then
    echo "❌ Error: No hay una cuenta autenticada en gcloud."
    echo "   Ejecuta 'gcloud auth login' para autenticarte."
    exit 1
fi

# Verificar buckets
echo "🪣 Verificando buckets GCS..."
BUCKETS=("${PROJECT_ID}-btc-artifacts" "${PROJECT_ID}-btc-raw-data" "${PROJECT_ID}-btc-processed-data" "${PROJECT_ID}-btc-models" "${PROJECT_ID}-btc-tensorboard")
MISSING_BUCKETS=0

for BUCKET in "${BUCKETS[@]}"; do
    if ! gsutil ls -b "gs://${BUCKET}" &> /dev/null; then
        echo "❌ Bucket no encontrado: ${BUCKET}"
        MISSING_BUCKETS=1
    else
        echo "✅ Bucket encontrado: ${BUCKET}"
    fi
done

# Verificar cuenta de servicio
echo "👤 Verificando cuenta de servicio..."
if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT}" --project="${PROJECT_ID}" &> /dev/null; then
    echo "❌ Cuenta de servicio no encontrada: ${SERVICE_ACCOUNT}"
    exit 1
else
    echo "✅ Cuenta de servicio encontrada: ${SERVICE_ACCOUNT}"
fi

# Si faltan buckets, advertir pero continuar
if [ $MISSING_BUCKETS -eq 1 ]; then
    echo "⚠️ Algunos buckets no existen. Esto puede causar errores en el entrenamiento."
    echo "   ¿Deseas continuar de todos modos? (y/n)"
    read -r CONTINUE
    if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
        echo "❌ Abortando."
        exit 1
    fi
fi

# Continuar con el entrenamiento de prueba
echo "🚀 Ejecutando script de entrenamiento de prueba..."
cd "$(dirname "$0")"
./run_test_training.sh
