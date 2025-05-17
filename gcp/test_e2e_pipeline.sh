#!/bin/bash
# Script de prueba E2E para verificar la integración del pipeline en GCP

set -e

# Cargar variables de entorno
export GCP_PROJECT_ID=${GCP_PROJECT_ID:-"btcbot276299"}
export GCP_REGION=${GCP_REGION:-"europe-southwest1"}

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}       PRUEBA E2E DEL PIPELINE DE TRADING EN GCP         ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Establecer parámetros para la prueba (valores pequeños para completar rápido)
SYMBOL="BTCUSDT"
TIMEFRAME="1h"
START_DATE="2023-01-01"
END_DATE="2023-02-01"  # Período corto para test
LOOKBACK_WINDOW=24     # Ventana más pequeña para test
TOTAL_TIMESTEPS=10000  # Menos pasos para test
ALGORITHM="SAC"
NUM_EVAL_EPISODES=2    # Menos episodios para evaluación
ENABLE_CACHING=true    # Habilitar caché para acelerar pruebas futuras

echo -e "\n${YELLOW}1. Verificando configuración del entorno GCP${NC}"
gcloud config list
echo -e "${GREEN}✓ Configuración GCP verificada${NC}"

echo -e "\n${YELLOW}2. Habilitando APIs necesarias${NC}"
source ./enable_apis.sh
echo -e "${GREEN}✓ APIs habilitadas${NC}"

echo -e "\n${YELLOW}3. Construyendo y publicando imagen Docker${NC}"
./deploy.sh --build-only --tag=e2e-test
echo -e "${GREEN}✓ Imagen Docker construida y publicada${NC}"

echo -e "\n${YELLOW}4. Actualizando config.py con variables necesarias${NC}"
# Todo esto ya está en common/config.py, no necesitamos actualizarlo aquí

echo -e "\n${YELLOW}5. Ejecutando pipeline de entrenamiento completo${NC}"
./run_pipeline.sh \
  --symbol $SYMBOL \
  --timeframe $TIMEFRAME \
  --start-date $START_DATE \
  --end-date $END_DATE \
  --lookback-window $LOOKBACK_WINDOW \
  --total-timesteps $TOTAL_TIMESTEPS \
  --algorithm $ALGORITHM \
  --eval-episodes $NUM_EVAL_EPISODES \
  --enable-cache \
  --wait

if [ $? -eq 0 ]; then
  echo -e "\n${GREEN}✅ PRUEBA E2E COMPLETADA EXITOSAMENTE${NC}"
  echo -e "Se ha ejecutado el pipeline completo con éxito. Puedes verificar los resultados en:"
  echo -e "- Consola GCP > Vertex AI > Pipelines"
  echo -e "- Buckets de almacenamiento GCS:\n  gs://${GCP_PROJECT_ID}-models-staging"
else
  echo -e "\n${YELLOW}❌ PRUEBA E2E FALLIDA${NC}"
  exit 1
fi
