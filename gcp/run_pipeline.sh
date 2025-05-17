#!/bin/bash
# Script para ejecutar el pipeline de entrenamiento en Vertex AI

# Colores para mejorar la legibilidad
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}     LANZAMIENTO DE PIPELINE DE ENTRENAMIENTO BTC-BOT     ${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Parámetros por defecto
SYMBOL="BTCUSDT"
TIMEFRAME="1h"
START_DATE="2020-01-01"
END_DATE=""
LOOKBACK_WINDOW=96
TOTAL_TIMESTEPS=500000
ALGORITHM="SAC"
NUM_EVAL_EPISODES=10
DEPLOY_MODEL=false
ENABLE_CACHING=false
WAIT_FOR_COMPLETION=false

# Procesar argumentos
while [[ $# -gt 0 ]]; do
  case $1 in
    --symbol)
      SYMBOL="$2"
      shift 2
      ;;
    --timeframe)
      TIMEFRAME="$2"
      shift 2
      ;;
    --start-date)
      START_DATE="$2"
      shift 2
      ;;
    --end-date)
      END_DATE="$2"
      shift 2
      ;;
    --lookback-window)
      LOOKBACK_WINDOW="$2"
      shift 2
      ;;
    --total-timesteps)
      TOTAL_TIMESTEPS="$2"
      shift 2
      ;;
    --algorithm)
      ALGORITHM="$2"
      shift 2
      ;;
    --eval-episodes)
      NUM_EVAL_EPISODES="$2"
      shift 2
      ;;
    --deploy)
      DEPLOY_MODEL=true
      shift
      ;;
    --enable-cache)
      ENABLE_CACHING=true
      shift
      ;;
    --wait)
      WAIT_FOR_COMPLETION=true
      shift
      ;;
    --help)
      echo "Uso: $0 [opciones]"
      echo "Opciones:"
      echo "  --symbol SYMBOL          Símbolo de criptomoneda (por defecto: BTCUSDT)"
      echo "  --timeframe TIMEFRAME    Intervalo de tiempo (por defecto: 1h)"
      echo "  --start-date DATE        Fecha de inicio para datos históricos (por defecto: 2020-01-01)"
      echo "  --end-date DATE          Fecha de fin para datos históricos (opcional)"
      echo "  --lookback-window N      Tamaño de la ventana de secuencia (por defecto: 96)"
      echo "  --total-timesteps N      Pasos totales de entrenamiento (por defecto: 500000)"
      echo "  --algorithm ALG          Algoritmo RL a utilizar (SAC, PPO, TD3) (por defecto: SAC)"
      echo "  --eval-episodes N        Número de episodios para evaluación (por defecto: 10)"
      echo "  --deploy                 Desplegar modelo si cumple criterios de calidad"
      echo "  --enable-cache           Habilitar caché para componentes del pipeline"
      echo "  --wait                   Esperar a que el pipeline termine su ejecución"
      exit 0
      ;;
    *)
      echo "Opción desconocida: $1"
      echo "Usa --help para ver las opciones disponibles."
      exit 1
      ;;
  esac
done

# Construir argumentos para el script de Python
ARGS=""
ARGS="$ARGS --symbol $SYMBOL"
ARGS="$ARGS --timeframe $TIMEFRAME"
ARGS="$ARGS --start_date $START_DATE"
[ ! -z "$END_DATE" ] && ARGS="$ARGS --end_date $END_DATE"
ARGS="$ARGS --lookback_window $LOOKBACK_WINDOW"
ARGS="$ARGS --total_timesteps $TOTAL_TIMESTEPS"
ARGS="$ARGS --algorithm $ALGORITHM"
ARGS="$ARGS --num_eval_episodes $NUM_EVAL_EPISODES"
[ "$DEPLOY_MODEL" = true ] && ARGS="$ARGS --deploy_model"
[ "$ENABLE_CACHING" = true ] && ARGS="$ARGS --enable_caching"
[ "$WAIT_FOR_COMPLETION" = true ] && ARGS="$ARGS --wait_for_completion"

# Mostrar resumen de configuración
echo -e "${YELLOW}Configuración del pipeline:${NC}"
echo -e "  Símbolo: ${GREEN}$SYMBOL${NC}"
echo -e "  Timeframe: ${GREEN}$TIMEFRAME${NC}"
echo -e "  Fecha de inicio: ${GREEN}$START_DATE${NC}"
[ ! -z "$END_DATE" ] && echo -e "  Fecha de fin: ${GREEN}$END_DATE${NC}"
echo -e "  Ventana de secuencia: ${GREEN}$LOOKBACK_WINDOW${NC}"
echo -e "  Pasos de entrenamiento: ${GREEN}$TOTAL_TIMESTEPS${NC}"
echo -e "  Algoritmo: ${GREEN}$ALGORITHM${NC}"
echo -e "  Episodios de evaluación: ${GREEN}$NUM_EVAL_EPISODES${NC}"
echo -e "  Despliegue automático: ${GREEN}$([ "$DEPLOY_MODEL" = true ] && echo "Sí" || echo "No")${NC}"
echo -e "  Caché habilitado: ${GREEN}$([ "$ENABLE_CACHING" = true ] && echo "Sí" || echo "No")${NC}"
echo -e "  Esperar finalización: ${GREEN}$([ "$WAIT_FOR_COMPLETION" = true ] && echo "Sí" || echo "No")${NC}"

echo -e "\n${YELLOW}Ejecutando el pipeline...${NC}"
echo -e "Comando: python 05_create_training_pipeline.py $ARGS\n"

# Ejecutar el script de Python con los argumentos
cd "$(dirname "$0")" # Asegurarse de estar en el directorio correcto
python3 05_create_training_pipeline.py $ARGS

# Verificar si la ejecución fue exitosa
if [ $? -eq 0 ]; then
  echo -e "\n${GREEN}✅ Pipeline lanzado correctamente${NC}"
else
  echo -e "\n${YELLOW}❌ Error al lanzar el pipeline${NC}"
  exit 1
fi
