#!/bin/bash
# deploy_to_k8s.sh
# Script para desplegar btcbot en Kubernetes utilizando las sustituciones definidas

# Colores para mensajes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Imprimir mensaje con color
print_message() {
  local color=$1
  local message=$2
  echo -e "${color}${message}${NC}"
}

# -----------------------------------------------------
# Parámetros configurables (puedes cambiar estos valores o pasarlos como argumentos)
# -----------------------------------------------------

# Valores por defecto
GCP_PROJECT_ID="lofty-complex-460416-r6"
GCS_BUCKET_NAME="lofty-complex-460416-r6_data"
GCP_REGION="europe-southwest1"
LIVE_TRADING_MODE="TESTNET"  # Usar "TESTNET" o "REAL"

# Directorio actual
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEMPLATE_FILE="${SCRIPT_DIR}/btcbot-deployment.template.yaml"
OUTPUT_FILE="${SCRIPT_DIR}/btcbot-deployment.yaml"

# -----------------------------------------------------
# Procesamiento de argumentos
# -----------------------------------------------------

# Procesar argumentos de línea de comandos
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --project) GCP_PROJECT_ID="$2"; shift ;;
    --bucket) GCS_BUCKET_NAME="$2"; shift ;;
    --region) GCP_REGION="$2"; shift ;;
    --mode) LIVE_TRADING_MODE="$2"; shift ;;
    --help)
      echo "Uso: $0 [opciones]"
      echo "Opciones:"
      echo "  --project ID     ID del proyecto de GCP (default: $GCP_PROJECT_ID)"
      echo "  --bucket NOMBRE  Nombre del bucket GCS (default: $GCS_BUCKET_NAME)"
      echo "  --region REGIÓN  Región de GCP (default: $GCP_REGION)"
      echo "  --mode MODO      Modo de trading: TESTNET o REAL (default: $LIVE_TRADING_MODE)"
      exit 0
      ;;
    *) echo "Opción desconocida: $1"; exit 1 ;;
  esac
  shift
done

# -----------------------------------------------------
# Validación de parámetros
# -----------------------------------------------------

# Validar el modo de trading
if [[ "$LIVE_TRADING_MODE" != "TESTNET" && "$LIVE_TRADING_MODE" != "REAL" ]]; then
  print_message "$RED" "ERROR: El modo de trading debe ser 'TESTNET' o 'REAL'. Recibido: $LIVE_TRADING_MODE"
  exit 1
fi

# -----------------------------------------------------
# Sustitución de valores
# -----------------------------------------------------

print_message "$YELLOW" "Generando archivo de despliegue con los siguientes valores:"
echo "  - GCP_PROJECT_ID: $GCP_PROJECT_ID"
echo "  - GCS_BUCKET_NAME: $GCS_BUCKET_NAME"
echo "  - GCP_REGION: $GCP_REGION"
echo "  - LIVE_TRADING_MODE: $LIVE_TRADING_MODE"

# Verificar que el archivo de plantilla existe
if [[ ! -f "$TEMPLATE_FILE" ]]; then
  print_message "$RED" "ERROR: No se encontró el archivo de plantilla en $TEMPLATE_FILE"
  exit 1
fi

# Crear una copia del archivo de plantilla
cp "$TEMPLATE_FILE" "$OUTPUT_FILE"

# Realizar las sustituciones
sed -i '' "s|\${_GCP_PROJECT_ID}|$GCP_PROJECT_ID|g" "$OUTPUT_FILE"
sed -i '' "s|\${_GCS_BUCKET_NAME}|$GCS_BUCKET_NAME|g" "$OUTPUT_FILE"
sed -i '' "s|\${_GCP_REGION}|$GCP_REGION|g" "$OUTPUT_FILE"
sed -i '' "s|\${_LIVE_TRADING_MODE}|$LIVE_TRADING_MODE|g" "$OUTPUT_FILE"

# Eliminar la sección de sustituciones
sed -i '' '/# --- SECCIÓN DE SUSTITUCIONES Y VALORES DE REFERENCIA ---/,/# __LIVE_TRADING_MODE_VALUE__ -> \${_LIVE_TRADING_MODE}/d' "$OUTPUT_FILE"

print_message "$GREEN" "✅ Archivo de despliegue generado: $OUTPUT_FILE"

# -----------------------------------------------------
# Despliegue a Kubernetes
# -----------------------------------------------------

# Preguntar al usuario si desea desplegar ahora
read -p "¿Deseas desplegar a Kubernetes ahora? (s/n): " DEPLOY_NOW

if [[ "$DEPLOY_NOW" == "s" || "$DEPLOY_NOW" == "S" ]]; then
  print_message "$YELLOW" "Desplegando a Kubernetes..."
  
  # Comprobar que kubectl está instalado
  if ! command -v kubectl &> /dev/null; then
    print_message "$RED" "ERROR: kubectl no está instalado o no se encuentra en el PATH"
    exit 1
  fi
  
  # Verificar conexión con el cluster
  if ! kubectl cluster-info &> /dev/null; then
    print_message "$RED" "ERROR: No se puede conectar al cluster de Kubernetes. Verifica tu configuración de kubectl"
    exit 1
  fi
  
  # Aplicar el archivo generado
  kubectl apply -f "$OUTPUT_FILE"
  
  if [ $? -eq 0 ]; then
    print_message "$GREEN" "✅ Despliegue completado correctamente!"
    
    # Mostrar información del deployment
    echo ""
    print_message "$YELLOW" "Información del deployment:"
    kubectl get deployment btcbot-live-trader
    
    echo ""
    print_message "$YELLOW" "Para monitorear los pods:"
    echo "kubectl get pods -l app=btcbot-trader"
    echo "kubectl logs -f -l app=btcbot-trader"
  else
    print_message "$RED" "❌ Error al desplegar a Kubernetes"
  fi
else
  print_message "$YELLOW" "No se realizó el despliegue. Puedes desplegarlo manualmente con:"
  echo "kubectl apply -f $OUTPUT_FILE"
fi
