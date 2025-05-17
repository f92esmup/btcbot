#!/bin/bash
# Script para construir y desplegar el contenedor Docker para GCP

set -e

# Establecer variables
PROJECT_ID=$(gcloud config get-value project)
REGION=${REGION:-"us-central1"}
REPOSITORY=${REPOSITORY:-"trading-bot"}
TAG=${TAG:-$(date +%Y%m%d%H%M%S)}
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/trading-bot:${TAG}"
MODE=${MODE:-"serve"}  # Modo por defecto

# Funciones
print_header() {
    echo "======================================================"
    echo "  $1"
    echo "======================================================"
}

# Verificar argumentos
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Uso: $0 [--build-only] [--deploy-only] [--mode=serve|train|download|preprocess|evaluate]"
    echo "Opciones:"
    echo "  --build-only     Sólo construir la imagen, no desplegar"
    echo "  --deploy-only    Sólo desplegar, no construir"
    echo "  --mode=MODE      Modo de ejecución (serve, train, download, preprocess, evaluate)"
    echo "  --tag=TAG        Etiqueta para la imagen (por defecto: timestamp)"
    echo "  --region=REGION  Región de GCP (por defecto: us-central1)"
    echo "  --help, -h       Mostrar esta ayuda"
    exit 0
fi

# Procesar argumentos
BUILD=true
DEPLOY=true

for arg in "$@"; do
    case $arg in
        --build-only)
            DEPLOY=false
            shift
            ;;
        --deploy-only)
            BUILD=false
            shift
            ;;
        --mode=*)
            MODE="${arg#*=}"
            shift
            ;;
        --tag=*)
            TAG="${arg#*=}"
            IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/trading-bot:${TAG}"
            shift
            ;;
        --region=*)
            REGION="${arg#*=}"
            IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/trading-bot:${TAG}"
            shift
            ;;
    esac
done

# Construir la imagen Docker
if [ "$BUILD" = true ]; then
    print_header "Construyendo imagen Docker: $IMAGE_NAME"
    docker build -f Dockerfile.cloud -t $IMAGE_NAME .
    
    print_header "Subiendo imagen a Artifact Registry"
    docker push $IMAGE_NAME
    
    echo "✅ Imagen construida y subida correctamente: $IMAGE_NAME"
fi

# Desplegar con Cloud Run
if [ "$DEPLOY" = true ]; then
    print_header "Desplegando en Cloud Run (modo: $MODE)"
    
    # Establecer variables de entorno según el modo
    ENV_VARS="MODE=$MODE"
    
    # Añadir variables de entorno adicionales según el modo
    case $MODE in
        serve)
            ENV_VARS="$ENV_VARS,PORT=8080,GCP_PROJECT_ID=$PROJECT_ID"
            SERVICE_NAME="trading-bot-service"
            ;;
        train)
            ENV_VARS="$ENV_VARS,GCP_PROJECT_ID=$PROJECT_ID"
            SERVICE_NAME="trading-bot-trainer"
            ;;
        download)
            ENV_VARS="$ENV_VARS,GCP_PROJECT_ID=$PROJECT_ID"
            SERVICE_NAME="trading-bot-downloader"
            ;;
        preprocess)
            ENV_VARS="$ENV_VARS,GCP_PROJECT_ID=$PROJECT_ID"
            SERVICE_NAME="trading-bot-preprocessor"
            ;;
        evaluate)
            ENV_VARS="$ENV_VARS,GCP_PROJECT_ID=$PROJECT_ID"
            SERVICE_NAME="trading-bot-evaluator"
            ;;
        *)
            echo "⚠️ Modo no reconocido: $MODE"
            exit 1
            ;;
    esac
    
    gcloud run deploy $SERVICE_NAME \
        --image=$IMAGE_NAME \
        --region=$REGION \
        --platform=managed \
        --allow-unauthenticated \
        --set-env-vars=$ENV_VARS
    
    echo "✅ Servicio desplegado: $SERVICE_NAME"
fi

print_header "Proceso completado con éxito"
echo "Imagen: $IMAGE_NAME"
echo "Modo: $MODE"
