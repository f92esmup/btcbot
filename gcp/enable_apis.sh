#!/bin/bash
# Activar las APIs necesarias antes de ejecutar el script principal

# Usar variable de entorno GCP_PROJECT_ID si está definida, de lo contrario usar el valor por defecto
PROJECT_ID=${GCP_PROJECT_ID:-"btcbot276299"}

echo "Activando APIs necesarias para el proyecto $PROJECT_ID..."

# Lista de APIs a habilitar
APIS=(
    "cloudresourcemanager.googleapis.com"
    "iam.googleapis.com"
    "iamcredentials.googleapis.com"
    "storage.googleapis.com"
    "secretmanager.googleapis.com"
    "aiplatform.googleapis.com"
    "artifactregistry.googleapis.com"
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "cloudscheduler.googleapis.com"
)

for api in "${APIS[@]}"; do
    echo "Activando API: $api"
    gcloud services enable $api --project=$PROJECT_ID
    
    # Esperar un momento para que los cambios se propaguen
    sleep 2
done

echo "Todas las APIs han sido activadas."
