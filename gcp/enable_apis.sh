#!/bin/bash
# Activar las APIs necesarias antes de ejecutar el script principal
echo "Activando APIs necesarias para el proyecto btcbot276299..."

# Lista de APIs a habilitar
APIS=(
    "cloudresourcemanager.googleapis.com"
    "iam.googleapis.com"
    "iamcredentials.googleapis.com"
    "storage.googleapis.com"
    "secretmanager.googleapis.com"
    "aiplatform.googleapis.com"
)

for api in "${APIS[@]}"; do
    echo "Activando API: $api"
    gcloud services enable $api --project=btcbot276299
    
    # Esperar un momento para que los cambios se propaguen
    sleep 2
done

echo "Todas las APIs han sido activadas. Ahora ejecutando el script principal..."
/Users/f92esmup/btcbot/.venv/bin/python /Users/f92esmup/btcbot/gcp/03_setup_iam.py
