#!/bin/bash
# gcp/kubernetes/assign_gcs_permissions.sh
# Script para asignar permisos de Storage a la cuenta de servicio

# Definir variables
PROJECT_ID="lofty-complex-460416-r6"
SERVICE_ACCOUNT="btcbot-inference-sa"
BUCKET_NAME="lofty-complex-460416-r6"

# Colores para formato de output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Asignando permisos de acceso a GCS a la cuenta de servicio...${NC}"
echo -e "Proyecto: ${GREEN}${PROJECT_ID}${NC}"
echo -e "Cuenta de servicio: ${GREEN}${SERVICE_ACCOUNT}${NC}"
echo -e "Bucket GCS: ${GREEN}${BUCKET_NAME}${NC}"
echo ""

# Verificar si la cuenta de servicio existe
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT}@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe ${SERVICE_ACCOUNT_EMAIL} --project ${PROJECT_ID} &>/dev/null; then
  SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT}"
  echo -e "${YELLOW}Usando nombre directo de cuenta de servicio: ${SERVICE_ACCOUNT}${NC}"
fi

# Asignar rol de administrador de objetos (Storage Object Admin)
echo -e "${BLUE}Asignando rol Storage Object Admin...${NC}"
gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role="roles/storage.objectAdmin" \
  --project ${PROJECT_ID}

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✓ Permisos Storage Object Admin asignados correctamente${NC}"
else
  echo -e "${RED}✗ Error al asignar permisos Storage Object Admin${NC}"
  exit 1
fi

# Asignar rol de visualizador del bucket (Storage Object Viewer)
echo -e "${BLUE}Asignando rol Storage Object Viewer...${NC}"
gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role="roles/storage.objectViewer" \
  --project ${PROJECT_ID}

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✓ Permisos Storage Object Viewer asignados correctamente${NC}"
else
  echo -e "${RED}✗ Error al asignar permisos Storage Object Viewer${NC}"
  exit 1
fi

# Asignar rol de administrador del bucket (Storage Admin) - opcional, para creación de buckets
echo -e "${BLUE}Asignando rol Storage Admin...${NC}"
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role="roles/storage.admin"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✓ Permisos Storage Admin asignados correctamente${NC}"
else
  echo -e "${RED}✗ Error al asignar permisos Storage Admin${NC}"
  exit 1
fi

echo -e "\n${GREEN}=== Permisos asignados correctamente ===${NC}"
echo -e "${BLUE}La cuenta de servicio ${SERVICE_ACCOUNT_EMAIL} ahora tiene acceso completo al bucket ${BUCKET_NAME}${NC}"
echo -e "${YELLOW}NOTA: Los cambios de permisos pueden tardar unos minutos en propagarse.${NC}"
