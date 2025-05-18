#!/bin/bash
# monitor_deployment.sh - Monitorear el despliegue y ejecución del Bitcoin Trading Bot en GCP
#
# Este script proporciona un panel interactivo para monitorear el estado de los recursos
# desplegados en GCP y los pipelines en ejecución.

set -e  # Exit immediately if a command exits with a non-zero status

# Get configuration from terraform.tfvars
cd "$(dirname "$0")/../terraform"
PROJECT_ID=$(grep 'project_id' terraform.tfvars | cut -d'"' -f2)
REGION=$(grep 'region' terraform.tfvars | cut -d'"' -f2)
cd ..

# Colores para la salida
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No color

# Función para mostrar el menú
show_menu() {
    clear
    echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       BITCOIN TRADING BOT - MONITOR DE GCP         ║${NC}"
    echo -e "${BLUE}╠════════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC} Proyecto: ${GREEN}$PROJECT_ID${NC}"
    echo -e "${BLUE}║${NC} Región:   ${GREEN}$REGION${NC}"
    echo -e "${BLUE}╠════════════════════════════════════════════════════╣${NC}"
    echo -e "${BLUE}║${NC} 1. Ver estado del despliegue                  ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} 2. Listar recursos en GCP                     ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} 3. Mostrar pipelines en ejecución             ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} 4. Ver logs de Cloud Build                    ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} 5. Ver contenido de buckets de almacenamiento ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} 6. Verificar secretos                         ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} 7. Recargar configuración                     ${BLUE}║${NC}"
    echo -e "${BLUE}║${NC} 0. Salir                                      ${BLUE}║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
    echo -e "Selecciona una opción: \c"
}

# Función para verificar estado de despliegue
check_deployment_status() {
    echo -e "${BLUE}Verificando estado del despliegue...${NC}"
    bash "$(dirname "$0")/check_deployment_status.sh"
    read -p "Presiona Enter para continuar..."
}

# Función para listar recursos
list_resources() {
    echo -e "${BLUE}Listando recursos en GCP...${NC}"
    bash "$(dirname "$0")/list_gcp_resources.sh"
    read -p "Presiona Enter para continuar..."
}

# Función para mostrar pipelines en ejecución
show_running_pipelines() {
    echo -e "${BLUE}Consultando pipelines en ejecución...${NC}"
    gcloud ai pipeline-jobs list --region="$REGION" --project="$PROJECT_ID" \
        --filter="state=PIPELINE_STATE_RUNNING" \
        --format="table(name, display_name, state, create_time, update_time)" || echo "No hay pipelines en ejecución."
    
    echo ""
    echo -e "${YELLOW}¿Deseas ver detalles de algún pipeline específico? (y/n)${NC}"
    read -r VIEW_DETAILS
    
    if [ "$VIEW_DETAILS" = "y" ] || [ "$VIEW_DETAILS" = "Y" ]; then
        echo -e "${YELLOW}Ingresa el ID del pipeline (última parte del nombre):${NC}"
        read -r PIPELINE_ID
        gcloud ai pipeline-jobs describe "projects/$PROJECT_ID/locations/$REGION/pipelineJobs/$PIPELINE_ID" \
            --project="$PROJECT_ID" --region="$REGION" || echo "Pipeline no encontrado."
    fi
    
    read -p "Presiona Enter para continuar..."
}

# Función para ver logs de Cloud Build
show_cloud_build_logs() {
    echo -e "${BLUE}Obteniendo historial de builds recientes...${NC}"
    gcloud builds list --project="$PROJECT_ID" --limit=5 \
        --format="table(id, status, source.repoSource.repoName, startTime, finishTime)" || echo "No hay builds recientes."
    
    echo ""
    echo -e "${YELLOW}¿Deseas ver logs de algún build específico? (y/n)${NC}"
    read -r VIEW_LOGS
    
    if [ "$VIEW_LOGS" = "y" ] || [ "$VIEW_LOGS" = "Y" ]; then
        echo -e "${YELLOW}Ingresa el ID del build:${NC}"
        read -r BUILD_ID
        gcloud builds log "$BUILD_ID" --project="$PROJECT_ID" || echo "Build no encontrado."
    fi
    
    read -p "Presiona Enter para continuar..."
}

# Función para ver contenido de buckets
show_bucket_contents() {
    echo -e "${BLUE}Buckets disponibles:${NC}"
    gsutil ls -p "$PROJECT_ID" | grep "$PROJECT_ID" || echo "No se encontraron buckets."
    
    echo ""
    echo -e "${YELLOW}Ingresa el nombre del bucket para ver su contenido (o Enter para volver):${NC}"
    read -r BUCKET_NAME
    
    if [ -n "$BUCKET_NAME" ]; then
        if gsutil ls -b "gs://$BUCKET_NAME" &>/dev/null; then
            echo -e "${BLUE}Contenido de gs://$BUCKET_NAME:${NC}"
            gsutil ls -r "gs://$BUCKET_NAME/**" | head -n 20
            count=$(gsutil ls -r "gs://$BUCKET_NAME/**" | wc -l)
            if [ "$count" -gt 20 ]; then
                echo "... y $(($count - 20)) archivos más."
            fi
        else
            echo -e "${RED}El bucket $BUCKET_NAME no existe.${NC}"
        fi
    fi
    
    read -p "Presiona Enter para continuar..."
}

# Función para verificar secretos
check_secrets() {
    echo -e "${BLUE}Verificando secretos en Secret Manager...${NC}"
    
    for secret in binance-api-key binance-api-secret; do
        if gcloud secrets describe "$secret" --project="$PROJECT_ID" &>/dev/null; then
            latest_version=$(gcloud secrets versions list "$secret" --project="$PROJECT_ID" --format="value(name)" | sort -rn | head -1)
            if [ -n "$latest_version" ]; then
                echo -e "${GREEN}✓${NC} Secreto $secret existe (última versión: $latest_version)"
            else
                echo -e "${YELLOW}⚠${NC} Secreto $secret existe pero no tiene versiones"
            fi
        else
            echo -e "${RED}✗${NC} Secreto $secret no existe"
        fi
    done
    
    echo ""
    echo -e "${YELLOW}¿Deseas actualizar los secretos? (y/n)${NC}"
    read -r UPDATE_SECRETS
    
    if [ "$UPDATE_SECRETS" = "y" ] || [ "$UPDATE_SECRETS" = "Y" ]; then
        bash "$(dirname "$0")/update_secrets.sh"
    fi
    
    read -p "Presiona Enter para continuar..."
}

# Bucle principal
while true; do
    show_menu
    read -r option
    
    case $option in
        1) check_deployment_status ;;
        2) list_resources ;;
        3) show_running_pipelines ;;
        4) show_cloud_build_logs ;;
        5) show_bucket_contents ;;
        6) check_secrets ;;
        7) 
            # Recargar configuración
            cd "$(dirname "$0")/../terraform"
            PROJECT_ID=$(grep 'project_id' terraform.tfvars | cut -d'"' -f2)
            REGION=$(grep 'region' terraform.tfvars | cut -d'"' -f2)
            cd ..
            echo "Configuración recargada."
            sleep 1
            ;;
        0) 
            echo "Saliendo..."
            exit 0
            ;;
        *) 
            echo -e "${RED}Opción inválida${NC}"
            sleep 1
            ;;
    esac
done