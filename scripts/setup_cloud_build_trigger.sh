#!/bin/bash
# setup_cloud_build_trigger.sh - Configure Cloud Build trigger for BTC Trading Bot
#
# This script sets up a Cloud Build trigger to automatically build and
# deploy the BTC Trading Bot when changes are pushed to the repository.

set -e  # Exit immediately if a command exits with a non-zero status

echo "🚀 Configurando el trigger de Cloud Build para BTC Trading Bot..."
echo "------------------------------------------------"

# --- Obtener configuración de terraform.tfvars ---
# Asegúrate de que este script esté en el directorio correcto o ajusta la ruta.
TERRAFORM_DIR="$(dirname "$0")/../terraform"
if [ ! -d "$TERRAFORM_DIR" ]; then
    echo "❌ Error: Directorio de Terraform no encontrado en '$TERRAFORM_DIR'."
    echo "   Asegúrate de que la ruta al directorio de Terraform sea correcta."
    exit 1
fi
if [ ! -f "$TERRAFORM_DIR/terraform.tfvars" ]; then
    echo "❌ Error: Archivo 'terraform.tfvars' no encontrado en '$TERRAFORM_DIR'."
    exit 1
fi

cd "$TERRAFORM_DIR"
PROJECT_ID=$(grep 'project_id' terraform.tfvars | cut -d'"' -f2)
REGION=$(grep 'region' terraform.tfvars | cut -d'"' -f2)
cd - > /dev/null # Volver al directorio original silenciosamente

# --- Solicitar información del repositorio GitHub ---
echo "📋 Introduce el propietario/nombre del repositorio GitHub (ej., 'usuario/nombre-repo'):"
read -r GITHUB_REPO

# Extraer propietario y nombre del repo
OWNER=$(echo "$GITHUB_REPO" | cut -d'/' -f1)
REPO=$(echo "$GITHUB_REPO" | cut -d'/' -f2)

# Solicitar la rama para activar las compilaciones
echo "📋 Introduce la rama en la que se activarán las compilaciones (ej., 'main'):"
read -r BRANCH

# Verificar que las variables no estén vacías
if [ -z "$PROJECT_ID" ] || [ -z "$REGION" ] || [ -z "$OWNER" ] || [ -z "$REPO" ] || [ -z "$BRANCH" ]; then
    echo "❌ Error: Falta información requerida."
    echo "   Project ID: $PROJECT_ID"
    echo "   Región: $REGION"
    echo "   Propietario GitHub: $OWNER"
    echo "   Repositorio GitHub: $REPO"
    echo "   Rama: $BRANCH"
    exit 1
fi

echo "------------------------------------------------"
echo "▶️ Usando la siguiente configuración:"
echo "   Project ID: $PROJECT_ID"
echo "   Región: $REGION"
echo "   Repo GitHub: $OWNER/$REPO"
echo "   Rama: $BRANCH"
echo "------------------------------------------------"

# --- Comprobar y configurar la conexión de GitHub ---
GITHUB_CONNECTION_NAME="github-connection" # Nombre común, ajústalo si usas otro.
echo "🔎 Comprobando el estado de la conexión de GitHub '$GITHUB_CONNECTION_NAME' en la región '$REGION'..."

CONNECTION_STATUS=$(gcloud builds connections describe "$GITHUB_CONNECTION_NAME" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --format="value(installationState.stage)" 2>/dev/null || echo "NOT_FOUND")

if [ "$CONNECTION_STATUS" = "NOT_FOUND" ]; then
    echo "📋 Necesitas conectar primero tu cuenta de GitHub a Cloud Build."
    echo "   Comando sugerido: gcloud builds connections create github $GITHUB_CONNECTION_NAME --project=$PROJECT_ID --region=$REGION"
    echo "   Luego sigue las instrucciones para autorizar a Cloud Build a acceder a tus repositorios de GitHub."
    
    echo "📋 ¿Te gustaría intentar crear la conexión de GitHub ahora? (s/n)"
    read -r CREATE_CONNECTION
    
    if [ "$CREATE_CONNECTION" = "s" ] || [ "$CREATE_CONNECTION" = "S" ]; then
        echo "🔧 Creando conexión de GitHub '$GITHUB_CONNECTION_NAME'..."
        gcloud builds connections create github "$GITHUB_CONNECTION_NAME" \
            --project="$PROJECT_ID" \
            --region="$REGION"
        
        GCLOUD_EXIT_STATUS=$?
        if [ $GCLOUD_EXIT_STATUS -ne 0 ]; then
            echo "❌ Error: Falló la creación de la conexión de GitHub (código de salida: $GCLOUD_EXIT_STATUS)."
            echo "   Por favor, revisa el mensaje de error anterior."
            exit 1
        fi
        echo "⚠️ Importante: Por favor, completa el proceso de autorización de GitHub en tu navegador."
        echo "   Después de que la autorización esté completa, ejecuta este script de nuevo."
        exit 0 # Salir para que el usuario complete la autorización
    else
        echo "ℹ️ Por favor, crea la conexión de GitHub manualmente y ejecuta este script de nuevo."
        exit 1
    fi
elif [ "$CONNECTION_STATUS" != "COMPLETE" ]; then
    echo "⚠️ La conexión de GitHub '$GITHUB_CONNECTION_NAME' existe pero su estado es: $CONNECTION_STATUS."
    echo "   Por favor, completa el proceso de autorización de GitHub visitando (o revisando la CLI):"
    echo "   https://console.cloud.google.com/cloud-build/triggers/connect?project=$PROJECT_ID"
    echo "   Después de que la autorización esté completa, ejecuta este script de nuevo."
    exit 1
fi
echo "✅ Conexión de GitHub '$GITHUB_CONNECTION_NAME' verificada y completa."
echo "------------------------------------------------"

# --- Crear o verificar el repositorio en Cloud Build bajo la conexión ---
# Este es el ID que tendrá el repositorio DENTRO de Cloud Build, puede ser diferente al nombre en GitHub.
CB_REPO_ID="btcbot-repo" # Puedes cambiar este ID si lo deseas.
REPO_REMOTE_URI="https://github.com/$OWNER/$REPO.git"

echo "🔎 Verificando si el repositorio '$CB_REPO_ID' ya está vinculado en Cloud Build bajo la conexión '$GITHUB_CONNECTION_NAME'..."

REPO_PATH_FOR_TRIGGER=$(gcloud builds repositories describe "$CB_REPO_ID" \
    --project="$PROJECT_ID" \
    --connection="$GITHUB_CONNECTION_NAME" \
    --region="$REGION" \
    --format="value(name)" 2>/dev/null || echo "NOT_FOUND")

if [ "$REPO_PATH_FOR_TRIGGER" = "NOT_FOUND" ]; then
    echo "⚠️ Repositorio '$CB_REPO_ID' no encontrado. Vinculando el repositorio '$REPO_REMOTE_URI'..."
    gcloud builds repositories create "$CB_REPO_ID" \
        --project="$PROJECT_ID" \
        --connection="$GITHUB_CONNECTION_NAME" \
        --region="$REGION" \
        --remote-uri="$REPO_REMOTE_URI"
        
    GCLOUD_EXIT_STATUS=$?
    if [ $GCLOUD_EXIT_STATUS -ne 0 ]; then
        echo "❌ Error: Falló la vinculación del repositorio '$CB_REPO_ID' (URI: $REPO_REMOTE_URI) en Cloud Build (código de salida: $GCLOUD_EXIT_STATUS)."
        exit 1
    fi
    echo "✅ Repositorio '$CB_REPO_ID' vinculado correctamente en Cloud Build."
    echo "⏳ Esperando un momento para la propagación..."
    sleep 5 # Dar tiempo para que se propague la creación/vinculación.
    
    # Obtener la ruta completa del recurso después de la creación
    REPO_PATH_FOR_TRIGGER=$(gcloud builds repositories describe "$CB_REPO_ID" \
        --project="$PROJECT_ID" \
        --connection="$GITHUB_CONNECTION_NAME" \
        --region="$REGION" \
        --format="value(name)" 2>/dev/null)

    if [ -z "$REPO_PATH_FOR_TRIGGER" ]; then
        echo "❌ Error: No se pudo obtener la ruta completa del recurso para el repositorio '$CB_REPO_ID' después de crearlo."
        exit 1
    fi
else
    echo "✅ Repositorio '$CB_REPO_ID' ya existe y está vinculado en Cloud Build."
fi

echo "ℹ️ Usando la siguiente ruta de repositorio para el trigger: $REPO_PATH_FOR_TRIGGER"
echo "------------------------------------------------"

# --- Comprobar y eliminar trigger existente ---
TRIGGER_NAME="btc-trading-bot-trigger" # Nombre del trigger a crear/gestionar

echo "🔎 Comprobando si ya existe un trigger llamado '$TRIGGER_NAME' en la región '$REGION'..."
EXISTING_TRIGGER_ID=$(gcloud builds triggers list \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --filter="name='$TRIGGER_NAME'" \
    --format="value(id)" 2>/dev/null)

GCLOUD_LIST_EXIT_STATUS=$? # Captura el estado de salida del comando 'list'

if [ $GCLOUD_LIST_EXIT_STATUS -ne 0 ]; then
    echo "⚠️ Advertencia: El comando 'gcloud builds triggers list' falló (código de salida: $GCLOUD_LIST_EXIT_STATUS)."
    echo "   Esto podría indicar problemas de permisos o configuración. Se intentará crear el trigger de todas formas."
    # No se puede confirmar si existe, así que no se intenta eliminar.
fi

if [ -n "$EXISTING_TRIGGER_ID" ]; then
    echo "⚠️ Un trigger llamado '$TRIGGER_NAME' (ID: $EXISTING_TRIGGER_ID) ya existe. Eliminándolo..."
    gcloud builds triggers delete "$EXISTING_TRIGGER_ID" \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --quiet # Suprime la confirmación
    
    GCLOUD_DELETE_EXIT_STATUS=$?
    if [ $GCLOUD_DELETE_EXIT_STATUS -ne 0 ]; then
        echo "❌ Error: Falló la eliminación del trigger existente '$TRIGGER_NAME' (ID: $EXISTING_TRIGGER_ID) (código de salida: $GCLOUD_DELETE_EXIT_STATUS)."
        echo "   Por favor, comprueba los permisos o elimínalo manualmente desde la Consola de Google Cloud."
        exit 1
    fi
    echo "✅ Trigger existente eliminado correctamente."
    sleep 3 # Darle un momento a GCP para procesar la eliminación
elif [ $GCLOUD_LIST_EXIT_STATUS -eq 0 ]; then
    # Solo si el comando 'list' fue exitoso y no se encontró ID
    echo "ℹ️ No se encontró ningún trigger existente con el nombre '$TRIGGER_NAME'."
fi
echo "------------------------------------------------"

# --- Crear el trigger de Cloud Build ---
echo "🔧 Creando el trigger de Cloud Build '$TRIGGER_NAME'..."
gcloud builds triggers create github \
    --name="$TRIGGER_NAME" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --description="Trigger para BTC Trading Bot" \
    --repository="$REPO_PATH_FOR_TRIGGER" \
    --branch-pattern="^${BRANCH}$" \
    --build-config="cloudbuild.yaml" # Asume que cloudbuild.yaml está en la raíz del repositorio

GCLOUD_CREATE_EXIT_STATUS=$?
if [ $GCLOUD_CREATE_EXIT_STATUS -eq 0 ]; then
    echo "✅ ¡Trigger de Cloud Build '$TRIGGER_NAME' creado correctamente!"
    echo "------------------------------------------------"
    echo "🔍 Puedes ver tus triggers en: https://console.cloud.google.com/cloud-build/triggers?project=$PROJECT_ID&region=$REGION"
else
    echo "❌ Error: Falló la creación del trigger de Cloud Build '$TRIGGER_NAME' (código de salida: $GCLOUD_CREATE_EXIT_STATUS)."
    echo "   Problemas comunes a verificar:"
    echo "   1. Asegúrate de que la ruta del repositorio '$REPO_PATH_FOR_TRIGGER' sea correcta y el repositorio esté accesible."
    echo "   2. Verifica que la conexión de GitHub '$GITHUB_CONNECTION_NAME' esté activa y autorizada."
    echo "   3. Comprueba que la cuenta de servicio de Cloud Build (o tu cuenta) tenga los permisos necesarios (ej. 'Editor de Cloud Build')."
    echo "   4. Asegúrate de que el archivo 'cloudbuild.yaml' exista en la raíz de la rama '$BRANCH' del repositorio '$OWNER/$REPO'."
    echo "   5. Revisa la salida anterior para mensajes de error específicos de gcloud."
    exit 1
fi

echo "------------------------------------------------"
echo "🎉 ¡Configuración del trigger completada!"
