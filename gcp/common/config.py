"""
Configuración centralizada para los scripts de GCP.
Prioriza la lectura de variables de entorno, con valores por defecto como fallback.
"""
import os
import logging

logger = logging.getLogger(__name__)

# --- Funciones de Ayuda para Leer Variables de Entorno ---
def get_env_variable(var_name: str, default_value: str = None, required: bool = False) -> str:
    """
    Obtiene una variable de entorno.
    Si es requerida y no se encuentra, lanza un error.
    Si no es requerida y no se encuentra, devuelve el valor por defecto.
    """
    value = os.getenv(var_name)
    if value is None:
        if required:
            logger.error(f"La variable de entorno requerida '{var_name}' no está configurada.")
            raise ValueError(f"La variable de entorno requerida '{var_name}' no está configurada.")
        logger.warning(f"La variable de entorno '{var_name}' no está configurada. Usando valor por defecto: {default_value}")
        return default_value
    return value

# --- Configuración del Proyecto GCP ---
PROJECT_ID = get_env_variable("GCP_PROJECT_ID", default_value="btcbot276299", required=True)
REGION = get_env_variable("GCP_REGION", default_value="europe-southwest1", required=True)
ZONE = get_env_variable("GCP_ZONE", default_value="europe-west3-a") # Menos crítico, puede tener default

# --- Nombres de Buckets ---
RAW_DATA_BUCKET = get_env_variable("RAW_DATA_BUCKET", default_value=f"btcbot-raw-data-{PROJECT_ID}", required=True)
PROCESSED_DATA_BUCKET = get_env_variable("PROCESSED_DATA_BUCKET", default_value=f"btcbot-processed-data-{PROJECT_ID}", required=True)
MODELS_STAGING_BUCKET = get_env_variable("MODELS_STAGING_BUCKET", default_value=f"btcbot-models-staging-{PROJECT_ID}", required=True)
EVALUATION_RESULTS_BUCKET = get_env_variable("EVALUATION_RESULTS_BUCKET", default_value=f"btcbot-evaluation-results-{PROJECT_ID}", required=True)
PIPELINE_ROOT_BUCKET = get_env_variable("PIPELINE_ROOT_BUCKET", default_value=f"gs://{MODELS_STAGING_BUCKET}/pipeline_root")

# --- Nombres para Recursos de IAM ---
SERVICE_ACCOUNT_NAME = get_env_variable("SERVICE_ACCOUNT_NAME", default_value="btcbot-service-account")
SERVICE_ACCOUNT_EMAIL = f"{SERVICE_ACCOUNT_NAME}@{PROJECT_ID}.iam.gserviceaccount.com"

# --- Repositorio de Artifact Registry ---
ARTIFACT_REPO = get_env_variable("ARTIFACT_REPO", default_value="btcbot-docker-repo")

# --- Nombres para Cloud Run/Functions ---
DATA_ACQUISITION_SERVICE_NAME = get_env_variable("DATA_ACQUISITION_SERVICE_NAME", default_value="btcbot-data-acquisition")

# --- Nombres para Secretos en Secret Manager ---
BINANCE_API_KEY_SECRET_NAME = get_env_variable("BINANCE_API_KEY_SECRET_NAME", default_value="binance-api-key")
BINANCE_API_SECRET_SECRET_NAME = get_env_variable("BINANCE_API_SECRET_SECRET_NAME", default_value="binance-api-secret")

# --- Nombres para Imágenes Docker ---
TRAINING_IMAGE_BASE_NAME = get_env_variable("TRAINING_IMAGE_BASE_NAME", default_value="btcbot-training")
PREPROCESSING_IMAGE_BASE_NAME = get_env_variable("PREPROCESSING_IMAGE_BASE_NAME", default_value="btcbot-preprocessing")

TRAINING_IMAGE_NAME = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REPO}/{TRAINING_IMAGE_BASE_NAME}"
PREPROCESSING_IMAGE_NAME = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REPO}/{PREPROCESSING_IMAGE_BASE_NAME}"

# --- Nombres para Trabajos y Pipelines en Vertex AI ---
TRAINING_JOB_NAME_PREFIX = get_env_variable("TRAINING_JOB_NAME_PREFIX", default_value="btcbot-training")
PIPELINE_NAME = get_env_variable("PIPELINE_NAME", default_value="btcbot-training-pipeline")

# --- Rutas de Código Fuente (para scripts de GCP que construyen Docker o pipelines) ---
# Estas son relativas a la raíz del proyecto
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))  # Asume que common está en gcp/, y gcp/ está en la raíz
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# --- Configuración específica para el Bot de Trading ---
DEFAULT_SYMBOL = get_env_variable("DEFAULT_SYMBOL", default_value="BTCUSDT")
DEFAULT_INTERVAL = get_env_variable("DEFAULT_INTERVAL", default_value="1h")
DEFAULT_HISTORICAL_START_DATE = get_env_variable("DEFAULT_HISTORICAL_START_DATE", default_value="2020-01-01")

logger.info(f"Configuración cargada para el proyecto: {PROJECT_ID} en la región: {REGION}")
