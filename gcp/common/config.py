"""
Configuración centralizada para los scripts de GCP.
"""
import os

# ID del proyecto GCP
PROJECT_ID = "btcbot276299"

# Región por defecto
REGION = "europe-southwest1"

# Zona por defecto
ZONE = "europe-west3-a"

# Nombres de los buckets
RAW_DATA_BUCKET = f"btcbot-raw-data-{PROJECT_ID}"
PROCESSED_DATA_BUCKET = f"btcbot-processed-data-{PROJECT_ID}"
MODELS_STAGING_BUCKET = f"btcbot-models-staging-{PROJECT_ID}"
EVALUATION_RESULTS_BUCKET = f"btcbot-evaluation-results-{PROJECT_ID}"

# Nombres para recursos de IAM
SERVICE_ACCOUNT_NAME = "btcbot-service-account"
SERVICE_ACCOUNT_EMAIL = f"{SERVICE_ACCOUNT_NAME}@{PROJECT_ID}.iam.gserviceaccount.com"

# Nombre del repositorio de Artifact Registry
ARTIFACT_REPO = "btcbot-docker-repo"

# Nombres para Cloud Functions/Run
DATA_ACQUISITION_SERVICE_NAME = "btcbot-data-acquisition"

# Nombres para secretos
BINANCE_API_KEY_SECRET_NAME = "binance-api-key"
BINANCE_API_SECRET_SECRET_NAME = "binance-api-secret"

# Nombres para imágenes de Docker
TRAINING_IMAGE_NAME = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REPO}/btcbot-training"
PREPROCESSING_IMAGE_NAME = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REPO}/btcbot-preprocessing"

# Nombre base para los trabajos de entrenamiento en Vertex AI
TRAINING_JOB_NAME_PREFIX = "btcbot-training"

# Ruta local del código fuente para empaquetar en Docker
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts")
