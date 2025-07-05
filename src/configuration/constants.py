"""
Constantes centralizadas del proyecto BTCBot.

Este módulo contiene todas las cadenas de texto y valores constantes utilizados
a lo largo del proyecto, organizados por categorías para mejorar la mantenibilidad
y evitar la duplicación de "magic strings".

Siguiendo el principio DRY (Don't Repeat Yourself) y facilitando el mantenimiento
centralizado de estas constantes.
"""

# =============================================================================
# DIRECTORY NAMES
# =============================================================================

DIR_DATA_RUNS = "data_runs"
DIR_TRAINING_RUNS = "training_runs" 
DIR_CHECKPOINTS = "checkpoints"
DIR_TENSORBOARD_LOGS = "tensorboard_logs"
DIR_SRC = "src"
DIR_CONFIGURATION = "configuration"
DIR_BEST_MODEL = "best_model"
DIR_FINAL_MODEL = "final_model"
DIR_EVALUATION = "evaluation"

# Directory path constants for frequently used literal strings
DIR_STORAGE = "storage"

# =============================================================================
# ARTIFACT FILENAMES (Data Processing)
# =============================================================================

FILE_NORMALIZED_DATAFRAME = "normalized_dataframe.pkl"
FILE_DATAFRAME_PKL = "normalized_dataframe.pkl"  # Alias for consistency
FILE_SCALER = "scaler.pkl"
FILE_SCALER_PKL = "scaler.pkl"  # Alias for consistency
FILE_PRICE_SCALER = "price_scaler.pkl"
FILE_PRICE_SCALER_PKL = "price_scaler.pkl"  # Alias for consistency

# =============================================================================
# CONFIGURATION FILENAMES
# =============================================================================

FILE_CONFIG_YAML = "config.yaml"
FILE_CONFIG_RUN_YAML = "config_run.yaml"
FILE_CONFIG_TRAINING_RUN_YAML = "config_training_run.yaml"
FILE_DATA_RUN_METADATA = "data_run_metadata.yaml"
FILE_DATA_RUN_METADATA_YAML = "data_run_metadata.yaml"  # Alias for consistency
FILE_EVALUATION_SUMMARY_JSON = "evaluation_summary.json"

# =============================================================================
# MODEL CHECKPOINT FILENAMES (Neural Networks)
# =============================================================================

FILE_ACTOR_PTH = "actor.pth"
FILE_CRITIC_1_PTH = "critic_1.pth"
FILE_CRITIC_2_PTH = "critic_2.pth"
FILE_CRITIC_TARGET_1_PTH = "critic_target_1.pth"
FILE_CRITIC_TARGET_2_PTH = "critic_target_2.pth"

# =============================================================================
# OPTIMIZER CHECKPOINT FILENAMES
# =============================================================================

FILE_ACTOR_OPTIMIZER_PTH = "actor_optimizer.pth"
FILE_CRITIC_1_OPTIMIZER_PTH = "critic_1_optimizer.pth"
FILE_CRITIC_2_OPTIMIZER_PTH = "critic_2_optimizer.pth"
FILE_ALPHA_OPTIMIZER_PTH = "alpha_optimizer.pth"

# =============================================================================
# ADDITIONAL MODEL FILENAMES
# =============================================================================

FILE_LOG_ALPHA_PTH = "log_alpha.pth"
FILE_METADATA_PTH = "metadata.pth"

# =============================================================================
# CONFIGURATION KEYS (YAML Structure)
# =============================================================================

KEY_ENVIRONMENT = "environment"
KEY_AGENT = "agent"
KEY_GCP = "gcp"
KEY_STORAGE_MODE = "storage_mode"
KEY_NORMALIZATION = "normalization"
KEY_DATA = "data"
KEY_STORAGE = "storage"
KEY_SECRETS = "secrets"
KEY_CONFIG = "config"
KEY_LINEAGE = "lineage"
KEY_DATA_RUN_ID = "data_run_id"

# Configuration sub-keys
KEY_BATCH_SIZE = "batch_size"
KEY_MIN_BUFFER_FOR_LEARNING = "min_buffer_for_learning"
KEY_REPLAY_BUFFER_SIZE = "replay_buffer_size"
KEY_REPLAY_BUFFER_CAPACITY = "replay_buffer_capacity"
KEY_BUCKET_NAME = "bucket_name"
KEY_SCALER_BLOB_NAME = "scaler_blob_name"
KEY_PRICE_SCALER_BLOB_NAME = "price_scaler_blob_name"
KEY_SEED = "seed"
KEY_EVAL_FREQUENCY = "eval_frequency"
KEY_SAVE_FREQUENCY = "save_frequency"
KEY_TENSORBOARD_DIR = "tensorboard_dir"

# Metadata keys for data runs
KEY_DATA_RUN_INFO = "data_run_info"
KEY_EXPERIMENT_PARAMETERS = "experiment_parameters"
KEY_DATA_PIPELINE_VERSION = "data_pipeline_version"
KEY_CREATION_TIMESTAMP = "creation_timestamp"
KEY_CREATED_BY = "created_by"
KEY_DESCRIPTION = "description"
KEY_SYMBOL = "symbol"
KEY_INTERVAL = "interval"
KEY_START_DATE = "start_date"
KEY_END_DATE = "end_date"
KEY_DATA_SOURCE = "data_source"
KEY_SCRIPT_VERSION = "script_version"
KEY_PIPELINE_MODULES = "pipeline_modules"

# =============================================================================
# DATA COLUMNS (OHLCV)
# =============================================================================

COLUMN_TIMESTAMP = "timestamp"
COLUMN_OPEN = "Open"
COLUMN_HIGH = "High"
COLUMN_LOW = "Low" 
COLUMN_CLOSE = "Close"
COLUMN_VOLUME = "Volume"

# List of OHLCV columns for convenience
COLUMNS_OHLCV = [COLUMN_OPEN, COLUMN_HIGH, COLUMN_LOW, COLUMN_CLOSE, COLUMN_VOLUME]
COLUMNS_OHLCV_WITH_TIMESTAMP = [COLUMN_TIMESTAMP] + COLUMNS_OHLCV

# =============================================================================
# STORAGE MODES
# =============================================================================

STORAGE_MODE_LOCAL = "local"
STORAGE_MODE_GCP = "gcp"

# =============================================================================
# FILE EXTENSIONS
# =============================================================================

EXT_PKL = ".pkl"
EXT_PTH = ".pth"
EXT_YAML = ".yaml"
EXT_JSON = ".json"

# =============================================================================
# PATH CONFIGURATIONS
# =============================================================================

CONFIG_PATH_DEFAULT = f"{DIR_SRC}/{DIR_CONFIGURATION}/{FILE_CONFIG_YAML}"

# =============================================================================
# BINANCE API ENDPOINTS AND SETTINGS
# =============================================================================

# Binance timeframe mappings
BINANCE_TIMEFRAMES = {
    "1m": "1m",
    "3m": "3m", 
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "8h": "8h",
    "12h": "12h",
    "1d": "1d",
    "3d": "3d",
    "1w": "1w",
    "1M": "1M"
}

# =============================================================================
# CHECKPOINT AND MODEL DIRECTORY PATTERNS
# =============================================================================

PATTERN_CHECKPOINT_EPISODE = "checkpoint_episode_{}"
PATTERN_CHECKPOINT_PREFIX = "checkpoint_episode"

# =============================================================================
# CHECKPOINT AND MODEL DIRECTORY PATTERNS
# =============================================================================

PATTERN_CHECKPOINT_EPISODE = "checkpoint_episode_{}"
PATTERN_CHECKPOINT_PREFIX = "checkpoint_episode"

# =============================================================================
# COMMON PATTERNS AND VALIDATION
# =============================================================================

# Pattern for run IDs (example: BTCUSDT_1h_20250704-092312)
PATTERN_RUN_ID = r"^[A-Z]+_\d+[mhd]_\d{8}-\d{6}$"

# Default values
DEFAULT_NETWORK_INTERFACE = "eth0"
DEFAULT_SCALER_FILENAME = FILE_SCALER
DEFAULT_PRICE_SCALER_FILENAME = FILE_PRICE_SCALER

# Common literal strings
VALUE_BINANCE_API = "Binance API"
VALUE_CREATE_DATASET_SCRIPT = "create_dataset.py"
VALUE_SCRIPT_VERSION_1_0_0 = "1.0.0"
ENCODING_UTF8 = "utf-8"
FILE_SUFFIX_YAML = ".yaml"
FILE_SUFFIX_PKL = ".pkl"
MODE_WRITE = "w"
ATTR_GCP = "gcp"

# Learning states
STATUS_LEARNING = "LEARNING"
STATUS_COLLECTING = "COLLECTING"

# Operation modes
OPERATION_MODE_NEW_TRAINING = "new_training"
OPERATION_MODE_FINE_TUNING = "fine_tuning"
OPERATION_MODE_RESUME_TRAINING = "resume_training"
