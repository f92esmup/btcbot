import logging
import sys
import os

# Añadir src al PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import ConfigManager
from src.data.binance_futures_downloader import BinanceFuturesDownloader

# Configuración básica del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Para ver logs en consola
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Iniciando script de descarga de datos históricos.")
    
    try:
        # Carga la configuración
        config_manager = ConfigManager(config_path="src/config.yaml", env_path=".env")
    except Exception as e:
        logger.error(f"Error al cargar la configuración: {e}")
        sys.exit(1)  # Salir con error
    
    # Verificar configuración de Google Cloud (obligatoria)
    gcp_project_id = config_manager.get_env_variable('GCP_PROJECT_ID')
    gcs_bucket_name = config_manager.get_env_variable('GCS_BUCKET_NAME')
    
    if not gcp_project_id or not gcs_bucket_name:
        logger.error("Configuración de Google Cloud incompleta. Ambos GCP_PROJECT_ID y GCS_BUCKET_NAME son obligatorios.")
        logger.error("Por favor, configure estas variables en su archivo .env")
        sys.exit(1)  # Salir con error
        
    logger.info(f"Usando Google Cloud Storage para almacenamiento. Proyecto: {gcp_project_id}, Bucket: {gcs_bucket_name}")
    
    # Verificar credenciales de Google Cloud
    credentials_path = config_manager.get_env_variable('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path:
        logger.error("No se ha configurado GOOGLE_APPLICATION_CREDENTIALS en el archivo .env")
        logger.error("Es obligatorio especificar la ruta al archivo de credenciales de servicio de Google Cloud")
        sys.exit(1)  # Salir con error
        
    if not os.path.exists(credentials_path):
        logger.error(f"No se encontró el archivo de credenciales de Google Cloud en: {credentials_path}")
        logger.error("Verifique que la ruta especificada en GOOGLE_APPLICATION_CREDENTIALS sea correcta")
        sys.exit(1)  # Salir con error
        
    logger.info(f"Credenciales de Google Cloud encontradas en: {credentials_path}")

    try:
        downloader = BinanceFuturesDownloader(config_manager)
    except Exception as e:
        logger.error(f"Error al inicializar BinanceFuturesDownloader: {e}")
        sys.exit(1)  # Salir con error

    symbol = config_manager.get_config_value('data_acquisition_defaults.symbol', "BTCUSDT")
    interval = config_manager.get_config_value('data_acquisition_defaults.interval', "1h")
    start_date = config_manager.get_config_value('data_acquisition_defaults.historical_start_date', "2020-01-01")

    try:
        downloader.fetch_historical_data(symbol, interval, start_date)
        logger.info("Proceso de descarga de datos finalizado exitosamente.")
    except Exception as e:
        logger.error(f"Ocurrió un error durante el proceso de descarga: {e}", exc_info=True)
        sys.exit(1)  # Salir con error

if __name__ == "__main__":
    main()
