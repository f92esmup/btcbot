import logging
import sys
import os
import argparse

# Añadir src al PYTHONPATH si es necesario
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if project_root not in sys.path:
   sys.path.insert(0, project_root)

from src.utils.config import ConfigManager
from src.data.preprocessor import DataPreprocessor

def setup_logging():
    """Configura el sistema de logging."""
    logging.basicConfig(
        level=logging.INFO,  # Cambiar a logging.DEBUG para más detalle
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Preprocesa datos de OHLCV y crea secuencias para training.')
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Nombre del archivo específico de datos crudos a procesar (ej. BTCUSDT_FUTURES_1h_20200101_20250516.csv).'
    )
    
    return parser.parse_args()

def main():
    """Función principal."""
    # Configurar logging
    logger = setup_logging()
    logger.info("Iniciando script de preprocesamiento de datos.")
    
    # Parsear argumentos
    args = parse_arguments()
    
    try:
        # Cargar configuración general 
        config_manager = ConfigManager(config_path="src/config.yaml", env_path=".env")
        logger.info("Configuración centralizada cargada correctamente")

        # Verificar si están disponibles las variables obligatorias para GCS
        gcp_project_id = config_manager.get_env_variable('GCP_PROJECT_ID')
        gcs_bucket_name = config_manager.get_env_variable('GCS_BUCKET_NAME')
        
        if not gcp_project_id or not gcs_bucket_name:
            logger.error("Variables de entorno obligatorias para GCS no configuradas. Verifique GCP_PROJECT_ID y GCS_BUCKET_NAME en el archivo .env")
            return

        logger.info(f"Usando Google Cloud Storage para procesamiento de datos. Bucket: {gcs_bucket_name}")

    except Exception as e:
        logger.error(f"Error al cargar la configuración: {e}")
        return

    try:
        # Inicializar el preprocesador con la configuración centralizada
        preprocessor = DataPreprocessor(config_manager)
    except Exception as e:
        logger.error(f"Error al inicializar DataPreprocessor: {e}", exc_info=True)
        return

    # Inicializar cliente Storage
    try:
        from google.cloud import storage
        storage_client = storage.Client(project=gcp_project_id)
        bucket = storage_client.bucket(gcs_bucket_name)
    except Exception as e:
        logger.error(f"Error al inicializar cliente de Google Cloud Storage: {e}", exc_info=True)
        return
    
    # Determinar el archivo de datos crudos a procesar en GCS
    gcs_raw_path = config_manager.get_config_value('data_paths.gcs_raw', 'raw')
    
    if args.file:
        # Usar el archivo específico proporcionado por el usuario
        raw_data_filename = args.file
        
        # Verificar si el archivo existe en GCS
        blob = bucket.blob(f"{gcs_raw_path}/{raw_data_filename}")
        if not blob.exists():
            logger.error(f"El archivo especificado {raw_data_filename} no existe en el bucket {gcs_bucket_name}, ruta {gcs_raw_path}")
            return
    else:
        # Lógica para seleccionar el archivo más reciente en GCS
        default_symbol = config_manager.get_config_value('data_acquisition_defaults.symbol', 'BTCUSDT')
        
        # Listar blobs en el bucket/carpeta
        gcs_prefix = f"{gcs_raw_path}/"
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))
        
        # Filtrar por archivos CSV que contienen el símbolo
        raw_files = [blob.name.split('/')[-1] for blob in blobs 
                  if blob.name.endswith('.csv') and default_symbol in blob.name]

        if not raw_files:
            logger.error(f"No se encontraron archivos de datos crudos para {default_symbol} en GCS {gcs_raw_path}. Ejecuta primero el script de adquisición.")
            return
        
        # Procesar el archivo más reciente (asumiendo que el nombre contiene fecha/hora o se ordena alfabéticamente)
        raw_data_filename = sorted(raw_files, reverse=True)[0]
    
    output_filename_base = os.path.splitext(raw_data_filename)[0]  # ej. BTCUSDT_FUTURES_1h_20200101_20250516

    try:
        logger.info(f"Procesando archivo de datos crudos desde GCS: {raw_data_filename}")
        preprocessor.process_data(raw_data_filename, output_filename_base)
        logger.info("Proceso de preprocesamiento de datos en GCS finalizado exitosamente.")
    except Exception as e:
        logger.error(f"Ocurrió un error crítico durante el proceso de preprocesamiento con GCS: {e}", exc_info=True)

if __name__ == "__main__":
    main()
