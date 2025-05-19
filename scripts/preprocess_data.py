import logging
import sys
import os
import yaml
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
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='src/data/preprocessing_config.yaml',
        help='Ruta al archivo de configuración de preprocesamiento. Por defecto: src/data/preprocessing_config.yaml'
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
        # Cargar configuración general y del módulo
        general_config_manager = ConfigManager(config_path="src/config.yaml", env_path=".env")
        
        preprocessing_config_path = args.config
        with open(preprocessing_config_path, 'r') as f:
            module_specific_config = yaml.safe_load(f)
        logger.info(f"Configuración de preprocesamiento cargada desde {preprocessing_config_path}")

    except FileNotFoundError as e:
        logger.error(f"Error: Archivo de configuración no encontrado. {e}")
        return
    except Exception as e:
        logger.error(f"Error al cargar la configuración: {e}")
        return

    try:
        preprocessor = DataPreprocessor(general_config_manager, module_specific_config)
    except Exception as e:
        logger.error(f"Error al inicializar DataPreprocessor: {e}", exc_info=True)
        return

    # Determinar el archivo de datos crudos a procesar
    raw_data_dir = general_config_manager.get_config_value('data_paths.raw')
    
    if args.file:
        # Usar el archivo específico proporcionado por el usuario
        raw_data_filename = args.file
        if not os.path.exists(os.path.join(raw_data_dir, raw_data_filename)):
            logger.error(f"El archivo especificado {raw_data_filename} no existe en {raw_data_dir}")
            return
    else:
        # Lógica para seleccionar el archivo más reciente
        default_symbol = general_config_manager.get_config_value('data_acquisition_defaults.symbol', 'BTCUSDT')
        raw_files = [f for f in os.listdir(raw_data_dir) if f.startswith(default_symbol) and f.endswith('.csv')]

        if not raw_files:
            logger.error(f"No se encontraron archivos de datos crudos para {default_symbol} en {raw_data_dir}. Ejecuta primero el script de adquisición.")
            return
        
        # Procesar el archivo más reciente (asumiendo que el nombre contiene fecha/hora o se ordena alfabéticamente)
        raw_data_filename = sorted(raw_files, reverse=True)[0] 
    
    output_filename_base = os.path.splitext(raw_data_filename)[0] # ej. BTCUSDT_FUTURES_1h_20200101_20250516

    try:
        logger.info(f"Procesando archivo de datos crudos: {raw_data_filename}")
        preprocessor.process_data(raw_data_filename, output_filename_base)
        logger.info("Proceso de preprocesamiento de datos finalizado exitosamente.")
    except Exception as e:
        logger.error(f"Ocurrió un error crítico durante el proceso de preprocesamiento: {e}", exc_info=True)

if __name__ == "__main__":
    main()
