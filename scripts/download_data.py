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
        # logging.FileHandler("download.log") # Para guardar logs en un archivo
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Iniciando script de descarga de datos históricos.")
    
    try:
        # Rutas relativas al script actual si config.yaml y .env están en la raíz del proyecto
        # o ajustar según la estructura final
        config_manager = ConfigManager(config_path="src/config.yaml", env_path=".env")
    except Exception as e:
        logger.error(f"Error al cargar la configuración: {e}")
        return

    try:
        downloader = BinanceFuturesDownloader(config_manager)
    except Exception as e:
        logger.error(f"Error al inicializar BinanceFuturesDownloader: {e}")
        return

    symbol = config_manager.get_config_value('data_acquisition_defaults.symbol', "BTCUSDT")
    interval = config_manager.get_config_value('data_acquisition_defaults.interval', "1h")
    start_date = config_manager.get_config_value('data_acquisition_defaults.historical_start_date', "2020-01-01")

    try:
        downloader.fetch_historical_data(symbol, interval, start_date)
        logger.info("Proceso de descarga de datos finalizado.")
    except Exception as e:
        logger.error(f"Ocurrió un error durante el proceso de descarga: {e}", exc_info=True)

if __name__ == "__main__":
    main()
