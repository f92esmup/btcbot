"""
Módulo de utilidades para la configuración del sistema de logging.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from typing import Dict, Any

from .config_model import AppConfig

def setup_logging() -> None:
    """Configura el sistema GLOBAL de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            #logging.FileHandler('trading_bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_date_format(date_string: str) -> bool:
    """
    Valida que la fecha tenga el formato correcto.
    
    Args:
        date_string (str): Fecha en formato YYYY-MM-DD
        
    Returns:
        bool: True si es válida, False en caso contrario
    """
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def load_system_config(config_path: str) -> AppConfig:
    """
    Carga la configuración principal del sistema desde config.yaml usando Pydantic.
    
    Returns:
        AppConfig: Configuración completa del sistema validada
        
    Raises:
        SystemExit: Si no se puede cargar la configuración
    """
    config_path = Path(config_path)

    try:
        config = AppConfig.from_yaml_file(config_path)
        logging.info(f"Configuración del sistema cargada desde: {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"No se encontró el archivo de configuración: {config_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error al cargar la configuración: {e}")
        sys.exit(1)

def generate_data_run_id(symbol: str, interval: str, start_date: str, end_date: Optional[str] = None) -> str:
    """
    Genera un identificador único para el data_run.
    
    Args:
        symbol: Símbolo del par de trading
        interval: Intervalo de tiempo
        start_date: Fecha de inicio
        end_date: Fecha de fin (opcional)
        
    Returns:
        str: ID único del data_run (ej: BTCUSDT_1m_20250101_20250703-153000)
    """
    timestamp = datetime.now().strftime("%H%M%S")
    date_str = start_date.replace("-", "")
    
    if end_date:
        end_date_str = end_date.replace("-", "")
        return f"{symbol}_{interval}_{date_str}_{end_date_str}-{timestamp}"
    else:
        today_str = datetime.now().strftime("%Y%m%d")
        return f"{symbol}_{interval}_{date_str}_{today_str}-{timestamp}"

def create_data_run_metadata(symbol: str, interval: str, start_date: str, 
                           end_date: Optional[str], data_run_id: str) -> Dict[str, Any]:
    """
    Crea el diccionario de metadatos para el data_run.
    
    Args:
        symbol: Símbolo del par de trading
        interval: Intervalo de tiempo
        start_date: Fecha de inicio
        end_date: Fecha de fin (opcional)
        data_run_id: Identificador único del data_run
        
    Returns:
        Dict[str, Any]: Metadatos del data_run
    """
    metadata = {
        "data_run_info": {
            "data_run_info": data_run_id,
            "creation_timestamp": datetime.now().isoformat(),
            "created_by": "create_dataset.py",
            "description": f'Dataset inmutable para {symbol} ({interval}) desde {start_date}'
        },
        "experiment_parameters": {
            "symbol": symbol,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "data_source": "Binance API"
        }
    }
    
    return metadata