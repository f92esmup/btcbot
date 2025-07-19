"""
Módulo de utilidades para la configuración del sistema de logging.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from .config_model import AppConfig

def setup_logging() -> None:
    """Configura el sistema GLOBAL de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading_bot.log'),
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
