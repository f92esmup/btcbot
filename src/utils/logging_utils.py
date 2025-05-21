import os
import logging
from datetime import datetime
import sys

def setup_logger(name):
    """
    Configura y devuelve un logger que envía los mensajes a la consola.
    Esta versión actualizada garantiza que los logs en Docker se envíen correctamente a stdout
    para que Google Cloud Logging los procese adecuadamente.
    
    Args:
        name (str): Nombre del logger
        
    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Limpiar handlers existentes para evitar duplicación
    if logger.handlers:
        logger.handlers.clear()
    
    # Configurar el formato de logging para consola
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_formatter = logging.Formatter(log_format)
    
    # Crear handler para stdout (no stderr)
    # En entornos Docker/Kubernetes, los logs a stderr se consideran errores
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    
    # Evitar que los logs se propaguen a la raíz
    logger.propagate = False
    
    logger.debug(f"Logger '{name}' configurado para mostrar logs en la terminal (stdout)")
    
    return logger
