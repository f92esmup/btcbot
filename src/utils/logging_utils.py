import os
import logging
from datetime import datetime

def setup_logger(name):
    """
    Configura y devuelve un logger que envía los mensajes a la consola.
    
    Args:
        name (str): Nombre del logger
        
    Returns:
        logging.Logger: Logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Si ya tiene handlers, no añadir más para evitar duplicación
    if logger.handlers:
        return logger
    
    # Configurar el formato de logging para consola
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_formatter = logging.Formatter(log_format)
    
    # Crear handler para salida estándar
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    
    logger.debug(f"Logger '{name}' configurado para mostrar logs en la terminal")
    
    return logger
