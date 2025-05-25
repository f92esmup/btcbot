import os
import logging
from datetime import datetime
import sys
import pytz

# Configurar timezone de Madrid como constante global
MADRID_TZ = pytz.timezone('Europe/Madrid')

def get_madrid_timestamp():
    """
    Obtiene el timestamp actual en timezone de Madrid.
    
    Returns:
        datetime: Timestamp con timezone de Madrid
    """
    return datetime.now(MADRID_TZ)

def get_madrid_timestamp_str():
    """
    Obtiene el timestamp actual en timezone de Madrid como string ISO.
    
    Returns:
        str: Timestamp en formato ISO con timezone de Madrid
    """
    return get_madrid_timestamp().isoformat()

def utc_to_madrid(utc_dt):
    """
    Convierte un datetime UTC a timezone de Madrid.
    
    Args:
        utc_dt (datetime): Datetime en UTC
        
    Returns:
        datetime: Datetime convertido a timezone de Madrid
    """
    if utc_dt.tzinfo is None:
        # Si no tiene timezone, asumimos que es UTC
        utc_dt = pytz.utc.localize(utc_dt)
    elif utc_dt.tzinfo != pytz.utc:
        # Si tiene otro timezone, convertimos primero a UTC
        utc_dt = utc_dt.astimezone(pytz.utc)
    
    return utc_dt.astimezone(MADRID_TZ)

def setup_logger(name):
    """
    Configura y devuelve un logger que envía los mensajes a la consola.
    Esta versión actualizada garantiza que los logs en Docker se envíen correctamente a stdout
    para que Google Cloud Logging los procese adecuadamente.
    Los timestamps se configuran para usar timezone de Madrid.
    
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
    
    # Configurar el formato de logging para consola con timezone de Madrid
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Crear un formatter personalizado que use timezone de Madrid
    class MadridFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=MADRID_TZ)
            if datefmt:
                return dt.strftime(datefmt)
            else:
                return dt.strftime('%Y-%m-%d %H:%M:%S,%f')[:-3] + ' CET/CEST'
    
    log_formatter = MadridFormatter(log_format)
    
    # Crear handler para stdout (no stderr)
    # En entornos Docker/Kubernetes, los logs a stderr se consideran errores
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    
    # Evitar que los logs se propaguen a la raíz
    logger.propagate = False
    
    logger.debug(f"Logger '{name}' configurado para mostrar logs en la terminal (stdout) con timezone Madrid")
    
    return logger
