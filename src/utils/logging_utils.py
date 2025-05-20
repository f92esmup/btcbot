import os
import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler

def setup_logger(name):
    """
    Configura y devuelve un logger que utiliza Google Cloud Logging en producción
    y logging estándar en desarrollo.
    
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
    
    # En producción, utilizamos Google Cloud Logging
    if os.environ.get('USE_CLOUD_LOGGING') == 'true':
        try:
            # Inicializar cliente de Google Cloud Logging
            client = google.cloud.logging.Client()
            
            # Crear un handler para Google Cloud Logging
            cloud_handler = CloudLoggingHandler(client, name=f"btcbot_{name}")
            
            # Configurar el logger para que use el handler de Google Cloud
            logger.addHandler(cloud_handler)
            logger.info(f"Logger '{name}' configurado para enviar logs a Google Cloud Logging")
        except Exception as e:
            # Si hay algún error al configurar Cloud Logging, fallback a logging básico
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger.error(f"Error al configurar Google Cloud Logging: {str(e)}. Usando configuración básica.")
    else:
        # En desarrollo, usamos la configuración básica que envía logs a stdout
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.debug(f"Logger '{name}' configurado para entorno de desarrollo")
    
    return logger
