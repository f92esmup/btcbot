import os
import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.api_core.exceptions import PermissionDenied

# Variable global para controlar si se debe intentar usar Cloud Logging
# Esto evita intentos repetidos cuando ya sabemos que fallarán por permisos
CLOUD_LOGGING_AVAILABLE = True

def setup_logger(name):
    """
    Configura y devuelve un logger que utiliza Google Cloud Logging en producción
    y logging estándar en desarrollo.
    
    Args:
        name (str): Nombre del logger
        
    Returns:
        logging.Logger: Logger configurado
    """
    global CLOUD_LOGGING_AVAILABLE
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Si ya tiene handlers, no añadir más para evitar duplicación
    if logger.handlers:
        return logger
    
    # Configurar el formato de logging básico que se usará por defecto o como fallback
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_formatter = logging.Formatter(log_format)
    
    # Obtener el modo de Cloud Logging (auto, enabled, disabled)
    cloud_logging_mode = os.environ.get('CLOUD_LOGGING_MODE', 'auto').lower()
    
    # Crear siempre un handler para salida estándar para asegurar visibilidad local
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    
    # Si Cloud Logging está deshabilitado o no está disponible, usar solo logging básico
    if cloud_logging_mode == 'disabled' or (cloud_logging_mode == 'auto' and not CLOUD_LOGGING_AVAILABLE):
        if cloud_logging_mode == 'disabled':
            logger.debug(f"Logger '{name}' configurado solo con logging básico (Cloud Logging desactivado explícitamente)")
        elif not CLOUD_LOGGING_AVAILABLE:
            logger.debug(f"Logger '{name}' configurado solo con logging básico (Cloud Logging no disponible)")
        return logger
        
    # Intentar configurar Cloud Logging si está habilitado o en modo auto
    if cloud_logging_mode in ('auto', 'enabled'):
        try:
            # Inicializar cliente de Google Cloud Logging
            client = google.cloud.logging.Client()
            
            # Crear un handler para Google Cloud Logging
            cloud_handler = CloudLoggingHandler(client, name=f"btcbot_{name}")
            
            # Añadir el handler al logger
            logger.addHandler(cloud_handler)
            logger.info(f"Logger '{name}' configurado para enviar logs a Google Cloud Logging")
            
            return logger
            
        except PermissionDenied as e:
            # Error específico de permisos
            error_msg = f"Permiso denegado para Google Cloud Logging: {str(e)}"
            
            if cloud_logging_mode == 'enabled':
                # Si el modo es 'enabled', propagamos el error porque el usuario lo quiere forzar
                raise PermissionDenied(f"{error_msg}. Configure CLOUD_LOGGING_MODE='auto' o 'disabled' para evitar este error.")
            
            # En modo 'auto', desactivamos Cloud Logging globalmente para futuros loggers
            CLOUD_LOGGING_AVAILABLE = False
            logger.warning(f"{error_msg}. Cloud Logging desactivado para toda la aplicación.")
            
        except Exception as e:
            # Otros errores de Cloud Logging
            error_msg = f"Error al configurar Google Cloud Logging: {str(e)}"
            
            if cloud_logging_mode == 'enabled':
                # Si el modo es 'enabled', propagamos el error
                raise Exception(f"{error_msg}. Configure CLOUD_LOGGING_MODE='auto' o 'disabled' para evitar este error.")
            
            # En modo 'auto', seguimos intentando para futuros loggers pero informamos del error
            logger.error(f"{error_msg}. Usando configuración básica.")
    
    return logger
