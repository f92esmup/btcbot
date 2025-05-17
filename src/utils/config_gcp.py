import os
import logging
import yaml
import json

# Configurar logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManagerCloud:
    """
    Gestor de configuración optimizado para entornos de Google Cloud.
    Prioriza variables de entorno sobre cualquier otra fuente de configuración.
    """
    
    def __init__(self):
        """
        Inicializa el gestor de configuración para Cloud.
        No carga configuraciones de archivos locales por defecto.
        """
        # Opcionalmente, se podría cargar una estructura base de YAML para valores no críticos
        self.config_yaml_base = {}
        
    def get_param(self, env_var_name: str, default=None, required: bool = True, var_type=str):
        """
        Obtiene un parámetro principalmente de una variable de entorno.
        
        Args:
            env_var_name: Nombre de la variable de entorno
            default: Valor por defecto si no se encuentra
            required: Si es True y no se encuentra, genera error
            var_type: Tipo al que convertir el valor (str, int, float, bool, list, dict)
        
        Returns:
            El valor convertido al tipo especificado
        """
        value_str = os.getenv(env_var_name)
        if value_str is None:
            if required:
                error_msg = f"Variable de entorno requerida '{env_var_name}' no configurada."
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.info(f"Variable de entorno '{env_var_name}' no encontrada. Usando default: {default}")
            return default

        try:
            if var_type == bool:
                return value_str.lower() in ['true', '1', 'yes', 'y']
            elif var_type == int:
                return int(value_str)
            elif var_type == float:
                return float(value_str)
            elif var_type == list or var_type == dict:
                try:
                    return json.loads(value_str)
                except json.JSONDecodeError:
                    logger.warning(f"Error decodificando JSON para '{env_var_name}'. Tratando como string.")
                    return value_str
            return value_str  # Devuelve como string por defecto
        except ValueError as e:
            error_msg = f"Error convirtiendo variable de entorno '{env_var_name}' (valor: '{value_str}') a tipo {var_type}: {e}"
            logger.error(error_msg)
            if required:
                raise
            return default


def get_secret_from_gcp(project_id: str, secret_name: str, version: str = "latest"):
    """
    Obtiene un secreto de Google Cloud Secret Manager.
    
    Args:
        project_id: ID del proyecto GCP
        secret_name: Nombre del secreto
        version: Versión del secreto, por defecto "latest"
    
    Returns:
        El valor del secreto como string
    """
    from google.cloud import secretmanager as sm
    
    client = sm.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
    
    try:
        response = client.access_secret_version(name=name)
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.error(f"Error accediendo al secreto '{secret_name}' en GCP: {e}")
        raise


def get_gcs_path(bucket_name: str, path: str = "") -> str:
    """
    Construye una ruta completa para Google Cloud Storage.
    
    Args:
        bucket_name: Nombre del bucket
        path: Ruta relativa dentro del bucket
        
    Returns:
        URI completa para GCS
    """
    # Asegurarse de que el bucket no incluya prefijo gs://
    clean_bucket = bucket_name.replace("gs://", "")
    
    # Asegurarse de que el path no comience con / 
    clean_path = path
    if clean_path.startswith("/"):
        clean_path = clean_path[1:]
        
    return f"gs://{clean_bucket}/{clean_path}"


# Funciones de utilidad para convertir tipos de datos específicos
def parse_int_list(env_var_name: str, default=None, required: bool = True, delimiter=","):
    """
    Convierte una variable de entorno en una lista de enteros.
    
    Args:
        env_var_name: Nombre de la variable de entorno
        default: Valor por defecto si no se encuentra
        required: Si es True y no se encuentra, genera error
        delimiter: Delimitador para separar los elementos de la lista
        
    Returns:
        Lista de enteros
    """
    value_str = os.getenv(env_var_name)
    if value_str is None:
        if required:
            error_msg = f"Variable de entorno requerida '{env_var_name}' no configurada."
            logger.error(error_msg)
            raise ValueError(error_msg)
        return default
    
    try:
        return [int(item.strip()) for item in value_str.split(delimiter)]
    except ValueError as e:
        error_msg = f"Error convirtiendo '{env_var_name}' a lista de enteros: {e}"
        logger.error(error_msg)
        if required:
            raise
        return default