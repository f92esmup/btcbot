import yaml
import os
from dotenv import load_dotenv
from google.cloud import secretmanager
import json
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    _instance = None

    def __new__(cls, config_path="src/config.yaml", env_path=".env"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            # Inicializar estos atributos primero para evitar errores
            cls._instance.config = None
            cls._instance.gcp_project_id = None 
            cls._instance.secret_client = None
            
            # Cargar variables de entorno primero (para GCP_PROJECT_ID, etc.)
            load_dotenv(dotenv_path=env_path)
            
            # Leer la configuración YAML
            try:
                with open(config_path, 'r') as f:
                    cls._instance.config = yaml.safe_load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Archivo de configuración {config_path} no encontrado.")
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error al parsear el archivo YAML {config_path}: {e}")
            
            # Obtener el ID del proyecto directamente de las variables de entorno
            cls._instance.gcp_project_id = os.getenv('GCP_PROJECT_ID')
            if not cls._instance.gcp_project_id:
                raise ValueError("GCP_PROJECT_ID no configurado en .env - es obligatorio para Secret Manager")
            
            # Inicializar Secret Manager (obligatorio)
            try:
                # Intentar inicializar el cliente de Secret Manager con ADC (autenticación por defecto)
                cls._instance.secret_client = secretmanager.SecretManagerServiceClient()
                logger.info(f"Cliente de Google Secret Manager inicializado para proyecto: {cls._instance.gcp_project_id}")
            except Exception as e:
                cls._instance.secret_client = None
                logger.error(f"No se pudo inicializar Google Secret Manager: {e}")
                raise ConnectionError(f"Error al inicializar Google Secret Manager: {e}. Revise su configuración de autenticación.")
        
        return cls._instance

    def get_env_variable(self, var_name: str, default=None):
        # Valores considerados secretos (claves de API, etc.) - SOLO obtenerse de Secret Manager
        secretos = ["BINANCE_API_KEY_FUTURES", "BINANCE_API_SECRET_FUTURES"]
        
        # Si es un valor secreto, solo buscarlo en Secret Manager
        if var_name in secretos:
            if not self.secret_client or not self.gcp_project_id:
                raise ValueError(f"Se requiere Google Secret Manager para obtener el secreto {var_name}, pero no está disponible")
                
            try:
                secret_name = f"projects/{self.gcp_project_id}/secrets/{var_name}/versions/latest"
                response = self.secret_client.access_secret_version(name=secret_name)
                secret_value = response.payload.data.decode('UTF-8')
                logger.info(f"Secreto {var_name} obtenido correctamente de Secret Manager")
                return secret_value
            except Exception as e:
                logger.error(f"No se pudo obtener el secreto {var_name} de Secret Manager: {e}")
                raise ValueError(f"Error al obtener el secreto {var_name} de Secret Manager: {e}")
        
        # Si no es un secreto, obtenerlo de las variables de entorno
        env_value = os.getenv(var_name, default)
        return env_value

    def get_config_value(self, key_path: str, default=None):
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            # logger.warning(f"Clave de configuración '{key_path}' no encontrada. Usando default: {default}")
            return default
        except TypeError: # En caso de que self.config no se haya cargado
             # logger.error(f"Configuración no cargada. Imposible obtener '{key_path}'.")
             raise TypeError(f"Configuración no cargada. Imposible obtener '{key_path}'.")


if __name__ == '__main__': # Para pruebas rápidas
    manager = ConfigManager(config_path='../../src/config.yaml', env_path='../../.env') # Ajusta paths si ejecutas directo
    print(f"Raw Data Path: {manager.get_config_value('data_paths.raw')}")
    print(f"API Key: {manager.get_env_variable('BINANCE_API_KEY_FUTURES')}")
    print(f"Default Symbol: {manager.get_config_value('data_acquisition_defaults.symbol')}")
