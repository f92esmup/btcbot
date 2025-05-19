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
            load_dotenv(dotenv_path=env_path)
            try:
                with open(config_path, 'r') as f:
                    cls._instance.config = yaml.safe_load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Archivo de configuración {config_path} no encontrado.")
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error al parsear el archivo YAML {config_path}: {e}")
            
            # Inicializar cliente de Secret Manager si se define un proyecto de GCP
            cls._instance.gcp_project_id = cls._instance.get_env_variable('GCP_PROJECT_ID')
            cls._instance.secret_client = None
            
            if cls._instance.gcp_project_id:
                try:
                    cls._instance.secret_client = secretmanager.SecretManagerServiceClient()
                    logger.info(f"Cliente de Google Secret Manager inicializado para proyecto: {cls._instance.gcp_project_id}")
                except Exception as e:
                    logger.warning(f"No se pudo inicializar Google Secret Manager: {e}")
        
        return cls._instance

    def get_env_variable(self, var_name: str, default=None):
        # Primero intentar obtener de Secret Manager si está configurado
        if self.secret_client and self.gcp_project_id:
            try:
                secret_name = f"projects/{self.gcp_project_id}/secrets/{var_name}/versions/latest"
                response = self.secret_client.access_secret_version(name=secret_name)
                secret_value = response.payload.data.decode('UTF-8')
                logger.debug(f"Secreto {var_name} obtenido correctamente de Secret Manager")
                return secret_value
            except Exception as e:
                logger.info(f"No se pudo obtener secreto {var_name} de Secret Manager: {e}")
                # Si el error es porque no existe este secreto específico, continuar con variables de entorno
                pass
        
        # Si no hay Secret Manager o no se encontró el secreto, intentar con variables de entorno
        env_value = os.getenv(var_name, default)
        if env_value is None:
            logger.warning(f"Variable {var_name} no encontrada en Secret Manager ni en variables de entorno")
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
