import yaml
import os
import logging
from dotenv import load_dotenv
from google.cloud import secretmanager

logger = logging.getLogger(__name__)

class ConfigManager:
    _instance = None
    _secret_client = None  # Cliente de Secret Manager

    def __new__(cls, config_path="src/config.yaml", env_path=".env", gcp_project_id: str = None):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            # Carga de .env (útil solo para obtener GOOGLE_CLOUD_PROJECT si no se pasa como parámetro)
            load_dotenv(dotenv_path=env_path)

            # Carga de config.yaml
            try:
                with open(config_path, 'r') as f:
                    cls._instance.config = yaml.safe_load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Archivo de configuración {config_path} no encontrado.")
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error al parsear el archivo YAML {config_path}: {e}")
            
            # Inicializar cliente de Secret Manager si estamos en GCP o se proporciona project_id
            # En GCP (Cloud Run, Vertex AI, GCF), el project_id a menudo se infiere.
            cls._instance.gcp_project_id = gcp_project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
            if cls._instance.gcp_project_id:
                try:
                    cls._secret_client = secretmanager.SecretManagerServiceClient()
                    logger.info("Secret Manager client inicializado.")
                except Exception as e:
                    raise RuntimeError(f"No se pudo inicializar Secret Manager client: {e}")
            else:
                raise ValueError("No se proporcionó GCP Project ID. Se requiere configurar GOOGLE_CLOUD_PROJECT.")
                
        return cls._instance

    def get_env_variable(self, var_name: str, default=None):
        # Este método debe ser usado sólo para variables no sensibles
        # Las credenciales y secretos deben obtenerse exclusivamente de Secret Manager
        value = os.getenv(var_name)
        if value is not None:
            return value
        return default

    def get_secret(self, secret_id: str, version_id: str = "latest") -> str:
        """Obtiene un secreto de Google Secret Manager."""
        if not self._secret_client or not self.gcp_project_id:
            raise ValueError(f"Secret Manager client no disponible. No se puede obtener el secreto '{secret_id}'. Asegúrate de configurar GOOGLE_CLOUD_PROJECT y tener permisos para Secret Manager.")

        secret_name = f"projects/{self.gcp_project_id}/secrets/{secret_id}/versions/{version_id}"
        try:
            response = self._secret_client.access_secret_version(name=secret_name)
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            raise ValueError(f"No se pudo acceder al secreto '{secret_id}' (versión '{version_id}') desde Secret Manager: {e}")

    def get_config_value(self, key_path: str, default=None):
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            logger.warning(f"Clave de configuración '{key_path}' no encontrada. Usando default: {default}")
            return default
        except TypeError: # En caso de que self.config no se haya cargado
             logger.error(f"Configuración no cargada. Imposible obtener '{key_path}'.")
             raise TypeError(f"Configuración no cargada. Imposible obtener '{key_path}'.")


if __name__ == '__main__': # Para pruebas rápidas
    manager = ConfigManager(config_path='../../src/config.yaml', env_path='../../.env') # Ajusta paths si ejecutas directo
    print(f"Raw Data Path: {manager.get_config_value('data_paths.raw')}")
    print(f"API Key: {manager.get_env_variable('BINANCE_API_KEY_FUTURES')}")
    print(f"Default Symbol: {manager.get_config_value('data_acquisition_defaults.symbol')}")
