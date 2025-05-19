import yaml
import os
from dotenv import load_dotenv

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
                # logger.error(f"Archivo de configuración {config_path} no encontrado.") # Necesitarías un logger aquí o lanzar excepción
                raise FileNotFoundError(f"Archivo de configuración {config_path} no encontrado.")
            except yaml.YAMLError as e:
                # logger.error(f"Error al parsear el archivo YAML {config_path}: {e}")
                raise yaml.YAMLError(f"Error al parsear el archivo YAML {config_path}: {e}")
        return cls._instance

    def get_env_variable(self, var_name: str, default=None):
        return os.getenv(var_name, default)

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
