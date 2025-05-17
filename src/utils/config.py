import yaml
import os
import logging
from dotenv import load_dotenv

# Configurar logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManager:
    _instance = None

    def __new__(cls, config_path="src/config.yaml", env_path=".env"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            # Cargar variables de entorno desde .env si existe
            load_dotenv(dotenv_path=env_path)
            try:
                with open(config_path, 'r') as f:
                    cls._instance.config = yaml.safe_load(f)
            except FileNotFoundError:
                logger.error(f"Archivo de configuración {config_path} no encontrado.")
                raise FileNotFoundError(f"Archivo de configuración {config_path} no encontrado.")
            except yaml.YAMLError as e:
                logger.error(f"Error al parsear el archivo YAML {config_path}: {e}")
                raise yaml.YAMLError(f"Error al parsear el archivo YAML {config_path}: {e}")
        return cls._instance

    def get_env_variable(self, var_name: str, default=None):
        """
        Obtiene una variable de entorno.
        Si no se encuentra, devuelve el valor por defecto.
        """
        value = os.getenv(var_name)
        if value is None and default is None:
            logger.warning(f"Variable de entorno '{var_name}' no encontrada y no se proporcionó valor por defecto.")
        return os.getenv(var_name, default)

    def get_config_value(self, key_path: str, default=None, env_var=None, required=False):
        """
        Obtiene un valor de configuración, priorizando variables de entorno sobre el archivo YAML.
        
        Args:
            key_path: Ruta de clave en el YAML (ej: 'data_paths.raw')
            default: Valor por defecto si no se encuentra
            env_var: Nombre de la variable de entorno que debe prevalecer sobre el YAML
            required: Si es True y no se encuentra el valor, genera error
        """
        # Si se proporciona una variable de entorno, intentar usarla primero
        if env_var:
            env_value = self.get_env_variable(env_var)
            if env_value is not None:
                return env_value
        
        # Si no hay variable de entorno o está vacía, intentar obtener del YAML
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            if required:
                logger.error(f"Clave de configuración requerida '{key_path}' no encontrada.")
                raise KeyError(f"Clave de configuración requerida '{key_path}' no encontrada.")
            logger.warning(f"Clave de configuración '{key_path}' no encontrada. Usando default: {default}")
            return default
        except TypeError: # En caso de que self.config no se haya cargado
             logger.error(f"Configuración no cargada. Imposible obtener '{key_path}'.")
             raise TypeError(f"Configuración no cargada. Imposible obtener '{key_path}'.")

    def get_agent_config(self, param_name, env_var=None, default=None):
        """
        Método específico para obtener configuración del agente, priorizando variables de entorno.
        Busca primero en variables de entorno, luego en agent_config.yaml.
        
        Args:
            param_name: Nombre del parámetro en agent_config.yaml
            env_var: Nombre de la variable de entorno a usar (si es diferente a AGENT_{param_name})
            default: Valor por defecto si no se encuentra
        """
        # Si no se especificó variable de entorno, usar convención AGENT_{param_name}
        if env_var is None:
            env_var = f"AGENT_{param_name.upper()}"
        
        # Intentar obtener de variables de entorno
        env_value = self.get_env_variable(env_var)
        if env_value is not None:
            # Convertir al tipo adecuado (simple)
            if isinstance(default, int) or (default is None and param_name in ['buffer_size', 'batch_size', 'n_steps']):
                return int(env_value)
            elif isinstance(default, float) or (default is None and param_name in ['learning_rate', 'gamma']):
                return float(env_value)
            elif isinstance(default, bool) or (default is None and param_name in ['normalize_advantage']):
                return env_value.lower() in ['true', '1', 'yes']
            else:
                return env_value
        
        # Si no hay en variables de entorno, intentar cargar de agent_config.yaml
        try:
            with open("src/agent/agent_config.yaml", 'r') as f:
                agent_config = yaml.safe_load(f)
            return agent_config.get(param_name, default)
        except (FileNotFoundError, yaml.YAMLError):
            logger.warning(f"No se pudo cargar agent_config.yaml. Usando default para {param_name}: {default}")
            return default


if __name__ == '__main__': # Para pruebas rápidas
    manager = ConfigManager(config_path='../../src/config.yaml', env_path='../../.env') # Ajusta paths si ejecutas directo
    print(f"Raw Data Path: {manager.get_config_value('data_paths.raw', env_var='DATA_RAW_PATH')}")
    print(f"API Key: {manager.get_env_variable('BINANCE_API_KEY_FUTURES')}")
    print(f"Default Symbol: {manager.get_config_value('data_acquisition_defaults.symbol', env_var='DEFAULT_SYMBOL')}")
    print(f"Agent Learning Rate: {manager.get_agent_config('learning_rate', default=0.0003)}")
