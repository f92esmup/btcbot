import yaml
import os
from dotenv import load_dotenv
from google.cloud import secretmanager
import json
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Clase singleton que proporciona acceso centralizado a toda la configuración del proyecto.
    
    Características:
    - Acceso a configuración desde un único archivo YAML centralizado
    - Acceso a variables de entorno
    - Acceso a secretos almacenados en Google Cloud Secret Manager
    - Métodos de conveniencia para acceder a secciones específicas de configuración
    """
    _instance = None

    def __new__(cls, config_path="src/config.yaml", env_path=".env"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            # Inicializar estos atributos primero para evitar errores
            cls._instance.config = None
            cls._instance.config_path = config_path  # Guardar el config_path
            cls._instance.gcp_project_id = None 
            cls._instance.secret_client = None
            
            # Cargar variables de entorno primero (para GCP_PROJECT_ID, etc.)
            load_dotenv(dotenv_path=env_path)
            
            # Leer la configuración YAML
            try:
                with open(config_path, 'r') as f:
                    cls._instance.config = yaml.safe_load(f)
                logger.info(f"Configuración cargada correctamente desde {config_path}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Archivo de configuración {config_path} no encontrado.")
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error al parsear el archivo YAML {config_path}: {e}")
            
            # Obtener y validar el ID del proyecto de GCP (OBLIGATORIO)
            cls._instance.gcp_project_id = os.getenv('GCP_PROJECT_ID')
            if not cls._instance.gcp_project_id:
                raise ValueError("GCP_PROJECT_ID no configurado en .env - es obligatorio para el funcionamiento del sistema")
            
            # Obtener y validar el nombre del bucket de GCS (OBLIGATORIO)
            cls._instance.gcs_bucket_name = os.getenv('GCS_BUCKET_NAME')
            if not cls._instance.gcs_bucket_name:
                raise ValueError("GCS_BUCKET_NAME no configurado en .env - es obligatorio para el funcionamiento del sistema")
            
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

    def get_env_variable(self, var_name: str, default=None) -> str:
        """
        Obtiene una variable de entorno o un secreto de Secret Manager.
        Para variables sensibles (credenciales, claves API), solo se obtienen de Secret Manager.
        
        Args:
            var_name: Nombre de la variable
            default: Valor por defecto si no se encuentra
            
        Returns:
            Valor de la variable o secreto
        """
        # Valores considerados secretos (claves de API, etc.) - SOLO obtenerse de Secret Manager
        secretos = [
            "BINANCE_API_KEY_FUTURES", 
            "BINANCE_API_SECRET_FUTURES",
            "TESTNET_BINANCE_API_KEY_FUTURES",    # Nombre del secreto para la API Key de Testnet
            "TESTNET_BINANCE_API_SECRET_FUTURES"  # Nombre del secreto para la API Secret de Testnet
        ]
        
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

    def get_config_value(self, key_path: str, default=None) -> Any:
        """
        Obtiene un valor de configuración mediante una ruta de claves separadas por puntos.
        
        Args:
            key_path: Ruta a la configuración (ej. "data_paths.raw")
            default: Valor por defecto si no se encuentra
            
        Returns:
            Valor de configuración o valor por defecto
        """
        try:
            keys = key_path.split('.')
            value = self.config
            for key in keys:
                value = value[key]
            return value
        except KeyError:
            logger.debug(f"Clave de configuración '{key_path}' no encontrada. Usando default: {default}")
            return default
        except TypeError:
            raise TypeError(f"Configuración no cargada. Imposible obtener '{key_path}'.")

    # Métodos de conveniencia para secciones específicas
    def get_data_paths(self) -> Dict[str, str]:
        """Obtiene todas las rutas de datos configuradas"""
        return self.get_config_value('data_paths', {})
    
    def get_binance_api_config(self) -> Dict[str, Any]:
        """Obtiene la configuración de la API de Binance"""
        return self.get_config_value('binance_api', {})
    
    def get_data_acquisition_defaults(self) -> Dict[str, Any]:
        """Obtiene la configuración por defecto para adquisición de datos"""
        return self.get_config_value('data_acquisition_defaults', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Obtiene la configuración de preprocesamiento"""
        return self.get_config_value('preprocessing', {})
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Obtiene la configuración del entorno de trading"""
        return self.get_config_value('environment', {})
    
    def get_agent_config(self) -> Dict[str, Any]:
        """Obtiene la configuración del agente RL"""
        return self.get_config_value('agent', {})

    def get_live_trading_config(self) -> Dict[str, Any]:
        """
        Devuelve la configuración específica para el modo de trading en vivo.
        """
        return self.get_config_value('live_trading', {})

    def get_testing_config(self) -> Dict[str, Any]:
        """
        Devuelve la configuración específica para testing y desarrollo.
        """
        return self.get_config_value('testing', {})

    def get_websocket_url(self, trading_mode: str = None) -> str:
        """
        Obtiene la URL base del WebSocket según el modo de trading.
        
        Args:
            trading_mode: 'TESTNET' o 'REAL'. Si no se especifica, usa LIVE_TRADING_MODE del entorno.
            
        Returns:
            URL base del WebSocket de Binance
        """
        if trading_mode is None:
            trading_mode = os.getenv('LIVE_TRADING_MODE', 'TESTNET').upper()
        
        websocket_urls = self.get_config_value('live_trading.websocket_urls', {})
        
        if trading_mode == 'TESTNET':
            return websocket_urls.get('testnet', 'wss://stream.binancefuture.com/ws')
        else:  # REAL
            return websocket_urls.get('real', 'wss://fstream.binance.com/ws')

    def get_full_config(self) -> Dict[str, Any]:
        """Obtiene la configuración completa"""
        return self.config

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file (static method for simple use cases).
        
        This method provides the same functionality as the deprecated config_loader.load_config
        for backward compatibility and simple use cases where you don't need the full ConfigManager.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the parsed configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise RuntimeError(f"Unexpected error loading configuration: {e}")


if __name__ == '__main__': # Para pruebas rápidas
    manager = ConfigManager(config_path='../../src/config.yaml', env_path='../../.env') # Ajusta paths si ejecutas directo
    print(f"Raw Data Path: {manager.get_config_value('data_paths.raw')}")
    print(f"API Key: {manager.get_env_variable('BINANCE_API_KEY_FUTURES')}")
    print(f"Default Symbol: {manager.get_config_value('data_acquisition_defaults.symbol')}")
    
    # Prueba los nuevos métodos de conveniencia
    print("\nMétodos de conveniencia:")
    print(f"Data Paths: {manager.get_data_paths()}")
    print(f"Binance API Config: {manager.get_binance_api_config()}")
    print(f"Preprocessing Sequence Length: {manager.get_preprocessing_config().get('sequence_length_L')}")
    print(f"Environment Initial Equity: {manager.get_environment_config().get('initial_equity')}")
    print(f"Agent Learning Rate: {manager.get_agent_config().get('learning_rate')}")
    print(f"Live Trading Config: {manager.get_live_trading_config()}")
    print(f"Testing Config: {manager.get_testing_config()}")
    print(f"WebSocket URL (TESTNET): {manager.get_websocket_url('TESTNET')}")
    print(f"WebSocket URL (REAL): {manager.get_websocket_url('REAL')}")
