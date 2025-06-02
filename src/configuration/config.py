"""
Módulo de configuración centralizada para el bot de trading.
Maneja la carga y acceso a parámetros desde config.yaml y Google Cloud Secret Manager.
"""

import yaml
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from google.cloud import secretmanager
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Clase para gestionar la configuración del proyecto."""
    
    def __init__(self):
        """Inicializa la configuración cargando el archivo YAML y configurando Secret Manager."""
        self.config_path = Path(__file__).parent / "config.yaml"
        self._config = self._load_config()
        self._secret_client = None
        self._api_keys = {}
        self._load_api_keys()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carga la configuración desde el archivo YAML."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error al parsear el archivo YAML: {e}")
    
    def _get_secret_client(self) -> secretmanager.SecretManagerServiceClient:
        """Obtiene el cliente de Secret Manager."""
        if self._secret_client is None:
            self._secret_client = secretmanager.SecretManagerServiceClient()
        return self._secret_client
    
    def _get_secret(self, secret_name: str, project_id: Optional[str] = None) -> str:
        """Obtiene un secreto de Google Cloud Secret Manager."""
        try:
            client = self._get_secret_client()
            
            # Usar el project_id de la configuración si no se proporciona uno
            if project_id is None:
                project_id = self._config.get('gcp', {}).get('project_id')
            
            # Si aún no hay project_id, intentar obtenerlo del entorno
            if project_id is None:
                project_id = os.environ.get('GOOGLE_CLOUD_PROJECT')
            
            if project_id is None:
                raise ValueError("No se pudo determinar el project_id de Google Cloud. "
                               "Configúralo en config.yaml o en la variable de entorno GOOGLE_CLOUD_PROJECT")
            
            # Construir el nombre del recurso del secreto
            name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            
            # Acceder al secreto
            response = client.access_secret_version(request={"name": name})
            secret_value = response.payload.data.decode("UTF-8")
            
            logger.info(f"Secreto '{secret_name}' cargado exitosamente")
            return secret_value
            
        except Exception as e:
            logger.error(f"Error al cargar el secreto '{secret_name}': {e}")
            raise RuntimeError(f"No se pudo cargar el secreto '{secret_name}': {e}")
    
    def _load_api_keys(self):
        """Carga las API keys desde Google Cloud Secret Manager según el modo testnet."""
        try:
            is_testnet = self.is_testnet
            
            if is_testnet:
                # Cargar claves de testnet
                self._api_keys['api_key'] = self._get_secret('TESTNET_BINANCE_API_KEY_FUTURES')
                self._api_keys['api_secret'] = self._get_secret('TESTNET_BINANCE_API_SECRET_FUTURES')
                logger.info("API keys de testnet cargadas exitosamente")
            else:
                # Cargar claves de producción
                self._api_keys['api_key'] = self._get_secret('BINANCE_API_KEY_FUTURES')
                self._api_keys['api_secret'] = self._get_secret('BINANCE_API_SECRET_FUTURES')
                logger.info("API keys de producción cargadas exitosamente")
                
        except Exception as e:
            logger.error(f"Error al cargar las API keys: {e}")
            # No lanzar excepción aquí para permitir que la aplicación continue
            # Las propiedades de API keys manejarán el error cuando se acceda a ellas
    
    # Propiedades para API de Binance
    @property
    def api_call_limit(self) -> int:
        """Límite de velas por llamada a la API."""
        return self._config['api']['call_limit']
    
    @property
    def max_api_retries(self) -> int:
        """Número máximo de reintentos para llamadas a la API."""
        return self._config['api']['max_retries']
    
    @property
    def retry_delay(self) -> int:
        """Tiempo de espera entre reintentos (segundos)."""
        return self._config['api']['retry_delay']
    
    @property
    def api_timeout(self) -> int:
        """Timeout para las peticiones a la API (segundos)."""
        return self._config['api']['timeout']
    
    # Propiedades para configuración de trading
    @property
    def is_testnet(self) -> bool:
        """Indica si está en modo testnet."""
        return self._config.get('trading', {}).get('testnet', False)
    
    # Propiedades para API keys de Binance
    @property
    def binance_api_key(self) -> str:
        """API Key de Binance (producción o testnet según configuración)."""
        if 'api_key' not in self._api_keys:
            raise RuntimeError("API key no disponible. Verifica la configuración de Google Cloud Secret Manager.")
        return self._api_keys['api_key']
    
    @property
    def binance_api_secret(self) -> str:
        """API Secret de Binance (producción o testnet según configuración)."""
        if 'api_secret' not in self._api_keys:
            raise RuntimeError("API secret no disponible. Verifica la configuración de Google Cloud Secret Manager.")
        return self._api_keys['api_secret']
    
    # Propiedades para datos
    @property
    def ohlcv_columns(self) -> List[str]:
        """Lista de columnas OHLCV a mantener."""
        return self._config['data']['columns']
    
    @property
    def data_dtypes(self) -> Dict[str, str]:
        """Tipos de datos para optimizar RAM."""
        return self._config['data']['dtypes']
    
    # Propiedades para zona horaria
    @property
    def target_timezone(self) -> str:
        """Zona horaria objetivo (Madrid)."""
        return self._config['timezone']['target']
    
    @property
    def source_timezone(self) -> str:
        """Zona horaria de origen (UTC)."""
        return self._config['timezone']['source']
    
    # Propiedades para interpolación
    @property
    def interpolation_method(self) -> str:
        """Método de interpolación para NaNs."""
        return self._config['interpolation']['method']
    
    @property
    def interpolation_limit_direction(self) -> str:
        """Dirección límite para interpolación."""
        return self._config['interpolation']['limit_direction']
    
    # Propiedades para normalización
    @property
    def normalization_config(self) -> Dict[str, Any]:
        """Configuración completa de normalización."""
        return self._config.get('normalization', {})
    
    @property
    def scaler_type(self) -> str:
        """Tipo de escalador a usar para normalización."""
        return self.normalization_config.get('scaler_type', 'MinMaxScaler')
    
    @property
    def feature_range(self) -> List[float]:
        """Rango de características para normalización."""
        return self.normalization_config.get('feature_range', [0, 1])
    
    @property
    def scaler_path(self) -> str:
        """Ruta donde guardar el scaler ajustado."""
        return self.normalization_config.get('scaler_path', 'models/scaler.pkl')
    
    @property
    def storage_mode(self) -> str:
        """Modo de almacenamiento del scaler: 'local' o 'gcp'."""
        return self.normalization_config.get('storage_mode', 'local')
    
    # Propiedades para Google Cloud Platform
    @property
    def project_id(self) -> str:
        """Project ID de Google Cloud."""
        return self._config.get('gcp', {}).get('project_id')
    
    @property
    def gcs_bucket_name(self) -> str:
        """Nombre del bucket de GCS para almacenar modelos."""
        return self._config.get('gcp', {}).get('storage', {}).get('bucket_name', 'btcbot-models')
    
    @property
    def gcs_scaler_blob_name(self) -> str:
        """Nombre del archivo scaler en GCS."""
        return self._config.get('gcp', {}).get('storage', {}).get('scaler_blob_name', 'scaler.pkl')
    
    # Propiedades para indicadores técnicos
    @property
    def indicators_config(self) -> Dict[str, Any]:
        """Configuración completa de indicadores técnicos."""
        return self._config.get('indicators', {})
    
    @property
    def trend_indicators(self) -> Dict[str, Any]:
        """Configuración de indicadores de tendencia."""
        return self.indicators_config.get('trend', {})
    
    @property
    def momentum_indicators(self) -> Dict[str, Any]:
        """Configuración de indicadores de momento."""
        return self.indicators_config.get('momentum', {})
    
    @property
    def volatility_indicators(self) -> Dict[str, Any]:
        """Configuración de indicadores de volatilidad."""
        return self.indicators_config.get('volatility', {})
    
    @property
    def volume_indicators(self) -> Dict[str, Any]:
        """Configuración de indicadores de volumen."""
        return self.indicators_config.get('volume', {})
    
    # Métodos adicionales
    def refresh_api_keys(self):
        """Recarga las API keys desde Google Cloud Secret Manager."""
        self._api_keys.clear()
        self._load_api_keys()
        logger.info("API keys recargadas exitosamente")
    
    def get_environment_info(self) -> Dict[str, str]:
        """Retorna información sobre el entorno actual."""
        return {
            'mode': 'testnet' if self.is_testnet else 'production',
            'has_api_key': 'api_key' in self._api_keys,
            'has_api_secret': 'api_secret' in self._api_keys,
            'project_id': self._config.get('gcp', {}).get('project_id', 'default')
        }


# Instancia global de configuración
config = Config()