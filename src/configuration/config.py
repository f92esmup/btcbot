"""
Módulo de configuración centralizada para el bot de trading.
Maneja la carga y acceso a parámetros desde config.yaml y Google Cloud Secret Manager.
"""

import yaml
import os
from typing import Dict, Any, List, Optional, Union
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
    # (is_testnet property removed - testnet mode should be determined explicitly by callers)
    
    # Métodos para API keys de Binance
    def get_binance_api_key(self, is_testnet: bool = False) -> str:
        """
        Obtiene la API Key de Binance desde Google Cloud Secret Manager.
        
        Args:
            is_testnet (bool): Si True, carga la clave de testnet. Si False, carga la de producción.
            
        Returns:
            str: La API key correspondiente
        """
        try:
            if is_testnet:
                secret_name = 'TESTNET_BINANCE_API_KEY_FUTURES'
                logger.info("Cargando API key de testnet")
            else:
                secret_name = 'BINANCE_API_KEY_FUTURES'
                logger.info("Cargando API key de producción")
            
            api_key = self._get_secret(secret_name)
            logger.info(f"API key {'de testnet' if is_testnet else 'de producción'} cargada exitosamente")
            return api_key
        except Exception as e:
            logger.error(f"Error al cargar la API key: {e}")
            raise RuntimeError(f"API key no disponible. Verifica la configuración de Google Cloud Secret Manager: {e}")
    
    def get_binance_api_secret(self, is_testnet: bool = False) -> str:
        """
        Obtiene el API Secret de Binance desde Google Cloud Secret Manager.
        
        Args:
            is_testnet (bool): Si True, carga el secreto de testnet. Si False, carga el de producción.
            
        Returns:
            str: El API secret correspondiente
        """
        try:
            if is_testnet:
                secret_name = 'TESTNET_BINANCE_API_SECRET_FUTURES'
                logger.info("Cargando API secret de testnet")
            else:
                secret_name = 'BINANCE_API_SECRET_FUTURES'
                logger.info("Cargando API secret de producción")
            
            api_secret = self._get_secret(secret_name)
            logger.info(f"API secret {'de testnet' if is_testnet else 'de producción'} cargado exitosamente")
            return api_secret
        except Exception as e:
            logger.error(f"Error al cargar el API secret: {e}")
            raise RuntimeError(f"API secret no disponible. Verifica la configuración de Google Cloud Secret Manager: {e}")
    
    # Propiedades para Telegram
    @property
    def telegram_bot_token(self) -> str:
        """Token del bot de Telegram desde Google Secret Manager."""
        try:
            return self._get_secret('TELEGRAM_BOT_TOKEN')
        except Exception as e:
            logger.error(f"Error al cargar el token del bot de Telegram: {e}")
            raise RuntimeError("Token del bot de Telegram no disponible. Verifica que el secreto 'TELEGRAM_BOT_TOKEN' exista en Google Cloud Secret Manager.")
    
    @property
    def telegram_chat_id(self) -> str:
        """ID del chat de Telegram desde Google Secret Manager."""
        try:
            return self._get_secret('TELEGRAM_CHAT_ID')
        except Exception as e:
            logger.error(f"Error al cargar el ID del chat de Telegram: {e}")
            raise RuntimeError("ID del chat de Telegram no disponible. Verifica que el secreto 'TELEGRAM_CHAT_ID' exista en Google Cloud Secret Manager.")

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
    
    @property
    def gcs_price_scaler_blob_name(self) -> str:
        """Nombre del archivo price_scaler en GCS."""
        return self._config.get('gcp', {}).get('storage', {}).get('price_scaler_blob_name', 'price_scaler.pkl')

    # Propiedades para Vertex AI TensorBoard
    @property
    def tensorboard_vertex_ai_config(self) -> Dict[str, Any]:
        """Configuración completa de Vertex AI TensorBoard."""
        return self._config.get('tensorboard_vertex_ai', {})

    @property
    def tensorboard_instance_name(self) -> Optional[str]:
        """Nombre de la instancia de Vertex AI TensorBoard."""
        return self.tensorboard_vertex_ai_config.get('instance_name')

    @property
    def tensorboard_experiment_name(self) -> str:
        """Nombre del experimento de TensorBoard."""
        return self.tensorboard_vertex_ai_config.get('experiment_name', 'default-experiment')

    @property
    def tensorboard_location(self) -> Optional[str]:
        """Ubicación/región de la instancia de TensorBoard."""
        return self.tensorboard_vertex_ai_config.get('location')
    
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
    
    # Propiedades para configuración del entorno de trading
    @property
    def environment_config(self) -> Dict[str, Any]:
        """Configuración completa del entorno de trading."""
        return self._config.get('environment', {})
    
    @property
    def capital_inicial(self) -> float:
        """Capital inicial en USDT."""
        return self.environment_config.get('capital_inicial', 10000.0)
    
    @property
    def apalancamiento(self) -> int:
        """Apalancamiento máximo."""
        return self.environment_config.get('apalancamiento', 10)
    
    @property
    def porcentaje_max_inversion_por_trade(self) -> float:
        """Porcentaje máximo del balance a usar como margen por trade."""
        return self.environment_config.get('porcentaje_max_inversion_por_trade', 0.05)
    
    @property
    def max_drawdown_configurado_cuenta(self) -> float:
        """Drawdown máximo antes de terminated."""
        return self.environment_config.get('max_drawdown_configurado_cuenta', 0.5)
    
    @property
    def comision_taker_porcentaje(self) -> float:
        """Comisión taker en porcentaje."""
        return self.environment_config.get('comision_taker_porcentaje', 0.0004)
    
    @property
    def slippage_porcentaje(self) -> float:
        """Slippage en porcentaje."""
        return self.environment_config.get('slippage_porcentaje', 0.00005)
    
    @property
    def ventana_observacion_size(self) -> int:
        """Tamaño de la ventana de observación (L)."""
        return self.environment_config.get('ventana_observacion_size', 24)
    
    @property
    def max_pasos_en_posicion(self) -> int:
        """Máximo pasos esperados en una posición."""
        return self.environment_config.get('max_pasos_en_posicion', 100)
    
    @property
    def min_clip_pnl_roe(self) -> float:
        """ROE mínimo para clip."""
        return self.environment_config.get('min_clip_pnl_roe', -0.50)
    
    @property
    def max_clip_pnl_roe(self) -> float:
        """ROE máximo para clip."""
        return self.environment_config.get('max_clip_pnl_roe', 0.50)
    
    @property
    def zona_muerta_mantener(self) -> float:
        """Zona muerta para mantener posición."""
        return self.environment_config.get('zona_muerta_mantener', 0.15)
    
    @property
    def peso_recompensa_paso(self) -> float:
        """Peso para recompensa por paso."""
        return self.environment_config.get('peso_recompensa_paso', 1.0)
    
    @property
    def peso_recompensa_cierre(self) -> float:
        """Peso para recompensa al cerrar operación."""
        return self.environment_config.get('peso_recompensa_cierre', 1.0)
    
    @property
    def peso_recompensa_episodio(self) -> float:
        """Peso para Sortino ratio."""
        return self.environment_config.get('peso_recompensa_episodio', 0.0)
    
    @property
    def usar_log1p_en_pnl(self) -> bool:
        """Si aplicar log1p al PNL para suavizar."""
        return self.environment_config.get('usar_log1p_en_pnl', True)
    
    @property
    def usar_max_pasos_episodio(self) -> bool:
        """Si limitar pasos por episodio."""
        return self.environment_config.get('usar_max_pasos_episodio', True)
    
    @property
    def max_pasos_episodio(self) -> int:
        """Máximo pasos por episodio."""
        return self.environment_config.get('max_pasos_episodio', 1000)

    @property
    def max_consecutive_losses(self) -> int:
        """Máximo de pérdidas consecutivas antes de activar el kill switch."""
        return self.environment_config.get('max_consecutive_losses', 5)

    # Propiedades para configuración del agente SAC
    @property
    def agent_config(self) -> Dict[str, Any]:
        """Configuración completa del agente."""
        return self._config.get('agent', {})
    
    @property
    def algorithm(self) -> str:
        """Algoritmo del agente."""
        return self.agent_config.get('algorithm', 'SAC')
    
    @property
    def replay_buffer_size(self) -> int:
        """Tamaño del buffer de replay."""
        return self.agent_config.get('replay_buffer_size', 1000000)
    
    @property
    def batch_size(self) -> int:
        """Tamaño del batch para entrenamiento."""
        return self.agent_config.get('batch_size', 256)
    
    @property
    def min_buffer_for_learning(self) -> int:
        """Mínimo número de experiencias antes de comenzar el aprendizaje."""
        return self.agent_config.get('min_buffer_for_learning', 100000)

    @property
    def hiperparametros_sac(self) -> Dict[str, Any]:
        """Hiperparámetros específicos de SAC."""
        return self.agent_config.get('hiperparametros_sac', {})
    
    @property
    def gamma(self) -> float:
        """Factor de descuento."""
        return self.hiperparametros_sac.get('gamma', 0.99)
    
    @property
    def tau(self) -> float:
        """Coeficiente para actualización suave de redes objetivo."""
        return self.hiperparametros_sac.get('tau', 0.005)
    
    @property
    def actor_learning_rate(self) -> float:
        """Tasa de aprendizaje del actor."""
        return self.hiperparametros_sac.get('actor_learning_rate', 0.0003)
    
    @property
    def critic_learning_rate(self) -> float:
        """Tasa de aprendizaje del crítico."""
        return self.hiperparametros_sac.get('critic_learning_rate', 0.0003)
    
    @property
    def learn_alpha(self) -> bool:
        """Si alpha es aprendible."""
        return self.hiperparametros_sac.get('learn_alpha', True)
    
    @property
    def alpha_learning_rate(self) -> float:
        """Tasa de aprendizaje de alpha."""
        return self.hiperparametros_sac.get('alpha_learning_rate', 0.0003)
    
    @property
    def target_entropy(self) -> Union[str, float]:
        """
        Entropía objetivo.
        
        Returns:
            'auto' para cálculo automático como -dim_action, o valor numérico específico
        """
        return self.hiperparametros_sac.get('target_entropy', 'auto')
    
    @property
    def initial_log_alpha(self) -> float:
        """Valor inicial de log alpha."""
        return self.hiperparametros_sac.get('initial_log_alpha', 0.0)
    
    @property
    def transformer_config(self) -> Dict[str, Any]:
        """Configuración del Transformer."""
        return self.agent_config.get('transformer', {})
    
    @property
    def d_model(self) -> int:
        """Dimensión interna del Transformer."""
        return self.transformer_config.get('d_model', 128)
    
    @property
    def n_head(self) -> int:
        """Número de cabezales de atención."""
        return self.transformer_config.get('n_head', 4)
    
    @property
    def num_encoder_layers(self) -> int:
        """Número de capas del encoder."""
        return self.transformer_config.get('num_encoder_layers', 3)
    
    @property
    def dim_feedforward(self) -> int:
        """Dimensión de la capa feedforward."""
        return self.transformer_config.get('dim_feedforward', 256)
    
    @property
    def dropout_rate(self) -> float:
        """Tasa de dropout."""
        return self.transformer_config.get('dropout_rate', 0.1)
    
    @property
    def mlp_heads_config(self) -> Dict[str, Any]:
        """Configuración de las cabezas MLP."""
        return self.agent_config.get('mlp_heads', {})
    
    @property
    def hidden_dims(self) -> List[int]:
        """Dimensiones de las capas ocultas del MLP."""
        return self.mlp_heads_config.get('hidden_dims', [256, 256])
    
    @property
    def learning_frequency(self) -> int:
        """Frecuencia de aprendizaje."""
        return self.agent_config.get('learning_frequency', 1)
    
    @property
    def update_target_frequency(self) -> int:
        """Frecuencia de actualización de redes objetivo."""
        return self.agent_config.get('update_target_frequency', 1)
    
    @property
    def model_save_frequency(self) -> int:
        """Frecuencia de guardado de modelos."""
        return self.agent_config.get('model_save_frequency', 10000)
    
    @property
    def models_directory(self) -> str:
        """Directorio para guardar modelos."""
        return self.agent_config.get('models_directory', 'models/agent')

    # Métodos adicionales
    def get_environment_info(self) -> Dict[str, str]:
        """Retorna información sobre el entorno actual."""
        # Note: testnet mode is now determined explicitly by callers
        return {
            'project_id': self._config.get('gcp', {}).get('project_id', 'default')
        }