"""
Modelos de datos Pydantic para la configuración del sistema.

Este módulo define las clases de configuración que reemplazan el acceso
basado en diccionarios por un modelo de datos validado y con tipado estático.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from pathlib import Path


class TrainingSetupConfig(BaseModel):
    """Configuración del setup de entrenamiento."""
    seed: int = Field(..., description="Semilla para la reproducibilidad del entrenamiento")


class APIConfig(BaseModel):
    """Configuración de la API de Binance."""
    call_limit: int = Field(1000, description="Número máximo de velas por llamada")
    max_retries: int = Field(3, description="Número máximo de reintentos en caso de error")
    retry_delay: int = Field(1, description="Segundos entre reintentos")
    timeout: int = Field(10, description="Timeout para las peticiones en segundos")


class TradingConfig(BaseModel):
    """Configuración de trading."""
    testnet: bool = Field(True, description="Si está en modo testnet (true) o producción (false)")


class GCPStorageConfig(BaseModel):
    """Configuración de almacenamiento en GCP."""
    bucket_name: str = Field(..., description="Nombre del bucket para almacenar modelos")
    scaler_blob_name: str = Field("scaler.pkl", description="Nombre del archivo scaler en GCS")
    price_scaler_blob_name: str = Field("price_scaler.pkl", description="Nombre del archivo price_scaler en GCS")


class GCPSecretsConfig(BaseModel):
    """Configuración de secretos en GCP Secret Manager."""
    binance_api_key_futures: str = Field(..., description="Nombre del secreto para API key de Binance Futures")
    binance_api_secret_futures: str = Field(..., description="Nombre del secreto para API secret de Binance Futures")
    testnet_binance_api_key_futures: str = Field(..., description="Nombre del secreto para API key de Binance Testnet")
    testnet_binance_api_secret_futures: str = Field(..., description="Nombre del secreto para API secret de Binance Testnet")
    telegram_bot_token: str = Field(..., description="Nombre del secreto para token del bot de Telegram")
    telegram_chat_id: str = Field(..., description="Nombre del secreto para chat ID de Telegram")


class GCPConfig(BaseModel):
    """Configuración de Google Cloud Platform."""
    project_id: str = Field(..., description="ID del proyecto de Google Cloud")
    storage: GCPStorageConfig = Field(..., description="Configuración de almacenamiento")
    secrets: GCPSecretsConfig = Field(..., description="Configuración de secretos")


class DataConfig(BaseModel):
    """Configuración de datos."""
    columns: List[str] = Field(..., description="Columnas OHLCV a mantener")
    dtypes: Dict[str, str] = Field(..., description="Tipos de datos para optimizar RAM")


class TimezoneConfig(BaseModel):
    """Configuración de zona horaria."""
    target: str = Field("Europe/Madrid", description="Zona horaria objetivo")
    source: str = Field("UTC", description="Zona horaria de origen")


class InterpolationConfig(BaseModel):
    """Configuración de interpolación."""
    method: str = Field("linear", description="Método de interpolación para NaNs")
    limit_direction: str = Field("both", description="Dirección para fill")


class NormalizationConfig(BaseModel):
    """Configuración de normalización."""
    scaler_type: str = Field("MinMaxScaler", description="Tipo de escalador a usar")
    feature_range: List[Union[int, float]] = Field([0, 1], description="Rango de normalización")
    storage_mode: str = Field("local", description="Modo de almacenamiento: 'local' o 'gcp'")


class IndicatorConfig(BaseModel):
    """Configuración base para indicadores."""
    enabled: bool = Field(True, description="Si el indicador está habilitado")


class EMAConfig(IndicatorConfig):
    """Configuración para indicadores EMA."""
    period: int = Field(..., description="Período del indicador")


class ADXConfig(IndicatorConfig):
    """Configuración para indicador ADX."""
    period: int = Field(14, description="Período del indicador")


class RSIConfig(IndicatorConfig):
    """Configuración para indicador RSI."""
    period: int = Field(14, description="Período del indicador")


class StochConfig(IndicatorConfig):
    """Configuración para indicador Stochastic."""
    k_period: int = Field(14, description="Período K")
    d_period: int = Field(3, description="Período D")
    smooth_k: int = Field(3, description="Suavizado K")


class ATRConfig(IndicatorConfig):
    """Configuración para indicador ATR."""
    period: int = Field(14, description="Período del indicador")


class OBVConfig(IndicatorConfig):
    """Configuración para indicador OBV."""
    pass  # OBV no requiere parámetros adicionales


class TrendIndicatorsConfig(BaseModel):
    """Configuración de indicadores de tendencia."""
    ema_20: EMAConfig = Field(..., description="Configuración EMA 20")
    ema_50: EMAConfig = Field(..., description="Configuración EMA 50")
    adx: ADXConfig = Field(..., description="Configuración ADX")


class MomentumIndicatorsConfig(BaseModel):
    """Configuración de indicadores de momento."""
    rsi: RSIConfig = Field(..., description="Configuración RSI")
    stoch: StochConfig = Field(..., description="Configuración Stochastic")


class VolatilityIndicatorsConfig(BaseModel):
    """Configuración de indicadores de volatilidad."""
    atr: ATRConfig = Field(..., description="Configuración ATR")


class VolumeIndicatorsConfig(BaseModel):
    """Configuración de indicadores de volumen."""
    obv: OBVConfig = Field(..., description="Configuración OBV")


class EnvironmentConfig(BaseModel):
    """Configuración del entorno de trading."""
    # Capital y riesgo
    capital_inicial: float = Field(..., description="Capital inicial en USDT")
    apalancamiento: int = Field(..., description="Apalancamiento máximo")
    porcentaje_max_inversion_por_trade: float = Field(..., description="Porcentaje máximo del balance como margen por trade")
    max_drawdown_configurado_cuenta: float = Field(..., description="Drawdown máximo antes de terminated")
    max_consecutive_losses: int = Field(..., description="Número máximo de pérdidas consecutivas permitidas")
    
    # Costos de trading
    comision_taker_porcentaje: float = Field(..., description="Comisión taker sobre valor nocional")
    slippage_porcentaje: float = Field(..., description="Slippage sobre precio de ejecución")
    
    # Observación
    ventana_observacion_size: int = Field(..., description="Número de velas pasadas en la observación")
    
    # Normalización de portfolio para observación
    max_pasos_en_posicion: int = Field(..., description="Máximo esperado de pasos en una posición")
    min_clip_pnl_roe: float = Field(..., description="ROE mínimo para clip")
    max_clip_pnl_roe: float = Field(..., description="ROE máximo para clip")
    
    # Zona muerta para acciones
    zona_muerta_mantener: float = Field(..., description="Zona muerta para mantener posición")
    
    # Función de recompensa
    peso_recompensa_paso: float = Field(..., description="Peso para recompensa por paso")
    peso_recompensa_cierre: float = Field(..., description="Peso para recompensa al cerrar operación")
    peso_recompensa_episodio: float = Field(..., description="Peso para Sortino")
    
    # Condiciones de finalización
    usar_max_pasos_episodio: bool = Field(..., description="Si limitar pasos por episodio")
    max_pasos_episodio: int = Field(..., description="Máximo pasos por episodio")


class ArchitectureConfig(BaseModel):
    """Configuración de la arquitectura del agente."""
    portfolio_features: int = Field(4, description="Número de características del portfolio")
    transformer_max_seq_len: int = Field(100, description="Longitud máxima de secuencia del transformer")
    positional_encoding_learnable: bool = Field(False, description="Si el encoding posicional es aprendible")


class SACHyperparametersConfig(BaseModel):
    """Hiperparámetros específicos de SAC."""
    gamma: float = Field(0.99, description="Factor de descuento")
    tau: float = Field(0.005, description="Factor de actualización suave")
    actor_learning_rate: float = Field(0.0003, description="Learning rate del actor")
    critic_learning_rate: float = Field(0.0003, description="Learning rate del critic")
    learn_alpha: bool = Field(True, description="Si aprender alpha automáticamente")
    alpha_learning_rate: float = Field(0.0003, description="Learning rate de alpha")
    target_entropy: str = Field("auto", description="Entropía objetivo")
    initial_log_alpha: float = Field(0.0, description="Log alpha inicial")
    log_std_min: int = Field(-20, description="Límite inferior para log_std")
    log_std_max: int = Field(2, description="Límite superior para log_std")


class TransformerConfig(BaseModel):
    """Configuración del Transformer."""
    d_model: int = Field(128, description="Dimensión del modelo")
    n_head: int = Field(4, description="Número de cabezas de atención")
    num_encoder_layers: int = Field(3, description="Número de capas del encoder")
    dim_feedforward: int = Field(256, description="Dimensión del feedforward")
    dropout_rate: float = Field(0.1, description="Tasa de dropout")


class MLPHeadsConfig(BaseModel):
    """Configuración de las cabezas MLP."""
    hidden_dims: List[int] = Field([256, 256], description="Dimensiones ocultas")


class AgentConfig(BaseModel):
    """Configuración del agente SAC."""
    algorithm: str = Field("SAC", description="Algoritmo del agente")
    
    # Arquitectura
    architecture: ArchitectureConfig = Field(..., description="Configuración de la arquitectura")
    
    # Buffer y entrenamiento
    replay_buffer_size: int = Field(..., description="Tamaño del replay buffer")
    batch_size: int = Field(..., description="Tamaño del batch")
    min_buffer_for_learning: int = Field(..., description="Mínimo del buffer para empezar a aprender")
    
    # Hiperparámetros SAC
    hiperparametros_sac: SACHyperparametersConfig = Field(..., description="Hiperparámetros de SAC")
    
    # Transformer
    transformer: TransformerConfig = Field(..., description="Configuración del Transformer")
    
    # Cabezas MLP
    mlp_heads: MLPHeadsConfig = Field(..., description="Configuración de las cabezas MLP")
    
    # Frecuencias
    learning_frequency: int = Field(16, description="Frecuencia de aprendizaje")
    update_target_frequency: int = Field(2, description="Frecuencia de actualización de target")
    
    # Guardado y carga
    model_save_frequency: int = Field(10000, description="Frecuencia de guardado del modelo")
    models_directory: str = Field("models/agent", description="Directorio de modelos")


class LiveTradingConfig(BaseModel):
    """Configuración de trading en vivo."""
    warm_up_candles: int = Field(200, description="Número de velas para warm-up")
    default_model_to_load: str = Field("best_model", description="Modelo por defecto a cargar")
    bigquery_dataset_id: str = Field("trading_logs", description="ID del dataset de BigQuery")
    bigquery_table_id: str = Field("live_trading_log", description="ID de la tabla de BigQuery")


class SystemConfig(BaseModel):
    """Configuración del sistema."""
    max_parallel_workers_api: int = Field(8, description="Número máximo de workers paralelos para API")
    nccl_socket_ifname: str = Field('eth0', description="Interfaz de red para NCCL")


class EvaluationConfig(BaseModel):
    """Configuración de evaluación."""
    temp_directory: str = Field("temp_evaluation", description="Directorio temporal de evaluación")
    default_model_to_load: str = Field("best", description="Modelo por defecto a cargar para evaluación")


class AppConfig(BaseModel):
    """Configuración principal de la aplicación."""
    training_setup: TrainingSetupConfig = Field(..., description="Configuración del setup de entrenamiento")
    api: APIConfig = Field(..., description="Configuración de la API")
    trading: TradingConfig = Field(..., description="Configuración de trading")
    gcp: GCPConfig = Field(..., description="Configuración de Google Cloud Platform")
    data: DataConfig = Field(..., description="Configuración de datos")
    timezone: TimezoneConfig = Field(..., description="Configuración de zona horaria")
    interpolation: InterpolationConfig = Field(..., description="Configuración de interpolación")
    normalization: NormalizationConfig = Field(..., description="Configuración de normalización")
    trend_indicators: TrendIndicatorsConfig = Field(..., description="Configuración de indicadores de tendencia")
    momentum_indicators: MomentumIndicatorsConfig = Field(..., description="Configuración de indicadores de momento")
    volatility_indicators: VolatilityIndicatorsConfig = Field(..., description="Configuración de indicadores de volatilidad")
    volume_indicators: VolumeIndicatorsConfig = Field(..., description="Configuración de indicadores de volumen")
    environment: EnvironmentConfig = Field(..., description="Configuración del entorno")
    agent: AgentConfig = Field(..., description="Configuración del agente")
    live_trading: LiveTradingConfig = Field(..., description="Configuración de trading en vivo")
    system: SystemConfig = Field(..., description="Configuración del sistema")
    evaluation: EvaluationConfig = Field(..., description="Configuración de evaluación")

    @classmethod
    def from_yaml_file(cls, yaml_path: Union[str, Path]) -> "AppConfig":
        """
        Carga la configuración desde un archivo YAML.
        
        Args:
            yaml_path: Ruta al archivo YAML de configuración
            
        Returns:
            AppConfig: Instancia de configuración validada
            
        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si la configuración es inválida
        """
        import yaml
        
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"El archivo de configuración no existe: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            return cls(**raw_config)
        except Exception as e:
            raise ValueError(f"Error al cargar la configuración desde {yaml_path}: {e}")

    @validator('normalization')
    def validate_storage_mode(cls, v):
        """Valida que storage_mode sea un valor válido."""
        if v.storage_mode not in ['local', 'gcp']:
            raise ValueError("storage_mode debe ser 'local' o 'gcp'")
        return v

    @validator('environment')
    def validate_environment(cls, v):
        """Valida parámetros del entorno."""
        if v.capital_inicial <= 0:
            raise ValueError("capital_inicial debe ser positivo")
        if v.apalancamiento <= 0:
            raise ValueError("apalancamiento debe ser positivo")
        if not 0 < v.porcentaje_max_inversion_por_trade <= 1:
            raise ValueError("porcentaje_max_inversion_por_trade debe estar entre 0 y 1")
        if not 0 < v.max_drawdown_configurado_cuenta <= 1:
            raise ValueError("max_drawdown_configurado_cuenta debe estar entre 0 y 1")
        return v

    @validator('agent')
    def validate_agent(cls, v):
        """Valida parámetros del agente."""
        if v.replay_buffer_size <= 0:
            raise ValueError("replay_buffer_size debe ser positivo")
        if v.batch_size <= 0:
            raise ValueError("batch_size debe ser positivo")
        if v.min_buffer_for_learning <= 0:
            raise ValueError("min_buffer_for_learning debe ser positivo")
        return v
