"""
Modelos de datos Pydantic para la configuración del sistema.

Este módulo define las clases de configuración que reemplazan el acceso
basado en diccionarios por un modelo de datos validado y con tipado estático.
"""
import yaml
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional, Union, Any, Tuple
from pydantic import BaseModel, Field, SecretStr, validator, root_validator, field_validator
from pathlib import Path

# Configuración para Create_dataset.py

class Dtypes(BaseModel):
    """Tipos de datos para optimizar el uso de RAM."""
    Open: str = Field(..., description="Tipo de dato para la columna 'open'")
    High: str = Field(..., description="Tipo de dato para la columna 'high'")
    Low: str = Field(..., description="Tipo de dato para la columna 'low'")
    Close: str = Field(..., description="Tipo de dato para la columna 'close'")
    Volume: str = Field(..., description="Tipo de dato para la columna 'volume'")


class BinaceAPI(BaseModel):
    call_limit: int = Field(..., description="Límite de llamadas a la API de Binance")
    max_retries: int = Field(..., description="Número máximo de reintentos en caso de error")
    retry_delay: int = Field(..., description="Tiempo de espera entre reintentos en segundos")
    timeout: int = Field(..., description="Tiempo máximo de espera para una llamada a la API en segundos")

class Timezone(BaseModel):
    """Configuración de zona horaria."""
    target: str = Field(..., description="Zona horaria objetivo para los datos")
    source: str = Field(..., description="Zona horaria de origen para los datos")

class Interpolation(BaseModel):
    """Configuración de interpolación."""
    method: str = Field(..., description="Método de interpolación a utilizar")
    limit_direction: str = Field(..., description="Dirección del límite de interpolación")

class Normalization(BaseModel):
    """Configuración de normalización."""
    scaler_type: str = Field(..., description="Tipo de escalador a utilizar para normalización")
    feature_range: Tuple[int, int] = Field(..., description="Rango de características para normalización")

class trend(BaseModel):
    """Configuración de la tendencia."""
    period: int = Field(..., description="Período para el cálculo de la tendencia")
    enabled: bool = Field(..., description="Habilitar cálculo de tendencia")

class TrendIndicators(BaseModel):
    """Configuración de indicadores de tendencia."""
    ema_20: trend = Field(..., description="Configuración del EMA de 20 períodos")
    ema_50: trend = Field(..., description="Configuración del EMA de 50 períodos")
    adx: trend = Field(..., description="Configuración del ADX")

class stochastic(BaseModel):
    """Configuración del Stochastic."""
    k_period: int = Field(..., description="Período K para el cálculo del Stochastic")
    d_period: int = Field(..., description="Período D para el cálculo del Stochastic")
    smooth_k: int = Field(..., description="Suavizado K para el Stochastic")
    enabled: bool = Field(..., description="Habilitar cálculo del Stochastic")

class MomentumIndicators(BaseModel):
    """Configuración de indicadores de momento."""
    rsi: trend = Field(..., description="Configuración del RSI")
    stoch: stochastic = Field(..., description="Configuración del Stochastic")

class VolatilityIndicators(BaseModel):
    """Configuración de indicadores de volatilidad."""
    atr: trend = Field(..., description="Configuración del ATR")

class Obv(BaseModel):
    """Configuración del On-Balance Volume (OBV)."""
    enabled: bool = Field(..., description="Habilitar cálculo del OBV")

class VolumeIndicators(BaseModel):
    """Configuración de indicadores de volumen."""
    obv: Obv = Field(..., description="Configuración del OBV")

class CreateDatasetConfig(BaseModel):
    """Configuración para el script Create_dataset.py."""
    columns: List[str] = Field(..., min_items = 5, description="Columnas OHLCV a mantener del dataframe dado por el cliente de Binance")
    dtypes: Dtypes = Field(..., description="Tipos de datos para optimizar RAM")
    binanceapi: BinaceAPI = Field(..., description="Parámetros de la API de Binance")
    timezone: Timezone = Field(..., description="Configuración de zona horaria")
    interpolation: Interpolation = Field(..., min_items = 2, description="Configuración de interpolación")
    normalization: Normalization = Field(..., min_items = 2, description="Configuración de normalización")
    trend_indicators: TrendIndicators = Field(..., min_items = 3, description="Indicadores de Tendencia")
    momentum_indicators: MomentumIndicators = Field(..., min_items = 2, description="Indicadores de Momento")
    volatility_indicators: VolatilityIndicators = Field(..., min_items = 1, description="Indicadores de Volatilidad")
    volume_indicators: VolumeIndicators = Field(..., min_items = 1, description="Indicadores de Volumen")


class Directories(BaseModel):
    data_runs: Path = Field(..., description="Directorio para almacenar los datos de las ejecuciones")

class Artifacts(BaseModel):
    normalized_dataframe: Path = Field(..., description="Ruta del archivo de datos normalizados")
    scaler: Path = Field(..., description="Ruta del archivo del escalador utilizado para normalización")
    price_scaler: Path = Field(..., description="Ruta del archivo del escalador de precios")
    dataset_metadata: Path = Field(..., description="Ruta del archivo de metadatos del dataset")
class BaseConfig(BaseModel):
    """Configuración base para el sistema de datos."""
    dir: Directories = Field(..., description="Rutas de los directorios del sistema")
    artifacts: Artifacts = Field(..., description="Rutas de los artefactos del sistema")

class Credenciales(BaseModel):
    api_key: SecretStr = Field(None, description="Clave API para acceso a servicios externos")
    api_secret: SecretStr = Field(None, description="Secreto de la API para acceso a servicios externos")
    testnet: bool = Field(True, description="Indica si se usa el entorno de prueba (testnet)")

    @field_validator('api_key', 'api_secret', mode='before')
    def not_empty(cls, v, info):
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError(f"La credencial '{info.field_name}' no puede ser None ni vacía.")
        return v
class AppConfig(BaseModel):
    """Configuración principal de la aplicación."""
    dataset: CreateDatasetConfig = Field(..., description="Configuración del dataset")
    base: BaseConfig = Field(..., description="Configuración base del sistema de datos")
    credenciales: Credenciales = Field(..., description="Credenciales para servicios externos")

    @classmethod
    def from_yaml_file(cls, yaml_path: Union[str, Path], testnet: bool = True) -> "AppConfig":
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

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"El archivo de configuración no existe: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            # Cargar variables de entorno desde .env
            load_dotenv()

            if testnet:
                api_key = os.getenv("TESTNET_BINANCE_API_KEY")
                api_secret = os.getenv("TESTNET_BINANCE_API_SECRET")
            else:
                api_key = os.getenv("BINANCE_API_KEY")
                api_secret = os.getenv("BINANCE_API_SECRET")

            raw_config["credenciales"] = {
                "api_key": api_key,
                "api_secret": api_secret,
                "testnet": testnet
            }

            return cls(**raw_config)
        except Exception as e:
            raise ValueError(f"Error al cargar la configuración desde {yaml_path}: {e}")
