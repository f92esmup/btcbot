"""
Modelos de datos Pydantic para la configuración del sistema.

Este módulo define las clases de configuración que reemplazan el acceso
basado en diccionarios por un modelo de datos validado y con tipado estático.
"""
import yaml
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from pathlib import Path

# Configuración para Create_dataset.py
class CreateDatasetConfig(BaseModel):
    """Configuración para el script Create_dataset.py."""
    columns: List[str] = Field(..., min_items = 5, description="Columnas OHLCV a mantener")
    dtypes: Dict[str, str] = Field(..., min_items = 5, description="Tipos de datos para optimizar RAM")
    binanceapi: Dict[str, int] = Field(..., min_items = 4, description="Parámetros de la API de Binance")
    timezone: Dict[str, str] = Field(..., min_items = 2, description="Configuración de zona horaria")
    interpolation: Dict[str, Any] = Field(..., min_items = 2, description="Configuración de interpolación")
    normalization: Dict[str, Any] = Field(..., min_items = 2, description="Configuración de normalización")
    trend_indicators: Dict[str, Any] = Field(..., min_items = 3, description="Indicadores de Tendencia")
    momentum_indicators: Dict[str, Any] = Field(..., min_items = 2, description="Indicadores de Momento")
    volatility_indicators: Dict[str, Any] = Field(..., min_items = 1, description="Indicadores de Volatilidad")
    volume_indicators: Dict[str, Any] = Field(..., min_items = 1, description="Indicadores de Volumen")


class BaseConfig(BaseModel):
    """Configuración base para el sistema de datos."""
    dir: Dict[str, Path] = Field(..., description="Rutas de los directorios del sistema")
    artifacts: Dict[str, Path] = Field(..., min_items=4, description="Rutas de los artefactos del sistema")

class AppConfig(BaseModel):
    """Configuración principal de la aplicación."""
    dataset: CreateDatasetConfig = Field(..., description="Configuración del dataset")
    base: BaseConfig = Field(..., description="Configuración base del sistema de datos")

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

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"El archivo de configuración no existe: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            return cls(**raw_config)
        except Exception as e:
            raise ValueError(f"Error al cargar la configuración desde {yaml_path}: {e}")
