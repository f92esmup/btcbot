"""
Módulo de pipeline de datos.
Unifica todo el preprocesamiento de datos incluyendo adquisición, indicadores y normalización.
"""

import logging
from typing import Tuple
import pandas as pd
from pathlib import Path

from .Adquisicion import Adquisicion
from .indicadores import Indicadores
from .normalization import Normalization


class DataPipeline:
    """Clase que unifica todo el preprocesamiento de datos."""
    
    def __init__(self, symbol: str, interval: str, start_date: str, run_id: str, base_path: str):
        """
        Inicializa el pipeline de datos.
        
        Args:
            symbol (str): Símbolo del par de trading (ej: 'BTCUSDT')
            interval (str): Intervalo de tiempo para las velas (ej: '1h', '4h')
            start_date (str): Fecha de inicio en formato YYYY-MM-DD
            run_id (str): Identificador único del entrenamiento
            base_path (str): Ruta base para guardar los artifacts del entrenamiento
        """
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.run_id = run_id
        self.base_path = base_path
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Pipeline de datos inicializado para {symbol} ({interval})")
        self.logger.info(f"Período: desde {start_date}")
        self.logger.info(f"Run ID: {run_id}")
        self.logger.info(f"Base path: {base_path}")
    
    def run(self) -> Tuple[pd.DataFrame, str]:
        """
        Ejecuta el pipeline completo de preprocesamiento de datos.
        
        Returns:
            Tuple[pd.DataFrame, str]: DataFrame normalizado y ruta del price_scaler
        """
        self.logger.info("=== Iniciando Pipeline de Datos ===")
        
        # Paso 1: Adquisición de datos
        self.logger.info("PASO 1: Adquisición de Datos")
        adquisicion = Adquisicion(
            symbol=self.symbol,
            interval=self.interval,
            start_date=self.start_date
        )
        dataframe = adquisicion.main()
        
        self.logger.info(f"Datos adquiridos exitosamente:")
        self.logger.info(f"  - Forma del DataFrame: {dataframe.shape}")
        self.logger.info(f"  - Rango temporal: {dataframe.index.min()} a {dataframe.index.max()}")
        self.logger.info(f"  - Columnas: {list(dataframe.columns)}")
        self.logger.info(f"  - Memoria utilizada: {dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Paso 2: Cálculo de indicadores técnicos
        self.logger.info("PASO 2: Cálculo de Indicadores Técnicos")
        indicadores = Indicadores(dataframe)
        dataframe_with_indicators = indicadores.main()
        
        self.logger.info(f"Indicadores calculados exitosamente:")
        self.logger.info(f"  - Forma del DataFrame: {dataframe_with_indicators.shape}")
        self.logger.info(f"  - Columnas totales: {len(dataframe_with_indicators.columns)}")
        self.logger.info(f"  - Nuevas columnas de indicadores: {len(dataframe_with_indicators.columns) - len(dataframe.columns)}")
        self.logger.info(f"  - Memoria utilizada: {dataframe_with_indicators.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Mostrar las nuevas columnas
        original_columns = set(dataframe.columns)
        new_columns = [col for col in dataframe_with_indicators.columns if col not in original_columns]
        if new_columns:
            self.logger.info(f"  - Indicadores añadidos: {new_columns}")
        
        # Paso 3: Normalización de datos
        self.logger.info("PASO 3: Normalización de Datos")
        normalization = Normalization(
            dataframe_with_indicators, 
            base_path=self.base_path, 
            run_id=self.run_id
        )
        normalized_dataframe, scaler = normalization.main()
        
        self.logger.info(f"Normalización completada exitosamente:")
        self.logger.info(f"  - Forma del DataFrame normalizado: {normalized_dataframe.shape}")
        self.logger.info(f"  - Rango de valores: [{normalized_dataframe.min().min():.6f}, {normalized_dataframe.max().max():.6f}]")
        self.logger.info(f"  - Scaler guardado en: {normalization.scaler_path}")
        self.logger.info(f"  - Price scaler guardado en: {normalization.price_scaler_path}")
        self.logger.info(f"  - Memoria utilizada: {normalized_dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Mostrar información del scaler
        feature_info = normalization.get_feature_info()
        self.logger.info(f"  - Características normalizadas: {feature_info['num_features']}")
        self.logger.info(f"  - Tipo de scaler: {feature_info['scaler_type']}")
        self.logger.info(f"  - Rango de normalización: {feature_info['feature_range']}")
        
        self.logger.info("=== Pipeline de Datos Completado ===")
        
        return normalized_dataframe, normalization.price_scaler_path
