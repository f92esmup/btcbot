"""
Módulo de pipeline de datos.
Unifica todo el preprocesamiento de datos incluyendo adquisición, indicadores y normalización.
"""

import logging
from typing import Tuple, Dict, Optional
import pandas as pd
from pathlib import Path

from .binance_source import BinanceDataSource
from .abstractions import DataSource
from .indicadores import Indicadores
from .normalization import Normalization


class DataPipeline:
    """Clase que unifica todo el preprocesamiento de datos."""
    
    def __init__(self, data_source: DataSource, symbol: str, interval: str, start_date: str, run_id: str, base_path: str,
                 full_config: Dict, end_date: str = None, save_artifacts: bool = True, 
                 gcs_utils=None):
        """
        Inicializa el pipeline de datos.
        
        Args:
            data_source (DataSource): Instancia de fuente de datos que implementa la interfaz DataSource
            symbol (str): Símbolo del par de trading (ej: 'BTCUSDT')
            interval (str): Intervalo de tiempo para las velas (ej: '1h', '4h')
            start_date (str): Fecha de inicio en formato YYYY-MM-DD
            run_id (str): Identificador único del entrenamiento
            base_path (str): Ruta base para guardar los artifacts del entrenamiento
            full_config (Dict): Configuración completa para todas las clases del pipeline
            end_date (str, optional): Fecha de fin en formato YYYY-MM-DD
            save_artifacts (bool): Si True, guarda artefactos como scalers
            gcs_utils: Instancia de GCSUtils para operaciones en la nube (opcional)
        """
        # Inyección de dependencia: fuente de datos
        self.data_source = data_source
        
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.run_id = run_id
        self.base_path = base_path
        self.save_artifacts = save_artifacts
        
        # Store injected configuration and dependencies
        self.full_config = full_config
        self.gcs_utils = gcs_utils
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Pipeline de datos inicializado para {symbol} ({interval})")
        self.logger.info(f"Período: desde {start_date}" + (f" hasta {end_date}" if end_date else " hasta ahora"))
        self.logger.info(f"Run ID: {run_id}")
        self.logger.info(f"Base path: {base_path}")
        self.logger.info(f"Guardar artefactos: {save_artifacts}")
        self.logger.info(f"Configuración inyectada: {len(self.full_config)} secciones")
    
    def run(self) -> Tuple[pd.DataFrame, object, object]:
        """
        Ejecuta el pipeline completo de preprocesamiento de datos.
        
        Returns:
            Tuple[pd.DataFrame, object, object]: DataFrame normalizado, scaler y price_scaler utilizados.
        """
        self.logger.info("=== Iniciando Pipeline de Datos ===")
        
        # Paso 1: Adquisición de datos
        self.logger.info("PASO 1: Adquisición de Datos")
        dataframe = self.data_source.fetch_data()

        self.logger.info(f"Datos adquiridos exitosamente:")
        self.logger.info(f"  - Forma del DataFrame: {dataframe.shape}")
        self.logger.info(f"  - Rango temporal: {dataframe.index.min()} a {dataframe.index.max()}")
        self.logger.info(f"  - Columnas: {list(dataframe.columns)}")
        self.logger.info(f"  - Memoria utilizada: {dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Paso 2: Cálculo de indicadores técnicos
        self.logger.info("PASO 2: Cálculo de Indicadores Técnicos")
        indicadores = Indicadores(
            dataframe, 
            config_dict=self.full_config
        )
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
            run_id=self.run_id,
            save_artifacts=False,  # ArtifactManager se encargará del guardado
            normalization_config=self.full_config.get('normalization', {}),
            gcs_utils=self.gcs_utils
        )
        normalized_dataframe, scaler, price_scaler = normalization.main()
        
        self.logger.info(f"Normalización completada exitosamente:")
        self.logger.info(f"  - Forma del DataFrame normalizado: {normalized_dataframe.shape}")
        self.logger.info(f"  - Rango de valores: [{normalized_dataframe.min().min():.6f}, {normalized_dataframe.max().max():.6f}]")
        
        # Los artefactos serán guardados por el ArtifactManager en el script principal
        self.logger.info("  - Artefactos disponibles para guardado por ArtifactManager")
            
        self.logger.info(f"  - Memoria utilizada: {normalized_dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Mostrar información del scaler
        feature_info = normalization.get_feature_info()
        self.logger.info(f"  - Características normalizadas: {feature_info['num_features']}")
        self.logger.info(f"  - Tipo de scaler: {feature_info['scaler_type']}")
        self.logger.info(f"  - Rango de normalización: {feature_info['feature_range']}")
        
        self.logger.info("=== Pipeline de Datos Completado ===")
        
        return normalized_dataframe, scaler, price_scaler
