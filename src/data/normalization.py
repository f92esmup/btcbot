"""
Módulo de normalización de datos.
Normaliza todas las características del DataFrame para optimizar el entrenamiento del modelo.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict
import logging

from src.config import (
    AppConfig
)


class Normalization:
    """Clase para normalizar datos de trading usando MinMaxScaler."""

    def __init__(self, dataframe: pd.DataFrame, config: AppConfig, base_path: str,
                 run_id: str):
        """
        Inicializa la clase de normalización.
        
        Args:
            dataframe (pd.DataFrame): DataFrame con datos OHLCV e indicadores técnicos
            config (AppConfig): Configuración de normalización
            base_path (str): Ruta base para guardar los artifacts del entrenamiento
            run_id (str): Identificador único del entrenamiento
        """
        self.dataframe = dataframe.copy()
        self.initial_length = len(self.dataframe)
        self.scaler = None
        self.price_scaler = None  # Nuevo: scaler específico para precios
        self.feature_columns = None
        
        # Store run configuration
        self.base_path = base_path
        self.run_id = run_id
        
        # Store injected configuration
        self.config = config
        
        # Configurar logging
        self.logger = logging.getLogger(__name__)
        
        # Validar que el dataframe no esté vacío
        if self.dataframe.empty:
            raise ValueError("El DataFrame proporcionado está vacío")
        
        # Obtener configuración de normalización desde el diccionario inyectado
        self.scaler_type = self.config.dataset.normalization.scaler_type
        self.feature_range = self.config.dataset.normalization.feature_range

        # Construir rutas dinámicamente
        self.scaler_path = Path(self.base_path) / self.config.base.artifacts.scaler
        self.price_scaler_path = Path(self.base_path) / self.config.base.artifacts.price_scaler

        # Crear directorio para el scaler si no existe (solo en modo local)
        scaler_dir = Path(self.scaler_path).parent
        scaler_dir.mkdir(parents=True, exist_ok=True)
    
        self.logger.info(f"Inicializada clase Normalization con {len(self.dataframe)} filas")
        self.logger.info(f"Tipo de scaler: {self.scaler_type}")
        self.logger.info(f"Rango de características: {self.feature_range}")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Base path: {self.base_path}")
        self.logger.info(f"Ruta del scaler: {self.scaler_path}")
        self.logger.info(f"Ruta del price_scaler: {self.price_scaler_path}")  
    
    def main(self) -> Tuple[pd.DataFrame, MinMaxScaler, MinMaxScaler]:
        """
        Método principal que orquesta todo el proceso de normalización.
        
        Returns:
            Tuple[pd.DataFrame, MinMaxScaler, MinMaxScaler]: DataFrame normalizado, scaler y price_scaler ajustados
        """
        self.logger.info("Iniciando proceso de normalización...")
        
        # Paso 1: Preparar las características para normalización
        self._prepare_features()
        
        # Paso 2: Crear y ajustar el scaler
        self._fit_scalers()

        # Paso 3: Transformar el dataset
        normalized_df = self._transform_datasets()
        
        self.logger.info("Proceso de normalización completado exitosamente")
        self.logger.info(f"DataFrame normalizado: {len(normalized_df)} filas, {len(normalized_df.columns)} columnas")
        
        return normalized_df, self.scaler, self.price_scaler
    
    def _prepare_features(self):
        """Prepara las características que serán normalizadas."""
        # Excluir cualquier columna que no deba ser normalizada
        # Por defecto, todas las columnas numéricas serán normalizadas
        numeric_columns = self.dataframe.select_dtypes(include=[np.number]).columns.tolist()
        
        # Verificar que hay columnas numéricas para normalizar
        if not numeric_columns:
            raise ValueError("No se encontraron columnas numéricas para normalizar")
        
        self.feature_columns = numeric_columns
        
        self.logger.info(f"Características preparadas para normalización: {len(self.feature_columns)} columnas")
        self.logger.debug(f"Columnas a normalizar: {self.feature_columns}")
        
        # Verificar que no hay valores infinitos o NaN
        if np.isinf(self.dataframe[self.feature_columns]).any().any():
            self.logger.warning("Se encontraron valores infinitos en los datos")
            # Reemplazar infinitos con NaN y luego eliminar
            self.dataframe[self.feature_columns] = self.dataframe[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        
        if self.dataframe[self.feature_columns].isna().any().any():
            initial_rows = len(self.dataframe)
            self.dataframe = self.dataframe.dropna()
            final_rows = len(self.dataframe)
            removed_rows = initial_rows - final_rows
            
            if removed_rows > 0:
                self.logger.warning(f"Se eliminaron {removed_rows} filas con valores NaN o infinitos")
            
            if self.dataframe.empty:
                raise ValueError("No quedan datos válidos después de eliminar NaN/infinitos")
    
    def _fit_scalers(self):
        """Crear y ajustar los objetos scaler y price_scaler con los datos de entrenamiento."""
        self.logger.info("Creando y ajustando los scalers...")

        if self.scaler_type == "MinMaxScaler":
            self.scaler = MinMaxScaler(feature_range=self.feature_range)
            self.price_scaler = MinMaxScaler(feature_range=self.feature_range)
        else:
            raise ValueError(f"Tipo de scaler no soportado: {self.scaler_type}")

        feature_data = self.dataframe[self.feature_columns]
        close_data = self.dataframe[["Close"]]

        self.logger.info(f"Ajustando {self.scaler_type} con {feature_data.shape[0]} muestras y {feature_data.shape[1]} características")

        try:
            self.scaler.fit(feature_data)
            self.price_scaler.fit(close_data)
            self.logger.info("Scaler y price_scaler ajustados exitosamente")

            # Log de estadísticas del scaler general
            if hasattr(self.scaler, 'data_min_') and hasattr(self.scaler, 'data_max_'):
                self.logger.info(f"Valores mínimos originales: {self.scaler.data_min_[:5]}...")
                self.logger.info(f"Valores máximos originales: {self.scaler.data_max_[:5]}...")

            # Log de estadísticas del price_scaler
            if hasattr(self.price_scaler, 'data_min_') and hasattr(self.price_scaler, 'data_max_'):
                self.logger.info("Price scaler ajustado exitosamente")
                self.logger.info(f"Rango original del precio {'Close'}: {self.price_scaler.data_min_[0]:.2f} - {self.price_scaler.data_max_[0]:.2f}")

        except Exception as e:
            self.logger.error(f"Error al ajustar los scalers: {str(e)}")
            raise

    def _transform_datasets(self) -> pd.DataFrame:
        """
        Aplicar la transformación de escalado a los conjuntos de datos.
        
        Returns:
            pd.DataFrame: DataFrame con características normalizadas
        """
        self.logger.info("Aplicando transformación de normalización...")
        
        if self.scaler is None:
            raise ValueError("El scaler no ha sido ajustado. Ejecute _fit_scaler() primero.")
        
        try:
            # Obtener los datos de características
            feature_data = self.dataframe[self.feature_columns]
            
            # Aplicar la transformación
            normalized_data = self.scaler.transform(feature_data)
            
            # Crear el DataFrame normalizado manteniendo el índice original
            normalized_df = pd.DataFrame(
                normalized_data,
                columns=self.feature_columns,
                index=self.dataframe.index
            )
            
            # Verificar que la normalización se aplicó correctamente
            self._validate_normalization(normalized_df)
            
            self.logger.info("Transformación aplicada exitosamente")
            return normalized_df
            
        except Exception as e:
            self.logger.error(f"Error al aplicar la transformación: {str(e)}")
            raise
    
    def _validate_normalization(self, normalized_df: pd.DataFrame):
        """
        Validar que la normalización se aplicó correctamente.
        
        Args:
            normalized_df (pd.DataFrame): DataFrame normalizado a validar
        """
        # Verificar que los valores están en el rango esperado
        min_vals = normalized_df.min()
        max_vals = normalized_df.max()
        
        expected_min, expected_max = self.feature_range
        
        # Tolerancia para errores de punto flotante
        tolerance = 1e-6
        
        # Verificar valores mínimos
        if (min_vals < expected_min - tolerance).any():
            problematic_cols = min_vals[min_vals < expected_min - tolerance].index.tolist()
            self.logger.warning(f"Algunas columnas tienen valores por debajo del mínimo esperado: {problematic_cols}")
        
        # Verificar valores máximos
        if (max_vals > expected_max + tolerance).any():
            problematic_cols = max_vals[max_vals > expected_max + tolerance].index.tolist()
            self.logger.warning(f"Algunas columnas tienen valores por encima del máximo esperado: {problematic_cols}")
        
        # Log de estadísticas de normalización
        self.logger.info(f"Rango de valores normalizados: [{min_vals.min():.6f}, {max_vals.max():.6f}]")
        self.logger.info(f"Rango esperado: [{expected_min}, {expected_max}]")
    