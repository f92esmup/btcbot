"""
Módulo de normalización de datos.
Normaliza todas las características del DataFrame para optimizar el entrenamiento del modelo.
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Tuple, Dict, Any
import logging

from src.configuration.constants import (
    FILE_SCALER, FILE_PRICE_SCALER, COLUMN_OPEN, COLUMN_HIGH, 
    COLUMN_LOW, COLUMN_CLOSE, COLUMN_VOLUME, COLUMNS_OHLCV
)


class Normalization:
    """Clase para normalizar datos de trading usando MinMaxScaler."""
    
    def __init__(self, dataframe: pd.DataFrame, normalization_config: Dict, base_path: str, 
                 run_id: str, save_artifacts: bool = True, gcs_utils=None):
        """
        Inicializa la clase de normalización.
        
        Args:
            dataframe (pd.DataFrame): DataFrame con datos OHLCV e indicadores técnicos
            normalization_config (Dict): Diccionario con la configuración de normalización
            base_path (str): Ruta base para guardar los artifacts del entrenamiento
            run_id (str): Identificador único del entrenamiento
            save_artifacts (bool): Si True, guarda los scalers; si False, solo los carga
            gcs_utils: Instancia de GCSUtils para operaciones en la nube (opcional)
        """
        self.dataframe = dataframe.copy()
        self.initial_length = len(self.dataframe)
        self.scaler = None
        self.price_scaler = None  # Nuevo: scaler específico para precios
        self.feature_columns = None
        
        # Store run configuration
        self.base_path = base_path
        self.run_id = run_id
        self.save_artifacts = save_artifacts
        
        # Store injected configuration and GCS utils
        self.config = normalization_config
        self.gcs_utils = gcs_utils
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Validar que el dataframe no esté vacío
        if self.dataframe.empty:
            raise ValueError("El DataFrame proporcionado está vacío")
        
        # Obtener configuración de normalización desde el diccionario inyectado
        self.scaler_type = self.config.get('scaler_type', 'MinMaxScaler')
        self.feature_range = tuple(self.config.get('feature_range', [0, 1]))
        self.storage_mode = self.config.get('storage_mode', 'local')

        # Construir rutas dinámicamente
        if self.storage_mode == "gcp":
            self.scaler_path = f"{self.base_path}/{FILE_SCALER}"
            self.price_scaler_path = f"{self.base_path}/{FILE_PRICE_SCALER}"
        else:
            self.scaler_path = str(Path(self.base_path) / FILE_SCALER)
            self.price_scaler_path = str(Path(self.base_path) / FILE_PRICE_SCALER)
        
        # Crear directorio para el scaler si no existe (solo en modo local)
        if self.storage_mode == "local":
            scaler_dir = Path(self.scaler_path).parent
            scaler_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Inicializada clase Normalization con {len(self.dataframe)} filas")
        self.logger.info(f"Tipo de scaler: {self.scaler_type}")
        self.logger.info(f"Rango de características: {self.feature_range}")
        self.logger.info(f"Modo de almacenamiento: {self.storage_mode}")
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
        self._fit_scaler()
        
        # Paso 3: Crear y ajustar el price_scaler específico
        self._fit_price_scaler()
        
        # Paso 4: Transformar el dataset
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
    
    def _fit_scaler(self):
        """Crear y ajustar el objeto scaler con los datos de entrenamiento."""
        self.logger.info("Creando y ajustando el scaler...")
        
        # Crear el scaler según la configuración
        if self.scaler_type == "MinMaxScaler":
            self.scaler = MinMaxScaler(feature_range=self.feature_range)
        else:
            raise ValueError(f"Tipo de scaler no soportado: {self.scaler_type}")
        
        # Ajustar el scaler con los datos
        feature_data = self.dataframe[self.feature_columns]
        
        self.logger.info(f"Ajustando {self.scaler_type} con {feature_data.shape[0]} muestras y {feature_data.shape[1]} características")
        
        try:
            self.scaler.fit(feature_data)
            self.logger.info("Scaler ajustado exitosamente")
            
            # Log de estadísticas del scaler
            if hasattr(self.scaler, 'data_min_') and hasattr(self.scaler, 'data_max_'):
                self.logger.info(f"Valores mínimos originales: {self.scaler.data_min_[:5]}...")  # Solo mostrar primeros 5
                self.logger.info(f"Valores máximos originales: {self.scaler.data_max_[:5]}...")  # Solo mostrar primeros 5
                
        except Exception as e:
            self.logger.error(f"Error al ajustar el scaler: {str(e)}")
            raise
    
    def _fit_price_scaler(self):
        """Crear y ajustar el price_scaler específico para la columna Close."""
        self.logger.info(f"Creando y ajustando el price_scaler para la columna {COLUMN_CLOSE}...")
        
        # Verificar que la columna Close existe
        if COLUMN_CLOSE not in self.feature_columns:
            raise ValueError(f"La columna '{COLUMN_CLOSE}' no se encuentra en las características a normalizar")
        
        # Crear el price_scaler
        if self.scaler_type == "MinMaxScaler":
            self.price_scaler = MinMaxScaler(feature_range=self.feature_range)
        else:
            raise ValueError(f"Tipo de scaler no soportado: {self.scaler_type}")
        
        try:
            # Obtener solo los valores de la columna Close
            close_data = self.dataframe[[COLUMN_CLOSE]]
            
            # Ajustar el price_scaler solo con los datos de Close
            self.price_scaler.fit(close_data)
            
            self.logger.info("Price scaler ajustado exitosamente")
            self.logger.info(f"Rango original del precio {COLUMN_CLOSE}: {self.price_scaler.data_min_[0]:.2f} - {self.price_scaler.data_max_[0]:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error al ajustar el price_scaler: {str(e)}")
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
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Obtener información sobre las características y el proceso de normalización.
        
        Returns:
            Dict[str, Any]: Información sobre características y normalización
        """
        info = {
            'num_features': len(self.feature_columns) if self.feature_columns else 0,
            'feature_columns': self.feature_columns,
            'scaler_type': self.scaler_type,
            'feature_range': self.feature_range,
            'scaler_path': self.scaler_path,
            'price_scaler_path': self.price_scaler_path,
            'original_data_shape': self.dataframe.shape,
            'scaler_fitted': self.scaler is not None,
            'price_scaler_fitted': self.price_scaler is not None
        }
        
        if self.scaler is not None and hasattr(self.scaler, 'data_min_'):
            info.update({
                'original_min_values': self.scaler.data_min_.tolist(),
                'original_max_values': self.scaler.data_max_.tolist(),
                'scale_factors': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
            })
        
        if self.price_scaler is not None and hasattr(self.price_scaler, 'data_min_'):
            info.update({
                'price_scaler_min': self.price_scaler.data_min_[0],
                'price_scaler_max': self.price_scaler.data_max_[0],
                'price_scaler_range': self.price_scaler.data_max_[0] - self.price_scaler.data_min_[0]
            })
        
        return info
    
    def scaler_exists(self) -> bool:
        """
        Verifica si existe un scaler previamente entrenado.
        
        Returns:
            bool: True si el scaler existe, False en caso contrario
        """
        if self.storage_mode == "gcp":
            return self.gcs_utils.scaler_exists_in_gcs() if self.gcs_utils else False
        else:
            return os.path.exists(self.scaler_path)
    
    def price_scaler_exists(self) -> bool:
        """
        Verifica si existe un price_scaler previamente entrenado.
        
        Returns:
            bool: True si el price_scaler existe, False en caso contrario
        """
        if self.storage_mode == "gcp":
            return self.gcs_utils.price_scaler_exists_in_gcs() if self.gcs_utils else False
        else:
            return os.path.exists(self.price_scaler_path)
    
    def get_scaler_storage_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el almacenamiento del scaler y price_scaler.
        
        Returns:
            Dict[str, Any]: Información sobre los scalers almacenados
        """
        info = {
            'storage_mode': self.storage_mode,
            'scaler_exists': self.scaler_exists(),
            'price_scaler_exists': self.price_scaler_exists()
        }
        
        if self.storage_mode == "gcp":
            if self.gcs_utils:
                gcs_info = self.gcs_utils.get_scaler_info()
                if gcs_info:
                    info.update({'scaler_info': gcs_info})
                
                # Intentar obtener información del price_scaler también
                price_scaler_info = self.gcs_utils.get_price_scaler_info()
                if price_scaler_info:
                    info.update({'price_scaler_info': price_scaler_info})
        else:
            if os.path.exists(self.scaler_path):
                stat = os.stat(self.scaler_path)
                info.update({
                    'scaler_path': self.scaler_path,
                    'scaler_size': stat.st_size,
                    'scaler_modified': stat.st_mtime
                })
            
            if os.path.exists(self.price_scaler_path):
                stat = os.stat(self.price_scaler_path)
                info.update({
                    'price_scaler_path': self.price_scaler_path,
                    'price_scaler_size': stat.st_size,
                    'price_scaler_modified': stat.st_mtime
                })
        
        return info