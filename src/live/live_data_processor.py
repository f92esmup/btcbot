import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple

from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger
from src.data.feature_engineering import compute_technical_indicators, compute_log_returns

logger = setup_logger('LiveDataProcessor')

class LiveFeatureProcessor:
    """
    Procesa datos de mercado en vivo para convertirlos en características utilizables por el modelo.
    Replica la lógica de preprocesamiento utilizada durante el entrenamiento.
    """
    def __init__(self, config_manager: ConfigManager):
        """
        Inicializa el procesador de datos en vivo.
        
        Args:
            config_manager: Instancia de ConfigManager para obtener la configuración
        """
        self.config_manager = config_manager
        
        # Obtener configuración de preprocesamiento
        self.preproc_config = config_manager.get_preprocessing_config()
        
        # Longitud de secuencia L
        self.sequence_length_L = self.preproc_config.get('sequence_length_L', 96)
        
        # Multiplicador para la ventana de normalización
        self.norm_window_mult = self.preproc_config.get('normalization_window_multiplier_for_L', 2)
        
        # Parámetros para indicadores técnicos
        self.ohlcv_processing = self.preproc_config.get('ohlcv_processing', {})
        self.indicators_config = self.preproc_config.get('indicators', {})
        
        # Columnas finales de características
        self.final_feature_columns = self.preproc_config.get('final_market_feature_columns', [])
        
        if not self.final_feature_columns:
            logger.warning("No se encontraron columnas de características definidas en la configuración")
        
        logger.info(f"Procesador de datos en vivo inicializado (L={self.sequence_length_L}, norm_window={self.sequence_length_L * self.norm_window_mult})")
    
    def _apply_live_normalization(self, df_with_features: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica normalización a las características de mercado en un contexto en vivo.
        Utiliza solo datos históricos (lookback) para la normalización.
        
        Args:
            df_with_features: DataFrame con características sin normalizar
            
        Returns:
            DataFrame con características normalizadas
        """
        # Obtener el tamaño de la ventana para normalización
        norm_window_size = self.sequence_length_L * self.norm_window_mult
        
        # Si no tenemos suficientes datos, usar todo lo que tenemos
        if len(df_with_features) < norm_window_size:
            logger.warning(f"Solo se tienen {len(df_with_features)} puntos para normalización (menor que {norm_window_size})")
            norm_window_size = len(df_with_features)
        
        # Tomar la ventana más reciente para la normalización
        norm_window = df_with_features.iloc[-norm_window_size:]
        
        # Crear una copia para almacenar los resultados normalizados
        df_normalized = df_with_features.copy()
        
        # Lista de columnas que NO se deben normalizar con z-score
        non_zscore_cols = []
        
        # Lista de columnas que ya están normalizadas o escaladas
        already_scaled_cols = ['RSI_scaled']
        
        # Para cada columna en el DataFrame, aplicar normalización adecuada
        for col in df_with_features.columns:
            # Si la columna ya está normalizada o es un retorno, no hacer nada
            if col in already_scaled_cols or col in non_zscore_cols:
                continue
            
            # Para retornos logarítmicos y características con "norm" en el nombre
            if 'log_ret' in col or 'norm' in col:
                mean = norm_window[col].mean()
                std = norm_window[col].std()
                
                # Evitar división por cero
                if std > 0:
                    df_normalized[col] = (df_with_features[col] - mean) / std
                else:
                    logger.warning(f"Desviación estándar cero para {col}. No se aplica normalización z-score.")
                    df_normalized[col] = df_with_features[col] - mean  # Solo centrar
        
        return df_normalized
    
    def process_market_data(self, raw_candles_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Procesa los datos de mercado crudos y los convierte en la matriz de características
        requerida por el modelo.
        
        Args:
            raw_candles_df: DataFrame con velas OHLCV
            
        Returns:
            Array numpy de características de mercado o None en caso de error
        """
        try:
            # Verificar que tenemos suficientes datos
            if len(raw_candles_df) < self.sequence_length_L:
                logger.error(f"Datos insuficientes para procesamiento. Se requieren al menos {self.sequence_length_L} velas.")
                return None
            
            # 1. Asegurarnos de que tenemos las columnas correctas
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in raw_candles_df.columns:
                    logger.error(f"Columna requerida '{col}' no encontrada en los datos")
                    return None
            
            # 2. Computar retornos logarítmicos
            df_with_returns = compute_log_returns(raw_candles_df)
            
            # 3. Computar indicadores técnicos
            df_with_features = compute_technical_indicators(
                df_with_returns, 
                indicators_config=self.indicators_config,
                ohlcv_processing=self.ohlcv_processing
            )
            
            # 4. Manejar valores NaN
            # Eliminar filas con NaN (típicamente las primeras filas por indicadores técnicos)
            df_with_features = df_with_features.dropna()
            
            # Verificar si hemos perdido demasiadas filas
            if len(df_with_features) < self.sequence_length_L:
                logger.error(f"Datos insuficientes después de eliminar NaN. Quedan {len(df_with_features)} filas.")
                return None
            
            # 5. Aplicar normalización
            df_normalized = self._apply_live_normalization(df_with_features)
            
            # 6. Seleccionar solo las columnas finales y las últimas L filas
            if not all(col in df_normalized.columns for col in self.final_feature_columns):
                missing_cols = [col for col in self.final_feature_columns if col not in df_normalized.columns]
                logger.error(f"Columnas requeridas no encontradas: {missing_cols}")
                return None
            
            final_df = df_normalized[self.final_feature_columns].iloc[-self.sequence_length_L:]
            
            # 7. Convertir a numpy array
            market_features_array = final_df.values
            
            # 8. Verificar la forma final
            if market_features_array.shape != (self.sequence_length_L, len(self.final_feature_columns)):
                logger.error(f"Forma incorrecta de características de mercado: {market_features_array.shape}, esperada: ({self.sequence_length_L}, {len(self.final_feature_columns)})")
                return None
            
            logger.info(f"Procesamiento de datos de mercado exitoso. Forma final: {market_features_array.shape}")
            return market_features_array
            
        except Exception as e:
            logger.error(f"Error en procesamiento de datos de mercado: {e}", exc_info=True)
            return None
