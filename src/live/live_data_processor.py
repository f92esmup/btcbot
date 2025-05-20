# src/live/live_data_processor.py
import pandas as pd
import numpy as np
import logging
from src.utils.config import ConfigManager
from src.data.feature_engineering import FeatureEngineer # Reutilizar

from src.utils.logging_utils import setup_logger
logger = setup_logger("LiveFeatureProcessor")

class LiveFeatureProcessor:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        
        # Cargar configuraciones relevantes
        preprocessing_config = self.config_manager.get_preprocessing_config()
        self.indicators_config = preprocessing_config.get('indicators', {})
        self.ohlcv_config = preprocessing_config.get('ohlcv_processing', {})
        self.sequence_length_L = preprocessing_config.get('sequence_length_L', 96)
        self.norm_window_multiplier = preprocessing_config.get('normalization_window_multiplier_for_L', 2)
        
        # Calcular tamaño de ventana de normalización
        self.normalization_window_size = self.sequence_length_L * self.norm_window_multiplier
        
        # Reutilizar FeatureEngineer del módulo de datos
        self.feature_engineer = FeatureEngineer(self.indicators_config, self.ohlcv_config)
        
        # Lista de características finales que se usarán
        self.final_feature_columns = preprocessing_config.get('final_market_feature_columns', [])
        if not self.final_feature_columns:
            logger.warning("No se encontraron columnas finales de características definidas en config.yaml!")

        logger.info(f"LiveFeatureProcessor inicializado. Secuencia L={self.sequence_length_L}, "
                   f"Ventana de normalización={self.normalization_window_size}")

    def process_klines_data(self, klines_df: pd.DataFrame) -> pd.DataFrame:
        """Procesa datos OHLCV para generar características técnicas completas.
        
        Args:
            klines_df: DataFrame con columnas Open, High, Low, Close, Volume
            
        Returns:
            DataFrame con todas las características calculadas y normalizadas
        """
        if klines_df is None or klines_df.empty:
            logger.error("DataFrame de klines vacío o nulo recibido para procesamiento.")
            return pd.DataFrame()
            
        logger.debug(f"Procesando {len(klines_df)} klines para generar características técnicas.")
        
        # 1. Añadir características básicas de OHLCV
        df = self.feature_engineer.add_ohlcv_features(klines_df)
        
        # 2. Calcular indicadores técnicos
        df = self.feature_engineer.add_technical_indicators(df)
        
        # 3. Normalización Z-score usando una ventana móvil
        df = self.apply_z_score_normalization(df)
        
        # 4. Filtrar y ordenar las columnas según la configuración
        if self.final_feature_columns:
            missing_columns = [col for col in self.final_feature_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Columnas finales no encontradas en el DataFrame procesado: {missing_columns}")
                
            # Usar solo las columnas que existen en el DataFrame
            valid_columns = [col for col in self.final_feature_columns if col in df.columns]
            df_final = df[valid_columns].copy()
            logger.debug(f"DataFrame reducido a {len(valid_columns)} columnas según configuración.")
        else:
            df_final = df
            logger.debug("Usando todas las columnas procesadas (ninguna especificada en config).")
            
        return df_final

    def apply_z_score_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica normalización z-score basada en ventana móvil a las características.
        
        Args:
            df: DataFrame con las características calculadas
            
        Returns:
            DataFrame con las características normalizadas añadidas (sufijo '_norm')
        """
        logger.debug(f"Aplicando normalización z-score con ventana={self.normalization_window_size}")
        
        # Crear copia del DataFrame para mantener originales
        df_out = df.copy()
        
        # Lista de columnas que queremos normalizar
        # Cualquier columna cuyo nombre termine en '_norm' (excepto las que ya tienen sufijo '_norm')
        cols_to_normalize = []
        for col in df.columns:
            # Para rasgos OHLCV básicos
            if col.startswith('log_ret_') and not col.endswith('_norm'):
                cols_to_normalize.append(col)
            # Para indicadores técnicos (algunos pueden requerir normalización especial)
            elif col in ['SMA_short', 'SMA_long', 'EMA_short', 'EMA_long', 'MACD_line', 'MACD_signal', 'MACD_hist']:
                cols_to_normalize.append(col)
            elif col == 'ATR': # ATR dividido por precio de cierre
                df_out['ATR_norm'] = df_out['ATR'] / df_out['Close']
            elif col == 'RSI': # RSI ya está entre 0-100, escalarlo
                if self.indicators_config.get('rsi_scaling_mode') == '0_1':
                    df_out['RSI_scaled'] = df_out['RSI'] / 100.0
                else: # -1_1 mode
                    df_out['RSI_scaled'] = (df_out['RSI'] / 50.0) - 1.0
        
        # Aplicar z-score usando ventana móvil para los rasgos seleccionados
        for col in cols_to_normalize:
            # Calcular media y desviación estándar móviles
            rolling_mean = df_out[col].rolling(window=self.normalization_window_size).mean()
            rolling_std = df_out[col].rolling(window=self.normalization_window_size).std()
            
            # Manejar desviaciones estándar de cero o NaN
            rolling_std = rolling_std.replace(0, np.nan)  # Convertir ceros a NaN 
            rolling_std = rolling_std.fillna(0.00001)  # Reemplazar NaNs con un valor pequeño
            
            # Calcular z-scores
            z_scores = (df_out[col] - rolling_mean) / rolling_std
            
            # Recortar valores extremos (opcional, pero útil para evitar outliers extremos)
            z_scores = np.clip(z_scores, -4, 4)
            
            # Agregar la nueva columna normalizada
            new_col_name = f"{col}_norm" if not col.endswith('_norm') else col
            df_out[new_col_name] = z_scores
            
        # Para Bandas de Bollinger, calculamos distancia normalizada del precio a las bandas
        if all(band in df_out.columns for band in ['BB_upper', 'BB_lower']):
            if 'ATR' in df_out.columns:
                # Normalizar distancia a bandas usando ATR
                df_out['BB_dist_upper_norm'] = (df_out['BB_upper'] - df_out['Close']) / df_out['ATR']
                df_out['BB_dist_lower_norm'] = (df_out['Close'] - df_out['BB_lower']) / df_out['ATR']
            else:
                logger.warning("ATR no encontrado para normalizar distancias de Bandas de Bollinger.")
        
        return df_out
        
    def get_latest_feature_sequence(self, df_features: pd.DataFrame) -> np.ndarray:
        """Extrae la secuencia más reciente de características para el modelo.
        
        Args:
            df_features: DataFrame con todas las características procesadas
            
        Returns:
            ndarray con la secuencia de características de longitud L, o None si no hay suficientes datos
        """
        if len(df_features) < self.sequence_length_L:
            logger.error(f"No hay suficientes muestras para crear una secuencia. Necesita {self.sequence_length_L}, tiene {len(df_features)}.")
            return None
            
        # Verificar si hay valores NaN en la parte final que usaremos
        last_L_rows = df_features.iloc[-self.sequence_length_L:].copy()
        if last_L_rows.isna().any().any():
            logger.warning("Se detectaron valores NaN en la secuencia a usar para predicción.")
            # Podríamos usar fillna aquí si es una situación esperada en algunos casos
            last_L_rows = last_L_rows.fillna(0)
            
        # Extraer solo las columnas finales requeridas
        valid_columns = [col for col in self.final_feature_columns if col in last_L_rows.columns]
        if len(valid_columns) != len(self.final_feature_columns):
            logger.warning(f"Faltan columnas en el conjunto de datos finales: {set(self.final_feature_columns) - set(valid_columns)}")
        
        features_sequence = last_L_rows[valid_columns].values
        
        # Verificar dimensiones correctas para el modelo
        if features_sequence.shape != (self.sequence_length_L, len(valid_columns)):
            logger.error(f"Dimensiones incorrectas de la secuencia: {features_sequence.shape}, esperado: ({self.sequence_length_L}, {len(valid_columns)})")
            return None
            
        logger.info(f"Secuencia de características extraída correctamente: forma={features_sequence.shape}")
        return features_sequence
