import pandas as pd
import numpy as np
import os
import logging
import yaml
from src.utils.config import ConfigManager
from src.data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.preprocessing_config = config_manager.get_preprocessing_config()

        self.raw_data_path = self.config_manager.get_config_value('data_paths.raw')
        self.processed_data_path = self.config_manager.get_config_value('data_paths.processed')
        os.makedirs(self.processed_data_path, exist_ok=True)

        self.L = self.preprocessing_config['sequence_length_L']
        self.norm_window = self.L * self.preprocessing_config['normalization_window_multiplier_for_L']
        
        self.feature_engineer = FeatureEngineer(
            indicators_config=self.preprocessing_config['indicators'],
            ohlcv_config=self.preprocessing_config['ohlcv_processing']
        )
        self.final_feature_columns = self.preprocessing_config['final_market_feature_columns']

    def _load_and_prepare_base_df(self, raw_data_filename: str) -> pd.DataFrame:
        """
        Carga y prepara el DataFrame base desde un archivo CSV de forma eficiente.
        
        Args:
            raw_data_filename: Nombre del archivo CSV con datos crudos
            
        Returns:
            DataFrame preparado y limpio
        """
        filepath = os.path.join(self.raw_data_path, raw_data_filename)
        logger.info(f"Cargando datos crudos desde: {filepath}")
        
        # Comprobar si debemos usar formato Parquet si el archivo existe
        parquet_path = f"{os.path.splitext(filepath)[0]}.parquet"
        use_float32 = self.preprocessing_config.get('use_float32', False)
        dtype_config = {col: 'float32' for col in ['Open', 'High', 'Low', 'Close', 'Volume']} if use_float32 else None
        
        try:
            # Intentar cargar desde Parquet si existe (más eficiente)
            if os.path.exists(parquet_path) and self.preprocessing_config.get('use_parquet_storage', False):
                logger.info(f"Cargando datos desde Parquet: {parquet_path}")
                df = pd.read_parquet(parquet_path)
                logger.info(f"Datos cargados desde Parquet con éxito: {df.shape}")
                return df
                
            # Si no hay Parquet, cargar desde CSV de forma optimizada
            logger.info(f"Cargando datos desde CSV: {filepath}")
            # Usar dtypes específicos y especificar parse_dates para eficiencia
            df = pd.read_csv(
                filepath,
                parse_dates=['Open_Time'],
                dtype=dtype_config
            )
            
            # Asegurar que Open_Time es datetime y UTC - optimizado
            df['Open_Time'] = pd.to_datetime(df['Open_Time'], utc=True)
            df.set_index('Open_Time', inplace=True)

            # 1. Optimizar para conjuntos de datos grandes
            # Primero verificar si el ordenamiento/deduplicación son realmente necesarios
            is_monotonic = df.index.is_monotonic_increasing
            has_duplicates = df.index.duplicated().any()
            
            if not is_monotonic:
                logger.warning(f"El índice de tiempo en {raw_data_filename} no está ordenado. Ordenando...")
                df.sort_index(inplace=True)
            
            if has_duplicates:
                logger.warning(f"Timestamps duplicados encontrados en {raw_data_filename}. Se eliminarán duplicados manteniendo la primera ocurrencia.")
                df = df[~df.index.duplicated(keep='first')]

            # 2. Convertir columnas OHLCV a numérico en una sola operación
            cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')

            # --- 3. Detección y Reporte de NaNs Iniciales ---
            nan_counts_initial = df[cols_to_numeric].isnull().sum()
            total_nans_initial = nan_counts_initial.sum()

            if total_nans_initial > 0:
                logger.warning(f"NaNs encontrados en columnas OHLCV de datos crudos ({raw_data_filename}) ANTES de la imputación:\n{nan_counts_initial[nan_counts_initial > 0]}")

                # --- 4. Imputación Limitada con Forward Fill (ffill) ---
                # Obtener el límite de ffill desde la configuración del módulo
                ffill_limit = self.preprocessing_config.get('raw_data_settings', {}).get('ffill_limit_for_nans', 0)  # Por defecto 0 (sin ffill)

                if ffill_limit > 0:
                    for col in cols_to_numeric:
                        df[col].ffill(limit=ffill_limit, inplace=True)
                    
                    nan_counts_after_ffill = df[cols_to_numeric].isnull().sum()
                    nans_filled_count = nan_counts_initial - nan_counts_after_ffill
                    logger.info(f"NaNs rellenados con ffill (limit={ffill_limit}):\n{nans_filled_count[nans_filled_count > 0]}")
                else:
                    logger.info("ffill para NaNs en datos crudos está desactivado (ffill_limit=0).")

            # --- 5. Eliminación de NaNs Restantes ---
            # Esto se aplica si ffill está desactivado, o para NaNs que ffill no pudo rellenar (al inicio o huecos > ffill_limit)
            nan_counts_before_dropna = df[cols_to_numeric].isnull().sum()
            total_nans_before_dropna = nan_counts_before_dropna.sum()

            if total_nans_before_dropna > 0:
                logger.warning(f"Eliminando {total_nans_before_dropna} NaNs restantes en OHLCV (o todos los NaNs si ffill está desactivado/no los cubrió).")
                df.dropna(subset=cols_to_numeric, inplace=True)
            
            # --- 6. Verificación de DataFrame Vacío ---
            if df.empty:
                logger.error(f"El DataFrame para {raw_data_filename} está vacío después del manejo de NaNs. No se puede continuar con este archivo.")
                # Devolver un DataFrame vacío para que el proceso principal lo maneje (ej. saltando el archivo)
                return pd.DataFrame() 

            # --- 7. Correcciones Finales (ej. Open = 0) ---
            # Es mejor hacerlo después de que los NaNs han sido manejados para asegurar que 'Open' existe y es numérico.
            df['Open'] = df['Open'].replace(0, 1e-9)  # Evitar log(0) o división por cero
            
            logger.info(f"Datos crudos cargados y preparados inicialmente para {raw_data_filename}. Forma final del DataFrame base: {df.shape}")
            return df

        except FileNotFoundError:
            logger.error(f"Archivo de datos crudos no encontrado: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error crítico durante la carga y preparación básica de datos desde {filepath}: {e}", exc_info=True)
            raise

    def _apply_feature_normalization(self, df_with_features: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica normalización a las características usando operaciones vectoriales de Pandas y NumPy.
        
        Args:
            df_with_features: DataFrame con características calculadas
            
        Returns:
            DataFrame con características normalizadas
        """
        logger.debug("Aplicando normalización/escalado final a las características usando operaciones vectorizadas.")
        
        # Crear una vista del dataframe sin copiar datos
        df_norm = df_with_features.copy(deep=False)

        # Convertir tipos a float32 para reducir uso de memoria
        use_float32 = self.preprocessing_config.get('use_float32', False)
        if use_float32:
            for col in df_norm.select_dtypes(include=['float64']).columns:
                df_norm[col] = df_norm[col].astype(np.float32)

        # Ventana para Z-score, asegurando min_periods para tener valores al inicio
        min_p = self.norm_window // 2

        # --- Normalización de características OHLCV (Z-score móvil) ---
        # Vectorización: Procesamos todos los retornos en un solo paso
        ohlcv_raw_cols = ['log_ret_C_O', 'log_ret_H_O', 'log_ret_L_O', 'log_ret_C_C_prev', 'log_ret_Vol_SMAVol']
        
        # Calculamos medias y desviaciones estándar para todos los retornos juntos
        # Esto reduce el número de llamadas a rolling(), que son costosas
        returns_df = df_norm[ohlcv_raw_cols]
        mean_returns = returns_df.rolling(window=self.norm_window, min_periods=min_p).mean()
        std_returns = returns_df.rolling(window=self.norm_window, min_periods=min_p).std()
        
        # Reemplazar ceros con epsilon para evitar divisiones por cero
        std_returns = std_returns.replace(0, 1e-9)
        
        # Calcular z-scores en una sola operación vectorizada
        zscore_returns = (returns_df - mean_returns) / std_returns
        
        # Renombrar columnas con sufijo _norm
        zscore_returns.columns = [f'{col}_norm' for col in ohlcv_raw_cols]
        
        # Añadir al DataFrame principal de manera eficiente
        for col in zscore_returns.columns:
            df_norm[col] = zscore_returns[col]

        # --- Normalización de Indicadores ---
        # Vectorizamos operaciones comunes
        atr = df_norm['ATR'].replace(0, 1e-9)  # Para evitar división por cero
        close = df_norm['Close'].replace(0, 1e-9)

        # Normalización de SMAs y EMAs (operaciones vectorizadas)
        sma_ema_cols = ['SMA_short', 'SMA_long', 'EMA_short', 'EMA_long']
        for col in sma_ema_cols:
            df_norm[f'{col}_norm'] = (df_norm[col] - close) / atr

        # RSI - Escalado con verificación de opciones
        if self.preprocessing_config['indicators']['rsi_scaling_mode'] == "0_1":
            df_norm['RSI_scaled'] = df_norm['RSI'] / 100.0
        else:  # "-1_1"
            df_norm['RSI_scaled'] = (df_norm['RSI'] - 50.0) / 50.0
        
        # ATR normalizado
        df_norm['ATR_norm'] = atr / close

        # MACD - normalización vectorizada
        macd_cols = ['MACD_line', 'MACD_signal', 'MACD_hist']
        for col in macd_cols:
            df_norm[f'{col}_norm'] = df_norm[col] / atr

        # Bandas de Bollinger - vectorización por grupos
        df_norm['BB_dist_upper_norm'] = (df_norm['BB_upper'] - close) / atr
        df_norm['BB_dist_lower_norm'] = (close - df_norm['BB_lower']) / atr
        df_norm['BB_width_norm'] = df_norm['BB_width'] / atr

        # CCI - Z-score vectorizado
        mean_cci = df_norm['CCI'].rolling(window=self.norm_window, min_periods=min_p).mean()
        std_cci = df_norm['CCI'].rolling(window=self.norm_window, min_periods=min_p).std().replace(0, 1e-9)
        df_norm['CCI_norm'] = (df_norm['CCI'] - mean_cci) / std_cci

        # Estocástico - escalado simple
        stoch_cols = ['STOCH_slowk', 'STOCH_slowd']
        for col in stoch_cols:
            df_norm[f'{col}_scaled'] = df_norm[col] / 100.0
        
        # Seleccionar solo las columnas finales especificadas en la configuración
        try:
            df_final_selection = df_norm[self.final_feature_columns]
        except KeyError as e:
            missing = list(set(self.final_feature_columns) - set(df_norm.columns))
            logger.error(f"Una o más columnas finales no se encontraron después de la normalización: {missing}. Error: {e}")
            raise
        return df_final_selection

    def _create_sequences(self, df_final_features: pd.DataFrame) -> tuple:
        """
        Crea secuencias de datos a partir del DataFrame procesado utilizando operaciones vectorizadas.
        
        Args:
            df_final_features: DataFrame con características finales
            
        Returns:
            Tupla con arrays de secuencias y timestamps
        """
        logger.info(f"Creando secuencias de longitud L={self.L} con operaciones vectorizadas.")
        
        # Convertir a NumPy array para eficiencia
        data_values = df_final_features.values
        timestamps_values = df_final_features.index.to_numpy()

        num_samples = len(data_values) - self.L + 1
        
        if num_samples <= 0:
            logger.warning("No hay suficientes datos para crear ni una sola secuencia después del preprocesamiento y recorte de NaNs.")
            return np.array([]), np.array([])

        # Método vectorizado para crear secuencias
        # Crear un array 3D directamente con la forma correcta (muestras, longitud de secuencia, features)
        n_features = data_values.shape[1]
        
        # Preasignar array para mejor rendimiento, usando float32 si está configurado
        dtype = np.float32 if self.preprocessing_config.get('use_float32', False) else np.float64
        X_sequences = np.zeros((num_samples, self.L, n_features), dtype=dtype)
        
        # Para cada posición en la secuencia, copiar los datos de manera eficiente
        for i in range(self.L):
            X_sequences[:, i, :] = data_values[i:i+num_samples]
            
        # Extraer timestamps de las últimas posiciones de cada secuencia
        ts_sequences = timestamps_values[self.L-1:self.L-1+num_samples]
        
        return X_sequences, ts_sequences

    def process_data(self, raw_data_filename: str, output_filename_base: str = None):
        logger.info(f"Iniciando preprocesamiento para el archivo: {raw_data_filename}")
        
        if output_filename_base is None:
            output_filename_base = os.path.splitext(raw_data_filename)[0]
            
        # 1. Cargar y preparación básica
        df_base = self._load_and_prepare_base_df(raw_data_filename)
        if df_base.empty:
            return

        # 2. Ingeniería de Características (cálculo de indicadores y features OHLCV)
        df_with_features = self.feature_engineer.add_ohlcv_features(df_base)
        df_with_features = self.feature_engineer.add_technical_indicators(df_with_features)
        
        # 3. Aplicar Normalización/Escalado Final
        df_normalized_features = self._apply_feature_normalization(df_with_features)

        # 4. Eliminar NaNs inducidos por lookback de indicadores y ventanas de normalización
        # El primer índice válido será aquel donde todas las features tengan un valor no-NaN.
        # Esto ocurre después del mayor periodo de lookback.
        df_cleaned = df_normalized_features.dropna()
        if df_cleaned.empty:
            logger.warning("El DataFrame está vacío después de eliminar NaNs (pos-normalización). No se pueden crear secuencias.")
            return
        
        logger.info(f"Forma del DataFrame después de la limpieza de NaNs y selección de features finales: {df_cleaned.shape}")

        # 5. Creación de Secuencias
        X_sequences, ts_sequences = self._create_sequences(df_cleaned)
        
        if X_sequences.shape[0] == 0:
             logger.warning("No se generaron secuencias válidas.")
             return

        # 6. Guardado de Datos Procesados
        output_filename = f"{output_filename_base}_L{self.L}_market_features.npz"
        output_path = os.path.join(self.processed_data_path, output_filename)
        try:
            # Guardamos también las series originales de 'Close' y 'ATR' sin normalizar para cálculos más precisos
            close_series = df_with_features['Close'].values
            atr_series = df_with_features['ATR'].values
            
            # Aseguramos que tengamos los mismos puntos de tiempo que en las secuencias
            # Tomamos los últimos valores de cada secuencia (el punto actual para cada secuencia)
            seq_count = X_sequences.shape[0]
            close_for_sequences = np.array([close_series[i + self.L - 1] for i in range(seq_count)], dtype=np.float32)
            atr_for_sequences = np.array([atr_series[i + self.L - 1] for i in range(seq_count)], dtype=np.float32)
            
            # Guardamos en formato comprimido
            np.savez_compressed(
                output_path, 
                X_market=X_sequences, 
                timestamps=ts_sequences,
                close_prices=close_for_sequences,
                atr_values=atr_for_sequences,
                feature_names=np.array(self.final_feature_columns)
            )
            logger.info(f"Secuencias procesadas ({X_sequences.shape[0]} muestras de forma {X_sequences.shape}) guardadas en: {output_path}")
            logger.info(f"También se guardaron series de Close y ATR sin normalizar para cálculos precisos de slippage y liquidación")
            
            # Guardar también en formato Parquet para datasets muy grandes
            if self.preprocessing_config.get('use_parquet_storage', False) and len(df_cleaned) > 100000:
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                parquet_filename = f"{output_filename_base}_processed.parquet"
                parquet_path = os.path.join(self.processed_data_path, parquet_filename)
                
                # Guardamos el DataFrame completo en formato Parquet
                df_cleaned.to_parquet(parquet_path, compression='snappy')
                logger.info(f"Datos también guardados en formato Parquet para mejor eficiencia: {parquet_path}")
        except Exception as e:
            logger.error(f"Error guardando las secuencias procesadas: {e}")
            raise
