import pandas as pd
import numpy as np
import os
import logging
import yaml
from src.utils.config import ConfigManager
from src.data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, general_config_manager: ConfigManager, module_specific_config: dict):
        self.gcfg = general_config_manager
        self.mcfg = module_specific_config  # Config específica del módulo de preprocesamiento

        self.raw_data_path = self.gcfg.get_config_value('data_paths.raw')
        self.processed_data_path = self.gcfg.get_config_value('data_paths.processed')
        os.makedirs(self.processed_data_path, exist_ok=True)

        self.L = self.mcfg['sequence_length_L']
        self.norm_window = self.L * self.mcfg['normalization_window_multiplier_for_L']
        
        self.feature_engineer = FeatureEngineer(
            indicators_config=self.mcfg['indicators'],
            ohlcv_config=self.mcfg['ohlcv_processing']
        )
        self.final_feature_columns = self.mcfg['final_market_feature_columns']
        if len(self.final_feature_columns) != 20:  # 5 OHLCV + 15 Indicadores
            logger.warning(f"El número de columnas finales ({len(self.final_feature_columns)}) no coincide con el esperado (20). Verifica 'final_market_feature_columns' en la config.")

    def _load_and_prepare_base_df(self, raw_data_filename: str) -> pd.DataFrame:
        filepath = os.path.join(self.raw_data_path, raw_data_filename)
        logger.info(f"Cargando datos crudos desde: {filepath}")
        try:
            df = pd.read_csv(
                filepath,
                parse_dates=['Open_Time']
            )
            # Asegurar que Open_Time es datetime y UTC
            df['Open_Time'] = pd.to_datetime(df['Open_Time'], utc=True)
            df.set_index('Open_Time', inplace=True)

            # 1. Asegurar que el índice es único y está ordenado
            if not df.index.is_monotonic_increasing:
                logger.warning(f"El índice de tiempo en {raw_data_filename} no está ordenado. Ordenando...")
                df.sort_index(inplace=True)
            if not df.index.is_unique:
                logger.warning(f"Timestamps duplicados encontrados en {raw_data_filename}. Se eliminarán duplicados manteniendo la primera ocurrencia.")
                df = df[~df.index.duplicated(keep='first')]

            # 2. Convertir columnas OHLCV a numérico, 'coerce' pone NaN si no puede convertir
            cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in cols_to_numeric:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # --- 3. Detección y Reporte de NaNs Iniciales ---
            nan_counts_initial = df[cols_to_numeric].isnull().sum()
            total_nans_initial = nan_counts_initial.sum()

            if total_nans_initial > 0:
                logger.warning(f"NaNs encontrados en columnas OHLCV de datos crudos ({raw_data_filename}) ANTES de la imputación:\n{nan_counts_initial[nan_counts_initial > 0]}")

                # --- 4. Imputación Limitada con Forward Fill (ffill) ---
                # Obtener el límite de ffill desde la configuración del módulo
                ffill_limit = self.mcfg.get('raw_data_settings', {}).get('ffill_limit_for_nans', 0)  # Por defecto 0 (sin ffill)

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
        logger.debug("Aplicando normalización/escalado final a las características.")
        df_norm = df_with_features.copy()

        # Ventana para Z-score, asegurando min_periods para tener valores al inicio
        min_p = self.norm_window // 2

        # Normalización de características OHLCV procesadas (Z-score móvil)
        ohlcv_raw_cols = ['log_ret_C_O', 'log_ret_H_O', 'log_ret_L_O', 'log_ret_C_C_prev', 'log_ret_Vol_SMAVol']
        for col in ohlcv_raw_cols:
            mean = df_norm[col].rolling(window=self.norm_window, min_periods=min_p).mean()
            std = df_norm[col].rolling(window=self.norm_window, min_periods=min_p).std().replace(0, 1e-9)  # Evitar división por cero
            df_norm[f'{col}_norm'] = (df_norm[col] - mean) / std

        # Normalización de Indicadores
        atr = df_norm['ATR'].replace(0, 1e-9)  # Para evitar división por cero
        close = df_norm['Close'].replace(0, 1e-9)

        df_norm['SMA_short_norm'] = (df_norm['SMA_short'] - close) / atr
        df_norm['SMA_long_norm'] = (df_norm['SMA_long'] - close) / atr
        df_norm['EMA_short_norm'] = (df_norm['EMA_short'] - close) / atr
        df_norm['EMA_long_norm'] = (df_norm['EMA_long'] - close) / atr

        if self.mcfg['indicators']['rsi_scaling_mode'] == "0_1":
            df_norm['RSI_scaled'] = df_norm['RSI'] / 100.0
        else:  # "-1_1"
            df_norm['RSI_scaled'] = (df_norm['RSI'] - 50.0) / 50.0
        
        df_norm['ATR_norm'] = atr / close

        df_norm['MACD_line_norm'] = df_norm['MACD_line'] / atr  # O Z-score
        df_norm['MACD_signal_norm'] = df_norm['MACD_signal'] / atr  # O Z-score
        df_norm['MACD_hist_norm'] = df_norm['MACD_hist'] / atr  # O Z-score
        
        # Distancias a BB normalizadas por ATR
        df_norm['BB_dist_upper_norm'] = (df_norm['BB_upper'] - close) / atr
        df_norm['BB_dist_lower_norm'] = (close - df_norm['BB_lower']) / atr
        # Ancho de BB normalizado por ATR o por la media móvil (BB_middle)
        df_norm['BB_width_norm'] = df_norm['BB_width'] / atr  # o df_norm['BB_width'] / df_norm['BB_middle'].replace(0,1e-9)

        # CCI (Z-score móvil puede ser bueno aquí, o dividir por constante empírica)
        mean_cci = df_norm['CCI'].rolling(window=self.norm_window, min_periods=min_p).mean()
        std_cci = df_norm['CCI'].rolling(window=self.norm_window, min_periods=min_p).std().replace(0, 1e-9)
        df_norm['CCI_norm'] = (df_norm['CCI'] - mean_cci) / std_cci

        df_norm['STOCH_slowk_scaled'] = df_norm['STOCH_slowk'] / 100.0
        df_norm['STOCH_slowd_scaled'] = df_norm['STOCH_slowd'] / 100.0
        
        # Seleccionar solo las columnas finales especificadas en la configuración
        try:
            df_final_selection = df_norm[self.final_feature_columns]
        except KeyError as e:
            missing = list(set(self.final_feature_columns) - set(df_norm.columns))
            logger.error(f"Una o más columnas finales no se encontraron después de la normalización: {missing}. Error: {e}")
            raise
        return df_final_selection

    def _create_sequences(self, df_final_features: pd.DataFrame) -> tuple:
        logger.info(f"Creando secuencias de longitud L={self.L}.")
        
        # Convertir a NumPy array para eficiencia
        data_values = df_final_features.values
        timestamps_values = df_final_features.index.to_numpy()

        num_samples = len(data_values) - self.L + 1
        
        if num_samples <= 0:
            logger.warning("No hay suficientes datos para crear ni una sola secuencia después del preprocesamiento y recorte de NaNs.")
            return np.array([]), np.array([])

        # Alternativa con bucle (más clara, potencialmente más lenta pero más segura para empezar)
        X_list, ts_list = [], []
        for i in range(num_samples):
            X_list.append(data_values[i : i + self.L])
            ts_list.append(timestamps_values[i + self.L - 1])  # Timestamp del último elemento

        sequences_X = np.array(X_list)
        sequences_ts = np.array(ts_list)
        
        return sequences_X, sequences_ts

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
            np.savez_compressed(output_path, X_market=X_sequences, timestamps=ts_sequences)
            logger.info(f"Secuencias procesadas ({X_sequences.shape[0]} muestras de forma {X_sequences.shape}) guardadas en: {output_path}")
        except Exception as e:
            logger.error(f"Error guardando las secuencias procesadas: {e}")
            raise
