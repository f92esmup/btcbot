import pandas as pd
import numpy as np
import os
import logging
import yaml
import io
from src.utils.config import ConfigManager
from src.data.feature_engineering import FeatureEngineer
from google.cloud import bigquery
from google.cloud import storage

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
        
        # Inicializar clientes de GCP 
        self.gcp_project_id = self.gcfg.gcp_project_id
        
        if not self.gcp_project_id:
            raise ValueError("Se requiere un ID de proyecto GCP para continuar. Establece GOOGLE_CLOUD_PROJECT en el entorno.")
            
        try:
            # Inicializar BigQuery
            self.bq_client = bigquery.Client(project=self.gcp_project_id)
            self.bq_raw_dataset_id = self.gcfg.get_config_value('gcp.bigquery.raw_dataset_id', 'market_data_raw')
            self.bq_raw_table_id_prefix = self.gcfg.get_config_value('gcp.bigquery.raw_table_id_prefix', 'klines_')
            logger.info(f"Cliente BigQuery inicializado para el proyecto {self.gcp_project_id}")
            
            # Inicializar GCS
            self.gcs_client = storage.Client(project=self.gcp_project_id)
            self.gcs_processed_bucket_name = self.gcfg.get_config_value('gcp.gcs.processed_bucket_name')
            
            if not self.gcs_processed_bucket_name:
                raise ValueError("No se ha configurado gcp.gcs.processed_bucket_name en config.yaml")
                
            # Verificar si el bucket existe
            try:
                bucket = self.gcs_client.get_bucket(self.gcs_processed_bucket_name)
                logger.info(f"Bucket GCS encontrado: {self.gcs_processed_bucket_name}")
            except Exception:
                # Crear bucket si no existe
                logger.info(f"Bucket {self.gcs_processed_bucket_name} no encontrado, creando...")
                bucket = self.gcs_client.create_bucket(self.gcs_processed_bucket_name, location="europe-southwest1")
                logger.info(f"Bucket GCS creado: {self.gcs_processed_bucket_name}")
                
            logger.info(f"Cliente GCS inicializado para el proyecto {self.gcp_project_id}")
            
            # Configurar el uso exclusivo de GCP
            self.use_gcp = True
            
        except Exception as e:
            raise ConnectionError(f"Error inicializando clientes GCP: {e}")

    def _load_and_prepare_base_df(self, raw_data_source: str, symbol: str = None, interval: str = None) -> pd.DataFrame:
        """
        Carga y prepara el DataFrame base desde BigQuery.
        
        Args:
            raw_data_source: Se ignora este parámetro cuando se usa BigQuery
            symbol: Símbolo para consulta BigQuery (requerido)
            interval: Intervalo para consulta BigQuery (requerido)
            
        Returns:
            DataFrame preparado y limpio
        """
        use_float32 = self.mcfg.get('use_float32', False)
        
        if not symbol or not interval:
            raise ValueError("Se requiere symbol e interval para cargar datos desde BigQuery")
        
        try:
            logger.info(f"Cargando datos desde BigQuery para {symbol}_{interval}")
            
            # Construir nombre de tabla de BigQuery
            raw_table_id = f"{self.bq_raw_table_id_prefix}{symbol}_{interval}"
            full_raw_table_id = f"{self.gcp_project_id}.{self.bq_raw_dataset_id}.{raw_table_id}"
            
            # Optimizar la consulta para manejar posibles problemas específicos de datos crudos
            ffill_limit = self.mcfg.get('raw_data_settings', {}).get('ffill_limit_for_nans', 0)
            query = f"""
            SELECT 
                Open_Time,
                -- Reemplazar 0 con NULL en Open para evitar problemas con log(0)
                NULLIF(Open, 0) as Open,
                High,
                Low,
                Close,
                Volume
            FROM `{full_raw_table_id}` 
            ORDER BY Open_Time
            """
            
            # Ejecutar query con opciones optimizadas
            df = self.bq_client.query(query).to_dataframe()
            
            if df.empty:
                raise ValueError(f"No se encontraron datos en BigQuery para {symbol}_{interval}")
            
            # Convertir tipos y optimizar según configuración
            cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
            if use_float32:
                for col in cols_to_numeric:
                    df[col] = df[col].astype('float32')
            
            # Procesar fechas y establecer índice
            df['Open_Time'] = pd.to_datetime(df['Open_Time'], utc=True)
            df.set_index('Open_Time', inplace=True)
            
            # Verificar la ordenación y duplicados
            is_monotonic = df.index.is_monotonic_increasing
            has_duplicates = df.index.duplicated().any()
            
            if not is_monotonic:
                logger.warning(f"El índice de tiempo para {symbol}_{interval} no está ordenado. Ordenando...")
                df.sort_index(inplace=True)
            
            if has_duplicates:
                logger.warning(f"Timestamps duplicados encontrados para {symbol}_{interval}. Se eliminarán duplicados manteniendo la primera ocurrencia.")
                df = df[~df.index.duplicated(keep='first')]
            
            # Manejar NaNs - detección y reporte
            nan_counts = df[cols_to_numeric].isnull().sum()
            total_nans = nan_counts.sum()
            
            if total_nans > 0:
                logger.warning(f"NaNs encontrados en columnas OHLCV de {symbol}_{interval}: {nan_counts[nan_counts > 0]}")
                
                # Aplicar forward fill si está configurado
                if ffill_limit > 0:
                    for col in cols_to_numeric:
                        df[col] = df[col].ffill(limit=ffill_limit)
                    
                    nan_counts_after = df[cols_to_numeric].isnull().sum()
                    logger.info(f"NaNs después de ffill: {nan_counts_after[nan_counts_after > 0]}")
                
                # Eliminar filas con NaNs restantes
                df_before = df.shape[0]
                df.dropna(subset=cols_to_numeric, inplace=True)
                df_after = df.shape[0]
                
                if df_before > df_after:
                    logger.warning(f"Se eliminaron {df_before - df_after} filas con NaNs no imputables.")
            
            # Verificación final y sustitución de valores problemáticos
            if df.empty:
                raise ValueError(f"El DataFrame para {symbol}_{interval} está vacío después del manejo de NaNs.")
            
            # Sustitución de valores que podrían causar problemas en cálculos posteriores
            for col in cols_to_numeric:
                df[col] = df[col].replace(0, 1e-9)  # Evitar divisiones por cero o log(0)
            
            logger.info(f"Datos cargados exitosamente desde BigQuery: {df.shape} filas para {symbol}_{interval}")
            return df
            
        except Exception as e:
            logger.error(f"Error cargando datos desde BigQuery: {e}")
            raise ValueError(f"No se pudieron cargar los datos desde BigQuery: {e}")
        
        try:
            # Intentar cargar desde Parquet si existe (más eficiente)
            if os.path.exists(parquet_path) and self.mcfg.get('use_parquet_storage', False):
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
        """
        Aplica normalización a las características usando operaciones vectoriales de Pandas y NumPy.
        
        Args:
            df_with_features: DataFrame con características calculadas
            
        Returns:
            DataFrame con características normalizadas
        """
        logger.info("Aplicando normalización/escalado final a las características usando operaciones vectorizadas.")
        
        # Crear una vista del dataframe sin copiar datos
        df_norm = df_with_features.copy(deep=False)

        # Convertir tipos a float32 para reducir uso de memoria
        use_float32 = self.mcfg.get('use_float32', False)
        if use_float32:
            for col in df_norm.select_dtypes(include=['float64']).columns:
                df_norm[col] = df_norm[col].astype(np.float32)

        # Ventana para Z-score, asegurando min_periods para tener valores al inicio
        min_p = max(2, self.norm_window // 2)  # Asegurar que min_periods sea al menos 2

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
        if self.mcfg['indicators']['rsi_scaling_mode'] == "0_1":
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
        
        # Verificar valores extremos en las características finales
        extreme_threshold = 10.0  # Umbral para considerar un valor como extremo
        for feat in self.final_feature_columns:
            if feat in df_norm.columns:
                # Reemplazar valores extremos con el umbral
                extremes = df_norm[feat].abs() > extreme_threshold
                if extremes.any():
                    extreme_count = extremes.sum()
                    if extreme_count > 0:
                        logger.warning(f"Se encontraron {extreme_count} valores extremos en {feat}. Aplicando recorte.")
                        df_norm.loc[df_norm[feat] > extreme_threshold, feat] = extreme_threshold
                        df_norm.loc[df_norm[feat] < -extreme_threshold, feat] = -extreme_threshold
        
        # Seleccionar solo las columnas finales especificadas en la configuración
        try:
            df_final_selection = df_norm[self.final_feature_columns]
            
            # Verificación final de NaNs
            nan_counts = df_final_selection.isna().sum()
            if nan_counts.sum() > 0:
                logger.warning(f"NaNs en características finales después de normalización: {nan_counts[nan_counts > 0]}")
                
        except KeyError as e:
            missing = list(set(self.final_feature_columns) - set(df_norm.columns))
            logger.error(f"Una o más columnas finales no se encontraron después de la normalización: {missing}. Error: {e}")
            raise
        
        logger.info(f"Normalización de características completada con éxito. Forma final: {df_final_selection.shape}")
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
            logger.warning(f"No hay suficientes datos para crear secuencias. Se requieren al menos {self.L} puntos de tiempo.")
            return np.array([]), np.array([])

        # Método vectorizado para crear secuencias
        # Crear un array 3D directamente con la forma correcta (muestras, longitud de secuencia, features)
        n_features = data_values.shape[1]
        
        # Preasignar array para mejor rendimiento, usando float32 si está configurado
        dtype = np.float32 if self.mcfg.get('use_float32', False) else np.float64
        X_sequences = np.zeros((num_samples, self.L, n_features), dtype=dtype)
        
        # Para cada posición en la secuencia, copiar los datos de manera eficiente
        for i in range(self.L):
            X_sequences[:, i, :] = data_values[i:i+num_samples]
            
        # Extraer timestamps de las últimas posiciones de cada secuencia
        ts_sequences = timestamps_values[self.L-1:self.L-1+num_samples]
        
        # Verificar y reportar información sobre las secuencias creadas
        nan_count = np.isnan(X_sequences).sum()
        if nan_count > 0:
            logger.warning(f"Se detectaron {nan_count} valores NaN en las secuencias creadas.")
            
        logger.info(f"Secuencias creadas exitosamente: {X_sequences.shape[0]} secuencias de longitud {self.L} con {n_features} características")
        
        # Información sobre memoria utilizada
        memory_usage_mb = X_sequences.nbytes / (1024 * 1024)
        logger.info(f"Uso de memoria para las secuencias: {memory_usage_mb:.2f} MB")
        
        return X_sequences, ts_sequences

    def process_data(self, raw_data_source: str, symbol: str = None, interval: str = None, output_filename_base: str = None):
        """
        Procesa datos crudos y crea secuencias para entrenamiento.
        
        Args:
            raw_data_source: Nombre del archivo CSV con datos crudos o tabla de BigQuery
            symbol: Símbolo para consulta BigQuery (requerido si se usa BigQuery)
            interval: Intervalo para consulta BigQuery (requerido si se usa BigQuery)
            output_filename_base: Base para el nombre del archivo de salida
            
        Returns:
            Path completo a los datos procesados en GCS si la operación fue exitosa
        """
        logger.info(f"Iniciando preprocesamiento para: {raw_data_source} {symbol} {interval}")
        
        if output_filename_base is None:
            if symbol and interval:
                output_filename_base = f"{symbol}_{interval}"
            else:
                output_filename_base = os.path.splitext(raw_data_source)[0]
            
        # 1. Cargar y preparación básica
        df_base = self._load_and_prepare_base_df(raw_data_source, symbol, interval)
        if df_base is None or df_base.empty:
            logger.warning("No se pudo cargar el DataFrame base. Abortando procesamiento.")
            return None

        # 2. Ingeniería de Características (cálculo de indicadores y features OHLCV)
        logger.info("Aplicando ingeniería de características...")
        df_with_features = self.feature_engineer.add_ohlcv_features(df_base)
        df_with_features = self.feature_engineer.add_technical_indicators(df_with_features)
        
        # 3. Aplicar Normalización/Escalado Final
        logger.info("Aplicando normalización a las características...")
        df_normalized_features = self._apply_feature_normalization(df_with_features)

        # 4. Eliminar NaNs inducidos por lookback de indicadores y ventanas de normalización
        nan_count_before = df_normalized_features.isna().sum().sum()
        df_cleaned = df_normalized_features.dropna()
        nan_count_after = nan_count_before - df_normalized_features.shape[0] + df_cleaned.shape[0]
        
        if nan_count_before > 0:
            logger.info(f"Se eliminaron {nan_count_after} filas con NaNs de {df_normalized_features.shape[0]} filas totales")
            
        if df_cleaned.empty:
            logger.warning("El DataFrame está vacío después de eliminar NaNs (pos-normalización). No se pueden crear secuencias.")
            return None
        
        logger.info(f"Forma del DataFrame después de la limpieza de NaNs y selección de features finales: {df_cleaned.shape}")

        # 5. Creación de Secuencias
        X_sequences, ts_sequences = self._create_sequences(df_cleaned)
        
        if X_sequences.shape[0] == 0:
             logger.warning("No se generaron secuencias válidas.")
             return None

        # 6. Guardado de Datos Procesados
        output_filename = f"{output_filename_base}_L{self.L}_market_features.npz"
        
        # Preparar datos para guardar
        close_series = df_with_features['Close'].values
        atr_series = df_with_features['ATR'].values
        
        # Aseguramos que tengamos los mismos puntos de tiempo que en las secuencias
        seq_count = X_sequences.shape[0]
        close_for_sequences = np.array([close_series[i + self.L - 1] for i in range(seq_count)], dtype=np.float32)
        atr_for_sequences = np.array([atr_series[i + self.L - 1] for i in range(seq_count)], dtype=np.float32)
        
        # Mostrar información del tamaño de los datos a guardar
        memory_size_mb = (X_sequences.nbytes + close_for_sequences.nbytes + atr_for_sequences.nbytes) / (1024*1024)
        logger.info(f"Tamaño total de datos a guardar: {memory_size_mb:.2f} MB")
        
        # Guardar solo en GCS (obligatorio)
        try:
            # Crear un stream de memoria para los datos NPZ
            in_memory_file = io.BytesIO()
            
            # Convertir todo a float32 para reducir tamaño
            X_sequences_f32 = X_sequences.astype(np.float32)
            close_for_sequences_f32 = close_for_sequences.astype(np.float32)
            atr_for_sequences_f32 = atr_for_sequences.astype(np.float32)
            
            # Comprimir con mayor eficiencia - optimizando tamaño
            logger.info(f"Comprimiendo {X_sequences_f32.shape[0]} secuencias con {X_sequences_f32.shape[2]} características cada una...")
            np.savez_compressed(
                in_memory_file, 
                X_market=X_sequences_f32, 
                timestamps=ts_sequences,
                close_prices=close_for_sequences_f32,
                atr_values=atr_for_sequences_f32,
                feature_names=np.array(self.final_feature_columns)
            )
            in_memory_file.seek(0)  # Rebobinar al inicio del stream
            
            # Guardar datos a local primero (backup)
            local_output_path = os.path.join(self.processed_data_path, output_filename)
            with open(local_output_path, 'wb') as f:
                f.write(in_memory_file.getvalue())
            logger.info(f"Datos primero guardados localmente como backup: {local_output_path}")
            
            # Rebobinar para subir a GCS
            in_memory_file.seek(0)
            
            # Cargar a GCS con manejo de errores mejorado
            bucket = self.gcs_client.bucket(self.gcs_processed_bucket_name)
            gcs_path = f"processed/{output_filename}"
            blob = bucket.blob(gcs_path)
            
            # Configurar mayores tiempos de espera y reintentos
            from google.cloud.storage.retry import DEFAULT_RETRY
            custom_retry = DEFAULT_RETRY.with_deadline(300.0).with_max_retries(10)
            
            # Intentar subir con método chunked
            file_size_mb = os.path.getsize(local_output_path)/(1024*1024)
            logger.info(f"Subiendo {file_size_mb:.2f} MB a GCS usando chunks de 5MB...")
            
            # Subir archivo en chunks para mejorar rendimiento y estabilidad
            chunk_size = 5 * 1024 * 1024  # 5MB chunks
            blob.upload_from_file(
                in_memory_file, 
                retry=custom_retry, 
                timeout=300, 
                chunk_size=chunk_size
            )
            
            logger.info(f"Datos procesados subidos exitosamente a GCS: gs://{self.gcs_processed_bucket_name}/{gcs_path}")
            
            # Guardar ruta completa para referencia
            gcs_full_path = f"gs://{self.gcs_processed_bucket_name}/{gcs_path}"
            return gcs_full_path
            
        except Exception as e:
            logger.error(f"Error al subir datos a GCS: {e}")
            
            # Si ya se guardó un backup local, informarlo
            local_output_path = os.path.join(self.processed_data_path, output_filename)
            if os.path.exists(local_output_path):
                logger.warning(f"Existe un backup local en: {local_output_path}")
                logger.warning("ADVERTENCIA: Este es solo un respaldo temporal. GCS es el almacenamiento principal.")
            else:
                # Intento de guardar localmente como respaldo temporal si no existe
                try:
                    np.savez_compressed(
                        local_output_path, 
                        X_market=X_sequences.astype(np.float32), 
                        timestamps=ts_sequences,
                        close_prices=close_for_sequences.astype(np.float32),
                        atr_values=atr_for_sequences.astype(np.float32),
                        feature_names=np.array(self.final_feature_columns)
                    )
                    logger.warning(f"Respaldo guardado localmente debido a error de GCS: {local_output_path}")
                    logger.warning("ADVERTENCIA: Este es solo un respaldo temporal. GCS es el almacenamiento principal.")
                except Exception as local_err:
                    logger.error(f"Error adicional al intentar guardar localmente: {local_err}")
            
            # Esta excepción debe ser manejada por el código que llama a esta función
            raise RuntimeError(f"No se pudieron guardar los datos procesados en GCS. El proceso no puede continuar: {e}")
