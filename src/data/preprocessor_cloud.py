import pandas as pd
import numpy as np
import os
import logging
import yaml
import json
import tempfile
import fsspec
from google.cloud import storage
from src.utils.config_cloud import ConfigManagerCloud, get_gcs_path
from src.data.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

class DataPreprocessorCloud:
    """
    Preprocesador de datos optimizado para ejecutarse en entornos GCP.
    Lee datos desde GCS, los procesa y guarda las secuencias resultantes en GCS.
    """
    
    def __init__(self, 
                 project_id: str,
                 raw_data_bucket: str, 
                 processed_data_bucket: str,
                 sequence_length_L: int, 
                 norm_window_multiplier: int,
                 indicators_config_dict: dict = None,
                 ohlcv_config_dict: dict = None,
                 final_market_feature_columns: list = None,
                 use_float32: bool = True):
        """
        Inicializa el preprocesador con los parámetros necesarios.
        
        Args:
            project_id: ID del proyecto GCP 
            raw_data_bucket: Nombre del bucket para datos crudos
            processed_data_bucket: Nombre del bucket para datos procesados
            sequence_length_L: Longitud de la secuencia para el Transformer
            norm_window_multiplier: Multiplicador para la ventana de normalización
            indicators_config_dict: Configuración para los indicadores técnicos
            ohlcv_config_dict: Configuración para el procesamiento OHLCV
            final_market_feature_columns: Lista de columnas de características finales
            use_float32: Si es True, usa float32 en lugar de float64
        """
        self.project_id = project_id
        self.raw_data_bucket = raw_data_bucket
        self.processed_data_bucket = processed_data_bucket
        
        # Configuración para el preprocesamiento
        self.L = sequence_length_L
        self.norm_window = self.L * norm_window_multiplier
        self.use_float32 = use_float32
        
        # Inicializar el cliente de Storage
        self.storage_client = storage.Client(project=project_id)
        
        # Configuración para indicadores técnicos
        if indicators_config_dict is None:
            # Valores por defecto para indicadores
            indicators_config_dict = {
                'sma_short_period': 20,
                'sma_long_period': 50,
                'ema_short_period': 12,
                'ema_long_period': 26,
                'rsi_period': 14,
                'rsi_scaling_mode': "0_1",
                'atr_period': 14,
                'macd_fast_period': 12,
                'macd_slow_period': 26,
                'macd_signal_period': 9,
                'bollinger_period': 20,
                'bollinger_std_dev': 2,
                'cci_period': 20,
                'stochastic_k_period': 14,
                'stochastic_d_period': 3,
                'stochastic_slowing_period': 3
            }
        
        # Configuración para procesamiento OHLCV
        if ohlcv_config_dict is None:
            ohlcv_config_dict = {'volume_sma_period': 20}
        
        # Inicializar FeatureEngineer con la configuración
        self.feature_engineer = FeatureEngineer(
            indicators_config=indicators_config_dict,
            ohlcv_config=ohlcv_config_dict
        )
        
        # Columnas de características finales
        if final_market_feature_columns is None:
            # Usar columnas por defecto que coinciden con la configuración predeterminada
            self.final_feature_columns = [
                'log_ret_C_O_norm', 'log_ret_H_O_norm', 'log_ret_L_O_norm',
                'log_ret_C_C_prev_norm', 'log_ret_Vol_SMAVol_norm',
                'sma_short_norm', 'sma_long_norm', 'ema_short_norm', 'ema_long_norm',
                'rsi_norm', 'atr_norm', 'macd_line_norm', 'macd_signal_norm', 'macd_hist_norm',
                'bb_upper_norm', 'bb_lower_norm', 'bb_width_norm', 'cci_norm',
                'stoch_k_norm', 'stoch_d_norm'
            ]
        else:
            self.final_feature_columns = final_market_feature_columns
        
        if len(self.final_feature_columns) != 20:
            logger.warning(f"El número de columnas finales ({len(self.final_feature_columns)}) no coincide con el esperado (20). Verifica la configuración de características.")

    def _load_from_gcs(self, gcs_path: str) -> pd.DataFrame:
        """
        Carga datos desde Google Cloud Storage.
        
        Args:
            gcs_path: Ruta completa al archivo en GCS
            
        Returns:
            DataFrame con los datos cargados
        """
        logger.info(f"Cargando datos desde GCS: {gcs_path}")
        
        # Determinar extensión para elegir el método de carga
        file_ext = os.path.splitext(gcs_path)[1].lower()
        
        # Configurar tipos de datos para mejorar rendimiento si es necesario
        dtype_config = {col: 'float32' for col in ['Open', 'High', 'Low', 'Close', 'Volume']} if self.use_float32 else None
        
        try:
            # Cargar según el tipo de archivo
            if file_ext == '.parquet':
                with fsspec.open(gcs_path, 'rb') as f:
                    df = pd.read_parquet(f)
            else:  # Por defecto, asumimos CSV
                with fsspec.open(gcs_path, 'rb') as f:
                    df = pd.read_csv(f, dtype=dtype_config, parse_dates=['Open_Time'])
            
            logger.info(f"Datos cargados exitosamente: {len(df)} filas, {df.columns.tolist()}")
            return df
            
        except Exception as e:
            logger.error(f"Error cargando datos desde {gcs_path}: {e}")
            raise
    
    def _save_to_gcs(self, data, output_gcs_path: str, is_npz: bool = True):
        """
        Guarda datos en Google Cloud Storage.
        
        Args:
            data: Datos a guardar (ya sea un DataFrame o un dict de arrays para NPZ)
            output_gcs_path: Ruta completa en GCS donde guardar
            is_npz: Si es True, guarda como archivo NPZ; si es False, guarda como CSV/Parquet
        """
        logger.info(f"Guardando datos en GCS: {output_gcs_path}")
        
        try:
            # Extraer bucket y blob de la URI de GCS
            bucket_name = output_gcs_path.replace("gs://", "").split("/")[0]
            blob_name = output_gcs_path.replace(f"gs://{bucket_name}/", "")
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            # Guardar los datos según el tipo
            if is_npz:
                # Para NPZ, primero guardar en un archivo temporal
                with tempfile.NamedTemporaryFile() as temp:
                    np.savez_compressed(temp.name, **data)
                    # Asegurarse de que se escriba todo
                    temp.flush()
                    # Reposicionar al inicio del archivo
                    temp.seek(0)
                    # Subir a GCS
                    blob.upload_from_file(temp)
            else:
                # Para DataFrame, guardar según la extensión
                if output_gcs_path.endswith('.parquet'):
                    with tempfile.NamedTemporaryFile() as temp:
                        data.to_parquet(temp.name)
                        temp.flush()
                        temp.seek(0)
                        blob.upload_from_file(temp)
                else:  # Por defecto, CSV
                    blob.upload_from_string(data.to_csv(index=False), 'text/csv')
            
            logger.info(f"Datos guardados exitosamente en: {output_gcs_path}")
            return output_gcs_path
            
        except Exception as e:
            logger.error(f"Error guardando datos en {output_gcs_path}: {e}")
            raise
    
    def create_z_normalized_sequences(self, df: pd.DataFrame, feature_columns: list) -> tuple:
        """
        Crea secuencias normalizadas con Z-score para el entrenamiento.
        
        Args:
            df: DataFrame con datos de características
            feature_columns: Lista de columnas a incluir en las secuencias
            
        Returns:
            Tupla de (secuencias_X, timestamps) para entrenamiento
        """
        logger.info(f"Creando secuencias normalizadas (L={self.L}, ventana_norm={self.norm_window})")
        
        # Asegurar que solo usamos las columnas de características solicitadas
        if not all(col in df.columns for col in feature_columns):
            missing = [col for col in feature_columns if col not in df.columns]
            logger.error(f"Columnas faltantes en el DataFrame: {missing}")
            raise ValueError(f"Columnas faltantes en el DataFrame: {missing}")
        
        # Seleccionar solo las características necesarias y timestamps
        data = df[feature_columns].values
        timestamps = df['Open_Time'].values
        
        # Si se especifica usar float32, convertir los datos
        if self.use_float32:
            data = data.astype(np.float32)
        
        total_samples = len(data)
        valid_indices = []
        X_sequences = []
        ts_sequences = []
        
        logger.info(f"Datos originales: {data.shape}")
        
        # Crear secuencias normalizadas con Z-score
        for i in range(self.norm_window, total_samples - self.L + 1):
            # Ventana de normalización (mira hacia atrás desde la posición actual)
            norm_window_data = data[max(0, i - self.norm_window):i]
            
            # Calcular media y desviación estándar para cada característica
            # usando solo la ventana de normalización
            means = np.mean(norm_window_data, axis=0)
            stds = np.std(norm_window_data, axis=0)
            
            # Reemplazar ceros en stds para evitar divisiones por cero
            stds = np.where(stds == 0, 1e-8, stds)
            
            # Secuencia actual a normalizar (L timesteps)
            current_sequence = data[i:i+self.L]
            
            # Normalizar la secuencia usando la media y std de la ventana
            normalized_sequence = (current_sequence - means) / stds
            
            X_sequences.append(normalized_sequence)
            ts_sequences.append(timestamps[i:i+self.L])
            valid_indices.append(i)
        
        if not X_sequences:
            logger.error(f"No se pudieron crear secuencias. Verifica que el dataset tenga suficientes datos (>= {self.norm_window + self.L})")
            raise ValueError(f"No se pudieron crear secuencias. Dataset insuficiente.")
        
        # Convertir a arrays de numpy
        X_sequences = np.array(X_sequences)
        ts_sequences = np.array(ts_sequences)
        
        logger.info(f"Secuencias creadas: {X_sequences.shape}")
        
        return X_sequences, ts_sequences
    
    def process_data(self, raw_data_gcs_path: str, output_gcs_prefix: str = None,
                 output_filename_base: str = None, extra_metadata: dict = None) -> str:
        """
        Procesa datos crudos y genera secuencias para entrenamiento.
        
        Args:
            raw_data_gcs_path: Ruta completa al archivo de datos crudos en GCS
            output_gcs_prefix: Prefijo opcional para la ruta de salida en GCS
            output_filename_base: Nombre base para el archivo de salida (si None, se infiere del nombre original)
            extra_metadata: Diccionario con metadatos adicionales para incluir en el archivo NPZ
            
        Returns:
            URI de GCS donde se guardaron los datos procesados
        """
        # Preparar la ruta de salida
        if output_gcs_prefix is None:
            output_gcs_prefix = f"gs://{self.processed_data_bucket}/data"
        
        # Extraer información básica del nombre del archivo para la salida
        filename = os.path.basename(raw_data_gcs_path)
        symbol = filename.split('_')[0]  # Asumimos formato BTCUSDT_FUTURES_...
        
        if output_filename_base is None:
            output_filename_base = f"{symbol}_processed"
        
        try:
            # 1. Cargar y preparar los datos base
            df = self._load_from_gcs(raw_data_gcs_path)
            
            # 2. Aplicar ingeniería de características
            logger.info("Aplicando ingeniería de características")
            df_features = self.feature_engineer.create_all_features(df)
            
            # 3. Limpiar y preparar para secuencias
            # Eliminar filas con NaN después de la ingeniería de características
            df_features.dropna(inplace=True)
            logger.info(f"Datos después de eliminar NaN: {len(df_features)} filas")
            
            # 4. Crear secuencias normalizadas
            X_sequences, ts_sequences = self.create_z_normalized_sequences(
                df_features, self.final_feature_columns
            )
            
            # 5. Guardar secuencias procesadas en GCS como NPZ
            output_npz_path = f"{output_gcs_prefix}/{output_filename_base}_L{self.L}_market_features.npz"
            
            # Estructura de datos a guardar
            data_dict = {
                'X_market': X_sequences,
                'timestamps': ts_sequences,
                'feature_names': np.array(self.final_feature_columns)
            }
            
            # Agregar metadatos adicionales si se proporcionan
            if extra_metadata is not None:
                for key, value in extra_metadata.items():
                    # Convertir valores a arrays de NumPy si es necesario
                    if isinstance(value, (list, tuple)):
                        data_dict[key] = np.array(value)
                    elif isinstance(value, (int, float, str, bool)):
                        data_dict[key] = np.array([value])
                    else:
                        data_dict[key] = value
                        
            # Guardar en GCS
            self._save_to_gcs(data_dict, output_npz_path, is_npz=True)
            logger.info(f"Secuencias guardadas en: {output_npz_path}")
            
            return output_npz_path
            
        except Exception as e:
            logger.error(f"Error procesando datos: {e}")
            raise