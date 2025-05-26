import pandas as pd
from binance.client import Client
from binance.async_client import AsyncClient
from datetime import datetime, timezone
import time
import logging
import os
import io
import asyncio
from google.cloud import storage
from src.utils.config import ConfigManager
from src.utils.logging_utils import get_madrid_timestamp, MADRID_TZ
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class BinanceFuturesDownloader:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
        # Obtener credenciales exclusivamente de Secret Manager
        try:
            self.api_key = self.config_manager.get_env_variable('BINANCE_API_KEY_FUTURES')
            self.api_secret = self.config_manager.get_env_variable('BINANCE_API_SECRET_FUTURES')
        except Exception as e:
            logger.error(f"Error al obtener credenciales de Binance de Secret Manager: {e}")
            raise ValueError(f"Error al obtener credenciales de Binance de Secret Manager: {e}")
        
        if not self.api_key or not self.api_secret:
            logger.error("API Key o Secret de Binance Futuros no configuradas en Google Secret Manager")
            raise ValueError("API Key o Secret de Binance Futuros no configuradas en Google Secret Manager.")

        # Inicializar cliente sincrónico para backward compatibility
        try:
            self.client = Client(self.api_key, self.api_secret)
            self.client.futures_ping()
            logger.info("Conexión con Binance Futures API establecida exitosamente.")
        except Exception as e:
            logger.error(f"Error al conectar con Binance Futures API: {e}")
            raise ConnectionError(f"Error al conectar con Binance Futures API: {e}")

        # Cliente asíncrono para nuevas funcionalidades
        self.async_client = None

        # Configuración de Google Cloud Storage (obligatoria)
        self.gcp_project_id = self.config_manager.get_env_variable('GCP_PROJECT_ID')
        if not self.gcp_project_id:
            logger.error("GCP_PROJECT_ID no configurado. Es obligatorio para el funcionamiento.")
            raise ValueError("GCP_PROJECT_ID no configurado. Es obligatorio para el funcionamiento.")
        
        self.gcs_bucket_name = self.config_manager.get_env_variable('GCS_BUCKET_NAME')
        if not self.gcs_bucket_name:
            logger.error("GCS_BUCKET_NAME no configurado. Es obligatorio para el funcionamiento.")
            raise ValueError("GCS_BUCKET_NAME no configurado. Es obligatorio para el funcionamiento.")
        
        self.gcs_data_path = self.config_manager.get_config_value('data_paths.gcs_raw', 'raw')
        
        logger.info(f"Usando Google Cloud Storage para almacenar datos. Bucket: {self.gcs_bucket_name}")
        try:
            self.storage_client = storage.Client(project=self.gcp_project_id)
            self.bucket = self.storage_client.bucket(self.gcs_bucket_name)
            
            if not self.bucket.exists():
                logger.error(f"El bucket {self.gcs_bucket_name} no existe. Por favor, crea el bucket desde la consola de Google Cloud primero.")
                raise ValueError(f"El bucket {self.gcs_bucket_name} no existe.")
            
            logger.info(f"Bucket {self.gcs_bucket_name} encontrado correctamente.")
        
        except Exception as e:
            logger.error(f"Error al conectar con Google Cloud Storage: {e}")
            raise ConnectionError(f"Error al conectar con Google Cloud Storage: {e}")

        self.request_limit = self.config_manager.get_config_value('binance_api.request_limit_per_call', 1000)
        self.request_delay = self.config_manager.get_config_value('binance_api.request_delay_seconds', 0.5)
        self.retry_attempts = self.config_manager.get_config_value('binance_api.retry_attempts', 5)
        self.retry_delay = self.config_manager.get_config_value('binance_api.retry_delay_seconds', 60)    

    async def _ensure_async_client(self):
        """Asegura que el cliente asíncrono esté inicializado."""
        if self.async_client is None:
            try:
                self.async_client = await AsyncClient.create(
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
                logger.info("Cliente asíncrono de Binance inicializado exitosamente.")
            except Exception as e:
                logger.error(f"Error al inicializar cliente asíncrono de Binance: {e}")
                raise ConnectionError(f"Error al inicializar cliente asíncrono de Binance: {e}")

    async def close_async_client(self):
        """Cierra el cliente asíncrono si está activo."""
        if self.async_client:
            await self.async_client.close_connection()
            self.async_client = None
            logger.info("Cliente asíncrono de Binance cerrado.")

    def _interval_to_milliseconds(self, interval_str: str) -> int:
        """Convierte el intervalo a milisegundos."""
        interval_dict = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
        }
        return interval_dict.get(interval_str, 60 * 60 * 1000)

    def _generate_filename(self, symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> str:
        """Genera nombre de archivo estandarizado para un chunk."""
        start_str = start_dt.strftime('%Y%m%d')
        end_str = end_dt.strftime('%Y%m%d%H%M')
        filename = f"{symbol}_FUTURES_{interval}_{start_str}_{end_str}.csv"
        return os.path.join(self.gcs_data_path, filename)

    def generate_chunk_filename(self, symbol: str, interval: str, chunk_period: str) -> str:
        """
        Genera nombre de archivo para un chunk usando formato de periodo.
        
        Args:
            symbol: Símbolo del activo (ej. BTCUSDT)
            interval: Intervalo de tiempo (ej. 1h)
            chunk_period: Periodo del chunk (ej. 2024-01, 2024-Q1)
            
        Returns:
            Nombre del archivo chunk
        """
        filename = f"{symbol}_FUTURES_{interval}_{chunk_period}.csv"
        return os.path.join(self.gcs_data_path, filename)

    async def download_chunk_data_async(
        self, 
        symbol: str, 
        interval: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Descarga datos históricos para un chunk específico de forma asíncrona.
        
        Args:
            symbol: Símbolo del activo
            interval: Intervalo de tiempo
            start_date: Fecha de inicio del chunk
            end_date: Fecha de fin del chunk
            
        Returns:
            DataFrame con datos OHLCV del chunk
        """
        await self._ensure_async_client()
        
        logger.info(f"Descargando chunk asíncrono para {symbol} ({interval}) desde {start_date.date()} hasta {end_date.date()}")
        
        start_timestamp_ms = int(start_date.timestamp() * 1000)
        end_timestamp_ms = int(end_date.timestamp() * 1000)
        
        all_klines_data = []
        fetch_start_time = start_timestamp_ms
        attempts = 0

        while fetch_start_time < end_timestamp_ms:
            try:
                logger.debug(f"Obteniendo velas para {symbol} desde {datetime.fromtimestamp(fetch_start_time/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC...")
                
                klines = await self.async_client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=fetch_start_time,
                    endTime=end_timestamp_ms,
                    limit=self.request_limit 
                )

                if not klines:
                    logger.info("No se encontraron más datos para el período o se alcanzó el límite de datos de Binance.")
                    break 

                all_klines_data.extend(klines)
                
                # Actualizar el timestamp de inicio para la siguiente solicitud
                last_kline_open_time = klines[-1][0]
                fetch_start_time = last_kline_open_time + 1

                if fetch_start_time >= end_timestamp_ms:
                    logger.info("Se alcanzó la fecha/hora de finalización.")
                    break
                
                attempts = 0  # Resetear intentos si la solicitud fue exitosa
                await asyncio.sleep(self.request_delay)  # Respetar los límites de la API

            except Exception as e:
                attempts += 1
                logger.warning(f"Error obteniendo datos para {symbol}: {e}. Intento {attempts}/{self.retry_attempts}.")
                if attempts >= self.retry_attempts:
                    logger.error(f"Máximo número de reintentos alcanzado para {symbol}. Abortando descarga para este chunk.")
                    raise
                await asyncio.sleep(self.retry_delay * attempts)  # Exponential backoff

        if not all_klines_data:
            logger.warning(f"No se descargaron datos para {symbol} en el rango especificado.")
            return pd.DataFrame()

        # Procesar datos descargados
        columns = [
            'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_Time', 'Quote_Asset_Volume', 'Number_of_Trades',
            'Taker_Buy_Base_Asset_Volume', 'Taker_Buy_Quote_Asset_Volume', 'Ignore'
        ]
        df = pd.DataFrame(all_klines_data, columns=columns)

        # Procesamiento básico
        df['Open_Time'] = pd.to_datetime(df['Open_Time'], unit='ms', utc=True)
        df = df[['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=numeric_cols, inplace=True)
        df.drop_duplicates(subset=['Open_Time'], keep='first', inplace=True)
        df.sort_values(by='Open_Time', inplace=True)
        df.reset_index(drop=True, inplace=True)

        logger.info(f"Descarga completa para chunk: {len(df)} velas procesadas")
        return df

    def save_chunk_to_gcs(self, df: pd.DataFrame, chunk_filename: str) -> str:
        """
        Guarda un DataFrame chunk en Google Cloud Storage.
        
        Args:
            df: DataFrame con datos del chunk
            chunk_filename: Nombre del archivo chunk
            
        Returns:
            Ruta GCS donde se guardó el chunk
        """
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            blob = self.bucket.blob(chunk_filename)
            blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
            
            logger.info(f"Chunk guardado en GCS: {chunk_filename} ({len(df)} velas)")
            return chunk_filename
        except Exception as e:
            logger.error(f"Error al guardar chunk en GCS: {e}")
            raise

    # Mantener método original para backward compatibility
    def fetch_historical_data(self, symbol: str, interval: str, start_date_str: str):
        """Método original para backward compatibility."""
        logger.info(f"Iniciando descarga de datos históricos para {symbol} ({interval}) desde {start_date_str}.")
        
        try:
            start_dt = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            logger.error(f"Formato de fecha de inicio incorrecto: {start_date_str}. Usar YYYY-MM-DD.")
            return

        current_time_utc = datetime.now(timezone.utc)
        current_time_madrid = get_madrid_timestamp()
        logger.info(f"Tiempo actual UTC: {current_time_utc.strftime('%Y-%m-%d %H:%M:%S')}, Madrid: {current_time_madrid.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Convertir a milisegundos para la API de Binance
        start_timestamp_ms = int(start_dt.timestamp() * 1000)
        end_timestamp_ms = int(current_time_utc.timestamp() * 1000)

        all_klines_data = []
        fetch_start_time = start_timestamp_ms
        attempts = 0

        while fetch_start_time < end_timestamp_ms:
            try:
                logger.debug(f"Obteniendo velas para {symbol} desde {datetime.fromtimestamp(fetch_start_time/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC...")
                
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=fetch_start_time,
                    endTime=end_timestamp_ms,
                    limit=self.request_limit 
                )

                if not klines:
                    logger.info("No se encontraron más datos para el período o se alcanzó el límite de datos de Binance.")
                    break 

                all_klines_data.extend(klines)
                
                last_kline_open_time = klines[-1][0]
                fetch_start_time = last_kline_open_time + 1

                if fetch_start_time >= end_timestamp_ms:
                    logger.info("Se alcanzó la fecha/hora de finalización.")
                    break
                
                attempts = 0
                time.sleep(self.request_delay)

            except Exception as e:
                attempts += 1
                logger.warning(f"Error obteniendo datos para {symbol}: {e}. Intento {attempts}/{self.retry_attempts}.")
                if attempts >= self.retry_attempts:
                    logger.error(f"Máximo número de reintentos alcanzado para {symbol}. Abortando descarga para este batch.")
                    return 
                time.sleep(self.retry_delay * attempts)

        if not all_klines_data:
            logger.warning(f"No se descargaron datos para {symbol} en el rango especificado.")
            return

        columns = [
            'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_Time', 'Quote_Asset_Volume', 'Number_of_Trades',
            'Taker_Buy_Base_Asset_Volume', 'Taker_Buy_Quote_Asset_Volume', 'Ignore'
        ]
        df = pd.DataFrame(all_klines_data, columns=columns)

        df['Open_Time'] = pd.to_datetime(df['Open_Time'], unit='ms', utc=True)
        df = df[['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume']]

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=numeric_cols, inplace=True)
        df.drop_duplicates(subset=['Open_Time'], keep='first', inplace=True)
        df.sort_values(by='Open_Time', inplace=True)
        df.reset_index(drop=True, inplace=True)

        output_filename = self._generate_filename(symbol, interval, start_dt, current_time_madrid)
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            blob = self.bucket.blob(output_filename)
            blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")
            
            logger.info(f"Datos para {symbol} ({len(df)} velas) guardados en GCS bucket: {self.gcs_bucket_name}, ruta: {output_filename}")
        except Exception as e:
            logger.error(f"Error al guardar los datos en Google Cloud Storage: {e}")
            raise Exception(f"Error al guardar los datos en Google Cloud Storage: {e}")
