import pandas as pd
from binance.um_futures import UMFutures
from datetime import datetime, timezone
import time
import logging
import os
from google.cloud import storage
from google.cloud import secretmanager
from src.utils.config_gcp import get_secret_from_gcp, get_gcs_path

# Configurar logger
logger = logging.getLogger(__name__)

class BinanceFuturesDownloaderCloud:
    """
    Clase para descargar datos históricos de futuros de Binance, optimizada para GCP.
    Guarda los datos directamente en Google Cloud Storage (GCS).
    """
    
    def __init__(self, project_id, api_key_secret_name, api_secret_secret_name, 
                 raw_data_bucket, request_limit=1000, request_delay=0.5, 
                 retry_attempts=5, retry_delay=60):
        """
        Inicializa el descargador de datos con parámetros de GCP.
        
        Args:
            project_id: ID del proyecto GCP
            api_key_secret_name: Nombre del secreto que contiene la API key
            api_secret_secret_name: Nombre del secreto que contiene el API secret
            raw_data_bucket: Nombre del bucket para datos crudos
            request_limit: Límite de velas por solicitud
            request_delay: Tiempo de espera entre solicitudes (segundos)
            retry_attempts: Número de intentos al fallar una solicitud
            retry_delay: Tiempo de espera base entre reintentos (segundos)
        """
        self.project_id = project_id
        self.api_key_secret_name = api_key_secret_name
        self.api_secret_secret_name = api_secret_secret_name
        self.raw_data_bucket = raw_data_bucket
        self.request_limit = request_limit
        self.request_delay = request_delay
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Inicializar cliente de Storage para GCS
        self.storage_client = storage.Client(project=project_id)
        
        # Obtener credenciales de Binance desde Secret Manager
        try:
            self.api_key = get_secret_from_gcp(project_id, api_key_secret_name)
            self.api_secret = get_secret_from_gcp(project_id, api_secret_secret_name)
            logger.info("Claves API de Binance cargadas desde GCP Secret Manager")
        except Exception as e:
            logger.error(f"Error al cargar claves API desde GCP Secret Manager: {e}")
            raise
        
        # Inicializar cliente de Binance Futures
        try:
            self.client = UMFutures(key=self.api_key, secret=self.api_secret)
            # Verificar conexión 
            self.client.time()
            logger.info("Conexión con Binance Futures API establecida exitosamente")
        except Exception as e:
            logger.error(f"Error al conectar con Binance Futures API: {e}")
            raise ConnectionError(f"Error al conectar con Binance Futures API: {e}")

    def _interval_to_milliseconds(self, interval_str: str) -> int:
        """
        Convierte un intervalo de tiempo (1m, 1h, etc) a milisegundos.
        """
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
        return interval_dict.get(interval_str, 60 * 60 * 1000)  # Default a 1h

    def _generate_gcs_path(self, symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> str:
        """
        Genera la ruta completa de GCS para el archivo CSV.
        """
        start_str = start_dt.strftime('%Y%m%d')
        end_str = end_dt.strftime('%Y%m%d%H%M')
        file_name = f"{symbol}_FUTURES_{interval}_{start_str}_{end_str}.csv"
        
        # Crear estructura de carpetas por año/mes
        folder_path = f"data/{start_dt.strftime('%Y/%m')}"
        return f"{folder_path}/{file_name}"

    def fetch_historical_data(self, symbol: str, interval: str, start_date_str: str, output_gcs_prefix: str = None, end_date_str: str = None):
        """
        Descarga datos históricos de futuros y los guarda en GCS.
        
        Args:
            symbol: Símbolo de trading (ej. 'BTCUSDT')
            interval: Intervalo de tiempo (ej. '1h')
            start_date_str: Fecha de inicio en formato 'YYYY-MM-DD'
            output_gcs_prefix: Prefijo opcional para la ruta en GCS
            end_date_str: Fecha de fin en formato 'YYYY-MM-DD' (opcional, default: fecha actual)
        
        Returns:
            URI de GCS donde se guardaron los datos o None si hay error
        """
        logger.info(f"Iniciando descarga de datos históricos para {symbol} ({interval}) desde {start_date_str}")
        
        try:
            start_dt = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            logger.error(f"Formato de fecha de inicio incorrecto: {start_date_str}. Usar YYYY-MM-DD")
            return None

        if end_date_str:
            try:
                end_dt = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                current_time_utc = end_dt
            except ValueError:
                logger.error(f"Formato de fecha de fin incorrecto: {end_date_str}. Usar YYYY-MM-DD")
                return None
        else:
            current_time_utc = datetime.now(timezone.utc)
        
        # Convertir a milisegundos para la API de Binance
        start_timestamp_ms = int(start_dt.timestamp() * 1000)
        end_timestamp_ms = int(current_time_utc.timestamp() * 1000)

        all_klines_data = []
        fetch_start_time = start_timestamp_ms
        attempts = 0

        while fetch_start_time < end_timestamp_ms:
            try:
                logger.debug(f"Obteniendo velas para {symbol} desde {datetime.fromtimestamp(fetch_start_time/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC...")
                
                # Usar el endpoint de klines de Binance Futures
                klines = self.client.klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=fetch_start_time,
                    endTime=end_timestamp_ms,
                    limit=self.request_limit
                )

                if not klines:
                    logger.info("No se encontraron más datos para el período o se alcanzó el límite de datos de Binance")
                    break 

                all_klines_data.extend(klines)
                
                # Actualizar el timestamp de inicio para la siguiente solicitud
                last_kline_open_time = klines[-1][0]
                fetch_start_time = last_kline_open_time + 1

                if fetch_start_time >= end_timestamp_ms:
                    logger.info("Se alcanzó la fecha/hora de finalización")
                    break
                
                attempts = 0  # Resetear intentos si la solicitud fue exitosa
                time.sleep(self.request_delay)  # Respetar los límites de la API

            except Exception as e:
                attempts += 1
                logger.warning(f"Error obteniendo datos para {symbol}: {e}. Intento {attempts}/{self.retry_attempts}")
                if attempts >= self.retry_attempts:
                    logger.error(f"Máximo número de reintentos alcanzado para {symbol}. Abortando descarga")
                    return None
                time.sleep(self.retry_delay * attempts)  # Exponential backoff

        if not all_klines_data:
            logger.warning(f"No se descargaron datos para {symbol} en el rango especificado")
            return None

        # Definir columnas según la documentación de la API de Futuros
        columns = [
            'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_Time', 'Quote_Asset_Volume', 'Number_of_Trades',
            'Taker_Buy_Base_Asset_Volume', 'Taker_Buy_Quote_Asset_Volume', 'Ignore'
        ]
        df = pd.DataFrame(all_klines_data, columns=columns)

        # Procesamiento básico: convertir timestamps, seleccionar columnas, convertir tipos
        df['Open_Time'] = pd.to_datetime(df['Open_Time'], unit='ms', utc=True)
        df = df[['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume']]  # Seleccionar OHLCV

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=numeric_cols, inplace=True)
        df.drop_duplicates(subset=['Open_Time'], keep='first', inplace=True)
        df.sort_values(by='Open_Time', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Obtener la ruta GCS
        gcs_relative_path = self._generate_gcs_path(symbol, interval, start_dt, current_time_utc)
        
        # Si se proporciona un prefijo específico, usarlo
        if output_gcs_prefix:
            # Asegurar que el prefijo no termine con /
            if output_gcs_prefix.endswith('/'):
                output_gcs_prefix = output_gcs_prefix[:-1]
                
            # Extraer solo el nombre del archivo de la ruta relativa
            filename = os.path.basename(gcs_relative_path)
            gcs_blob_name = f"{output_gcs_prefix}/{filename}"
        else:
            gcs_blob_name = gcs_relative_path

        # Guardar en GCS usando una subida directa
        try:
            bucket = self.storage_client.get_bucket(self.raw_data_bucket)
            blob = bucket.blob(gcs_blob_name)
            
            # Guardar a GCS
            blob.upload_from_string(df.to_csv(index=False), 'text/csv')
            
            gcs_uri = f"gs://{self.raw_data_bucket}/{gcs_blob_name}"
            logger.info(f"Datos para {symbol} ({len(df)} velas) guardados exitosamente en: {gcs_uri}")
            return gcs_uri
            
        except Exception as e:
            logger.error(f"Error al guardar datos en GCS: {e}")
            return None