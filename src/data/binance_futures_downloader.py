import pandas as pd
from binance.client import Client
from datetime import datetime, timezone
import time
import logging
import os
from src.utils.config import ConfigManager
from google.cloud import bigquery

logger = logging.getLogger(__name__) # Configurar el logger a nivel de script o aplicación

class BinanceFuturesDownloader:
    def __init__(self, config_manager): # ConfigManager debería ser inyectado
        self.config_manager = config_manager
        
        # Obtener credenciales de API exclusivamente de Secret Manager
        try:
            self.api_key = self.config_manager.get_secret('binance_api_key')
            self.api_secret = self.config_manager.get_secret('binance_api_secret')
        except Exception as e:
            logger.error(f"No se pudieron obtener credenciales de Binance desde Secret Manager: {e}")
            raise ValueError("Se requieren las credenciales de Binance en Secret Manager (binance_api_key y binance_api_secret)")

        try:
            self.client = Client(self.api_key, self.api_secret)
            self.client.futures_ping()  # Verificar conexión con la API de futuros
            logger.info("Conexión con Binance Futures API establecida exitosamente.")
        except Exception as e:
            logger.error(f"Error al conectar con Binance Futures API: {e}")
            raise ConnectionError(f"Error al conectar con Binance Futures API: {e}")


        self.raw_data_path = self.config_manager.get_config_value('data_paths.raw')
        self.request_limit = self.config_manager.get_config_value('binance_api.request_limit_per_call', 1000)
        self.request_delay = self.config_manager.get_config_value('binance_api.request_delay_seconds', 0.5)
        self.retry_attempts = self.config_manager.get_config_value('binance_api.retry_attempts', 5)
        self.retry_delay = self.config_manager.get_config_value('binance_api.retry_delay_seconds', 60)

        # Inicializar cliente BigQuery
        self.gcp_project_id = self.config_manager.gcp_project_id
        
        if not self.gcp_project_id:
            raise ValueError("Se requiere un ID de proyecto GCP para continuar. Establece GOOGLE_CLOUD_PROJECT en el entorno.")
            
        try:
            self.bq_client = bigquery.Client(project=self.gcp_project_id)
            self.bq_dataset_id = self.config_manager.get_config_value('gcp.bigquery.raw_dataset_id', 'market_data_raw')
            self.bq_raw_table_id_prefix = self.config_manager.get_config_value('gcp.bigquery.raw_table_id_prefix', 'klines_')
            logger.info(f"Cliente BigQuery inicializado para el proyecto {self.gcp_project_id}")
        except Exception as e:
            raise ConnectionError(f"Error al inicializar el cliente BigQuery: {e}")

        # Verificar si el dataset existe y crearlo si es necesario
        try:
            dataset_ref = self.bq_client.dataset(self.bq_dataset_id)
            self.bq_client.get_dataset(dataset_ref)
            logger.info(f"Dataset {self.bq_dataset_id} encontrado en BigQuery")
        except Exception:
            # Crear dataset si no existe
            logger.info(f"Dataset {self.bq_dataset_id} no encontrado, creando...")
            dataset = bigquery.Dataset(f"{self.gcp_project_id}.{self.bq_dataset_id}")
            dataset.location = "europe-southwest1"  # Usar la ubicación de Madrid
            dataset = self.bq_client.create_dataset(dataset)
            logger.info(f"Dataset {self.bq_dataset_id} creado en BigQuery")
            
        # Mantener directorio local para logs pero no es el almacenamiento principal
        os.makedirs(self.raw_data_path, exist_ok=True)


    def _interval_to_milliseconds(self, interval_str: str) -> int:
        # Función auxiliar para convertir el intervalo a milisegundos
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
        return interval_dict.get(interval_str, 60 * 60 * 1000)  # Default a 1h si no se encuentra

    def _generate_filename(self, symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> str:
        start_str = start_dt.strftime('%Y%m%d')
        end_str = end_dt.strftime('%Y%m%d%H%M') # Incluir hora y minuto para la fecha de fin actual
        # Añadir "_FUTURES" para distinguir de datos spot si alguna vez los usas.
        return os.path.join(self.raw_data_path, f"{symbol}_FUTURES_{interval}_{start_str}_{end_str}.csv")

    def fetch_historical_data(self, symbol: str, interval: str, start_date_str: str):
        logger.info(f"Iniciando descarga de datos históricos para {symbol} ({interval}) desde {start_date_str}.")
        
        try:
            start_dt = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        except ValueError:
            logger.error(f"Formato de fecha de inicio incorrecto: {start_date_str}. Usar YYYY-MM-DD.")
            return

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
                
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=fetch_start_time,
                    endTime=end_timestamp_ms, # Binance puede ignorar endTime si startTime + limit < endTime
                    limit=self.request_limit 
                )

                if not klines:
                    logger.info("No se encontraron más datos para el período o se alcanzó el límite de datos de Binance.")
                    break 

                all_klines_data.extend(klines)
                
                # Actualizar el timestamp de inicio para la siguiente solicitud
                # El timestamp de la última vela recibida + 1ms
                last_kline_open_time = klines[-1][0]
                fetch_start_time = last_kline_open_time + 1

                if fetch_start_time >= end_timestamp_ms:
                    logger.info("Se alcanzó la fecha/hora de finalización.")
                    break
                
                attempts = 0 # Resetear intentos si la solicitud fue exitosa
                time.sleep(self.request_delay) # Respetar los límites de la API

            except Exception as e:
                attempts += 1
                logger.warning(f"Error obteniendo datos para {symbol}: {e}. Intento {attempts}/{self.retry_attempts}.")
                if attempts >= self.retry_attempts:
                    logger.error(f"Máximo número de reintentos alcanzado para {symbol}. Abortando descarga para este batch.")
                    return 
                time.sleep(self.retry_delay * attempts) # Exponential backoff

        if not all_klines_data:
            logger.warning(f"No se descargaron datos para {symbol} en el rango especificado.")
            return

        # Definir columnas según la documentación de la API de Futuros
        columns = [
            'Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_Time', 'Quote_Asset_Volume', 'Number_of_Trades',
            'Taker_Buy_Base_Asset_Volume', 'Taker_Buy_Quote_Asset_Volume', 'Ignore'
        ]
        df = pd.DataFrame(all_klines_data, columns=columns)

        # Procesamiento básico: convertir timestamps, seleccionar columnas, convertir tipos
        df['Open_Time'] = pd.to_datetime(df['Open_Time'], unit='ms', utc=True)
        df = df[['Open_Time', 'Open', 'High', 'Low', 'Close', 'Volume']] # Seleccionar OHLCV

        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=numeric_cols, inplace=True) # Eliminar filas donde OHLCV no sea numérico
        df.drop_duplicates(subset=['Open_Time'], keep='first', inplace=True)
        df.sort_values(by='Open_Time', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Guardar en CSV (local, solo como registro)
        output_filename = self._generate_filename(symbol, interval, start_dt, current_time_utc)
        try:
            df.to_csv(output_filename, index=False)
            logger.info(f"Datos para {symbol} ({len(df)} velas) guardados localmente en: {output_filename}")
        except IOError as e:
            logger.warning(f"Error al guardar el archivo CSV local {output_filename}: {e}")
        
        # Guardar en BigQuery (obligatorio)
        try:
            # Definir el nombre de la tabla
            output_table_id = f"{self.bq_raw_table_id_prefix}{symbol}_{interval}"
            full_table_id = f"{self.gcp_project_id}.{self.bq_dataset_id}.{output_table_id}"
            
            # Configurar job para BigQuery
            job_config = bigquery.LoadJobConfig(
                schema=[
                    bigquery.SchemaField("Open_Time", "TIMESTAMP"),
                    bigquery.SchemaField("Open", "FLOAT64"),
                    bigquery.SchemaField("High", "FLOAT64"),
                    bigquery.SchemaField("Low", "FLOAT64"),
                    bigquery.SchemaField("Close", "FLOAT64"),
                    bigquery.SchemaField("Volume", "FLOAT64")
                ],
                write_disposition="WRITE_APPEND",  # Append para mantener el historial
            )
            
            # Cargar a BigQuery
            job = self.bq_client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
            job.result()  # Esperar a que el job termine
            logger.info(f"Datos para {symbol} ({len(df)} velas) cargados exitosamente en BigQuery: {full_table_id}")
            
            return df, full_table_id
        except Exception as e:
            logger.error(f"Error al cargar datos a BigQuery {output_table_id}: {e}")
            raise RuntimeError(f"No se pudieron cargar los datos a BigQuery. El proceso no puede continuar: {e}")
