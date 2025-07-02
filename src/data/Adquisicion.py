"""
Módulo de adquisición de datos de la API de Binance.
Descarga y procesa datos OHLCV para futuros de Bitcoin usando python-binance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
from typing import List, Optional
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException


def _download_kline_chunk(start_timestamp: int, symbol: str, interval: str, 
                         api_key: Optional[str] = None, api_secret: Optional[str] = None,
                         is_testnet: bool = False, api_call_limit: int = 1000,
                         max_retries: int = 3, retry_delay: float = 1.0) -> List:
    """
    Función worker para descargar un trozo de datos de velas de la API de Binance.
    Esta función debe estar definida fuera de la clase para ser utilizada con multiprocessing.
    
    Args:
        start_timestamp (int): Timestamp de inicio en milisegundos
        symbol (str): Símbolo del par de trading
        interval (str): Intervalo de tiempo
        api_key (str, optional): API key de Binance
        api_secret (str, optional): API secret de Binance
        is_testnet (bool): Si usar testnet o producción
        api_call_limit (int): Límite de velas por llamada
        max_retries (int): Número máximo de reintentos
        retry_delay (float): Delay entre reintentos
        
    Returns:
        List: Lista de klines descargados
    """
    # Configurar logging para el worker
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(f"worker_{start_timestamp}")
    
    # Crear cliente de Binance específico para este worker
    try:
        if api_key and api_secret:
            if is_testnet:
                client = Client(api_key=api_key, api_secret=api_secret, testnet=True)
            else:
                client = Client(api_key=api_key, api_secret=api_secret)
        else:
            client = Client()
    except Exception as e:
        logger.warning(f"Error inicializando cliente, usando cliente público: {e}")
        client = Client()
    
    # Realizar descarga con reintentos
    retry_count = 0
    while retry_count <= max_retries:
        try:
            logger.info(f"Worker descargando desde timestamp: {start_timestamp}")
            
            klines = client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_timestamp,
                limit=api_call_limit
            )
            
            if not klines:
                logger.info(f"Worker no recibió datos para timestamp: {start_timestamp}")
                return []
            
            logger.info(f"Worker descargó {len(klines)} velas desde timestamp: {start_timestamp}")
            return klines
            
        except BinanceAPIException as e:
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Worker falló después de {max_retries} reintentos con API error: {e}")
                return []
            else:
                logger.warning(f"Worker API error (intento {retry_count}/{max_retries}): {e}")
                time.sleep(retry_delay * retry_count)
                
        except BinanceRequestException as e:
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Worker falló después de {max_retries} reintentos con request error: {e}")
                return []
            else:
                logger.warning(f"Worker request error (intento {retry_count}/{max_retries}): {e}")
                time.sleep(retry_delay * retry_count)
                
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Worker falló después de {max_retries} reintentos con error inesperado: {e}")
                return []
            else:
                logger.warning(f"Worker error inesperado (intento {retry_count}/{max_retries}): {e}")
                time.sleep(retry_delay * retry_count)
    
    return []


class Adquisicion:
    """Clase para adquirir y procesar datos OHLCV de Binance."""
    
    def __init__(self, symbol: str, interval: str, start_date: str, end_date: str = None, 
                 config_dict: dict = None, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Inicializa la clase de adquisición.
        
        Args:
            symbol (str): Símbolo del par de trading (ej: 'BTCUSDT')
            interval (str): Intervalo de tiempo (ej: '1h', '4h', '1d')
            start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'
            end_date (str, optional): Fecha de fin en formato 'YYYY-MM-DD'
            config_dict (dict): Diccionario con la configuración necesaria
            api_key (str, optional): API key de Binance
            api_secret (str, optional): API secret de Binance
        """
        self.symbol = symbol
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = []
        self.dataframe = None
        
        # Almacenar configuración inyectada
        self.config = config_dict or {}
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Inicializar cliente de Binance con API keys pasadas como argumentos
        try:
            if api_key and api_secret:
                is_testnet = self.config.get('api', {}).get('is_testnet', False)
                
                if is_testnet:
                    # Configurar cliente para testnet
                    self.client = Client(
                        api_key=api_key,
                        api_secret=api_secret,
                        testnet=True
                    )
                    self.logger.info("Cliente de Binance inicializado en modo TESTNET")
                else:
                    # Configurar cliente para producción
                    self.client = Client(
                        api_key=api_key,
                        api_secret=api_secret
                    )
                    self.logger.info("Cliente de Binance inicializado en modo PRODUCCIÓN")
            else:
                self.logger.info("No se proporcionaron API keys - Inicializando cliente solo para datos públicos")
                self.client = Client()
                
        except Exception as e:
            self.logger.warning(f"Error inicializando cliente de Binance: {e}")
            self.logger.info("Inicializando cliente sin API keys (solo datos públicos)")
            self.client = Client()
        
    def main(self) -> pd.DataFrame:
        """
        Orquesta todo el proceso de adquisición y procesamiento de datos.
        
        Returns:
            pd.DataFrame: DataFrame procesado con datos OHLCV limpios
        """
        self.logger.info(f"Iniciando adquisición de datos para {self.symbol}")
        
        # 1. Descargar datos de la API (secuencial por defecto, paralelo opcional)
        self._download_klines_from_api()
        
        # 2. Crear y estructurar DataFrame
        self._create_and_structure_dataframe()
        
        # 3. Eliminar duplicados
        self._remove_duplicates()
        
        # 4. Interpolar NaNs parciales
        self._interpolate_partial_nans()
        
        # 5. Reconstruir secuencia completa
        self._reconstruct_full_sequence()
        
        # 6. Interpolar velas faltantes
        self._reconstruct_missing_candles()
        
        # 7. Limpieza final de NaNs
        self._final_nan_cleanup()
        
        self.logger.info(f"Proceso completado. DataFrame final: {self.dataframe.shape}")
        return self.dataframe
    
    def main_parallel(self) -> pd.DataFrame:
        """
        Orquesta todo el proceso de adquisición y procesamiento de datos usando descarga paralela.
        
        Returns:
            pd.DataFrame: DataFrame procesado con datos OHLCV limpios
        """
        self.logger.info(f"Iniciando adquisición PARALELA de datos para {self.symbol}")
        
        # 1. Descargar datos de la API en paralelo
        self._download_klines_parallel()
        
        # 2. Crear y estructurar DataFrame
        self._create_and_structure_dataframe()
        
        # 3. Eliminar duplicados
        self._remove_duplicates()
        
        # 4. Interpolar NaNs parciales
        self._interpolate_partial_nans()
        
        # 5. Reconstruir secuencia completa
        self._reconstruct_full_sequence()
        
        # 6. Interpolar velas faltantes
        self._reconstruct_missing_candles()
        
        # 7. Limpieza final de NaNs
        self._final_nan_cleanup()
        
        self.logger.info(f"Proceso PARALELO completado. DataFrame final: {self.dataframe.shape}")
        return self.dataframe
    
    def _download_klines_from_api(self) -> None:
        """Descarga datos de velas de la API de Binance usando múltiples llamadas secuenciales."""
        self.logger.info("Descargando datos de la API de Binance...")
        
        # Convertir fecha de inicio a timestamp en milisegundos
        start_date_obj = datetime.strptime(self.start_date, '%Y-%m-%d')
        start_timestamp = int(start_date_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        # Calcular timestamp final
        if self.end_date is not None:
            # Si se especifica end_date, usarlo como límite
            end_date_obj = datetime.strptime(self.end_date, '%Y-%m-%d')
            end_timestamp = int(end_date_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
        else:
            # Si no se especifica end_date, usar el timestamp actual
            end_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        current_timestamp = end_timestamp
        
        all_klines = []
        current_start = start_timestamp
        call_count = 0
        
        end_date_display = self.end_date if self.end_date else "ahora"
        self.logger.info(f"Descargando datos desde {self.start_date} hasta {end_date_display}...")
        
        api_call_limit = self.config.get('api', {}).get('call_limit', 1000)
        self.logger.info(f"Límite por llamada: {api_call_limit} velas")
        
        while current_start < current_timestamp:
            call_count += 1
            retry_count = 0
            max_retries = self.config.get('api', {}).get('max_retries', 3)
            klines = []  # Inicializar klines para cada iteración
            
            while retry_count <= max_retries:
                try:
                    self.logger.info(f"Llamada #{call_count} - Descargando desde timestamp: {current_start}")
                    
                    # Hacer llamada a la API con límite configurado
                    klines = self.client.futures_klines(
                        symbol=self.symbol,
                        interval=self.interval,
                        startTime=current_start,
                        limit=api_call_limit
                    )
                    
                    if not klines:
                        self.logger.info("No se recibieron más datos de la API")
                        break
                    
                    # Añadir las velas recibidas a la lista total
                    all_klines.extend(klines)
                    
                    # Actualizar el timestamp de inicio para la siguiente llamada
                    # Usar el timestamp de cierre de la última vela + 1ms
                    last_close_time = int(klines[-1][6])  # close_time está en índice 6
                    current_start = last_close_time + 1
                    
                    self.logger.info(f"Descargadas {len(klines)} velas. Total acumulado: {len(all_klines)}")
                    
                    # Delay entre llamadas para respetar rate limits
                    retry_delay = self.config.get('api', {}).get('retry_delay', 1.0)
                    if retry_delay > 0:
                        time.sleep(retry_delay)
                    
                    break  # Salir del bucle de reintentos si fue exitoso
                    
                except BinanceAPIException as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        self.logger.error(f"Error de API de Binance después de {max_retries} reintentos: {e}")
                        raise
                    else:
                        self.logger.warning(f"Error de API (intento {retry_count}/{max_retries}): {e}")
                        time.sleep(retry_delay * retry_count)  # Delay incremental
                        
                except BinanceRequestException as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        self.logger.error(f"Error de petición a Binance después de {max_retries} reintentos: {e}")
                        raise
                    else:
                        self.logger.warning(f"Error de petición (intento {retry_count}/{max_retries}): {e}")
                        time.sleep(retry_delay * retry_count)  # Delay incremental
                        
                except Exception as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        self.logger.error(f"Error inesperado después de {max_retries} reintentos: {e}")
                        raise
                    else:
                        self.logger.warning(f"Error inesperado (intento {retry_count}/{max_retries}): {e}")
                        time.sleep(retry_delay * retry_count)  # Delay incremental
            
            # Si no obtuvimos datos en esta iteración, salir del bucle principal
            if not klines:
                break
                
            # Verificar si ya llegamos al presente para evitar llamadas innecesarias
            if len(klines) < api_call_limit:
                self.logger.info("Recibidas menos velas que el límite, probablemente llegamos al presente")
                break
        
        self.raw_data = all_klines
        self.logger.info(f"Descarga completada: {len(self.raw_data)} velas totales en {call_count} llamadas")
        
        if not self.raw_data:
            raise ValueError("No se pudieron descargar datos de la API de Binance")
    
    def _download_klines_parallel(self) -> None:
        """
        Descarga datos de velas de la API de Binance usando múltiples procesos en paralelo.
        Esta implementación divide el rango de tiempo en trozos y los descarga simultáneamente.
        """
        self.logger.info("Descargando datos de la API de Binance en PARALELO...")
        
        # Convertir fecha de inicio a timestamp en milisegundos
        start_date_obj = datetime.strptime(self.start_date, '%Y-%m-%d')
        start_timestamp = int(start_date_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        # Calcular timestamp final
        if self.end_date is not None:
            # Si se especifica end_date, usarlo como límite
            end_date_obj = datetime.strptime(self.end_date, '%Y-%m-%d')
            end_timestamp = int(end_date_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
        else:
            # Si no se especifica end_date, usar el timestamp actual
            end_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        current_timestamp = end_timestamp
        
        # Calcular el intervalo en milisegundos para determinar el tamaño de los trozos
        interval_ms = self._get_interval_in_ms()
        api_call_limit = self.config.get('api', {}).get('call_limit', 1000)
        chunk_size_ms = interval_ms * api_call_limit  # Cada trozo será de api_call_limit velas
        
        # Crear lista de timestamps de inicio para cada trozo
        start_timestamps = []
        current_start = start_timestamp
        
        while current_start < current_timestamp:
            start_timestamps.append(current_start)
            current_start += chunk_size_ms
        
        self.logger.info(f"Dividiendo descarga en {len(start_timestamps)} trozos")
        end_date_display = self.end_date if self.end_date else "ahora"
        self.logger.info(f"Rango total: {self.start_date} hasta {end_date_display}")
        
        # Determinar número de procesos (usar CPU count pero limitado para no sobrecargar la API)
        max_workers = self.config.get('system', {}).get('max_parallel_workers_api', 8)
        num_workers = min(cpu_count(), len(start_timestamps), max_workers)
        self.logger.info(f"Usando {num_workers} procesos paralelos (máximo configurado: {max_workers})")
        
        # Preparar parámetros para la función worker
        # En este método, no tenemos acceso directo a API keys desde config
        # Se necesitarían pasar como parámetros al método o usar otra estrategia
        api_key = None  # Por seguridad, el multiprocessing no usa API keys
        api_secret = None
        
        # Obtener parámetros de configuración
        is_testnet = self.config.get('api', {}).get('is_testnet', False)
        max_retries = self.config.get('api', {}).get('max_retries', 3)
        retry_delay = self.config.get('api', {}).get('retry_delay', 1.0)
        
        # Crear función worker con parámetros fijos usando partial
        worker_task = partial(
            _download_kline_chunk,
            symbol=self.symbol,
            interval=self.interval,
            api_key=api_key,
            api_secret=api_secret,
            is_testnet=is_testnet,
            api_call_limit=api_call_limit,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        
        # Ejecutar descarga paralela
        all_klines = []
        try:
            with Pool(processes=num_workers) as pool:
                self.logger.info("Iniciando descarga paralela...")
                
                # Distribuir trabajo entre procesos
                results = pool.map(worker_task, start_timestamps)
                
                # Aplanar resultados (cada resultado es una lista de klines)
                for chunk_klines in results:
                    if chunk_klines:  # Solo agregar si el chunk no está vacío
                        all_klines.extend(chunk_klines)
                
        except Exception as e:
            self.logger.error(f"Error durante descarga paralela: {e}")
            self.logger.info("Fallback a descarga secuencial...")
            self._download_klines_from_api()
            return
        
        if not all_klines:
            raise ValueError("No se pudieron descargar datos usando descarga paralela")
        
        # Ordenar por timestamp de apertura para asegurar orden cronológico
        self.logger.info("Ordenando datos por timestamp...")
        all_klines.sort(key=lambda x: int(x[0]))  # timestamp está en índice 0
        
        # Eliminar duplicados que puedan haber surgido por solapamiento entre trozos
        self.logger.info("Eliminando posibles duplicados de la descarga paralela...")
        seen_timestamps = set()
        unique_klines = []
        
        for kline in all_klines:
            timestamp = int(kline[0])
            if timestamp not in seen_timestamps:
                seen_timestamps.add(timestamp)
                unique_klines.append(kline)
        
        duplicates_removed = len(all_klines) - len(unique_klines)
        if duplicates_removed > 0:
            self.logger.info(f"Eliminados {duplicates_removed} duplicados de la descarga paralela")
        
        self.raw_data = unique_klines
        self.logger.info(f"Descarga paralela completada: {len(self.raw_data)} velas únicas")
    
    def _get_interval_in_ms(self) -> int:
        """
        Convierte el intervalo de string a milisegundos.
        
        Returns:
            int: Intervalo en milisegundos
        """
        interval_map = {
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
            '1M': 30 * 24 * 60 * 60 * 1000  # Aproximación para 1 mes
        }
        
        if self.interval not in interval_map:
            raise ValueError(f"Intervalo no soportado: {self.interval}")
        
        return interval_map[self.interval]
    
    def _create_and_structure_dataframe(self) -> None:
        """Convierte los datos crudos en un DataFrame estructurado."""
        self.logger.info("Creando y estructurando DataFrame...")
        
        # Crear DataFrame con las columnas de Binance
        columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 
                  'close_time', 'quote_asset_volume', 'number_of_trades',
                  'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
        
        df = pd.DataFrame(self.raw_data, columns=columns)
        
        # Seleccionar solo las columnas OHLCV necesarias
        ohlcv_columns = self.config.get('data', {}).get('ohlcv_columns', ['Open', 'High', 'Low', 'Close', 'Volume'])
        df = df[['timestamp'] + ohlcv_columns]
        
        # Convertir timestamp a datetime con zona horaria
        target_timezone = self.config.get('data', {}).get('target_timezone', 'Europe/Madrid')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['timestamp'] = df['timestamp'].dt.tz_convert(target_timezone)
        
        # Establecer timestamp como índice
        df.set_index('timestamp', inplace=True)
        
        # Convertir columnas a tipos eficientes para RAM
        data_dtypes = self.config.get('data', {}).get('data_dtypes', {})
        for col in ohlcv_columns:
            if col in data_dtypes:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(data_dtypes[col])
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.dataframe = df
        self.logger.info(f"DataFrame creado con {len(df)} filas y {len(df.columns)} columnas")
    
    def _remove_duplicates(self) -> None:
        """Elimina filas con timestamps duplicados, conservando la primera aparición."""
        self.logger.info("Eliminando duplicados...")
        
        initial_count = len(self.dataframe)
        self.dataframe = self.dataframe[~self.dataframe.index.duplicated(keep='first')]
        final_count = len(self.dataframe)
        
        duplicates_removed = initial_count - final_count
        if duplicates_removed > 0:
            self.logger.info(f"Eliminados {duplicates_removed} duplicados")
        else:
            self.logger.info("No se encontraron duplicados")
    
    def _interpolate_partial_nans(self) -> None:
        """Identifica y rellena valores NaN esporádicos dentro de velas existentes."""
        self.logger.info("Interpolando NaNs parciales...")
        
        # Contar NaNs antes de interpolar
        nan_count_before = self.dataframe.isnull().sum().sum()
        
        # Interpolar usando el método configurado
        interpolation_method = self.config.get('data', {}).get('interpolation_method', 'linear')
        interpolation_limit_direction = self.config.get('data', {}).get('interpolation_limit_direction', 'forward')
        
        self.dataframe = self.dataframe.interpolate(
            method=interpolation_method,
            limit_direction=interpolation_limit_direction
        )
        
        nan_count_after = self.dataframe.isnull().sum().sum()
        interpolated = nan_count_before - nan_count_after
        
        if interpolated > 0:
            self.logger.info(f"Interpolados {interpolated} valores NaN parciales")
        else:
            self.logger.info("No se encontraron NaNs parciales para interpolar")
    
    def _reconstruct_full_sequence(self) -> None:
        """Asegura que la serie temporal tenga una entrada para cada intervalo esperado."""
        self.logger.info("Reconstruyendo secuencia completa...")
        
        # Determinar la frecuencia basada en el intervalo
        freq_map = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
        }
        
        if self.interval not in freq_map:
            raise ValueError(f"Intervalo no soportado: {self.interval}")
        
        freq = freq_map[self.interval]
        
        # Crear rango completo de timestamps
        start_time = self.dataframe.index.min()
        end_time = self.dataframe.index.max()
        target_timezone = self.config.get('data', {}).get('target_timezone', 'Europe/Madrid')
        complete_range = pd.date_range(start=start_time, end=end_time, freq=freq, tz=target_timezone)
        
        # Reindexar con el rango completo
        initial_count = len(self.dataframe)
        self.dataframe = self.dataframe.reindex(complete_range)
        final_count = len(self.dataframe)
        
        missing_candles = final_count - initial_count
        if missing_candles > 0:
            self.logger.info(f"Agregadas {missing_candles} filas para velas faltantes")
        else:
            self.logger.info("Secuencia temporal ya estaba completa")
    
    def _reconstruct_missing_candles(self) -> None:
        """Interpola datos para las velas que estaban completamente ausentes."""
        self.logger.info("Interpolando velas faltantes...")
        
        # Contar NaNs antes de interpolar
        nan_count_before = self.dataframe.isnull().sum().sum()
        
        # Interpolar las filas completamente NaN
        interpolation_method = self.config.get('data', {}).get('interpolation_method', 'linear')
        interpolation_limit_direction = self.config.get('data', {}).get('interpolation_limit_direction', 'forward')
        
        self.dataframe = self.dataframe.interpolate(
            method=interpolation_method,
            limit_direction=interpolation_limit_direction
        )
        
        nan_count_after = self.dataframe.isnull().sum().sum()
        interpolated = nan_count_before - nan_count_after
        
        if interpolated > 0:
            self.logger.info(f"Interpolados {interpolated} valores en velas faltantes")
        else:
            self.logger.info("No se encontraron velas faltantes para interpolar")
    
    def _final_nan_cleanup(self) -> None:
        """Aplica limpieza final de NaNs usando forward fill y backward fill."""
        self.logger.info("Aplicando limpieza final de NaNs...")
        
        # Contar NaNs antes de la limpieza
        nan_count_before = self.dataframe.isnull().sum().sum()
        
        # Aplicar forward fill seguido de backward fill
        self.dataframe = self.dataframe.ffill().bfill()
        
        nan_count_after = self.dataframe.isnull().sum().sum()
        cleaned = nan_count_before - nan_count_after
        
        if cleaned > 0:
            self.logger.info(f"Limpiados {cleaned} NaNs finales")
        
        if nan_count_after > 0:
            self.logger.warning(f"Aún quedan {nan_count_after} NaNs después de la limpieza")
        else:
            self.logger.info("DataFrame completamente limpio, sin NaNs")
    