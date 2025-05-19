¡Hola Pedro\! Es un placer colaborar contigo en este ambicioso proyecto. Tu enfoque metódico y tu sólida formación son una excelente base para construir un bot de trading de Bitcoin de última generación.

Comencemos por el **Módulo 1: Adquisición de Datos en Tiempo Real e Históricos de Binance**, tal como has propuesto. Tus ideas iniciales sobre la estructura y la gestión de la configuración son muy acertadas y sientan las bases para un sistema robusto y mantenible.

Vamos a desglosar este módulo en detalle:

**Módulo 1: Adquisición de Datos de Binance**

**1. Responsabilidades y Funcionalidades Exactas:**

  * **Conexión Segura y Autenticada:** Establecer y gestionar la conexión con la API de Binance, específicamente para los datos de futuros (ya que operas con BTC/USDT futuros). Esto incluye el manejo seguro de las claves API.
  * **Descarga de Datos Históricos (OHLCV):**
      * Obtener datos de velas (Open, High, Low, Close, Volume) para el par BTC/USDT (configurable) del mercado de futuros de Binance.
      * Permitir la especificación de una fecha de inicio (`START_DATE`) configurable.
      * La fecha de finalización será siempre la fecha y hora actuales (o la última vela completa disponible), calculada automáticamente por el sistema.
      * Manejar la paginación de la API de Binance para recuperar grandes conjuntos de datos históricos (Binance limita la cantidad de datos por solicitud, típicamente a 500-1500 velas).
      * Permitir la configuración de la granularidad (intervalo) de las velas (ej. "1m", "5m", "1h", "4h", "1d").
  * **Manejo de Errores y Límites de la API:**
      * Implementar una lógica robusta para gestionar errores de conexión, timeouts, y especialmente los límites de tasa (`rate limits`) impuestos por la API de Binance.
      * Incluir estrategias de reintento (ej. `exponential backoff`) para manejar fallos temporales o alcanzar los `rate limits`.
  * **Validación de Datos Descargados:** Realizar verificaciones básicas para asegurar la integridad de los datos (ej. formatos correctos, ausencia de `NaNs` inesperados en columnas críticas, orden cronológico).
  * **Almacenamiento de Datos Crudos:**
      * Guardar los datos históricos descargados en formato CSV en la ruta especificada (`data/raw/`).
      * Utilizar una convención de nomenclatura clara y estandarizada para los archivos (ej. `BTCUSDT_FUTURES_1h_20200101_20250516.csv`).
  * **Gestión de Configuración:** Cargar todos los parámetros necesarios (claves API, rutas, símbolo, intervalo, fecha de inicio, etc.) desde variables de entorno (`.env`) y un archivo de configuración (`config.yaml`) a través de un módulo centralizado (`src/utils/config.py`).
  * **Logging Detallado:** Registrar todas las operaciones significativas, advertencias, errores y el progreso de la descarga para facilitar la depuración y el seguimiento.

**2. Entradas que Recibirá y Salidas que Producirá:**

  * **Entradas (Principalmente desde `config.py` que lee `.env` y `config.yaml`):**
      * `BINANCE_API_KEY_FUTURES`: Clave API para futuros de Binance (desde `.env`).
      * `BINANCE_API_SECRET_FUTURES`: Clave secreta API para futuros de Binance (desde `.env`).
      * `DATA_RAW_PATH`: Ruta para guardar los archivos CSV (ej. "data/raw/", desde `config.yaml` o `.env`).
      * `SYMBOL`: Par de trading de futuros (ej. "BTCUSDT", desde `config.yaml`).
      * `INTERVAL`: Granularidad de las velas (ej. "1h", desde `config.yaml`).
      * `HISTORICAL_START_DATE`: Fecha de inicio para la descarga de datos (ej. "2020-01-01", desde `config.yaml`).
      * Parámetros de control de la API: `API_REQUEST_LIMIT` (ej. 1000 velas por solicitud), `API_REQUEST_DELAY` (segundos entre solicitudes), `API_RETRY_ATTEMPTS`, `API_RETRY_DELAY` (desde `config.yaml`).
  * **Salidas:**
      * Archivos CSV conteniendo los datos OHLCV históricos, guardados en `data/raw/` con un nombre estandarizado. Cada fila representará una vela. Columnas típicas: `Open_Time` (timestamp), `Open`, `High`, `Low`, `Close`, `Volume`.
      * Mensajes de log detallando el proceso, incluyendo cualquier error o advertencia.
      * (Opcional) Un resumen o estado de la descarga al finalizar (ej. número de velas descargadas, rango de fechas cubierto).

**3. Interacciones y Dependencias con Otros Módulos:**

  * **Módulo de Configuración (`src/utils/config.py`):** Este módulo de adquisición de datos dependerá directamente del módulo de configuración para obtener todos sus parámetros operativos, incluyendo credenciales, rutas y parámetros de la API.
  * **Orquestador de Scripts (`raiz/scripts/download_data.py`):** El script orquestador en `raiz/scripts/` importará y utilizará la clase principal de este módulo (ej. `BinanceFuturesDownloader`) para iniciar y gestionar el proceso de descarga.
  * **Módulo de Preprocesamiento de Datos (Siguiente Módulo):** Los archivos CSV generados por este módulo serán la entrada principal para el módulo de preprocesamiento, que limpiará, transformará y posiblemente enriquecerá estos datos crudos.
  * **Módulo de Logging Centralizado (Potencial):** Si se implementa un sistema de logging más avanzado para todo el proyecto, este módulo se integrará con él.

**4. Tecnologías, Librerías o Frameworks Específicos:**

  * **Lenguaje de Programación:** Python 3.x.
  * **Interacción con API de Binance:**
      * **`python-binance`:** Librería popular y bien mantenida para interactuar con la API de Binance. Es importante asegurarse de que se utiliza la funcionalidad específica para la API de Futuros (puede ser un cliente diferente como `FuturesClient` o métodos específicos dentro del `Client` general, como `futures_historical_klines`).
  * **Gestión de Variables de Entorno:**
      * **`python-dotenv`:** Para cargar variables desde el archivo `.env` en el entorno de ejecución.
  * **Gestión de Archivos de Configuración:**
      * **`PyYAML`:** Para leer y parsear el archivo `config.yaml`.
  * **Manipulación de Datos:**
      * **`pandas`:** Fundamental para estructurar los datos OHLCV en DataFrames, realizar conversiones de tipos (ej. timestamps a datetime, números a float) y guardar los datos en formato CSV.
  * **Manejo de Fechas y Horas:**
      * **`datetime` (módulo estándar de Python):** Para gestionar las fechas de inicio/fin, convertir timestamps, y realizar cálculos de tiempo.
      * **`pytz` (opcional pero recomendado):** Para un manejo explícito y correcto de las zonas horarias (Binance opera en UTC, por lo que estandarizar a UTC es una buena práctica).
  * **Logging:**
      * **`logging` (módulo estándar de Python):** Para implementar un sistema de logging configurable y robusto.
  * **Manejo de Solicitudes HTTP (si no se usa `python-binance` directamente o para funcionalidades avanzadas):**
      * **`requests`:** Para realizar llamadas HTTP directas si fuera necesario.

**5. Métricas Clave para Evaluar su Rendimiento:**

  * **Tiempo Total de Descarga:** El tiempo que tarda en completarse una descarga histórica para un rango de fechas y una granularidad dados.
  * **Velocidad de Descarga:** Velas descargadas por unidad de tiempo (ej. velas/segundo o velas/minuto).
  * **Número de Solicitudes a la API:** Para monitorizar la eficiencia y el respeto a los `rate limits`.
  * **Número de Errores y Reintentos:** Indica la robustez de la conexión y el manejo de errores.
  * **Completitud de los Datos:** Verificar que no haya lagunas inesperadas en los datos descargados (comparar el número de velas obtenidas con el esperado para el periodo).
  * **Tamaño de los Datos Almacenados:** Para tener una idea del consumo de almacenamiento.

**Estructura de Código Propuesta (basada en tus ideas):**

  * **`.env` (en la raíz del proyecto):**

    ```
    BINANCE_API_KEY_FUTURES="TU_API_KEY_FUTURES"
    BINANCE_API_SECRET_FUTURES="TU_API_SECRET_FUTURES"
    PYTHONPATH="${PYTHONPATH}:./src"
    ```

  * **`src/config.yaml`:**

    ```yaml
    data_paths:
      raw: "data/raw"
      processed: "data/processed"
      # ... otros paths

    binance_api:
      futures_base_url: "https://fapi.binance.com" # O el que corresponda
      request_limit_per_call: 1000 # Velas por solicitud
      request_delay_seconds: 0.5 # Pausa entre solicitudes
      retry_attempts: 5
      retry_delay_seconds: 60

    data_acquisition_defaults:
      symbol: "BTCUSDT"
      interval: "1h" # 1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
      historical_start_date: "2020-01-01" # YYYY-MM-DD
    ```

  * **`src/utils/config.py`:**

    ```python
    import yaml
    import os
    from dotenv import load_dotenv

    class ConfigManager:
        _instance = None

        def __new__(cls, config_path="src/config.yaml", env_path=".env"):
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                load_dotenv(dotenv_path=env_path)
                try:
                    with open(config_path, 'r') as f:
                        cls._instance.config = yaml.safe_load(f)
                except FileNotFoundError:
                    # logger.error(f"Archivo de configuración {config_path} no encontrado.") # Necesitarías un logger aquí o lanzar excepción
                    raise FileNotFoundError(f"Archivo de configuración {config_path} no encontrado.")
                except yaml.YAMLError as e:
                    # logger.error(f"Error al parsear el archivo YAML {config_path}: {e}")
                    raise yaml.YAMLError(f"Error al parsear el archivo YAML {config_path}: {e}")
            return cls._instance

        def get_env_variable(self, var_name: str, default=None):
            return os.getenv(var_name, default)

        def get_config_value(self, key_path: str, default=None):
            try:
                keys = key_path.split('.')
                value = self.config
                for key in keys:
                    value = value[key]
                return value
            except KeyError:
                # logger.warning(f"Clave de configuración '{key_path}' no encontrada. Usando default: {default}")
                return default
            except TypeError: # En caso de que self.config no se haya cargado
                 # logger.error(f"Configuración no cargada. Imposible obtener '{key_path}'.")
                 raise TypeError(f"Configuración no cargada. Imposible obtener '{key_path}'.")


    # Ejemplo de cómo podrías usarlo para obtener valores específicos de forma más estructurada
    # class AppConfig:
    #     def __init__(self, config_manager: ConfigManager):
    #         self.raw_data_path = config_manager.get_config_value('data_paths.raw')
    #         self.api_key = config_manager.get_env_variable('BINANCE_API_KEY_FUTURES')
    #         # ... y así sucesivamente

    if __name__ == '__main__': # Para pruebas rápidas
        manager = ConfigManager(config_path='../../src/config.yaml', env_path='../../.env') # Ajusta paths si ejecutas directo
        print(f"Raw Data Path: {manager.get_config_value('data_paths.raw')}")
        print(f"API Key: {manager.get_env_variable('BINANCE_API_KEY_FUTURES')}")
        print(f"Default Symbol: {manager.get_config_value('data_acquisition_defaults.symbol')}")
    ```

  * **`src/data/binance_futures_downloader.py` (Clase principal del módulo):**

    ```python
    import pandas as pd
    from binance.client import Client # Asegúrate que es el cliente correcto para futuros o usa uno específico.
                                  # from binance.futures import Futures as Client # Ejemplo, verifica la lib.
    from datetime import datetime, timezone
    import time
    import logging
    import os
    # from src.utils.config import ConfigManager # Importarías tu ConfigManager

    logger = logging.getLogger(__name__) # Configurar el logger a nivel de script o aplicación

    class BinanceFuturesDownloader:
        def __init__(self, config_manager): # ConfigManager debería ser inyectado
            self.config_manager = config_manager
            self.api_key = self.config_manager.get_env_variable('BINANCE_API_KEY_FUTURES')
            self.api_secret = self.config_manager.get_env_variable('BINANCE_API_SECRET_FUTURES')
            
            if not self.api_key or not self.api_secret:
                logger.error("API Key o Secret de Binance Futuros no configuradas en .env")
                raise ValueError("API Key o Secret de Binance Futuros no configuradas.")

            # Es crucial verificar la documentación de python-binance para el cliente de futuros
            # y si se requiere un endpoint específico para futuros.
            # self.client = Client(self.api_key, self.api_secret, tld='com', testnet=False) # Para spot
            # Para futuros, podría ser algo como:
            # self.client = Client(self.api_key, self.api_secret, base_url="https://fapi.binance.com") # o similar
            # O un cliente específico: from binance.um_futures import UMFutures
            # self.client = UMFutures(key=self.api_key, secret=self.api_secret)
            
            # Usaremos UMFutures como ejemplo, DEBES verificar la librería que uses.
            try:
                from binance.um_futures import UMFutures
                self.client = UMFutures(key=self.api_key, secret=self.api_secret)
                self.client.ping() # Verificar conexión
                logger.info("Conexión con Binance UMFutures API establecida exitosamente.")
            except Exception as e:
                logger.error(f"Error al conectar con Binance UMFutures API: {e}")
                raise ConnectionError(f"Error al conectar con Binance UMFutures API: {e}")


            self.raw_data_path = self.config_manager.get_config_value('data_paths.raw')
            self.request_limit = self.config_manager.get_config_value('binance_api.request_limit_per_call', 1000)
            self.request_delay = self.config_manager.get_config_value('binance_api.request_delay_seconds', 0.5)
            self.retry_attempts = self.config_manager.get_config_value('binance_api.retry_attempts', 5)
            self.retry_delay = self.config_manager.get_config_value('binance_api.retry_delay_seconds', 60)

            os.makedirs(self.raw_data_path, exist_ok=True)


        def _interval_to_milliseconds(self, interval_str: str) -> int:
            # Función auxiliar para convertir el intervalo a milisegundos si es necesario por la API
            # (Algunas funciones de la API toman el intervalo como string directamente)
            # Ejemplo: 1m = 60000 ms
            # Esta función puede no ser necesaria si la librería maneja los strings de intervalo
            pass 

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
                    
                    # Usar el método correcto para futuros históricos, ej. klines o historical_klines
                    # Para UMFutures: self.client.klines(...) o self.client.continuous_klines(...)
                    # o self.client.historical_klines(...) <- preferido si existe y maneja bien la paginación
                    # La función get_historical_klines suele ser más de alto nivel y maneja paginación
                    # pero para futuros, podría ser client.futures_historical_klines o similar.
                    # Vamos a asumir un método como el de Spot/UMFutures: client.klines
                    # Necesitas verificar la documentación de la librería para el método correcto de UMFutures.
                    # `klines` para UMFutures toma startTime, endTime, limit.
                    
                    klines = self.client.klines(
                        symbol=symbol,
                        interval=interval,
                        startTime=fetch_start_time,
                        endTime=end_timestamp_ms, # Binance puede ignorar endTime si startTime + limit < endTime
                        limit=self.request_limit 
                    )
                    # Ejemplo alternativo si existe un método más directo para datos históricos de futuros:
                    # klines = self.client.futures_historical_klines(symbol, interval, fetch_start_time, limit=self.request_limit)

                    if not klines:
                        logger.info("No se encontraron más datos para el período o se alcanzó el límite de datos de Binance.")
                        break 

                    all_klines_data.extend(klines)
                    
                    # Actualizar el timestamp de inicio para la siguiente solicitud
                    # El timestamp de la última vela recibida + 1ms (o + duración del intervalo)
                    last_kline_open_time = klines[-1][0]
                    fetch_start_time = last_kline_open_time + 1 # Siguiente vela después de la última recibida
                                                            # O podrías usar la duración del intervalo:
                                                            # fetch_start_time = last_kline_open_time + self._interval_to_milliseconds(interval)

                    if fetch_start_time >= end_timestamp_ms:
                        logger.info("Se alcanzó la fecha/hora de finalización.")
                        break
                    
                    attempts = 0 # Resetear intentos si la solicitud fue exitosa
                    time.sleep(self.request_delay) # Respetar los límites de la API

                except Exception as e: # Ser más específico con las excepciones de Binance (ej. BinanceAPIException)
                    attempts += 1
                    logger.warning(f"Error obteniendo datos para {symbol}: {e}. Intento {attempts}/{self.retry_attempts}.")
                    if attempts >= self.retry_attempts:
                        logger.error(f"Máximo número de reintentos alcanzado para {symbol}. Abortando descarga para este batch.")
                        # Aquí podrías decidir si continuar con el siguiente batch o parar todo.
                        # Por ahora, paramos. Se podría implementar guardado parcial.
                        return 
                    time.sleep(self.retry_delay * attempts) # Exponential backoff

            if not all_klines_data:
                logger.warning(f"No se descargaron datos para {symbol} en el rango especificado.")
                return

            # Definir columnas según la documentación de la API de Futuros
            # Para UMFutures klines:
            # [open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades,
            #  taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]
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

            # Guardar en CSV
            output_filename = self._generate_filename(symbol, interval, start_dt, current_time_utc)
            try:
                df.to_csv(output_filename, index=False)
                logger.info(f"Datos para {symbol} ({len(df)} velas) guardados exitosamente en: {output_filename}")
            except IOError as e:
                logger.error(f"Error al guardar el archivo CSV {output_filename}: {e}")
    ```

  * **`raiz/scripts/download_data.py` (Orquestador):**

    ```python
    import logging
    import sys
    # Asegurarse de que src está en el PYTHONPATH
    # Esto se puede manejar con la variable de entorno PYTHONPATH definida en .env
    # o añadiendo explícitamente al path:
    # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


    from src.utils.config import ConfigManager
    from src.data.binance_futures_downloader import BinanceFuturesDownloader

    # Configuración básica del logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout) # Para ver logs en consola
            # logging.FileHandler("download.log") # Para guardar logs en un archivo
        ]
    )
    logger = logging.getLogger(__name__)

    def main():
        logger.info("Iniciando script de descarga de datos históricos.")
        
        try:
            # Rutas relativas al script actual si config.yaml y .env están en la raíz del proyecto
            # o ajustar según la estructura final
            config_manager = ConfigManager(config_path="src/config.yaml", env_path=".env")
        except Exception as e:
            logger.error(f"Error al cargar la configuración: {e}")
            return

        try:
            downloader = BinanceFuturesDownloader(config_manager)
        except Exception as e:
            logger.error(f"Error al inicializar BinanceFuturesDownloader: {e}")
            return

        symbol = config_manager.get_config_value('data_acquisition_defaults.symbol', "BTCUSDT")
        interval = config_manager.get_config_value('data_acquisition_defaults.interval', "1h")
        start_date = config_manager.get_config_value('data_acquisition_defaults.historical_start_date', "2020-01-01")

        try:
            downloader.fetch_historical_data(symbol, interval, start_date)
            logger.info("Proceso de descarga de datos finalizado.")
        except Exception as e:
            logger.error(f"Ocurrió un error durante el proceso de descarga: {e}", exc_info=True)

    if __name__ == "__main__":
        main()
    ```

**Puntos Clave de esta Propuesta y Discusión Adicional:**

1.  **Cliente de Futuros de Binance:** He usado `UMFutures` como ejemplo, pero es **CRUCIAL** que verifiques la documentación de la versión de `python-binance` que estés utilizando para asegurarte de que usas el cliente y los métodos correctos para los datos de futuros (UM o COIN-M según necesites, BTCUSDT es un perpetuo de UM). El endpoint base y los métodos específicos (`klines`, `historical_klines`, `continuous_klines` etc.) pueden variar.
2.  **Manejo del `startTime` en el Bucle:** La lógica para actualizar `Workspace_start_time` (usando `klines[-1][0] + 1`) es una forma común de paginar. Asegúrate de que el timestamp `klines[-1][0]` es el de apertura de la vela y que la API espera el siguiente `startTime` de esta forma.
3.  **Robustez:** El manejo de errores y reintentos es básico. Se podría mejorar con una librería como `tenacity` para reintentos más sofisticados. También, para descargas muy largas, se podría considerar guardar datos en trozos temporales y luego unirlos.
4.  **Formato de Archivo CSV vs. Parquet:** Mantengo CSV según tu propuesta. Si los volúmenes de datos crecen mucho (ej. datos de 1 minuto durante varios años), Parquet ofrecerá un rendimiento de I/O y compresión significativamente mejores. El cambio en el código sería mínimo (`df.to_parquet(...)`).
5.  **Atomicidad y Actualizaciones Incrementales:** Por ahora, la lógica descarga todo el rango desde `start_date` hasta "ahora". Si este script se ejecuta regularmente, volverá a descargar muchos datos. Para actualizaciones incrementales, necesitarías una lógica para determinar la última fecha descargada y empezar desde ahí. Esto podría ser una mejora futura.
6.  **Timestamp de `END_DATE` en el Nombre de Archivo:** He incluido `%H%M` en el formato de `end_str` para el nombre del archivo, ya que "ahora" cambia. Esto ayuda a versionar las descargas si se ejecutan varias veces.
7.  **Configuración de `PYTHONPATH`:** Para que los imports como `from src.utils.config import ConfigManager` funcionen desde `raiz/scripts/`, `src` debe estar en `PYTHONPATH`. Incluir `PYTHONPATH="${PYTHONPATH}:./src"` en tu `.env` y usar `python-dotenv` es una buena manera de manejarlo. Alternativamente, puedes manipular `sys.path` al inicio de tus scripts en `raiz/scripts/`.

Este es un desglose detallado del primer módulo. Estoy listo para tus comentarios, ajustes, o si prefieres, podemos avanzar al siguiente módulo (Preprocesamiento de Datos), donde empezaremos a preparar estos datos crudos para tu modelo de Reinforcement Learning.