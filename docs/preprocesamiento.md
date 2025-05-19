Pedro, la definición que has proporcionado para el **Módulo 2: Preprocesamiento y Gestión de Datos** es excepcionalmente clara y detallada. Cubre todos los aspectos cruciales, desde la carga de datos hasta la ingeniería de características y la preparación de secuencias para el Transformer. Tu visión sobre la normalización causal y la estructura de las características es muy sólida.

Procederemos a definir este módulo basándonos en tus especificaciones, integrando las mejores prácticas y las tecnologías que has mencionado.

**Módulo 2: Preprocesamiento y Gestión de Datos**

Este módulo se encargará de transformar los datos crudos de mercado en un formato estructurado y normalizado, listo para ser consumido por los modelos de aprendizaje y el framework de backtesting.

**1. Responsabilidades y Funcionalidades Exactas:**

  * **Carga de Datos Crudos:**
      * Leer los datos históricos OHLCV (en formato CSV, según Módulo 1) desde la ruta especificada (`data/raw/`).
      * Manejar la conversión de tipos de datos (ej. timestamps, valores numéricos).
  * **Limpieza de Datos:**
      * Verificar y manejar valores faltantes (NaN) que pudieran existir en los datos crudos (aunque la API de Binance suele proveer datos completos). Se puede optar por `ffill` para datos faltantes aislados o eliminar filas si el problema es mayor (con la debida notificación).
      * Identificación y manejo de outliers (opcional y con extrema cautela): Para datos financieros, los outliers pueden ser eventos de mercado genuinos. Si se implementa, se usarían técnicas robustas como el `winsorizing` limitado a un número razonable de desviaciones estándar, aplicadas de forma causal.
  * **Ingeniería de Características de Mercado:**
      * **Procesamiento de Klines OHLCV:** Calcular las 5 características basadas en retornos logarítmicos que especificaste:
          * `log_ret(C/O)`
          * `log_ret(H/O)`
          * `log_ret(L/O)`
          * `log_ret(C/C_prev)`
          * `log_ret(Vol/SMA(Vol,20))`
      * **Cálculo de Indicadores Técnicos:** Implementar el cálculo de los 15 indicadores técnicos con sus respectivos periodos configurables:
          * SMA(20), SMA(50)
          * EMA(12), EMA(26)
          * RSI(14)
          * ATR(14)
          * MACD(12,26,9): Línea MACD, Línea Señal, Histograma.
          * Bandas de Bollinger(20,2): Banda Superior, Banda Media, Banda Inferior. De estas se derivarán: Distancia a Superior, Distancia a Inferior, Ancho de Bandas.
          * CCI(20)
          * Oscilador Estocástico(14,3,3): %K lento, %D.
  * **Normalización/Escalado Causal de Características de Mercado:**
      * **Z-score sobre Ventana Móvil:** Aplicar a características que no tienen un rango intrínseco o cuya magnitud puede variar considerablemente (ej. precios transformados, algunas diferencias de indicadores). La ventana será de `L * normalization_window_multiplier` (ej. `96 * 2 = 192` periodos).
          * $z = (x\_t - \\text{mean}(x\_{t-W : t-1})) / \\text{std}(x\_{t-W : t-1})$
      * **Escalado Específico para Indicadores con Rango Definido:**
          * **RSI(14):** Escalar a `[0, 1]` (dividiendo por 100) o `[-1, 1]` (aplicando `(RSI - 50) / 50`). Optaremos por `[0, 1]` por simplicidad inicial.
          * **Oscilador Estocástico(14,3,3) %K, %D:** Escalar a `[0, 1]` (dividiendo por 100).
      * **Normalización Relativa para Indicadores de Nivel/Precio (SMAs, EMAs, Bandas de Bollinger):**
          * En lugar de un Z-score directo, normalizarlos en relación con el precio actual (ej. `(Indicador - Close) / Close` o `(Indicador - Close) / ATR`) o simplemente `Indicador / Close - 1`. Esto los convierte en una desviación porcentual o relativa.
          * **Ancho de Bandas de Bollinger:** Normalizar dividiendo por la banda media (SMA) o por el ATR. `Ancho_BB / SMA_BB` o `Ancho_BB / ATR`.
      * **ATR:** A menudo se utiliza para normalizar otras características o se normaliza dividiéndolo por el precio de cierre (`ATR / Close`).
      * **MACD, CCI:** Pueden normalizarse con Z-score móvil o, si su rango es relativamente estable, por un factor empírico o por ATR.
      * El objetivo es que todas las características finales tengan escalas comparables y comportamientos estacionarios en la medida de lo posible.
  * **Construcción de Secuencias de Estado (Solo Características de Mercado):**
      * Transformar el DataFrame de series temporales (con las `N_features` de mercado calculadas y normalizadas por KLine) en secuencias tridimensionales de forma `(N_samples, L, N_features_mercado)`.
      * `L` (Longitud de la secuencia): Configurable, por defecto 96.
      * `N_features_mercado`: Total de características de mercado (5 OHLCV procesados + 15 Indicadores = 20 features).
      * Utilizar una ventana deslizante para generar las secuencias.
      * Asegurar que no haya `lookahead bias`: todos los cálculos (indicadores, normalización) deben usar solo datos disponibles hasta el paso temporal `t-1` para predecir o formar el estado en `t`. Pandas `rolling()` y `shift()` son clave aquí.
  * **Manejo de NaNs Inducidos:**
      * Los cálculos de indicadores y las normalizaciones con ventana móvil introducirán NaNs al principio del dataset.
      * Estas filas iniciales (y las secuencias que las contengan) se descartarán. El número de filas a descartar dependerá del periodo de `lookback` más largo utilizado en cualquier cálculo (incluyendo la ventana de normalización).
  * **Almacenamiento de Datos Procesados:**
      * Guardar las secuencias de características de mercado (`X_market`) y sus correspondientes timestamps (del último KLine de cada secuencia) en disco.
      * Formato: NumPy arrays comprimidos (`.npz`) es una excelente opción para datos numéricos homogéneos 3D. Un archivo `.npz` puede contener `X_market_sequences` y `timestamps_sequences`.
      * Ruta de salida: `data/processed/`.
  * **Gestión de Configuración:**
      * Cargar todos los parámetros específicos del preprocesamiento (longitud de secuencia, periodos de indicadores, multiplicador de ventana de normalización) desde un archivo YAML dedicado (ej. `src/data/preprocessing_config.yaml`).

**Nota sobre Características de Cartera y Portafolio:**
Como bien indicaste, las 8 características de cartera y portafolio (`Estado Posición`, `Tamaño Posición norm.`, etc.) son dinámicas y se generarán *dentro* del entorno de Reinforcement Learning (Módulo 4) o del framework de backtesting (Módulo 7), ya que dependen de las acciones tomadas por el agente y el estado de la cuenta. El preprocesador actual se enfoca en preparar las características de *mercado*. El estado completo para el agente Transformer será una combinación de estas características de mercado preprocesadas y las características de cartera generadas en tiempo de ejecución/simulación.

**2. Entradas que Recibirá y Salidas que Producirá:**

  * **Entradas:**
      * **Archivo de Datos Crudos:** Un archivo CSV (ej. `BTCUSDT_FUTURES_1h_20200101_20250516.csv`) ubicado en `data/raw/`. El nombre del archivo a procesar deberá ser localizable (ej. el más reciente o uno específico).
      * **Archivo de Configuración de Preprocesamiento:** Un archivo YAML (ej. `src/data/preprocessing_config.yaml`) que contiene todos los parámetros del módulo.
      * **(Implícito) `ConfigManager`:** Para acceder a rutas globales (ej. `data_paths.raw`, `data_paths.processed`).
  * **Salidas:**
      * **Archivo de Secuencias Procesadas:** Un archivo `.npz` (ej. `BTCUSDT_FUTURES_1h_20200101_20250516_L96_market_features.npz`) en `data/processed/`. Este archivo contendrá:
          * Un array NumPy `X_market` de forma `(N_samples, L, N_features_mercado)`.
          * Un array NumPy `timestamps` de forma `(N_samples,)` con los timestamps UTC del último KLine de cada secuencia.
      * **Logs de Ejecución:** Mensajes detallados sobre el proceso, advertencias y errores.
      * **(Opcional) Archivo de Metadatos:** Un pequeño archivo JSON o YAML que describa las características incluidas en `N_features_mercado`, su orden, y los parámetros de preprocesamiento utilizados para generar el archivo de secuencias.

**3. Interacciones y Dependencias con Otros Módulos:**

  * **Módulo 1 (Adquisición de Datos):** Consume la salida (archivos CSV crudos) de este módulo.
  * **Módulo de Configuración (`src/utils/config.py`):** Utilizado para obtener rutas de datos base y potencialmente otras configuraciones globales.
  * **Módulo 4 (Entorno de Reinforcement Learning):** Consumirá las secuencias de características de mercado preprocesadas. El entorno combinará estas con las características de cartera generadas dinámicamente.
  * **Módulo 7 (Framework de Backtesting):** Similar al entorno de RL, utilizará estas secuencias de mercado para simular estrategias.

**4. Tecnologías, Librerías o Frameworks Específicos:**

  * **Lenguaje de Programación:** Python 3.x.
  * **Manipulación de Datos:**
      * **`pandas`:** Para la carga de datos, manipulación de series temporales, y cálculos basados en ventanas móviles.
      * **`numpy`:** Para operaciones numéricas eficientes y la creación/gestión de los arrays de secuencias multidimensionales.
  * **Cálculo de Indicadores Técnicos:**
      * **`TA-Lib` (Python wrapper):** Librería estándar y de alto rendimiento para una amplia gama de indicadores técnicos. Se debe asegurar su correcta instalación.
      * Alternativas (si TA-Lib presenta problemas de instalación o por preferencia): `pandas_ta` es una excelente opción que es puramente Python y muy completa.
  * **Normalización/Escalado:**
      * Implementación directa con `pandas` para Z-score móvil (`.rolling().mean()`, `.rolling().std()`).
      * Fórmulas directas para escalado Min-Max de RSI/Estocástico.
      * `scikit-learn` (`StandardScaler`, `MinMaxScaler`) podría usarse si se aplica con cuidado en un contexto de ventana deslizante expansiva (entrenando el scaler progresivamente), pero la implementación directa con Pandas es más transparente para la causalidad en ventanas móviles fijas.
  * **Almacenamiento:**
      * `numpy` para guardar y cargar archivos `.npz`.

**5. Métricas Clave para Evaluar su Rendimiento (del módulo):**

  * **Tiempo de Procesamiento:** Duración total para procesar un archivo de datos crudos de un tamaño determinado.
  * **Consumo de Memoria:** Pico de uso de memoria durante el preprocesamiento, especialmente al generar secuencias.
  * **Número de Secuencias Válidas Generadas:** Para verificar que el proceso produce la cantidad esperada de datos.
  * **Estadísticas Descriptivas de Características Normalizadas:** Verificar medias, desviaciones estándar, mínimos y máximos de las características finales para asegurar que la normalización ha funcionado como se esperaba.
  * **Porcentaje de NaNs Eliminados:** Confirmar que los NaNs se manejan correctamente.

**Estructura de Código Detallada:**

Seguiremos la estructura que delineaste, con clases en `src/data/` y un script orquestador en `raiz/scripts/`.

  * **`src/data/preprocessing_config.yaml`:**

    ```yaml
    # Longitud de la secuencia para el Transformer
    sequence_length_L: 96

    # Multiplicador para la ventana de normalización Z-score (ventana = L * multiplicador)
    normalization_window_multiplier_for_L: 2

    # Parámetros para el procesamiento de OHLCV
    ohlcv_processing:
      volume_sma_period: 20

    # Parámetros para los indicadores técnicos
    indicators:
      sma_short_period: 20
      sma_long_period: 50
      ema_short_period: 12
      ema_long_period: 26
      rsi_period: 14
      rsi_scaling_mode: "0_1" # "0_1" o "-1_1"
      atr_period: 14
      macd_fast_period: 12
      macd_slow_period: 26
      macd_signal_period: 9
      bollinger_period: 20
      bollinger_std_dev: 2
      cci_period: 20
      stochastic_k_period: 14 # fastk_period en TA-Lib
      stochastic_d_period: 3  # slowd_period en TA-Lib
      stochastic_slowing_period: 3 # slowk_period en TA-Lib (para %K suavizado)

    # Lista de características de mercado finales a incluir en las secuencias (el orden importa)
    # Estos nombres deben coincidir con las columnas generadas después de la ingeniería y normalización.
    final_market_feature_columns:
      - 'log_ret_C_O_norm'
      - 'log_ret_H_O_norm'
      - 'log_ret_L_O_norm'
      - 'log_ret_C_C_prev_norm'
      - 'log_ret_Vol_SMAVol_norm'
      - 'SMA_short_norm'          # Ej. (SMA_short - Close) / ATR
      - 'SMA_long_norm'           # Ej. (SMA_long - Close) / ATR
      - 'EMA_short_norm'          # Ej. (EMA_short - Close) / ATR
      - 'EMA_long_norm'           # Ej. (EMA_long - Close) / ATR
      - 'RSI_scaled'              # Ej. RSI / 100
      - 'ATR_norm'                # Ej. ATR / Close
      - 'MACD_line_norm'          # Ej. MACD_line / ATR o Z-score
      - 'MACD_signal_norm'        # Ej. MACD_signal / ATR o Z-score
      - 'MACD_hist_norm'          # Ej. MACD_hist / ATR o Z-score
      - 'BB_dist_upper_norm'      # Ej. (BB_upper - Close) / ATR
      - 'BB_dist_lower_norm'      # Ej. (Close - BB_lower) / ATR
      - 'BB_width_norm'           # Ej. (BB_upper - BB_lower) / ATR o (BB_upper - BB_lower) / BB_middle
      - 'CCI_norm'                # Ej. Z-score o CCI / (constante empírica)
      - 'STOCH_slowk_scaled'      # Ej. STOCH_slowk / 100
      - 'STOCH_slowd_scaled'      # Ej. STOCH_slowd / 100

    output_data:
      format: "npz" # "npz", "hdf5" (npz preferido por ahora)
    ```

  * **`src/data/feature_engineering.py`:**
    Esta clase se dedicará exclusivamente al cálculo de características (OHLCV procesado e indicadores técnicos) *antes* de la normalización final que las prepara para la secuencia. La normalización se aplicará en la clase `DataPreprocessor` principal.

    ```python
    import pandas as pd
    import numpy as np
    import talib
    import logging

    logger = logging.getLogger(__name__)

    class FeatureEngineer:
        def __init__(self, indicators_config: dict, ohlcv_config: dict):
            self.ic = indicators_config
            self.oc = ohlcv_config

        def add_ohlcv_features(self, df: pd.DataFrame) -> pd.DataFrame:
            logger.debug("Calculando características OHLCV procesadas.")
            df_out = df.copy()
            df_out['log_ret_C_O'] = np.log(df_out['Close'] / df_out['Open'])
            df_out['log_ret_H_O'] = np.log(df_out['High'] / df_out['Open'])
            df_out['log_ret_L_O'] = np.log(df_out['Low'] / df_out['Open'])
            df_out['log_ret_C_C_prev'] = np.log(df_out['Close'] / df_out['Close'].shift(1))
            
            vol_sma_period = self.oc.get('volume_sma_period', 20)
            sma_volume = talib.SMA(df_out['Volume'], timeperiod=vol_sma_period)
            # Evitar división por cero o log de cero/negativo si sma_volume es 0 o Volume es 0
            df_out['log_ret_Vol_SMAVol'] = np.log(df_out['Volume'].replace(0, 1e-9) / sma_volume.replace(0, 1e-9))
            return df_out

        def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
            logger.debug("Calculando indicadores técnicos.")
            df_out = df.copy() # Asegurarse de operar sobre una copia si df se reutiliza

            # SMAs
            df_out['SMA_short'] = talib.SMA(df_out['Close'], timeperiod=self.ic['sma_short_period'])
            df_out['SMA_long'] = talib.SMA(df_out['Close'], timeperiod=self.ic['sma_long_period'])
            # EMAs
            df_out['EMA_short'] = talib.EMA(df_out['Close'], timeperiod=self.ic['ema_short_period'])
            df_out['EMA_long'] = talib.EMA(df_out['Close'], timeperiod=self.ic['ema_long_period'])
            # RSI
            df_out['RSI'] = talib.RSI(df_out['Close'], timeperiod=self.ic['rsi_period'])
            # ATR
            df_out['ATR'] = talib.ATR(df_out['High'], df_out['Low'], df_out['Close'], timeperiod=self.ic['atr_period'])
            # MACD
            macd, macdsignal, macdhist = talib.MACD(df_out['Close'],
                                                    fastperiod=self.ic['macd_fast_period'],
                                                    slowperiod=self.ic['macd_slow_period'],
                                                    signalperiod=self.ic['macd_signal_period'])
            df_out['MACD_line'] = macd
            df_out['MACD_signal'] = macdsignal
            df_out['MACD_hist'] = macdhist
            # Bandas de Bollinger
            upper, middle, lower = talib.BBANDS(df_out['Close'],
                                                timeperiod=self.ic['bollinger_period'],
                                                nbdevup=self.ic['bollinger_std_dev'],
                                                nbdevdn=self.ic['bollinger_std_dev'],
                                                matype=0) # 0 para SMA
            df_out['BB_upper'] = upper
            df_out['BB_middle'] = middle # Es la SMA(periodo_bollinger)
            df_out['BB_lower'] = lower
            df_out['BB_width'] = upper - lower # Ancho absoluto
            # CCI
            df_out['CCI'] = talib.CCI(df_out['High'], df_out['Low'], df_out['Close'], timeperiod=self.ic['cci_period'])
            # Stochastic Oscillator (%K lento, %D)
            # TA-Lib STOCH: fastk_period, slowk_period (slowing), slowd_period (smoothing de slowk)
            slowk, slowd = talib.STOCH(df_out['High'], df_out['Low'], df_out['Close'],
                                       fastk_period=self.ic['stochastic_k_period'],
                                       slowk_period=self.ic['stochastic_slowing_period'],
                                       slowk_matype=0, # SMA para suavizar %K rápido
                                       slowd_period=self.ic['stochastic_d_period'],
                                       slowd_matype=0) # SMA para suavizar %K lento (que es %D)
            df_out['STOCH_slowk'] = slowk
            df_out['STOCH_slowd'] = slowd
            return df_out
    ```

  * **`src/data/preprocessor.py`:**

    ```python
    import pandas as pd
    import numpy as np
    import os
    import logging
    from src.utils.config import ConfigManager # Asumiendo importación desde src
    from src.data.feature_engineering import FeatureEngineer

    logger = logging.getLogger(__name__)

    class DataPreprocessor:
        def __init__(self, general_config_manager: ConfigManager, module_specific_config: dict):
            self.gcfg = general_config_manager
            self.mcfg = module_specific_config # Config específica del módulo de preprocesamiento

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
            if len(self.final_feature_columns) != 20: # 5 OHLCV + 15 Indicadores
                 logger.warning(f"El número de columnas finales ({len(self.final_feature_columns)}) no coincide con el esperado (20). Verifica 'final_market_feature_columns' en la config.")


        def _load_and_prepare_base_df(self, raw_data_filename: str) -> pd.DataFrame:
            filepath = os.path.join(self.raw_data_path, raw_data_filename)
            logger.info(f"Cargando datos crudos desde: {filepath}")
            try:
                df = pd.read_csv(filepath, parse_dates=['Open_Time'], date_parser=lambda x: pd.to_datetime(x, utc=True))
                df.set_index('Open_Time', inplace=True)
                cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in cols_to_numeric:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Reemplazar ceros en 'Open' para evitar problemas con log(H/O) y log(L/O) si H o L son 0
                # Esto es un parche; idealmente, los datos no deberían tener Open=0 si hay H/L/C.
                df['Open'] = df['Open'].replace(0, 1e-9) 
                df.dropna(subset=cols_to_numeric, inplace=True) # Crucial después de la conversión
                df.sort_index(inplace=True) # Asegurar orden cronológico
                return df
            except FileNotFoundError:
                logger.error(f"Archivo de datos crudos no encontrado: {filepath}"); raise
            except Exception as e:
                logger.error(f"Error cargando o en preparación básica de datos desde {filepath}: {e}"); raise

        def _apply_feature_normalization(self, df_with_features: pd.DataFrame) -> pd.DataFrame:
            logger.debug("Aplicando normalización/escalado final a las características.")
            df_norm = df_with_features.copy()

            # Ventana para Z-score, asegurando min_periods para tener valores al inicio
            min_p = self.norm_window // 2

            # Normalización de características OHLCV procesadas (Z-score móvil)
            ohlcv_raw_cols = ['log_ret_C_O', 'log_ret_H_O', 'log_ret_L_O', 'log_ret_C_C_prev', 'log_ret_Vol_SMAVol']
            for col in ohlcv_raw_cols:
                mean = df_norm[col].rolling(window=self.norm_window, min_periods=min_p).mean()
                std = df_norm[col].rolling(window=self.norm_window, min_periods=min_p).std().replace(0, 1e-9) # Evitar división por cero
                df_norm[f'{col}_norm'] = (df_norm[col] - mean) / std

            # Normalización de Indicadores
            atr = df_norm['ATR'].replace(0, 1e-9) # Para evitar división por cero
            close = df_norm['Close'].replace(0, 1e-9)

            df_norm['SMA_short_norm'] = (df_norm['SMA_short'] - close) / atr
            df_norm['SMA_long_norm'] = (df_norm['SMA_long'] - close) / atr
            df_norm['EMA_short_norm'] = (df_norm['EMA_short'] - close) / atr
            df_norm['EMA_long_norm'] = (df_norm['EMA_long'] - close) / atr

            if self.mcfg['indicators']['rsi_scaling_mode'] == "0_1":
                df_norm['RSI_scaled'] = df_norm['RSI'] / 100.0
            else: # "-1_1"
                df_norm['RSI_scaled'] = (df_norm['RSI'] - 50.0) / 50.0
            
            df_norm['ATR_norm'] = atr / close

            df_norm['MACD_line_norm'] = df_norm['MACD_line'] / atr # O Z-score
            df_norm['MACD_signal_norm'] = df_norm['MACD_signal'] / atr # O Z-score
            df_norm['MACD_hist_norm'] = df_norm['MACD_hist'] / atr # O Z-score
            
            # Distancias a BB normalizadas por ATR
            df_norm['BB_dist_upper_norm'] = (df_norm['BB_upper'] - close) / atr
            df_norm['BB_dist_lower_norm'] = (close - df_norm['BB_lower']) / atr
            # Ancho de BB normalizado por ATR o por la media móvil (BB_middle)
            df_norm['BB_width_norm'] = df_norm['BB_width'] / atr # o df_norm['BB_width'] / df_norm['BB_middle'].replace(0,1e-9)

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

        def _create_sequences(self, df_final_features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
            logger.info(f"Creando secuencias de longitud L={self.L}.")
            
            # Convertir a NumPy array para eficiencia
            data_values = df_final_features.values
            timestamps_values = df_final_features.index.to_numpy()

            num_samples = len(data_values) - self.L + 1
            
            if num_samples <= 0:
                logger.warning("No hay suficientes datos para crear ni una sola secuencia después del preprocesamiento y recorte de NaNs.")
                return np.array([]), np.array([])

            # Usar conform_array de numpy.lib.stride_tricks es más eficiente para grandes arrays
            # pero requiere cuidado porque crea vistas. Un bucle es más seguro si la memoria no es una limitación extrema.
            # shape = (num_samples, self.L, data_values.shape[1])
            # strides = (data_values.strides[0], data_values.strides[0], data_values.strides[1])
            # sequences_X = np.lib.stride_tricks.as_strided(data_values, shape=shape, strides=strides)
            # sequences_ts = timestamps_values[self.L - 1:] # Timestamps del último elemento de cada secuencia

            # Alternativa con bucle (más clara, potencialmente más lenta pero más segura para empezar)
            X_list, ts_list = [], []
            for i in range(num_samples):
                X_list.append(data_values[i : i + self.L])
                ts_list.append(timestamps_values[i + self.L - 1]) # Timestamp del último elemento

            sequences_X = np.array(X_list)
            sequences_ts = np.array(ts_list)
            
            return sequences_X, sequences_ts

        def process_data(self, raw_data_filename: str, output_filename_base: str):
            logger.info(f"Iniciando preprocesamiento para el archivo: {raw_data_filename}")
            # 1. Cargar y preparación básica
            df_base = self._load_and_prepare_base_df(raw_data_filename)
            if df_base.empty: return

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
                logger.error(f"Error guardando las secuencias procesadas: {e}"); raise
    ```

  * **`raiz/scripts/preprocess_data.py` (Orquestador):**

    ```python
    import logging
    import sys
    import os
    import yaml

    # Añadir src al PYTHONPATH si es necesario (mejor configurar PYTHONPATH en .env)
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    # if project_root not in sys.path:
    #    sys.path.insert(0, project_root)

    from src.utils.config import ConfigManager
    from src.data.preprocessor import DataPreprocessor

    logging.basicConfig(
        level=logging.INFO, # Cambiar a logging.DEBUG para más detalle
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    def main():
        logger.info("Iniciando script de preprocesamiento de datos.")
        
        try:
            # Cargar configuración general y del módulo
            # Asumir que config.yaml y .env están en la raíz (o donde ConfigManager los espere)
            # y preprocessing_config.yaml está en src/data/
            general_config_manager = ConfigManager(config_path="src/config.yaml", env_path=".env")
            
            preprocessing_config_path = "src/data/preprocessing_config.yaml" 
            with open(preprocessing_config_path, 'r') as f:
                module_specific_config = yaml.safe_load(f)
            logger.info(f"Configuración de preprocesamiento cargada desde {preprocessing_config_path}")

        except FileNotFoundError as e:
            logger.error(f"Error: Archivo de configuración no encontrado. {e}"); return
        except Exception as e:
            logger.error(f"Error al cargar la configuración: {e}"); return

        try:
            preprocessor = DataPreprocessor(general_config_manager, module_specific_config)
        except Exception as e:
            logger.error(f"Error al inicializar DataPreprocessor: {e}", exc_info=True); return

        # Determinar el archivo de datos crudos a procesar
        raw_data_dir = general_config_manager.get_config_value('data_paths.raw')
        # Lógica para seleccionar el archivo: el más reciente, uno específico por parámetro, etc.
        # Aquí un ejemplo simple: buscar el primer CSV que empiece por el símbolo por defecto.
        default_symbol = general_config_manager.get_config_value('data_acquisition_defaults.symbol', 'BTCUSDT')
        raw_files = [f for f in os.listdir(raw_data_dir) if f.startswith(default_symbol) and f.endswith('.csv')]

        if not raw_files:
            logger.error(f"No se encontraron archivos de datos crudos para {default_symbol} en {raw_data_dir}. Ejecuta primero el script de adquisición."); return
        
        # Procesar el archivo más reciente (asumiendo que el nombre contiene fecha/hora o se ordena alfabéticamente)
        raw_data_filename = sorted(raw_files, reverse=True)[0] 
        output_filename_base = os.path.splitext(raw_data_filename)[0] # ej. BTCUSDT_FUTURES_1h_20200101_20250516

        try:
            logger.info(f"Procesando archivo de datos crudos: {raw_data_filename}")
            preprocessor.process_data(raw_data_filename, output_filename_base)
            logger.info("Proceso de preprocesamiento de datos finalizado exitosamente.")
        except Exception as e:
            logger.error(f"Ocurrió un error crítico durante el proceso de preprocesamiento: {e}", exc_info=True)

    if __name__ == "__main__":
        main()
    ```

**Consideraciones Adicionales y Próximos Pasos:**

1.  **Refinar Estrategias de Normalización:** La sección `_apply_feature_normalization` es clave.
      * **Z-score vs. División por ATR/Close:** Para indicadores como MACD lines, CCI, SMAs, EMAs, etc., la normalización relativa (dividir por ATR o Close) suele ser más robusta a cambios de régimen de volatilidad que un Z-score móvil simple. La configuración actual (`final_market_feature_columns` y la implementación en `_apply_feature_normalization`) refleja una mezcla. Esto es un buen punto de partida y se puede experimentar.
      * **Constantes Empíricas:** Para algunos indicadores como CCI, a veces se divide por constantes (ej. 100 o 200) si su rango típico es conocido, además o en lugar de Z-score.
2.  **Completitud de `final_market_feature_columns`:** Es vital que esta lista en `preprocessing_config.yaml` y su construcción en `_apply_feature_normalization` sean consistentes y produzcan exactamente las 20 características esperadas con los nombres correctos.
3.  **Verificación y Debugging:**
      * Después de la primera ejecución, inspecciona las estadísticas (media, std, min, max) de cada una de las 20 características en el array `X_market` resultante.
      * Visualiza algunas secuencias y características para asegurar que los patrones y valores tengan sentido.
4.  **TA-Lib vs. pandas\_ta:** Si encuentras dificultades con la instalación de TA-Lib (que a veces es problemática por su dependencia C), `pandas_ta` es una excelente alternativa que es puramente Python y muy fácil de instalar (`pip install pandas_ta`). La sintaxis es ligeramente diferente pero cubre todos los indicadores que necesitas.
5.  **Manejo de Errores y Robustez:** Se ha añadido logging básico y manejo de excepciones. Se puede mejorar con reintentos para ciertas operaciones si fuera necesario (aunque para preprocesamiento batch no suele ser tan crítico como para la adquisición de datos).

Este diseño detallado para el Módulo 2 debería proporcionarte una base sólida. ¿Te parece bien esta estructura y el nivel de detalle? ¿Hay algún aspecto específico, especialmente en la normalización, que quieras ajustar o discutir más a fondo antes de pasar al siguiente módulo?


Perfecto, tu puntualización sobre el manejo de NaNs en los datos crudos es muy pertinente y crucial. Mi propuesta anterior ya incluía una sección de manejo de NaNs, pero vamos a refinarla y detallarla aún más, considerando las implicaciones para la continuidad temporal y la calidad de los datos.

El objetivo es ser robustos ante posibles datos faltantes, manteniendo la integridad de las series temporales tanto como sea posible, sin introducir sesgos indebidos.

**Estrategia Detallada para el Manejo de NaNs en `_load_and_prepare_base_df` (Módulo 2):**

1.  **Carga Inicial y Conversión Numérica:**

      * Cargar los datos desde el CSV.
      * Convertir las columnas OHLCV a tipos numéricos, usando `errors='coerce'` para que los valores no convertibles se transformen en `NaN`.
      * Asegurar que el índice `Open_Time` (timestamp UTC) sea único y esté ordenado cronológicamente. Los duplicados se eliminan (manteniendo la primera ocurrencia) y se reordena el índice si es necesario.

2.  **Detección y Reporte de NaNs:**

      * Contar la cantidad de NaNs en cada una de las columnas OHLCV.
      * Si se encuentran NaNs, registrar una advertencia detallada indicando en qué columnas y cuántos NaNs hay.

3.  **Imputación Limitada con Forward Fill (`ffill`):**

      * **Razón:** Para huecos pequeños y esporádicos en los datos, `ffill` es una estrategia común en series temporales financieras. Preserva el último valor conocido, lo cual es una suposición razonable para periodos cortos donde el mercado podría no haber cambiado drásticamente o donde la falta de datos es momentánea. Ayuda a mantener la continuidad del índice temporal.
      * **Implementación:** Aplicar `ffill` a las columnas OHLCV.
      * **Límite de `ffill` (`ffill_limit`):** Es crucial no propagar un valor obsoleto durante demasiado tiempo. Se introducirá un parámetro configurable (ej. `raw_data_ffill_limit` en `preprocessing_config.yaml`) para controlar el número máximo de periodos consecutivos que `ffill` puede rellenar. Un valor típico podría ser entre 1 y 4 periodos. Si el intervalo es de 15 minutos, un límite de 4 rellenaría hasta una hora de datos faltantes. Si `ffill_limit` es 0, este paso se omite.
      * Registrar cuántos NaNs fueron rellenados mediante este método.

4.  **Eliminación de NaNs Restantes:**

      * **Razón:** Si después de `ffill` (o si `ffill` no se aplicó o no pudo rellenar todos los NaNs, por ejemplo, NaNs al inicio del dataset o huecos más grandes que `ffill_limit`), todavía existen filas con NaNs en las columnas OHLCV críticas, estas filas deben ser eliminadas. Intentar cálculos de indicadores o normalizaciones con NaNs en OHLCV puede llevar a errores o resultados incorrectos.
      * **Implementación:** Aplicar `dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])`.
      * Registrar una advertencia si se eliminan filas en este paso, indicando cuántas.

5.  **Verificación de DataFrame Vacío:**

      * Después de todos los pasos de limpieza, si el DataFrame resultante está vacío, registrar un error crítico y detener el procesamiento para ese archivo, ya que no hay datos válidos para continuar.

6.  **Correcciones Finales (ej. `Open` = 0):**

      * Una vez que los NaNs han sido manejados, proceder con otras correcciones menores, como reemplazar `Open == 0` con un valor muy pequeño (`1e-9`) para evitar errores en cálculos logarítmicos, asumiendo que si `Open` es cero pero `High/Low/Close` no lo son, es una anomalía del dato.

**Implementación en `_load_and_prepare_base_df` (Clase `DataPreprocessor`):**

```python
# Dentro de la clase DataPreprocessor, en src/data/preprocessor.py

        def _load_and_prepare_base_df(self, raw_data_filename: str) -> pd.DataFrame:
            filepath = os.path.join(self.raw_data_path, raw_data_filename)
            logger.info(f"Cargando datos crudos desde: {filepath}")
            try:
                df = pd.read_csv(
                    filepath,
                    parse_dates=['Open_Time'],
                    # El date_parser ya no es necesario con pandas > 2.0 si el formato es estándar
                    # date_parser=lambda x: pd.to_datetime(x, utc=True) # Mantener por compatibilidad o control explícito
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
                    # Ejemplo de estructura en preprocessing_config.yaml:
                    # raw_data_settings:
                    #   ffill_limit_for_nans: 4 # 0 para desactivar
                    ffill_limit = self.mcfg.get('raw_data_settings', {}).get('ffill_limit_for_nans', 0) # Por defecto 0 (sin ffill)

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
                df['Open'] = df['Open'].replace(0, 1e-9) # Evitar log(0) o división por cero
                
                logger.info(f"Datos crudos cargados y preparados inicialmente para {raw_data_filename}. Forma final del DataFrame base: {df.shape}")
                return df

            except FileNotFoundError:
                logger.error(f"Archivo de datos crudos no encontrado: {filepath}"); raise
            except Exception as e:
                logger.error(f"Error crítico durante la carga y preparación básica de datos desde {filepath}: {e}", exc_info=True); raise

```

**Configuración en `src/data/preprocessing_config.yaml`:**

Añade o asegúrate de tener una sección para estos parámetros:

```yaml
# ... (otras configuraciones del Módulo 2 como sequence_length_L, indicators, etc.) ...

# Configuraciones para el manejo de datos crudos
raw_data_settings:
  # Límite para ffill en NaNs de OHLCV en datos crudos. 
  # 0 deshabilita ffill, y todos los NaNs en OHLCV se eliminarían (dropna).
  # Un valor > 0 intentará rellenar hasta N periodos consecutivos.
  ffill_limit_for_nans: 4 
```

**Impacto de esta Estrategia:**

  * **Prioriza la Continuidad para Pequeños Huecos:** Al usar `ffill` con un límite, los pequeños "hipos" en los datos (ej. una o dos velas faltantes) se pueden tapar sin romper la secuencia temporal. Esto es generalmente preferible a eliminar la vela y crear un salto de tiempo irregular.
  * **Robustez ante Huecos Grandes o NaNs Persistentes:** Si los huecos son mayores que `ffill_limit`, o si los NaNs están al inicio del conjunto de datos (donde `ffill` no tiene un valor previo para propagar), esas filas problemáticas se eliminan. Esto evita que datos de muy mala calidad se propaguen.
  * **Transparencia:** El logging detallado informa sobre cuántos NaNs se encontraron, cuántos se rellenaron y cuántos (si los hay) llevaron a la eliminación de filas.
  * **Configurabilidad:** El `ffill_limit` permite ajustar la agresividad de la imputación. Si prefieres no hacer ninguna imputación, puedes establecer `ffill_limit_for_nans: 0`.

Este manejo más granular y configurable de los NaNs en la etapa de carga de datos crudos fortalecerá la robustez de tu pipeline de preprocesamiento. Es un excelente punto el que has planteado.