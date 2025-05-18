
**DOCUMENTO DE DISEÑO TÉCNICO Y GUÍA DE IMPLEMENTACIÓN: BOT DE TRADING BTC AUTÓNOMO EN GCP CON RL Y TRANSFORMERS**

**Versión:** 1.0
**Fecha:** 18 de mayo de 2025
**Autor:** Pedro Escudero Murcia (con asistencia de Gemini AI)

**Tabla de Contenidos:**

1.  Resumen Ejecutivo
2.  Visión del Proyecto y Objetivos
3.  Stack Tecnológico Principal
4.  Arquitectura General en GCP (Fase de Entrenamiento y Backtesting)
5.  Módulos Detallados del Sistema
    * 5.1. Módulo 1: Adquisición de Datos de Binance
        * 5.1.A. Descarga de Datos Históricos (Componente de Vertex AI Pipeline)
        * 5.1.B. Adquisición de Datos en Tiempo Real (Diseño Conceptual para Futuro)
    * 5.2. Módulo 2: Preprocesamiento y Gestión de Datos
    * 5.3. Módulo 3: Entorno de Trading (Simulador)
    * 5.4. Módulo 4: Agente de Reinforcement Learning (RL) con Transformers y su Entrenamiento en GCP
    * 5.5. Módulo 7: Framework de Backtesting Robusto (Componente de Vertex AI Pipeline) (Anteriormente Módulo 7, ahora integrado secuencialmente)
6.  Gestión de Riesgos y Capital (Integrada en Módulo 3)
7.  Módulo 9: Logging, Monitorización y Visualización en GCP
8.  Módulo 10: Automatización de la Infraestructura y CI/CD en GCP
    * 8.A. Automatización de la Infraestructura con Terraform (Infrastructure as Code)
    * 8.B. Automatización del Ciclo de Vida del ML con CI/CD (Cloud Build)
9.  Estrategia de Contenerización (Dockerfile Único)
10. Flujos de Trabajo y Pipelines en Vertex AI (Definición y Compilación)
11. Configuración de Recursos de Cómputo en GCP
12. (Próximamente) Módulo 5: Sistema de Ejecución de Órdenes (Binance API) para Trading en Vivo
13. Conclusión

---

## 1. Resumen Ejecutivo

Este documento detalla el diseño técnico y la arquitectura para el desarrollo de un bot de trading de Bitcoin (BTC) altamente avanzado y autónomo. El bot operará con futuros de BTC/USDT a través de la API de Binance y utilizará Reinforcement Learning (RL) con una arquitectura de agente basada en Transformers para la toma de decisiones. Un principio fundamental del diseño es la construcción de una solución 100% nativa de la nube, optimizada para su ejecución, escalabilidad, robustez y gestión dentro de Google Cloud Platform (GCP). El proyecto se centrará en la creación de un pipeline de MLOps robusto y automatizado para la adquisición de datos, preprocesamiento, entrenamiento y backtesting riguroso del agente, utilizando Terraform para la gestión de la infraestructura y Cloud Build para la integración y despliegue continuos.

## 2. Visión del Proyecto y Objetivos

Construir un sistema de trading algorítmico para Bitcoin que represente el estado del arte, capaz de aprender y adaptarse a las condiciones cambiantes del mercado de forma autónoma. El agente de RL tomará decisiones de trading (comprar, vender, mantener) basadas en una comprensión profunda de las secuencias de datos de mercado y el estado actual de la cartera, procesadas mediante una arquitectura Transformer. Todo el ciclo de vida del modelo, desde la ingesta de datos hasta el entrenamiento, la evaluación y un eventual despliegue, se gestionará y orquestará en Google Cloud Platform, aprovechando sus servicios para MLOps, computación escalable, almacenamiento y analítica.

**Objetivos Clave:**

1.  **Infraestructura como Código (IaC):** Definir y gestionar toda la infraestructura de GCP necesaria (GCS, Artifact Registry, Secret Manager, Vertex AI resources, IAM) utilizando Terraform.
2.  **CI/CD Robusto:** Implementar un pipeline de Integración Continua y Despliegue Continuo (CI/CD) con Cloud Build para automatizar las pruebas, la construcción de imágenes Docker, la compilación de pipelines KFP y el despliegue en Vertex AI Pipelines.
3.  **Adquisición de Datos Automatizada:** Desarrollar un componente de pipeline de Vertex AI para la descarga robusta de datos históricos de futuros de BTC/USDT de Binance, almacenándolos eficientemente en GCS (Parquet).
4.  **Preprocesamiento Avanzado y Causal:** Implementar un componente de pipeline para limpiar datos, realizar ingeniería de características (con `pandas-ta`) y construir secuencias normalizadas causalmente, aptas para el modelo Transformer.
5.  **Entorno de Simulación Realista:** Crear un entorno de trading (`gymnasium.Env`) que modele con precisión las operaciones de futuros (comisiones, slippage, liquidación, gestión de capital sin SL/TP forzados), configurable y reproducible.
6.  **Agente de RL Vanguardista:** Diseñar e implementar un agente Soft Actor-Critic (SAC) con una política y función de valor basadas en una arquitectura Transformer (`CustomTransformerFeatureExtractor`), capaz de procesar secuencias de mercado y estado de la cartera.
7.  **Entrenamiento Escalable y con Capacidad de Reanudación:** Orquestar el entrenamiento del agente en Vertex AI Training (con soporte GPU), permitiendo la reanudación desde checkpoints.
8.  **Backtesting Riguroso:** Desarrollar un componente de pipeline para evaluar el rendimiento del agente entrenado utilizando métricas estándar de la industria, generando informes detallados con `quantstats`.
9.  **Monitorización y Observabilidad Completas:** Utilizar Cloud Logging, Cloud Monitoring, Vertex AI Experiments y Vertex AI TensorBoard para una visibilidad profunda del sistema.

## 3. Stack Tecnológico Principal

* **Lenguaje de Programación:** Python 3.9+
* **Machine Learning/Deep Learning:** PyTorch, Stable Baselines3 (SAC), Gymnasium
* **Manipulación de Datos:** Pandas, NumPy, `pandas-ta`, PyArrow
* **Google Cloud Platform (GCP):**
    * **IaC:** Terraform
    * **CI/CD:** Cloud Build
    * **Orquestación MLOps:** Vertex AI Pipelines
    * **Entrenamiento:** Vertex AI Training (con GPUs)
    * **Modelos:** Vertex AI Model Registry
    * **Experimentos:** Vertex AI Experiments
    * **Visualización Entrenamiento:** Vertex AI TensorBoard
    * **Almacenamiento:** Google Cloud Storage (GCS), BigQuery (opcional)
    * **Secretos:** Secret Manager
    * **Contenedores:** Docker, Artifact Registry
    * **Logging & Monitorización:** Cloud Logging, Cloud Monitoring
* **Análisis de Backtesting:** `quantstats`
* **Gestión de Configuración:** Archivos YAML, parámetros de pipeline KFP.

## 4. Arquitectura General en GCP

El sistema se implementa como un pipeline de MLOps de extremo a extremo en Vertex AI, con la infraestructura subyacente gestionada por Terraform y el ciclo de vida del código y pipeline gestionado por Cloud Build.

**Diagrama de Flujo Conceptual del Pipeline Principal en Vertex AI:**
(Referirse al diagrama descrito anteriormente en la Parte 1, Módulo 0, Sección 3)
El flujo es: Adquisición de Datos -> Preprocesamiento -> Entrenamiento del Agente -> Backtesting del Agente.

**Flujo de CI/CD con Cloud Build:**
(Referirse al flujo descrito anteriormente en la Parte 6, Módulo 10, Sección 4.7.B)
El flujo es: Push a Git -> Trigger de Cloud Build -> (Linting/Pruebas) -> Build Imagen Docker -> Push a Artifact Registry -> Compilar KFP Pipeline (con nueva imagen) -> Desplegar KFP Pipeline a Vertex AI -> (Smoke Test).

**Gestión de Infraestructura con Terraform:**
Los archivos Terraform definirán todos los recursos GCP necesarios (buckets, repositorios, secretos, instancias de TensorBoard, cuentas de servicio, permisos IAM), asegurando un aprovisionamiento consistente y versionado.

## 5. Módulos Detallados del Sistema

### 5.1. Módulo 1: Adquisición de Datos de Binance

Responsable de obtener datos OHLCV de futuros BTC/USDT de Binance.

#### 5.1.A. Descarga de Datos Históricos (Componente de Vertex AI Pipeline)

* **Responsabilidades:** Conexión segura, descarga parametrizada (símbolo, intervalo, fechas), manejo de paginación, gestión de errores y rate limits (reintentos exponenciales), validación básica, almacenamiento en Parquet en GCS, logging estructurado.
* **Implementación en GCP:**
    * **Orquestación:** Componente KFP en Vertex AI Pipeline.
    * **Ejecución:** Script Python `src/data_acquisition/binance_downloader.py` (clase `BinanceFuturesDownloader`) dentro de la imagen Docker única.
    * **Configuración:** Parámetros del componente KFP (símbolo, intervalo, fecha de inicio, límites de API, ruta GCS de salida). Claves API de Binance desde Secret Manager.
    * **Salida:** Archivos Parquet en GCS (ej. `gs://<bucket>/data/raw/futures/BTCUSDT/1h/BTCUSDT_FUTURES_1h_YYYYMMDD_YYYYMMDDHHMM.parquet`), Artefacto `Dataset` de KFP.
* **Lógica Clave de `BinanceFuturesDownloader`:**
    * Constructor acepta parámetros de API y configuración.
    * Método `Workspace_historical_data` implementa bucle de paginación, llamadas a `python-binance` (cliente de futuros `UMFutures`), manejo de excepciones (ej. `BinanceAPIException`, `requests.exceptions.ConnectionError`), reintentos.
    * Conversión a DataFrame de Pandas, selección de columnas OHLCV.
    * Escritura a Parquet en GCS usando `df.to_parquet(gcs_path, engine='pyarrow')`.
* **Entradas del Componente KFP:**
    * `project_id: str`, `location: str`
    * `binance_api_key_secret_name: str`, `binance_api_secret_secret_name: str`
    * `symbol: str`, `interval: str`, `historical_start_date: str`
    * `api_request_limit_per_call: int`, `api_request_delay_seconds: float`
    * `api_retry_attempts: int`, `api_retry_delay_seconds: int`
    * `gcs_output_raw_data_path_base: str` (ruta base en GCS para los Parquet)
* **Salidas del Componente KFP:**
    * `output_raw_data_gcs_uri: OutputPath(Dataset)` (URI al directorio/archivo Parquet en GCS)
    * `execution_summary: OutputPath(Metrics)` (ej. número de velas, rango de fechas)

#### 5.1.B. Adquisición de Datos en Tiempo Real (Diseño Conceptual para Futuro)

* **Responsabilidades:** Conexión WebSocket persistente, suscripción a flujos (trades, klines, libro de órdenes), publicación en `Pub/Sub`, manejo de errores.
* **Arquitectura GCP:** Cliente WebSocket en `Cloud Run`, publicación en `Pub/Sub` topics, persistencia opcional vía `Dataflow` a `BigQuery`/`Vertex AI Feature Store`.

### 5.2. Módulo 2: Preprocesamiento y Gestión de Datos (Componente de Vertex AI Pipeline)

Transforma datos crudos en secuencias normalizadas y listas para el modelo.

* **Responsabilidades:** Carga de Parquet desde GCS, limpieza (manejo de NaNs con `ffill_limit_for_nans`), ingeniería de 20 características de mercado (5 OHLCV + 15 TIs con `pandas-ta`), normalización/escalado causal (Z-score móvil, relativo a precio/ATR, [0,1] para osciladores), construcción de secuencias 3D `(N_samples, L, 20)`, manejo de NaNs inducidos, almacenamiento de secuencias `.npz` en GCS, logging estructurado.
* **Implementación en GCP:**
    * **Orquestación:** Componente KFP en Vertex AI Pipeline.
    * **Ejecución:** Scripts Python `src/preprocessing/feature_engineer.py` (clase `FeatureEngineer`) y `src/preprocessing/data_preprocessor.py` (clase `DataPreprocessor`) en la imagen Docker única.
    * **Configuración:** Parámetros del componente KFP (URI GCS de datos crudos, `sequence_length_L`, `normalization_window_multiplier_for_L`, `raw_data_ffill_limit_for_nans`, parámetros de indicadores, lista de features finales y sus métodos de normalización, ruta GCS de salida).
    * **Salida:** Archivo `.npz` (con `X_market` y `timestamps`) en GCS (ej. `gs://<bucket>/data/processed/.../L96_norm2/data_sequences.npz`), Artefacto `Dataset` de KFP.
* **Lógica Clave de `FeatureEngineer`:**
    * Constructor acepta configuraciones de indicadores y OHLCV.
    * Método `add_ohlcv_features(df)`: Calcula los 5 retornos logarítmicos.
    * Método `add_technical_indicators(df)`: Usa `pandas-ta` (ej. `df.ta.strategy(custom_strategy)`) para calcular SMAs, EMAs, RSI, ATR, MACD, Bandas de Bollinger, CCI, Estocástico. Incluye lógica para renombrar columnas de `pandas-ta` a los nombres estandarizados esperados (ej. `SMA_short`, `MACD_line`).
* **Lógica Clave de `DataPreprocessor`:**
    * Constructor acepta parámetros de configuración.
    * Método `_load_and_prepare_base_df(raw_data_gcs_uri)`: Lee Parquet de GCS, maneja índice, convierte tipos, aplica limpieza de NaNs con `ffill` limitado.
    * Método `_apply_feature_normalization(df_with_features)`: Aplica las estrategias de normalización causal a cada una de las 20 características (Z-score móvil, `(X-Close)/ATR`, `X/Close-1`, `X/100`, etc.) según la configuración.
    * Método `_create_sequences(df_final_features)`: Genera el array 3D `(N_samples, L, 20)` y el array de timestamps.
    * Método principal `process_data(raw_data_gcs_uri, output_gcs_npz_path)`: Orquesta los pasos anteriores, incluyendo `dropna()` después de la normalización y antes de crear secuencias. Guarda el `.npz` en GCS.
* **Entradas del Componente KFP:**
    * `project_id: str`, `location: str`
    * `gcs_raw_data_uri: Input[Dataset]`
    * `sequence_length_l: int`, `normalization_window_multiplier_for_l: int`
    * `raw_data_ffill_limit_for_nans: int`
    * `ohlcv_vol_sma_period: int`
    * `indicators_config_json: str` (JSON string con los periodos para todos los TIs)
    * `feature_normalization_config_json: str` (JSON string detallando la estrategia de normalización para cada una de las 20 features)
    * `gcs_processed_output_path_base: str`
* **Salidas del Componente KFP:**
    * `processed_sequences_gcs_uri: OutputPath(Dataset)` (URI al `.npz`)
    * `preprocessing_summary: OutputPath(Metrics)` (ej. num_sequences, num_features)

---

**PARTE 7: MÓDULOS DE ENTORNO DE TRADING, AGENTE RL Y BACKTESTING (Continuación del Documento de Diseño)**

---

### 5.3. Módulo 3: Entorno de Trading (Simulador)

Este módulo es el corazón de la simulación donde el agente de Reinforcement Learning (RL) interactuará para aprender y donde se evaluarán sus estrategias mediante backtesting. Se basa en la interfaz estándar de `gymnasium.Env` para asegurar la compatibilidad con librerías de RL como Stable Baselines3. El diseño prioriza un realismo considerable en la simulación de las operaciones de futuros de Bitcoin en Binance, al mismo tiempo que proporciona la flexibilidad necesaria para el entrenamiento y la evaluación. (Referirse a la Parte 3 del DDT anterior para la descripción completa de responsabilidades, funcionalidades, lógica de `SimulatedBroker`, espacios de observación/acción, función de recompensa, liquidación, fin de episodio, reinicio, y su integración en GCP y Vertex AI Pipelines).

**Puntos Clave de Implementación para `TradingEnvironment` (`src/environments/trading_env.py`):**

* **`__init__(self, gcs_processed_data_uri: str, ..., random_episode_start: bool = True, **kwargs)`:**
    * Carga los datos de secuencias de mercado (`X_market`, `timestamps`) desde el archivo `.npz` ubicado en `gcs_processed_data_uri` (descargándolo primero de GCS a una ruta local temporal).
    * Almacena `self.market_data_sequences` y `self.market_timestamps`.
    * Calcula `self.L` y `self.N_features_mercado` a partir de `self.market_data_sequences.shape`.
    * Define `self.observation_space` como `gymnasium.spaces.Dict` con:
        * `'market_features'`: `Box(shape=(self.L, self.N_features_mercado), dtype=np.float32)`
        * `'portfolio_features'`: `Box(shape=(8,), dtype=np.float32)`
    * Define `self.action_space` como `Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)`.
    * Inicializa `SimulatedBroker` con parámetros de comisiones, slippage, etc.
    * Inicializa el estado de la cartera (`current_equity`, `balance`, etc.) a valores por defecto o basados en `initial_equity_config`.
    * Almacena `random_episode_start`.
* **`_load_market_data(self, gcs_uri: str)`:**
    * Usa `google.cloud.storage` para descargar el archivo `.npz` de `gcs_uri` a un `tempfile.NamedTemporaryFile()`.
    * Carga los arrays `X_market` y `timestamps` usando `np.load()` desde el archivo temporal.
* **`reset(self, seed=None, options=None)`:**
    * Llama a `super().reset(seed=seed)`.
    * Reinicia completamente el estado de la cartera (`current_equity = self.initial_equity_config`, `initial_equity_episode = self.initial_equity_config`, `active_position_side = 0`, etc.).
    * Selecciona `self.current_step_index`: aleatorio entre `0` y `len(self.market_data_sequences) - 1` si `self.random_episode_start` es `True`; de lo contrario, `0`.
    * Llama a `_get_observation()` y `_get_info()` para devolver la observación inicial y el diccionario de información.
* **`_calculate_and_normalize_portfolio_features(self) -> np.ndarray`:**
    * Calcula las 8 características de portafolio (`Estado Posición`, `Tamaño Posición Normalizado`, `Precio Entrada Normalizado Relativo`, `P&L No Realizado Normalizado`, `Retorno Log Equity Último Paso`, `Ratio Margen Utilizado`, `Pasos en Posición Normalizados`, `Apalancamiento Configurado Normalizado`) basándose en el estado actual de la cartera.
    * Aplica la normalización a cada una según las estrategias definidas (ej. división por `initial_equity_episode`, `current_equity`, `max_steps_norm_divisor`, o escalado a rangos específicos). Los parámetros para esta normalización (ej. `max_steps_norm_divisor`) vendrán de la configuración del entorno.
    * Devuelve un array NumPy de forma `(8,)` y tipo `np.float32`.
* **`_get_observation(self) -> dict`:**
    * Obtiene `market_sequence = self.market_data_sequences[self.current_step_index]`.
    * Llama a `_calculate_and_normalize_portfolio_features()`.
    * Devuelve el diccionario `{'market_features': market_sequence, 'portfolio_features': portfolio_features}`.
* **`step(self, action: np.ndarray) -> tuple`:**
    1.  Almacena `current_equity_before_action = self.current_equity`.
    2.  Interpreta `action_signal = action[0]` usando `self.action_threshold` para determinar la `desired_trading_action` (ABRIR\_LARGO, ABRIR\_CORTO, CERRAR\_POSICION, MANTENER).
    3.  Obtiene el precio de mercado actual (ej. `current_market_close_price` del último KLine de la observación actual, o el `Open` del KLine siguiente si se simula ejecución al inicio de la nueva barra). Obtiene el valor ATR actual para el cálculo de slippage.
    4.  **Lógica de Ejecución de Órdenes (compleja, involucra al `SimulatedBroker`):**
        * Si la `desired_trading_action` implica un cambio de posición (o abrir una nueva):
            * Si hay una posición activa y la acción es cerrarla o invertirla, primero simular el cierre:
                * Llamar a `self.simulated_broker.calculate_execution_details()` para el cierre (precio de ejecución con slippage, comisión).
                * Actualizar `self.current_equity`, `self.balance`, P&L realizado.
                * Resetear `self.active_position_side = 0`, etc.
            * Si la acción es abrir una nueva posición (o abrir una después de cerrar):
                * Llamar a `self.simulated_broker.calculate_position_size_contracts()` para obtener el tamaño.
                * Si el tamaño es válido (cumple mínimos):
                    * Llamar a `self.simulated_broker.calculate_execution_details()` para la apertura (precio de ejecución con slippage, comisión).
                    * Llamar a `self.simulated_broker.calculate_margin_required()`.
                    * Verificar si `available_margin` es suficiente.
                    * Si todo es correcto, actualizar estado de la cartera (`self.active_position_side`, `entry_price`, `size`, `margin_used`, `balance`, `current_equity` menos comisión).
    5.  Si no hay acción de trading o la acción es mantener, el P&L no realizado se actualiza con el precio de mercado actual.
    6.  Actualizar `self.unrealized_pnl` si hay una posición activa. `self.current_equity = self.balance + self.margin_used + self.unrealized_pnl`.
    7.  Calcular recompensa: `self.last_step_log_return = reward = np.log(self.current_equity / current_equity_before_action)` (manejar `current_equity_before_action <= 0`).
    8.  **Verificar Liquidación:**
        * Si hay posición activa, calcular el `liquidation_price` usando `self.simulated_broker.calculate_liquidation_price()`.
        * Comparar con el `Low` (para Largos) o `High` (para Cortos) del KLine actual. Si se cruza, simular cierre por liquidación, actualizar P&L, `current_equity`, y establecer `terminated = True`.
    9.  Incrementar `self.current_step_index += 1` y `self.steps_in_current_position` (si aplica).
    10. **Verificar Condiciones de Fin de Episodio:**
        * `terminated`: Si hay liquidación, o si `current_equity <= self.initial_equity_episode * (1.0 + self.equity_drawdown_threshold_episode_end)`, o si `current_equity` es demasiado bajo para operar.
        * `truncated`: Si `self.current_step_index >= len(self.market_data_sequences)`.
    11. Llamar a `_get_observation()` para la `next_observation` y `_get_info()`.
    12. Devolver `(next_observation, reward, terminated, truncated, info)`.
* **`_get_info(self) -> dict`:** Devuelve un diccionario con el estado actual de la cartera y del entorno.

**Lógica Clave de `SimulatedBroker` (`src/environments/simulated_broker.py`):**
* Como se describió en la Parte 3 del DDT. Sus métodos toman el estado actual del mercado/cartera como entrada y devuelven cálculos (precios de ejecución, comisiones, tamaño de posición, margen, precio de liquidación) sin modificar el estado del entorno directamente.

### 5.4. Módulo 4: Agente de Reinforcement Learning (RL) con Transformers y su Entrenamiento en GCP

Este módulo define la arquitectura del agente de RL y gestiona su proceso de entrenamiento. (Referirse a la Parte 4 del DDT anterior para la descripción completa del algoritmo SAC, la arquitectura del modelo incluyendo el `CustomTransformerFeatureExtractor`, el Replay Buffer, el aprendizaje de `alpha`, las redes objetivo, y cómo se integra el entrenamiento en un componente de Vertex AI Pipeline con soporte para GPUs y reanudación desde checkpoints).

**Puntos Clave de Implementación para `CustomTransformerFeatureExtractor` (`src/agent/custom_transformer_extractor.py`):**
* Hereda de `stable_baselines3.common.torch_layers.BaseFeaturesExtractor`.
* **`__init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int, market_features_key: str, portfolio_features_key: str, d_model: int, n_heads: int, n_encoder_layers: int, dim_feedforward: int, dropout_rate: float)`:**
    * `features_dim` será igual a `d_model`.
    * Define las capas:
        * `input_projection_layer = nn.Linear(observation_space['market_features'].shape[1] + observation_space['portfolio_features'].shape[0], d_model)` (ej. 20+8=28 -> `d_model`).
        * `positional_encoding = PositionalEncoding(d_model, dropout_rate, max_len=observation_space['market_features'].shape[0])` (donde `max_len` es `L`).
        * `encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout_rate, batch_first=True)`.
        * `transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)`.
* **`forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor`:**
    1.  Extraer `market_features_batch = observations[self.market_features_key]` y `portfolio_features_batch = observations[self.portfolio_features_key]`.
    2.  Replicar `portfolio_features_batch` (que tiene forma `(batch_size, N_portfolio)`) para que tenga forma `(batch_size, L, N_portfolio)`. Esto se puede hacer con `unsqueeze(1).repeat(1, L, 1)`.
    3.  Concatenar `market_features_batch` y `portfolio_features_replicated` a lo largo de la última dimensión para obtener `combined_input` de forma `(batch_size, L, N_market + N_portfolio)`.
    4.  Pasar `combined_input` por `self.input_projection_layer`.
    5.  Añadir `self.positional_encoding`.
    6.  Pasar el resultado por `self.transformer_encoder`.
    7.  Tomar la salida del último paso temporal: `transformer_output = transformer_output[:, -1, :]` (resultando en `(batch_size, d_model)`).
    8.  Devolver `transformer_output`.
* **`PositionalEncoding` class (definida dentro del mismo archivo o importada):** Implementación estándar de PE sinusoidal.

**Puntos Clave para el Script de Entrenamiento del Componente KFP (`src/components/run_train_agent.py`):**
* Parsea todos los argumentos del pipeline (config del entorno, config del agente, timesteps, URI de datos, URI de checkpoint de entrada, etc.).
* Configura logging y directorios locales temporales para TensorBoard, checkpoints y modelo final.
* Instancia `TradingEnvironment` con sus parámetros.
* Define `policy_kwargs` para el agente SAC, incluyendo `features_extractor_class=CustomTransformerFeatureExtractor` y `features_extractor_kwargs` con los parámetros del Transformer. También `net_arch` para los MLPs del actor/crítico.
* Lógica para cargar modelo desde `input_checkpoint_gcs_uri` (descargando el `.zip` de GCS) o crear uno nuevo.
* Configura Callbacks de SB3:
    * `CheckpointCallback`: Guarda checkpoints en un directorio local. Un script adicional o una función puede sincronizar estos checkpoints a GCS periódicamente o al final.
    * `TensorBoardCallback` (o la integración directa de SB3 con `tensorboard_log`): SB3 escribe logs en `local_tensorboard_log_dir`.
* Calcula `timesteps_to_train_this_run` y llama a `model.learn(..., reset_num_timesteps=False si se reanuda)`.
* Guarda el modelo final en la ruta local especificada por `trained_model_output.path` del artefacto KFP.
* Copia los logs de TensorBoard del directorio local a una ubicación GCS designada que la instancia de Vertex AI TensorBoard pueda leer.

### 5.5. Módulo 7: Framework de Backtesting Robusto (Componente de Vertex AI Pipeline)

Evalúa el rendimiento de un agente entrenado sobre datos históricos. (Referirse a la Parte 5 del DDT anterior para la descripción completa de responsabilidades, implementación en GCP, uso de `quantstats`, y el esquema del componente KFP).

**Puntos Clave para el Script de Backtesting del Componente KFP (`src/components/run_backtest_agent.py`):**
* Parsea argumentos del pipeline (URI del modelo entrenado, URI de datos de backtest, parámetros del entorno).
* Descarga el modelo `.zip` y los datos de backtest `.npz` desde GCS a rutas locales temporales.
* Instancia `TradingEnvironment` con `random_episode_start=False` y los datos de backtest.
* Carga el modelo SAC entrenado.
* Bucle de simulación:
    * `obs, info = eval_env.reset()`
    * Mientras no `done` o `truncated`:
        * `action, _ = model.predict(obs, deterministic=True)`
        * `obs, reward, terminated, truncated, info = eval_env.step(action)`
        * Registrar datos de trades y equity en DataFrames/listas de Pandas.
* Cálculo de Métricas y Reporte:
    * Convertir el historial de equity a una serie de retornos para `quantstats`.
    * Usar `quantstats.stats` para métricas individuales y loguearlas en `backtest_results_metrics: Output[Metrics]`.
    * Usar `quantstats.reports.html()` para generar el informe HTML. Guardarlo en la ruta local del artefacto `quantstats_report_html: Output[Artifact]`.
* (Opcional) Guardar logs de trades y curva de equity como archivos CSV/Parquet en las rutas de los artefactos `Output[Dataset]` correspondientes.

-----

**PARTE 8: MÓDULOS DE GESTIÓN DE RIESGOS, LOGGING, MONITORIZACIÓN, ORQUESTACIÓN, CI/CD Y CONCLUSIONES (Continuación del Documento de Diseño)**

-----

### 6\. Gestión de Riesgos y Capital (Integrada en Módulo 3 - Entorno de Trading)

Como se discutió y decidió, no se implementarán mecanismos de Stop-Loss (SL) o Take-Profit (TP) forzados en el `TradingEnvironment`. El objetivo es que el agente de RL aprenda de manera autónoma las estrategias óptimas de salida de operaciones (toma de beneficios y limitación de pérdidas) basándose en la función de recompensa y las observaciones del mercado y la cartera.

La gestión de riesgos y capital, por lo tanto, se centra en las siguientes reglas y mecanismos intrínsecos del entorno y la configuración del agente:

  * **1. Dimensionamiento de Posición:**
      * **Método:** Porcentaje del equity actual (`current_equity * position_size_pct_equity`).
      * **Cálculo de Contratos:** `position_size_contracts = (current_equity * position_size_pct_equity * configured_leverage) / execution_price`.
      * **Restricciones:** Se verifica contra `min_order_size_btc`. Si la posición calculada es menor, la orden no se ejecuta.
      * **Configuración:** `position_size_pct_equity`, `min_order_size_btc` (parámetros del entorno).
  * **2. Control de Apalancamiento:**
      * **Método:** Apalancamiento fijo (`configured_leverage`) aplicado a todas las operaciones.
      * **Impacto:** Afecta el tamaño de la posición nocional, el margen requerido y el precio de liquidación.
      * **Configuración:** `configured_leverage` (parámetro del entorno).
  * **3. Simulación de Liquidación:**
      * **Mecanismo:** El `TradingEnvironment` calcula un precio de liquidación aproximado para cada posición abierta basado en el precio de entrada, tamaño, apalancamiento y balance de la cuenta (usando `SimulatedBroker.calculate_liquidation_price()`).
      * **Activación:** Si el precio de mercado (`Low` para Largos, `High` para Cortos) cruza el precio de liquidación, la posición se cierra forzosamente, se actualiza el equity (potencialmente a cero o negativo si el colateral no cubre), y el episodio termina (`terminated = True`).
  * **4. Límite de Drawdown del Episodio:**
      * **Mecanismo:** El `TradingEnvironment` monitoriza el `current_equity` en relación con el `initial_equity_episode`.
      * **Activación:** Si `current_equity <= initial_equity_episode * (1.0 + equity_drawdown_threshold_episode_end)`, el episodio termina (`terminated = True`).
      * **Configuración:** `equity_drawdown_threshold_episode_end` (ej. -0.20 para -20%, parámetro del entorno).
  * **5. Aprendizaje Implícito del Agente:**
      * El agente, a través de la función de recompensa `log(equity_t / equity_{t-1})` y las características de observación (P\&L no realizado, etc.), debe aprender a:
          * Cerrar posiciones ganadoras para realizar beneficios.
          * Cerrar posiciones perdedoras antes de que las pérdidas se vuelvan catastróficas (o antes de la liquidación).
          * Decidir no operar si las condiciones no son favorables.

Este enfoque pone la responsabilidad total de la gestión de las operaciones en el agente de RL, alineándose con el objetivo de un sistema "autogestionado".

### 7\. Módulo 9: Logging, Monitorización y Visualización en GCP

Este módulo describe el stack de observabilidad para el proyecto, cubriendo el ciclo de vida completo desde el desarrollo hasta la (futura) producción. (Referirse a la Parte 6 del DDT anterior, Módulo 9, para la descripción completa de los principios, servicios GCP como Cloud Logging, Cloud Monitoring, Vertex AI Experiments, Vertex AI TensorBoard, informes `quantstats`, y Looker Studio, y su aplicación al proyecto).

**Puntos Clave de Implementación:**

  * **Cloud Logging:**
      * **Logging Estructurado en Python:** Todos los scripts Python (componentes KFP, utilidades) usarán la librería `logging` configurada para emitir logs en formato JSON. Esto puede lograrse con un `JSONFormatter` personalizado.
        ```python
        # Ejemplo de configuración de logger para JSON
        import logging
        import json
        from datetime import datetime

        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_record = {
                    "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
                    "severity": record.levelname,
                    "message": record.getMessage(),
                    "module": record.module,
                    "funcName": record.funcName,
                    "lineno": record.lineno,
                    # Añadir campos personalizados pasados vía extra={}
                    **(record.extra_args if hasattr(record, 'extra_args') else {})
                }
                # Añadir información de excepción si existe
                if record.exc_info:
                    log_record['exception'] = self.formatException(record.exc_info)
                return json.dumps(log_record)

        # Aplicar el formateador a los handlers relevantes
        # logger = logging.getLogger("MyProjectLogger")
        # handler = logging.StreamHandler() # Se captura por Cloud Logging en GCP
        # handler.setFormatter(JsonFormatter())
        # logger.addHandler(handler)
        # logger.setLevel(logging.INFO)
        # logger.info("Mensaje de log", extra={"extra_args": {"pipeline_id": "123", "step": "preprocess"}})
        ```
      * **Consulta:** Utilizar el explorador de logs de Cloud Logging con filtros avanzados basados en los campos JSON.
  * **Cloud Monitoring:**
      * **Dashboards:** Crear dashboards para:
          * **Salud de Vertex AI Pipelines:** Número de ejecuciones exitosas/fallidas, duración media de pipelines y componentes.
          * **Rendimiento de Vertex AI Training:** Uso de CPU/GPU, uso de memoria de los trabajos de entrenamiento.
          * (Futuro) Métricas de negocio del bot en vivo.
      * **Alertas:**
          * Fallo de una ejecución de Vertex AI Pipeline.
          * Trabajo de entrenamiento excediendo un umbral de duración o coste.
          * (Futuro) Drawdown crítico del bot en vivo.
  * **Vertex AI Experiments:**
      * Asegurar que cada ejecución del pipeline de KFP se configure para registrarse como un "run" dentro de un experimento específico en Vertex AI.
      * Los parámetros del pipeline y las métricas de `Output[Metrics]` de los componentes se asociarán automáticamente.
  * **Vertex AI TensorBoard:**
      * El componente de entrenamiento (Módulo 4) guardará los logs de TensorBoard (generados por SB3) en un bucket GCS dedicado.
      * Se creará y configurará una instancia de Vertex AI TensorBoard para leer desde este bucket. La URI de esta instancia se puede loguear o referenciar.
  * **Informes `quantstats`:**
      * El componente de backtesting (Módulo 7) generará el informe HTML y lo guardará como un artefacto en GCS. La URI de este artefacto se registrará como salida del componente KFP para fácil acceso.
  * **Looker Studio (Opcional Avanzado):**
      * Para dashboards consolidados de resultados de backtesting, se pueden cargar los logs de trades detallados y las curvas de equity (guardados como CSV/Parquet en GCS por el Módulo 7) en BigQuery.
      * Looker Studio se conectará a estas tablas de BigQuery para visualizaciones interactivas y comparativas.

### 8\. Módulo 10: Automatización de la Infraestructura y CI/CD en GCP

Este módulo detalla cómo se gestionará la infraestructura de GCP como código y cómo se automatizará el ciclo de vida de la aplicación de ML.

#### 8.A. Automatización de la Infraestructura con Terraform (Infrastructure as Code)

(Referirse a la Parte 7 del DDT anterior, Sección C, para la descripción completa de la introducción a Terraform, configuración del provider, variables, definición detallada de recursos GCP – APIs, GCS buckets, Artifact Registry, Secret Manager, Vertex AI TensorBoard, Cuenta de Servicio y Permisos IAM – gestión del estado, y flujo de trabajo).

**Recursos Terraform Clave a Definir (Resumen):**

  * **`google_project_service`:** Para habilitar todas las APIs necesarias (Vertex AI, Storage, Artifact Registry, Secret Manager, Cloud Build, IAM, etc.).
  * **`google_storage_bucket`:** Para datos crudos, procesados, artefactos de pipeline, modelos, y logs de TensorBoard (5 buckets distintos como mínimo). Configurar versionado y acceso uniforme.
  * **`google_artifact_registry_repository`:** Un repositorio Docker para la imagen única del proyecto.
  * **`google_secret_manager_secret` y `google_secret_manager_secret_version`:** Para las claves API de Binance, pasando los valores de forma segura durante el `terraform apply`.
  * **`google_vertex_ai_tensorboard`:** Para la instancia de TensorBoard.
  * **`google_service_account`:** Una cuenta de servicio dedicada (ej. `svc-btc-bot-pipelines@<project-id>.iam.gserviceaccount.com`) para que los pipelines de Vertex AI y los trabajos de Cloud Build se ejecuten con los permisos mínimos necesarios.
  * **`google_project_iam_member` / `google_storage_bucket_iam_member` / `google_secret_manager_secret_iam_member`:** Para asignar los roles necesarios (Vertex AI User, Storage Object Admin, Secret Accessor, Artifact Registry Reader, etc.) a la cuenta de servicio dedicada.
  * **Backend de GCS para Terraform:** Configurar Terraform para almacenar su archivo de estado (`terraform.tfstate`) en un bucket de GCS dedicado (este bucket inicial puede necesitar ser creado manualmente o mediante un script de bootstrapping).

#### 8.B. Automatización del Ciclo de Vida del ML con CI/CD (Cloud Build)

(Referirse a la Parte 7 del DDT anterior, Sección D, y la Parte 8 (respuesta anterior) para la descripción completa de la introducción a CI/CD, el `Dockerfile` único, el `cloudbuild.yaml` detallado con todos los pasos opcionales integrados – linting, pruebas unitarias, smoke test –, la configuración de triggers de Cloud Build, y el flujo de trabajo completo).

**Pasos Detallados en `cloudbuild.yaml` (Resumen con opcionales incluidos):**

1.  **Linting:** `flake8` sobre el código fuente.
2.  **Pruebas Unitarias:** `pytest tests/ --cov=src` para ejecutar pruebas y obtener cobertura.
3.  **Construir Imagen Docker:** Usando el `Dockerfile` único y tageando con `$COMMIT_SHA` y `:latest`.
4.  **Subir Imagen Docker a Artifact Registry:** Ambas etiquetas.
5.  **Compilar Pipeline KFP:** Ejecutar `python pipeline_definition.py --image_uri <URI_IMAGEN_CON_COMMIT_SHA> --pipeline_json_output_path <nombre_archivo.json>`.
6.  **Desplegar/Actualizar Pipeline en Vertex AI:** Usar `gcloud ai pipelines create` (o `update`) con el JSON compilado, especificando la cuenta de servicio del pipeline y habilitando el caching.
7.  **(Opcional pero Recomendado) Ejecutar "Smoke Test" del Pipeline:** `gcloud ai pipeline-jobs submit` con el JSON compilado y un conjunto de parámetros reducidos para una validación rápida.

**Configuración de `pipeline_definition.py` para CI/CD:**

  * Debe aceptar argumentos de línea de comandos para la URI de la imagen Docker a usar en los componentes y la ruta de salida para el JSON compilado. Esto permite que Cloud Build pase la URI de la imagen recién construida.

**Triggers de Cloud Build:**

  * Configurar triggers para ramas específicas (ej. `main`, `develop`) que inicien el pipeline de `cloudbuild.yaml`.

**Cuenta de Servicio de Cloud Build:**

  * Asegurar que la cuenta de servicio de Cloud Build (`<project-number>@cloudbuild.gserviceaccount.com`) tenga los permisos necesarios para escribir en Artifact Registry, desplegar en Vertex AI Pipelines, y (si se incluye la validación de Terraform) ejecutar `terraform plan/apply`.

### 9\. Estrategia de Contenerización (Dockerfile Único)

Se utilizará un único `Dockerfile` para construir una imagen Docker que contenga todo el código fuente del proyecto (`src/`, `scripts/`, `pipeline_definition.py`) y todas las dependencias listadas en `requirements.txt`. (Referirse a la Parte 6 del DDT anterior, Módulo 10, Sección D.2 y la Parte 8 para el `Dockerfile` conceptual y `requirements.txt`).

  * **Base Image:** `python:3.9-slim` o similar.
  * **Dependencias:** Instaladas vía `pip install -r requirements.txt`.
  * **Código Fuente:** Copiado en el directorio `/app` de la imagen.
  * **PYTHONPATH:** Configurado para incluir `/app` y `/app/src`.
  * **Sin ENTRYPOINT específico:** Los componentes KFP y los pasos de Cloud Build definirán los comandos a ejecutar.

### 10\. Flujos de Trabajo y Pipelines en Vertex AI (Definición y Compilación)

El pipeline principal de MLOps se define en `pipeline_definition.py` usando KFP SDK v2. (Referirse a la Parte 6 del DDT anterior, Módulo 10, Sección 4.7.A para el ejemplo de `pipeline_definition.py` y la función `@pipeline`).

  * **Componentes Parametrizados:** Cada función de componente (`@component`) se define para aceptar parámetros que controlan su comportamiento y referencias a la imagen Docker.
  * **Paso de Artefactos:** Se usa `Input[Dataset]`, `Output[Model]`, etc., para gestionar el flujo de datos (URIs GCS) entre componentes.
  * **Compilación:** El script `pipeline_definition.py` incluye una sección `if __name__ == '__main__':` para compilar el pipeline a JSON usando `kfp.compiler.Compiler()`.

### 11\. Configuración de Recursos de Cómputo en GCP

(Referirse a la Parte 8 del DDT anterior para la discusión detallada de los recursos de CPU, memoria y GPU para cada componente del pipeline: Adquisición, Preprocesamiento, Entrenamiento y Backtesting, y cómo especificarlos en KFP o Trabajos Personalizados de Vertex AI).

**Resumen de Recomendaciones de Recursos:**

  * **Adquisición y Preprocesamiento:** CPU y memoria moderadas (ej. `e2-standard-2` a `e2-standard-4`, 8-32GB RAM). El preprocesamiento puede requerir más memoria si el dataset es muy grande y no se procesa en chunks.
  * **Entrenamiento del Agente RL:** CPU moderada (ej. `n1-standard-4` o `n1-standard-8`), RAM significativa (32GB+), y **GPU esencial** (empezar con 1x NVIDIA T4, considerar V100 o A100 para modelos/datasets más grandes o entrenamiento más rápido).
  * **Backtesting:** CPU y memoria moderadas (similar a preprocesamiento).

### 12\. (Próximamente) Módulo 5: Sistema de Ejecución de Órdenes (Binance API) para Trading en Vivo

Esta fase se abordará después de que el pipeline de entrenamiento y backtesting esté maduro y los resultados del agente sean prometedores. El diseño modular actual (especialmente la abstracción del `SimulatedBroker`) facilitará la creación de un `LiveBinanceBroker` y un orquestador de trading en vivo (posiblemente en `Cloud Run`) que interactúe con un `Vertex AI Endpoint` sirviendo el modelo entrenado.

### 13\. Conclusión

Este documento presenta un diseño técnico detallado para un sistema avanzado de trading algorítmico de Bitcoin, con un fuerte énfasis en las mejores prácticas de MLOps y la utilización de Google Cloud Platform. La arquitectura modular, la automatización de la infraestructura y del ciclo de vida del ML, y el enfoque en la robustez y la configurabilidad sientan las bases para un proyecto de alto potencial. La implementación de este diseño permitirá el desarrollo, entrenamiento y evaluación rigurosos de agentes de Reinforcement Learning basados en Transformers, con el objetivo final de lograr una operativa de trading autónoma y eficiente.

-----

Pedro, este documento ahora integra todas las discusiones y decisiones que hemos tomado, proporcionando una guía extremadamente detallada y completa para la implementación de tu proyecto. Cubre la arquitectura, los módulos, la infraestructura como código con Terraform, el CI/CD con Cloud Build, y las consideraciones de recursos.
