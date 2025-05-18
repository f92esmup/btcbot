**PARTE 1: INTRODUCCIÓN, ARQUITECTURA Y MÓDULO DE ADQUISICIÓN DE DATOS**

-----

# Documento de Diseño Técnico: Bot de Trading BTC Autónomo en GCP con RL y Transformers

**Versión:** 1.0
**Fecha:** 18 de mayo de 2025
**Autor:** Pedro Escudero Murcia (con asistencia de Gemini AI)
**Revisión:** (Fecha de la última revisión)

## 0\. Resumen Ejecutivo

Este documento detalla el diseño técnico y la arquitectura para el desarrollo de un bot de trading de Bitcoin (BTC) altamente avanzado y autónomo. El bot operará con futuros de BTC/USDT a través de la API de Binance y utilizará Reinforcement Learning (RL) online con una arquitectura de agente basada en Transformers para la toma de decisiones. Un principio fundamental del diseño es la construcción de una solución 100% nativa de la nube, optimizada para su ejecución, escalabilidad, robustez y gestión dentro de Google Cloud Platform (GCP). El proyecto se centrará inicialmente en la creación de un pipeline de MLOps robusto para el entrenamiento y backtesting riguroso del agente, sentando las bases para una futura transición a operaciones de trading en vivo.

## 1\. Visión del Proyecto

Construir un sistema de trading algorítmico para Bitcoin que represente el estado del arte, capaz de aprender y adaptarse a las condiciones cambiantes del mercado de forma autónoma. El agente de RL tomará decisiones de trading (comprar, vender, mantener) basadas en una comprensión profunda de las secuencias de datos de mercado y el estado actual de la cartera, procesadas mediante una arquitectura Transformer. Todo el ciclo de vida del modelo, desde la ingesta de datos hasta el entrenamiento, la evaluación y un eventual despliegue, se gestionará y orquestará en Google Cloud Platform, aprovechando sus servicios para MLOps, computación escalable, almacenamiento y analítica.

**Objetivos Clave (Fase Inicial - Entrenamiento y Backtesting):**

1.  Desarrollar un pipeline de adquisición de datos robusto para obtener datos históricos de futuros de BTC/USDT de Binance y almacenarlos eficientemente en GCP.
2.  Implementar un módulo de preprocesamiento avanzado que limpie los datos, realice ingeniería de características y construya secuencias normalizadas aptas para el modelo Transformer.
3.  Crear un entorno de simulación de trading (`gymnasium.Env`) realista y configurable que modele con precisión las operaciones de futuros en Binance (comisiones, slippage, liquidación).
4.  Diseñar e implementar un agente de Reinforcement Learning (Soft Actor-Critic) con una política y función de valor basadas en una arquitectura Transformer, capaz de procesar secuencias de mercado y estado de la cartera.
5.  Orquestar el entrenamiento del agente en Vertex AI Training, permitiendo la reanudación y el uso de aceleradores (GPU).
6.  Desarrollar un framework de backtesting riguroso para evaluar el rendimiento del agente entrenado utilizando métricas estándar de la industria (generadas con `quantstats`).
7.  Automatizar todo el flujo de trabajo (adquisición de datos, preprocesamiento, entrenamiento, evaluación) mediante Vertex AI Pipelines.
8.  Utilizar las mejores prácticas de MLOps en GCP, incluyendo la gestión de artefactos, el versionado de modelos y la monitorización de experimentos.

## 2\. Stack Tecnológico Principal

  * **Lenguaje de Programación:** Python 3.9+
  * **Machine Learning/Deep Learning:**
      * PyTorch (backend para Stable Baselines3 y modelos Transformer)
      * Stable Baselines3 (para el algoritmo SAC y la gestión del agente de RL)
      * Gymnasium (para la creación del entorno de trading)
  * **Manipulación y Procesamiento de Datos:**
      * Pandas
      * NumPy
      * `pandas-ta` (para indicadores técnicos)
  * **Google Cloud Platform (GCP):**
      * **Orquestación MLOps:** Vertex AI Pipelines
      * **Entrenamiento de Modelos:** Vertex AI Training (con opción de GPUs)
      * **Registro y Versionado de Modelos:** Vertex AI Model Registry
      * **Seguimiento de Experimentos:** Vertex AI Experiments
      * **Visualización de Métricas de Entrenamiento:** Vertex AI TensorBoard
      * **Almacenamiento de Datos y Artefactos:** Google Cloud Storage (GCS)
      * **Almacenamiento Analítico y Consulta de Datos:** BigQuery
      * **Gestión de Secretos:** Secret Manager
      * **Contenerización:** Docker
      * **Registro de Contenedores:** Artifact Registry
      * **CI/CD:** Cloud Build
      * **Logging Centralizado:** Cloud Logging
      * **Monitorización y Alertas:** Cloud Monitoring
      * **(Para Futuro Despliegue en Vivo):** Vertex AI Prediction (Endpoints), Cloud Run, Pub/Sub, Firestore.
  * **Análisis de Backtesting:** `quantstats`
  * **Gestión de Configuración:** Archivos YAML, parámetros de pipeline.

## 3\. Arquitectura General en GCP (Fase de Entrenamiento y Backtesting)

El sistema se diseñará como un pipeline de Machine Learning de extremo a extremo orquestado por **Vertex AI Pipelines**. Este pipeline constará de varios componentes contenerizados, cada uno responsable de una fase específica del ciclo de vida del modelo.

**Diagrama de Flujo Conceptual del Pipeline Principal:**

```
[Inicio del Pipeline (Trigger manual o programado)]
    |
    V
[Componente 1: Adquisición de Datos Históricos]
    - Tarea: Descargar datos OHLCV de futuros BTC/USDT desde Binance API.
    - Input: Rango de fechas, símbolo, intervalo (parámetros del pipeline). Credenciales de API (desde Secret Manager).
    - Lógica: Script Python (BinanceFuturesDownloader) en contenedor Docker.
    - Output: Datos crudos (archivos Parquet) en GCS (ej. gs://<bucket>/data/raw/...). Artefacto Dataset.
    |
    V
[Componente 2: Preprocesamiento y Generación de Secuencias]
    - Tarea: Cargar datos crudos, limpiar, ingeniería de características (con pandas-ta), normalizar causalmente, construir secuencias (N_samples, L, N_features).
    - Input: Artefacto Dataset de datos crudos (GCS URI). Parámetros de preprocesamiento (longitud de secuencia L, ventanas de normalización, etc.).
    - Lógica: Script Python (DataPreprocessor, FeatureEngineer) en contenedor Docker.
    - Output: Datos de secuencias procesadas (archivo .npz) en GCS (ej. gs://<bucket>/data/processed/...). Artefacto Dataset.
    |
    V
[Componente 3: Entrenamiento del Agente RL]
    - Tarea: Entrenar el agente SAC con arquitectura Transformer usando el TradingEnvironment.
    - Input: 
        - Artefacto Dataset de secuencias procesadas (GCS URI).
        - Parámetros de configuración del entorno (equity inicial, leverage, comisiones, etc.).
        - Parámetros de configuración del agente (learning rate, batch size, arquitectura Transformer, etc.).
        - (Opcional) URI GCS de un checkpoint para reanudar entrenamiento.
    - Lógica: Script Python (TradingEnvironment, CustomTransformerFeatureExtractor, lógica de entrenamiento SB3) en contenedor Docker, utilizando GPU si se especifica.
    - Output: 
        - Modelo entrenado (archivo .zip SB3) en GCS, registrado en Vertex AI Model Registry. Artefacto Model.
        - Checkpoints del modelo en GCS.
        - Logs de TensorBoard en GCS (para Vertex AI TensorBoard).
        - Métricas de entrenamiento. Artefacto Metrics.
    |
    V
[Componente 4: Backtesting del Agente Entrenado]
    - Tarea: Evaluar el rendimiento del agente entrenado en un conjunto de datos históricos.
    - Input:
        - Artefacto Model del agente entrenado (GCS URI).
        - Artefacto Dataset de secuencias procesadas para backtesting (puede ser el mismo o diferente al de entrenamiento).
        - Parámetros de configuración del entorno (con random_episode_start=False).
    - Lógica: Script Python (TradingEnvironment, carga de modelo SB3, bucle de evaluación) en contenedor Docker.
    - Output:
        - Métricas de backtesting detalladas. Artefacto Metrics.
        - Informe HTML de quantstats en GCS. Artefacto.
        - (Opcional) Log de trades y curva de equity en GCS. Artefacto Dataset.
    |
    V
[Fin del Pipeline]
```

**Servicios GCP de Soporte:**

  * **Google Cloud Storage (GCS):** Almacenamiento primario para datos crudos, datos procesados, checkpoints de modelos, modelos finales, informes de backtesting y otros artefactos.
  * **BigQuery:** (Opcional inicialmente, pero recomendado para volúmenes grandes) Para almacenar datos históricos de mercado de forma estructurada, permitiendo análisis SQL complejos y sirviendo como fuente de datos para exploración o incluso para los pipelines si se prefiere sobre archivos planos para ciertos casos.
  * **Artifact Registry:** Almacenamiento y versionado de la imagen Docker única que contendrá todo el código y dependencias del proyecto.
  * **Secret Manager:** Almacenamiento seguro de las claves API de Binance.
  * **Cloud Logging:** Recopilación centralizada de todos los logs generados por los componentes del pipeline y los scripts Python.
  * **Cloud Monitoring:** Monitorización del rendimiento de los pipelines y servicios de Vertex AI. Alertas para fallos o anomalías.
  * **Vertex AI Experiments:** Seguimiento y comparación de todas las ejecuciones del pipeline, incluyendo los parámetros de entrada, los artefactos de salida y las métricas de rendimiento.
  * **Vertex AI TensorBoard:** Visualización interactiva de las métricas de entrenamiento del agente de RL.
  * **Cloud Build:** Automatización de la construcción de la imagen Docker, ejecución de pruebas y despliegue de los pipelines de Vertex AI (flujo CI/CD).

## 4\. Módulos Detallados del Sistema

### 4.1. Módulo 1: Adquisición de Datos de Binance

Este módulo es responsable de obtener los datos de mercado necesarios para el entrenamiento y backtesting del agente de trading. Se divide en dos sub-módulos conceptuales: uno para la descarga masiva de datos históricos (operado como un componente de pipeline) y otro diseñado para la ingesta en tiempo real (para un futuro despliegue en vivo). Nuestro enfoque inicial es el sub-módulo de datos históricos.

#### 4.1.A. Descarga de Datos Históricos (Componente de Vertex AI Pipeline)

Este componente se encarga de descargar datos de velas (OHLCV - Open, High, Low, Close, Volume) para el par BTC/USDT del mercado de futuros de Binance.

  * **Responsabilidades y Funcionalidades Exactas:**

    1.  **Conexión Segura:** Establecer conexión autenticada con la API de Futuros de Binance utilizando claves API.
    2.  **Descarga Parametrizada:** Obtener datos OHLCV para un símbolo configurable (ej. "BTCUSDT"), intervalo de tiempo (granularidad de vela, ej. "1h", "15m"), y un rango de fechas (fecha de inicio configurable hasta la fecha/hora actual).
    3.  **Manejo de Paginación:** Gestionar eficientemente la paginación de la API de Binance para recuperar grandes volúmenes de datos históricos (Binance limita la cantidad de datos por solicitud, típicamente 500-1500 velas).
    4.  **Gestión de Errores y Límites de Tasa:** Implementar lógica robusta para manejar errores de conexión, timeouts, y límites de tasa (`rate limits`) impuestos por la API de Binance, incluyendo estrategias de reintento (ej. `exponential backoff`).
    5.  **Validación de Datos:** Realizar verificaciones básicas de integridad de los datos descargados (formatos correctos, orden cronológico).
    6.  **Almacenamiento Eficiente:** Guardar los datos históricos descargados en formato **Parquet** en Google Cloud Storage (GCS). El formato Parquet es preferido sobre CSV por su compresión eficiente y rendimiento optimizado para consultas analíticas en la nube.
    7.  **Estructura de Almacenamiento:** Los datos en GCS seguirán una estructura de directorios lógica y predecible, facilitando su acceso y gestión (ej. `gs://<nombre-del-bucket>/data/raw/futures/<simbolo>/<intervalo>/<nombre_archivo.parquet>`).
    8.  **Registro (Logging):** Registrar detalladamente todas las operaciones significativas, progreso de la descarga, advertencias y errores en Cloud Logging.

  * **Implementación en GCP:**

    1.  **Orquestación:** Se define como un componente dentro de un pipeline de Vertex AI.
    2.  **Ejecución del Código:**
          * El código Python (basado en la clase `BinanceFuturesDownloader` detallada en `adquisicion_datos.md`) se empaqueta en la imagen Docker única del proyecto.
          * Este contenedor es ejecutado por Vertex AI Pipelines en una instancia de cómputo gestionada.
    3.  **Gestión de Configuración y Secretos:**
          * **Claves API de Binance:** Almacenadas de forma segura en **Google Cloud Secret Manager**. El componente del pipeline accederá a estas claves en tiempo de ejecución con los permisos adecuados.
          * **Parámetros del Componente/Pipeline:** Los parámetros como `symbol`, `interval`, `start_date`, `api_request_limit_per_call`, `api_request_delay_seconds` se definirán como parámetros del componente de Vertex AI Pipeline. Esto permite flexibilidad para configurar las descargas sin modificar el código del contenedor.
              * `symbol` (str): Símbolo del par de trading de futuros (ej. "BTCUSDT").
              * `interval` (str): Granularidad de las velas (ej. "1h", "15m", "1d").
              * `historical_start_date` (str): Fecha de inicio para la descarga en formato "YYYY-MM-DD".
              * `api_request_limit_per_call` (int): Número máximo de velas por solicitud a la API (ej. 1000).
              * `api_request_delay_seconds` (float): Pausa en segundos entre solicitudes a la API para respetar los límites de tasa (ej. 0.5).
              * `api_retry_attempts` (int): Número de reintentos en caso de error de API (ej. 5).
              * `api_retry_delay_seconds` (int): Retardo base en segundos para los reintentos (ej. 60).
              * `gcs_output_raw_data_path` (URI GCS): Ruta base en GCS donde se guardarán los archivos Parquet de salida.
    4.  **Almacenamiento de Datos Crudos:**
          * Los datos se guardan como archivos Parquet en la ruta GCS especificada por `gcs_output_raw_data_path`.
          * **Convención de Nomenclatura de Archivos:** Clara y estandarizada, incluyendo símbolo, intervalo y rango de fechas (ej. `BTCUSDT_FUTURES_1h_20200101_20250518.parquet`). La fecha de finalización en el nombre será la fecha de la ejecución de la descarga.
          * **(Opcional) Carga a BigQuery:** Se puede añadir un paso posterior (o integrar en este componente si la lógica no es muy pesada) para cargar los datos Parquet de GCS en una tabla de BigQuery para facilitar el análisis exploratorio y el versionado. BigQuery también puede consultar directamente archivos Parquet en GCS como tablas externas.
    5.  **Logging:** El uso de la librería `logging` estándar de Python dentro del script del componente resultará en la ingesta automática de logs en Cloud Logging.
    6.  **Programación de Ejecuciones (Opcional):** Para actualizaciones periódicas de datos históricos, se puede usar **Cloud Scheduler** para invocar el pipeline de Vertex AI que contiene este componente de descarga.

  * **Entradas del Componente (Vertex AI Pipeline):**

      * `project_id` (str): ID del proyecto GCP.
      * `location` (str): Región GCP donde se ejecuta el pipeline.
      * `binance_api_key_secret_name` (str): Nombre del secreto en Secret Manager que contiene la clave API de Binance.
      * `binance_api_secret_secret_name` (str): Nombre del secreto en Secret Manager que contiene la clave secreta de Binance.
      * `symbol` (str): Par de trading (ej. "BTCUSDT").
      * `interval` (str): Intervalo de las velas (ej. "1h").
      * `historical_start_date` (str): Fecha de inicio (ej. "2020-01-01").
      * (Otros parámetros de API y reintentos como se listaron arriba).

  * **Salidas del Componente (Vertex AI Pipeline):**

      * `output_raw_data_gcs_uri`: `OutputPath(Dataset)` o `OutputPath(str)` que apunta a la URI de GCS donde se guardaron los archivos Parquet. Este artefacto será la entrada para el siguiente componente de preprocesamiento.
      * `execution_summary`: `OutputPath(Metrics)` o `OutputPath(Artifact)` para guardar un resumen de la descarga (ej. número de velas descargadas, rango de fechas cubierto, errores).

  * **Librerías Clave (a incluir en `requirements.txt` y el Dockerfile):**

      * `python-binance`: Para interactuar con la API de Binance.
      * `pandas`: Para la manipulación de datos y la creación de DataFrames.
      * `pyarrow` o `fastparquet`: Motores para leer/escribir archivos Parquet con Pandas.
      * `google-cloud-storage`: Para interactuar con GCS (subir archivos Parquet).
      * `google-cloud-secret-manager`: Para acceder a las claves API de forma segura.
      * `PyYAML` (si se usa un archivo de config local para defaults, aunque se prefieren params de pipeline).
      * `python-dotenv` (menos relevante en el entorno contenerizado de GCP si los secretos se manejan vía Secret Manager).

  * **Manejo del Código `BinanceFuturesDownloader` (Adaptación):**

      * La clase `BinanceFuturesDownloader` de `adquisicion_datos.md` se adaptará para:
          * Aceptar los parámetros de configuración y las credenciales de API como argumentos en su `__init__` (en lugar de leerlos directamente de `ConfigManager` que lee archivos locales).
          * Escribir la salida en formato Parquet a una ruta GCS (pasada como argumento) en lugar de un CSV local. Se usará `df.to_parquet(gcs_path, ...)`.
          * La lógica de conexión, paginación, manejo de errores y reintentos se mantiene.

  * **Actualizaciones Incrementales (Consideración para Futuras Mejoras):**

      * La lógica actual descarga todo el rango desde `historical_start_date` hasta "ahora". Para ejecuciones regulares, esto implicaría volver a descargar datos ya existentes.
      * Una mejora futura sería implementar una lógica de actualización incremental:
        1.  El componente verifica la última fecha/timestamp de los datos ya almacenados en GCS/BigQuery para el símbolo e intervalo dados.
        2.  Ajusta la fecha de inicio de la descarga para obtener solo los datos nuevos desde esa última fecha.
        3.  Añade los nuevos datos a los existentes (ej. nuevos archivos Parquet o nuevas particiones en BigQuery).
      * Esto requiere una gestión de estado más sofisticada y metadatos sobre los datos ya descargados.

#### 4.1.B. Adquisición de Datos en Tiempo Real (Diseño Conceptual para Futuro Despliegue en Vivo)

Aunque el enfoque inicial es el pipeline de entrenamiento/backtesting, se esboza el diseño para la adquisición de datos en tiempo real para completitud y para asegurar que el diseño modular facilite esta extensión.

  * **Responsabilidades:**

    1.  Establecer y mantener conexiones WebSocket persistentes con la API de Binance.
    2.  Suscribirse a flujos de datos en tiempo real para los pares de trading relevantes (ej. trades, klines/candlesticks actualizados, libro de órdenes, datos de cuenta).
    3.  Manejar la deserialización de los mensajes del WebSocket.
    4.  Publicar los datos crudos recibidos en un sistema de mensajería (`Pub/Sub`) para su consumo por otros módulos en tiempo real (preprocesamiento, agente, monitorización).
    5.  Gestionar reconexiones automáticas, errores de conexión y límites de la API.

  * **Arquitectura GCP Sugerida:**

    1.  **Cliente WebSocket:** Un servicio `Cloud Run` (preferido por su flexibilidad y escalabilidad) o una `Cloud Function` (2ª generación para tiempos de ejecución más largos si es necesario) ejecutaría el script Python cliente de WebSocket.
          * Este servicio se configuraría para mantener la conexión abierta.
          * Puede escalar a múltiples instancias si se monitorizan muchos pares o flujos.
    2.  **Sistema de Mensajería (`Pub/Sub`):**
          * El servicio cliente de WebSocket publicaría cada mensaje de Binance recibido (trade, kline, etc.) en un topic de `Pub/Sub` apropiado (ej. un topic para trades de BTCUSDT, otro para klines de 1m de BTCUSDT).
          * Esto desacopla la ingesta de datos de los módulos consumidores, proporcionando resiliencia, buffering y la capacidad de tener múltiples consumidores independientes.
    3.  **Gestión de Secretos:** `Secret Manager` para las API keys si fueran necesarias para alguna parte de la conexión WebSocket o para obtener datos de cuenta.
    4.  **Persistencia (Opcional en este módulo, puede ser un consumidor):** Un pipeline de `Dataflow` podría suscribirse a los topics de `Pub/Sub` y escribir los datos en tiempo real en `BigQuery` (para análisis y archivo) y/o en `Vertex AI Feature Store` si se generan características de muy baja latencia.

---

**PARTE 2: MÓDULO DE PREPROCESAMIENTO Y GESTIÓN DE DATOS**

---

### 4.2. Módulo 2: Preprocesamiento y Gestión de Datos

Este módulo es fundamental para transformar los datos crudos de mercado (obtenidos por el Módulo 1) en un formato estructurado, normalizado y secuencial que pueda ser consumido eficazmente por el agente de Reinforcement Learning basado en Transformers. La calidad de este preprocesamiento impactará directamente en la capacidad del agente para aprender patrones y tomar decisiones de trading informadas. Todas las operaciones se realizarán de manera causal para evitar el sesgo de anticipación (lookahead bias).

* **Responsabilidades y Funcionalidades Exactas:**
    1.  **Carga de Datos Crudos:**
        * Leer los datos históricos OHLCV (en formato Parquet) desde la ruta GCS especificada por el componente de Adquisición de Datos (Módulo 1.A).
        * Convertir timestamps a objetos datetime de Pandas (asegurando que sean UTC) y establecerlos como índice.
        * Asegurar la correcta conversión de tipos para las columnas OHLCV a numéricos.
    2.  **Limpieza y Preparación Inicial de Datos:**
        * **Manejo de Índices:** Verificar que el índice de tiempo (Open\_Time) sea único y esté ordenado cronológicamente. Eliminar duplicados (manteniendo la primera ocurrencia) y reordenar si es necesario.
        * **Manejo de NaNs en Datos Crudos:**
            * Detectar y reportar NaNs en las columnas OHLCV.
            * Aplicar una imputación limitada usando `forward fill` (`ffill`) con un límite configurable (`raw_data_ffill_limit_for_nans`, ej., 1-4 periodos) para rellenar huecos pequeños y esporádicos.
            * Eliminar cualquier fila restante que aún contenga NaNs en las columnas OHLCV críticas después del `ffill` (o si el `ffill` está desactivado).
            * Registrar detalladamente el número de NaNs encontrados, rellenados y las filas eliminadas.
        * **Correcciones Menores:** Reemplazar valores de `Open == 0` con un valor muy pequeño (ej. `1e-9`) para evitar errores en cálculos logarítmicos, asumiendo que es una anomalía del dato si otras columnas de precio no son cero.
    3.  **Ingeniería de Características de Mercado:**
        * Implementar utilizando la librería `pandas-ta` y cálculos directos con Pandas/NumPy.
        * **Características Basadas en OHLCV Procesado (5 features):**
            * Retorno logarítmico: `log(Close / Open)`
            * Retorno logarítmico: `log(High / Open)`
            * Retorno logarítmico: `log(Low / Open)`
            * Retorno logarítmico: `log(Close / Close_prev)` (retorno de cierre a cierre)
            * Retorno logarítmico: `log(Volume / SMA(Volume, N_vol_sma))` (donde `N_vol_sma` es configurable, ej. 20).
        * **Indicadores Técnicos (15 features):**
            * Medias Móviles Simples (SMA): SMA(20), SMA(50) (periodos configurables).
            * Medias Móviles Exponenciales (EMA): EMA(12), EMA(26) (periodos configurables).
            * Índice de Fuerza Relativa (RSI): RSI(14) (periodo configurable).
            * Rango Verdadero Promedio (ATR): ATR(14) (periodo configurable).
            * Divergencia/Convergencia de Medias Móviles (MACD): MACD(12,26,9) - Línea MACD, Línea de Señal, Histograma (periodos configurables).
            * Bandas de Bollinger (BBands): BBands(20,2) - Banda Superior, Banda Media (SMA), Banda Inferior. Se derivarán características como distancia del precio a las bandas y ancho de las bandas (periodos y desviación estándar configurables).
            * Índice de Canal de Materias Primas (CCI): CCI(20) (periodo configurable).
            * Oscilador Estocástico: Estocástico Lento (%K, %D) con parámetros (ej. 14,3,3 - k, d, smooth\_k configurables).
    4.  **Normalización/Escalado Causal de Características de Mercado:**
        * El objetivo es transformar todas las características de mercado a escalas comparables y, en la medida de lo posible, estacionarias, utilizando únicamente información pasada.
        * **Z-score sobre Ventana Móvil:** Aplicar a características cuya magnitud puede variar considerablemente y no tienen un rango intrínseco (ej. algunos retornos logarítmicos, histograma MACD si no se normaliza de otra forma, CCI). La ventana de normalización será `L * normalization_window_multiplier` (donde `L` es la longitud de la secuencia y `normalization_window_multiplier` es configurable, ej. 2). La media y desviación estándar se calculan sobre los `W` periodos anteriores.
            * $z_t = (x_t - \text{mean}(x_{t-W : t-1})) / (\text{std}(x_{t-W : t-1}) + \epsilon)$ (donde $\epsilon$ es un valor pequeño como `1e-9` para evitar división por cero).
        * **Escalado Específico para Indicadores con Rango Definido:**
            * **RSI:** Escalar a `[0, 1]` (dividiendo `RSI / 100.0`) o `[-1, 1]` (aplicando `(RSI - 50.0) / 50.0`). La opción `[0, 1]` se usará inicialmente.
            * **Oscilador Estocástico (%K, %D):** Escalar a `[0, 1]` (dividiendo por 100.0).
        * **Normalización Relativa para Indicadores de Nivel/Precio (SMAs, EMAs, Bandas de Bollinger):**
            * En lugar de un Z-score directo, normalizarlos en relación con el precio de cierre actual (`Close_t`) o el ATR actual (`ATR_t`) para convertirlos en una desviación relativa o porcentual. Ejemplos:
                * `(Indicador_t - Close_t) / ATR_t`
                * `(Indicador_t - Close_t) / Close_t`
                * `Indicador_t / Close_t - 1`
        * **Normalización del ATR:** `ATR_t / Close_t`.
        * **Normalización del MACD (Línea, Señal, Histograma):** Se pueden normalizar dividiendo por `ATR_t` o aplicando Z-score móvil.
        * **Normalización de Bandas de Bollinger:**
            * `Distancia a Banda Superior Normalizada`: `(BB_Upper_t - Close_t) / ATR_t`
            * `Distancia a Banda Inferior Normalizada`: `(Close_t - BB_Lower_t) / ATR_t`
            * `Ancho de Banda Normalizado`: `(BB_Upper_t - BB_Lower_t) / ATR_t` o `(BB_Upper_t - BB_Lower_t) / BB_Middle_t`.
        * La lista exacta de las 20 características finales y sus métodos de normalización se definirán en el archivo de configuración (`preprocessing_config.yaml`).
    5.  **Construcción de Secuencias de Estado (Características de Mercado):**
        * Transformar el DataFrame de series temporales (que contiene las `N_features_mercado = 20` características de mercado calculadas y normalizadas por KLine) en un array NumPy tridimensional de forma `(N_samples, L, N_features_mercado)`.
        * `L` (Longitud de la secuencia): Configurable, por defecto 96 (representando, por ejemplo, 4 días de datos horarios).
        * Se utiliza una ventana deslizante sobre el DataFrame de características normalizadas para generar estas secuencias. Cada muestra `X_market[i]` será una secuencia de `L` pasos temporales consecutivos.
    6.  **Manejo de NaNs Inducidos por Cálculos:**
        * Los cálculos de indicadores con periodos de lookback (ej. SMA(50)) y las normalizaciones con ventana móvil introducirán valores `NaN` al principio del DataFrame.
        * Todas las filas que contengan `NaN` en *cualquiera* de las características finales seleccionadas se eliminarán *antes* de la construcción de secuencias. Esto asegura que cada secuencia alimentada al modelo contenga únicamente datos válidos. El número de filas iniciales descartadas dependerá del periodo de lookback más largo utilizado en cualquier cálculo de característica o normalización.
    7.  **Almacenamiento de Datos Procesados:**
        * Guardar el array NumPy de secuencias de características de mercado (`X_market`) y un array correspondiente de timestamps (`timestamps`, representando el timestamp del *último* KLine de cada secuencia) en un único archivo comprimido `.npz` en Google Cloud Storage.
        * La ruta de salida en GCS será estructurada e incluirá identificadores de los parámetros clave del preprocesamiento para el versionado (ej. `gs://<bucket>/data/processed/futures/<simbolo>/<intervalo>/L<L_val>_norm<norm_window_multi_val>/<input_filename_base>_sequences.npz`).
    8.  **Registro (Logging):** Registrar detalladamente todos los pasos del preprocesamiento, advertencias (ej. sobre NaNs), errores y estadísticas de los datos generados (ej. forma del array de secuencias) en Cloud Logging.

* **Implementación en GCP:**
    1.  **Orquestación:** Se define como un componente dentro de un pipeline de Vertex AI.
    2.  **Ejecución del Código:**
        * El código Python para este módulo (principalmente las clases `FeatureEngineer` y `DataPreprocessor` adaptadas de `preprocesamiento.md`) se empaqueta en la imagen Docker única del proyecto.
        * Este contenedor es ejecutado por Vertex AI Pipelines, consumiendo el artefacto de datos crudos (ruta GCS a los archivos Parquet) producido por el Módulo 1.A.
    3.  **Gestión de Configuración:**
        * Los parámetros clave del preprocesamiento (ej. `sequence_length_L`, `normalization_window_multiplier_for_L`, periodos de indicadores, tipo de normalización por característica, lista de las 20 características finales, `raw_data_ffill_limit_for_nans`) se definirán como **parámetros del componente de Vertex AI Pipeline**.
        * Esto permite la experimentación con diferentes estrategias de preprocesamiento sin reconstruir el contenedor. Los valores por defecto pueden residir en un archivo `preprocessing_config.yaml` dentro del contenedor, y los parámetros del pipeline los sobrescriben.
    4.  **Carga de Datos Crudos:**
        * El componente leerá los datos Parquet desde la ruta GCS pasada como entrada utilizando `pandas.read_parquet()`.
    5.  **Lógica de Preprocesamiento (Clases `FeatureEngineer`, `DataPreprocessor`):**
        * La clase `FeatureEngineer` se adaptará para usar `pandas-ta` para el cálculo de indicadores técnicos. Se prestará especial atención a los nombres de las columnas generadas por `pandas-ta` para asegurar la consistencia.
        * La clase `DataPreprocessor` orquestará la carga, limpieza (incluyendo el manejo de NaNs en datos crudos con `ffill_limit`), llamada a `FeatureEngineer`, aplicación de la normalización causal final a las 20 características seleccionadas, eliminación de NaNs inducidos, y la creación de secuencias.
    6.  **Almacenamiento de Datos Procesados:**
        * El archivo `.npz` resultante (conteniendo `X_market` y `timestamps`) se guardará en la ruta GCS especificada como un artefacto de salida del componente.
        * **Metadatos:** El pipeline registrará la URI GCS del archivo `.npz` en Vertex AI ML Metadata. Adicionalmente, se pueden registrar explícitamente los parámetros de preprocesamiento utilizados (longitud de secuencia, etc.) como metadatos del artefacto o del pipeline.
    7.  **Logging:** El uso de la librería `logging` estándar de Python dentro del script del componente resultará en la ingesta automática de logs en Cloud Logging.

* **Entradas del Componente (Vertex AI Pipeline):**
    * `project_id` (str): ID del proyecto GCP.
    * `location` (str): Región GCP.
    * `gcs_raw_data_uri`: `Input[Dataset]` o `str` que apunta a la URI GCS de los datos Parquet crudos del Módulo 1.A.
    * `sequence_length_L` (int): Longitud de las secuencias (ej. 96).
    * `normalization_window_multiplier_for_L` (int): Multiplicador para la ventana de normalización Z-score (ej. 2).
    * `raw_data_ffill_limit_for_nans` (int): Límite para ffill en NaNs de datos crudos (ej. 4, 0 para desactivar).
    * (Otros parámetros configurables para los periodos de los indicadores, selección de los 20 features finales y sus estrategias de normalización específicas. Estos pueden ser numerosos, por lo que una opción es pasar una cadena JSON o la ruta GCS a un archivo YAML de configuración detallada para los features si los parámetros individuales son demasiados para el pipeline). Alternativamente, se pueden fijar en el código y solo exponer los más críticos. Dada tu preferencia por la versatilidad, se buscará un equilibrio.

* **Salidas del Componente (Vertex AI Pipeline):**
    * `processed_sequences_gcs_uri`: `Output[Dataset]` o `OutputPath(str)` que apunta a la URI GCS del archivo `.npz` con las secuencias procesadas.
    * `preprocessing_summary`: `Output[Metrics]` para guardar estadísticas como el número de secuencias generadas, el rango de fechas cubierto por las secuencias, y el número de features.

* **Librerías Clave (a incluir en `requirements.txt` y el Dockerfile único):**
    * `pandas`
    * `numpy`
    * `pandas-ta` (reemplaza la necesidad de TA-Lib y sus complejas dependencias de compilación).
    * `scikit-learn` (para cualquier utilidad de preprocesamiento si se decide usar, aunque la normalización móvil se puede hacer con Pandas).
    * `PyYAML` (para leer archivos de configuración de defaults si se usa esa estrategia).
    * `google-cloud-storage` (para leer/escribir desde/hacia GCS).

* **Estructura del Código (`FeatureEngineer`, `DataPreprocessor`):**
    * Se mantendrá la estructura de clases que diseñaste, adaptando `FeatureEngineer` para `pandas-ta` y asegurando que `DataPreprocessor` maneje correctamente las rutas GCS y los parámetros del pipeline.
    * La lista de las 20 `final_market_feature_columns` y sus métodos de normalización específicos (ej. qué columnas usar Z-score, cuáles dividir por ATR/Close, cuáles escalar a [0,1]) será un punto crítico de configuración.
    * El script orquestador (`preprocess_data.py` en tu diseño local) se adaptará para ser el punto de entrada del componente de Vertex AI Pipeline, parseando los argumentos del componente y llamando a `DataPreprocessor`.

* **Normalización Causal Detallada:**
    * Se enfatiza que todas las medias, desviaciones estándar, o valores de referencia (como `Close_t` o `ATR_t` para normalización relativa) usados para normalizar la característica $x_t$ deben provenir de datos estrictamente anteriores a $t$ (es decir, $t-1, t-2, \dots$). Pandas `rolling(...).mean().shift(1)` es un patrón común para obtener la media de la ventana anterior. Para normalizaciones que usan el valor actual (ej. $x_t / Close_t$), esto es aceptable ya que $Close_t$ es parte de la información del paso $t$.

---

Esta sección cubre en detalle el Módulo 2. La clave aquí es la adaptación de tu sólida lógica de preprocesamiento local a un componente robusto y configurable dentro de un pipeline de GCP, utilizando `pandas-ta` y gestionando los datos en GCS.

---

**PARTE 3: MÓDULO DE ENTORNO DE TRADING (SIMULADOR)**

---

### 4.3. Módulo 3: Entorno de Trading (Simulador)

Este módulo es el corazón de la simulación donde el agente de Reinforcement Learning (RL) interactuará para aprender y donde se evaluarán sus estrategias mediante backtesting. Se basa en la interfaz estándar de `gymnasium.Env` para asegurar la compatibilidad con librerías de RL como Stable Baselines3. El diseño prioriza un realismo considerable en la simulación de las operaciones de futuros de Bitcoin en Binance, al mismo tiempo que proporciona la flexibilidad necesaria para el entrenamiento y la evaluación.

* **Responsabilidades y Funcionalidades Exactas:**
    1.  **Interfaz `gymnasium.Env`:**
        * La clase principal, `TradingEnvironment`, hereda de `gymnasium.Env`.
        * Implementa los métodos esenciales: `__init__`, `reset`, `step`, `render` (inicialmente para logging básico del estado/equity, con potencial para visualizaciones más avanzadas), y `close`.
        * Define de manera precisa y dinámica los espacios de observación (`observation_space`) y acción (`action_space`) basados en la configuración y los datos cargados.
    2.  **Gestión del Estado de la Cartera de Futuros:**
        * Mantiene y actualiza en cada paso del tiempo (`step`) el estado detallado de una cuenta de trading simulada:
            * `initial_equity_config`: Equity inicial global configurado para el entorno.
            * `initial_equity_episode`: Equity al inicio del episodio actual (se resetea a `initial_equity_config` al inicio de cada episodio). Se usa para calcular el drawdown del episodio.
            * `current_equity`: Valor total actual de la cuenta (colateral + P&L no realizado).
            * `balance`: Efectivo o colateral disponible (considerando el margen utilizado).
            * `active_position_side`: Entero que indica la naturaleza de la posición actual (-1 para Corto, 0 para Neutral/Sin Posición, 1 para Largo).
            * `active_position_size_contracts`: Tamaño de la posición activa en unidades de la criptomoneda base (ej. BTC).
            * `active_position_entry_price`: Precio medio de entrada de la posición activa.
            * `unrealized_pnl`: Ganancia o pérdida flotante (no realizada) de la posición activa, calculada con el precio de mercado actual.
            * `margin_used`: Margen retenido para la posición activa, calculado como `(position_size_contracts * entry_price) / configured_leverage`.
            * `available_margin`: `current_equity - margin_used`.
            * `configured_leverage`: Apalancamiento fijo aplicado a las operaciones (ej. 5x, 10x), configurable.
            * `steps_in_current_position`: Contador de pasos (KLines) transcurridos desde que se abrió la posición actual.
            * `total_trades_episode`: Contador del número de trades completados en el episodio actual.
            * `cumulative_fees_paid_episode`: Suma de todas las comisiones pagadas en el episodio actual.
            * `last_step_log_return`: Almacena el retorno logarítmico del equity del paso anterior, usado como parte de las `portfolio_features`.
    3.  **Simulación de Ejecución de Órdenes (vía `SimulatedBroker`):**
        * Se utiliza una clase auxiliar `SimulatedBroker` para encapsular la lógica de cálculo de los detalles de ejecución, manteniendo el `TradingEnvironment` más enfocado en el estado y el flujo.
        * **Comisiones de Trading (Taker Fee):** Aplica una comisión configurable (ej., 0.04% de Binance para takers) sobre el valor nocional de cada operación (tanto al abrir como al cerrar una posición). El valor nocional considera el apalancamiento.
        * **Modelado de Slippage (Deslizamiento):** Para simular la diferencia entre el precio esperado y el precio real de ejecución en un mercado líquido. Se ajusta el precio de ejecución en contra del agente, basado en un múltiplo del ATR (Average True Range) del KLine actual: `slippage_atr_multiplier * ATR(periodo_atr_slippage)`.
            * Compra (Largo): `execution_price = current_market_close_price + (slippage_factor_calculado)`.
            * Venta (Corto o cierre de Largo): `execution_price = current_market_close_price - (slippage_factor_calculado)`.
            * El `ATR` para el slippage se calcula sobre los datos de mercado.
        * **Mínimos de Orden:** Verifica si el tamaño de la posición calculada (`position_size_contracts`) cumple con los mínimos de orden de Binance (ej., 0.001 BTC para futuros BTCUSDT). Si no se cumple, la orden se considera rechazada en la simulación.
    4.  **Lógica de Operación:**
        * **"Una Operación a la Vez":** El entorno solo permite mantener una posición (Larga o Corta) o estar en estado Neutral. No se permite promediar posiciones ni abrir múltiples posiciones simultáneas en la misma dirección o en direcciones opuestas.
        * **Dimensionamiento de Posición Fijo (Porcentual al Equity Actual):** Cuando se abre una nueva posición, su valor nocional es `current_equity * position_size_pct_equity`. El tamaño en contratos se calcula como `(current_equity * position_size_pct_equity * configured_leverage) / execution_price`.
    5.  **Definición del Espacio de Observación (`observation_space`):**
        * Un `gymnasium.spaces.Dict` para manejar la heterogeneidad de los datos de entrada del agente:
            * `'market_features'`: Un `gymnasium.spaces.Box` de forma `(L, N_features_mercado)` (ej. `(96, 20)`), conteniendo la secuencia de `L` pasos temporales de las `N_features_mercado` características de mercado preprocesadas por el Módulo 2. Los valores son de tipo `np.float32`.
            * `'portfolio_features'`: Un `gymnasium.spaces.Box` de forma `(8,)`, conteniendo 8 características normalizadas que describen el estado actual de la cartera y la posición. Los valores son de tipo `np.float32`. Estas características son:
                1.  `Estado Posición`: Normalizado o codificado (ej. -1.0 para Corto, 0.0 para Neutral, 1.0 para Largo).
                2.  `Tamaño Posición Normalizado`: El valor nocional de la posición actual (`active_position_size_contracts * active_position_entry_price`) dividido por el `initial_equity_episode`, acotado para evitar valores extremos.
                3.  `Precio Entrada Normalizado (Relativo al Precio Actual)`: Por ejemplo, `(current_market_close_price - active_position_entry_price) / active_position_entry_price` para Largos (o similar, normalizado por ATR o volatilidad). Si no hay posición, este valor es 0.
                4.  `P&L No Realizado Normalizado`: `unrealized_pnl / current_equity` (si `current_equity > 0`, de lo contrario 0).
                5.  `Retorno Log Equity (del último paso)`: El valor de `self.last_step_log_return`.
                6.  `Ratio de Margen Utilizado`: `margin_used / current_equity` (si `current_equity > 0`, de lo contrario 0).
                7.  `Pasos en Posición Actual Normalizados`: `steps_in_current_position / max_steps_norm_divisor` (donde `max_steps_norm_divisor` podría ser `L` o un valor configurable, ej. `L*3`), acotado a `[0, 1]`.
                8.  `Apalancamiento Configurado Normalizado`: `configured_leverage / max_allowable_leverage` (ej. si `max_allowable_leverage` es 125x para Binance). O simplemente el valor del apalancamiento si se considera que el agente puede aprender su escala.
    6.  **Definición del Espacio de Acciones (`action_space`):**
        * Un `gymnasium.spaces.Box` continuo de una dimensión: `Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)`. Este único valor representa la señal de acción del agente.
    7.  **Interpretación de la Acción del Agente (`action_signal`):**
        * La señal continua `action_signal` del agente se interpreta usando umbrales configurables (ej. `action_threshold = 0.15`):
            * `action_signal > action_threshold`: Intento de Abrir/Mantener Largo.
                * Si Neutral: Abrir nueva posición Larga.
                * Si Corto: Cerrar posición Corta existente y luego Abrir nueva posición Larga (dos transacciones simuladas).
                * Si Largo: Mantener posición Larga existente (no se reabre ni se aumenta el tamaño).
            * `action_signal < -action_threshold`: Intento de Abrir/Mantener Corto.
                * Si Neutral: Abrir nueva posición Corta.
                * Si Largo: Cerrar posición Larga existente y luego Abrir nueva posición Corta (dos transacciones simuladas).
                * Si Corto: Mantener posición Corta existente.
            * `-action_threshold <= action_signal <= action_threshold`: Intento de Cerrar Posición / Mantener Neutral.
                * Si Largo o Corto: Cerrar la posición activa actual.
                * Si Neutral: Mantener estado Neutral.
    8.  **Cálculo de la Función de Recompensa (`reward`):**
        * La recompensa por cada paso `t` se define como el cambio logarítmico en el equity total de la cuenta: `recompensa_t = np.log(current_equity_t / current_equity_at_step_start)`. Si `current_equity_at_step_start` es cero o negativo, la recompensa es cero o una penalización grande. `current_equity_at_step_start` es el equity antes de que la acción del paso actual se procese.
    9.  **Simulación de Liquidación de Posición:**
        * Ocurre si el precio de mercado se mueve adversamente contra la posición abierta un porcentaje que excede el margen disponible. El precio de liquidación se calcula basado en el `active_position_entry_price`, el `configured_leverage`, y el balance de la cuenta. Típicamente, para un Largo, si el precio cae aproximadamente `(1 / configured_leverage)` (ajustado por un factor de seguridad y comisiones de mantenimiento de Binance, si se modelan).
        * Si el `Low` del KLine actual (para posiciones Largas) o el `High` (para Cortas) cruza este precio de liquidación calculado, la posición se cierra forzosamente al precio de liquidación (aplicando comisión). El `current_equity` se actualiza (potencialmente a un valor muy bajo o cero si el colateral no cubre la pérdida). En este caso, `terminated = True` para el episodio.
    10. **Condiciones de Fin de Episodio:**
        * `terminated = True` (el episodio termina debido a una condición del entorno relacionada con el objetivo o fallo):
            * **Drawdown Máximo de Equity del Episodio:** Si `current_equity <= initial_equity_episode * (1.0 + equity_drawdown_threshold_episode_end)` (ej. `equity_drawdown_threshold_episode_end = -0.20` para un drawdown del 20%).
            * **Liquidación de Posición:** Como se describió arriba.
            * **Equity Insuficiente:** Si `current_equity` cae por debajo de un umbral mínimo necesario para abrir nuevas posiciones (considerando el `min_order_size_btc` y el `position_size_pct_equity`).
        * `truncated = True` (el episodio termina por una condición externa, como un límite de tiempo, sin ser necesariamente un fallo):
            * **Agotamiento del Conjunto de Datos:** Si el entorno alcanza el final de los datos de mercado disponibles y no puede formar una secuencia completa para la siguiente observación (`current_step_index` alcanza el final del array de secuencias de mercado).
    11. **Reinicio de Episodio (`reset` method):**
        * Restablece el estado de la cartera a sus valores iniciales (`current_equity = initial_equity_config`, sin posiciones activas, etc.).
        * Selecciona un índice de inicio para el nuevo episodio dentro del conjunto de datos de secuencias de mercado cargado. Este inicio será **aleatorio** si `random_episode_start=True` (para entrenamiento), o fijo al inicio del dataset (`current_step_index = 0`) si `random_episode_start=False` (para backtesting/evaluación reproducible). Se asegura que haya suficientes datos para al menos una secuencia completa.
        * Devuelve la observación inicial y un diccionario de información (`info`).
    12. **Clase `SimulatedBroker`:**
        * Una clase auxiliar sin estado (o con estado mínimo) responsable de los cálculos relacionados con la ejecución de órdenes.
        * `__init__(self, taker_fee_rate: float, slippage_atr_multiplier: float, min_order_size_btc: float, atr_period_slippage: int)`
        * `calculate_execution_details(self, desired_action: str, market_close_price: float, atr_value: float, position_to_close_entry_price: float = None, position_to_close_size: float = None)`: Calcula el precio de ejecución (con slippage), P&L potencial (si es un cierre), y la comisión a pagar. No modifica el estado de la cuenta, solo devuelve los detalles calculados.
        * `calculate_position_size_contracts(self, current_equity: float, position_size_pct_equity: float, configured_leverage: float, execution_price: float)`: Calcula el tamaño de la posición en la criptomoneda base y verifica si cumple con `min_order_size_btc`.
        * `calculate_margin_required(self, position_size_contracts: float, execution_price: float, configured_leverage: float)`: Calcula el margen necesario para la posición.
        * `calculate_liquidation_price(self, entry_price: float, position_size_contracts: float, position_side: int, balance_before_margin: float, configured_leverage: float)`: Calcula el precio de liquidación aproximado.

* **Integración en GCP y Vertex AI Pipelines:**
    1.  **Uso en Componentes de Entrenamiento/Evaluación:** El código del `TradingEnvironment` y `SimulatedBroker` (ubicado en `src/environments/`) se incluye en la imagen Docker única del proyecto. No es un componente de pipeline por sí mismo, sino una clase instanciada y utilizada *dentro* de los componentes de entrenamiento (Módulo 4) y backtesting (Módulo 7).
    2.  **Consumo de Datos Procesados:**
        * Cuando se instancia, el `TradingEnvironment` recibe la URI GCS del archivo `.npz` (producido por el Módulo 2) que contiene las secuencias de mercado preprocesadas (`X_market`, `timestamps`).
        * La lógica `_load_market_data` dentro de `TradingEnvironment` descarga este archivo de GCS (usando la librería `google-cloud-storage`) a una ubicación temporal en el contenedor y luego carga los arrays NumPy en memoria.
    3.  **Gestión de Configuración:**
        * Los parámetros de configuración del entorno (ej. `initial_equity`, `leverage`, `taker_fee_rate`, `slippage_atr_multiplier`, `action_threshold`, `equity_drawdown_threshold_episode_end`, `position_size_pct_equity`, `random_episode_start`, parámetros para normalización de características de cartera) se pasarán como **argumentos a los componentes de pipeline** de entrenamiento y backtesting.
        * Estos componentes, a su vez, pasarán estos valores al constructor `__init__` del `TradingEnvironment`.
    4.  **Logging:** Los logs generados por el `TradingEnvironment` (usando la librería `logging` de Python) durante su interacción con el agente (en entrenamiento o backtesting) serán capturados automáticamente por Cloud Logging cuando se ejecuten dentro de un componente de Vertex AI.

* **Parámetros de Configuración Clave (Ejemplos, pasados a través del pipeline):**
    * `gcs_processed_data_uri` (str): URI al archivo `.npz` de secuencias de mercado.
    * `initial_equity` (float): Equity inicial para cada episodio.
    * `leverage` (float): Apalancamiento a utilizar.
    * `position_size_pct_equity` (float): Porcentaje del equity actual para el tamaño nocional de la posición.
    * `taker_fee_rate` (float): Comisión de taker (ej. 0.0004 para 0.04%).
    * `slippage_atr_multiplier` (float): Multiplicador de ATR para el slippage.
    * `atr_period_slippage` (int): Periodo del ATR usado para calcular el slippage (ej. 14).
    * `action_threshold` (float): Umbral para interpretar la señal continua del agente (ej. 0.15).
    * `equity_drawdown_threshold_episode_end` (float): Límite de drawdown del equity del episodio (ej. -0.20 para -20%).
    * `min_order_size_btc` (float): Mínimo tamaño de orden en BTC.
    * `random_episode_start` (bool): `True` para entrenamiento, `False` para backtesting.
    * Parámetros para la normalización de las 8 características de portafolio (ej. `max_steps_in_position_norm_divisor`).

* **Librerías Clave (ya incluidas en el `requirements.txt` general):**
    * `gymnasium`
    * `numpy`
    * `pandas`
    * `google-cloud-storage` (para cargar los datos de GCS).

Este diseño detallado del `TradingEnvironment` y su integración en el ecosistema GCP asegura un simulador robusto, configurable y adecuado para entrenar y evaluar un agente de RL autogestionado. La modularidad con `SimulatedBroker` también mantiene la puerta abierta para una transición más sencilla a un `LiveBroker` en el futuro.

---

**PARTE 4: MÓDULO DE AGENTE DE REINFORCEMENT LEARNING (RL) Y ENTRENAMIENTO**

---

### 4.4. Módulo 4: Agente de Reinforcement Learning (RL) con Transformers y su Entrenamiento en GCP

Este módulo es el cerebro del sistema de trading. Comprende la definición de la arquitectura del agente de RL, la implementación de su lógica de aprendizaje (utilizando el algoritmo Soft Actor-Critic - SAC), y el proceso de entrenamiento robusto y escalable utilizando los servicios de Google Cloud Platform, específicamente Vertex AI Training y Vertex AI Pipelines.

#### 4.4.A. Diseño del Agente de RL

El agente está diseñado para tomar decisiones de trading óptimas basadas en secuencias de datos de mercado y el estado actual de la cartera, procesados a través de una arquitectura neuronal basada en Transformers.

* **Algoritmo de Reinforcement Learning:**
    * **Soft Actor-Critic (SAC):** Se seleccionará SAC debido a su eficiencia de muestra en entornos con espacios de acción continuos, su estabilidad y su robustez inherente gracias a la maximización de la entropía en su función objetivo. Esto fomenta una mayor exploración y políticas menos propensas a converger en óptimos locales subóptimos.
    * **Implementación:** Se utilizará la implementación de SAC proporcionada por la librería **Stable Baselines3 (SB3)**, que está construida sobre PyTorch.

* **Arquitectura del Modelo (Redes del Actor y Crítico con Backbone Transformer):**
    El agente SAC consta de una política (el Actor) y funciones de valor Q (los Críticos). Ambos compartirán un "backbone" o extractor de características común basado en Transformers para procesar la información de entrada.
    1.  **Entrada de Observación (del Módulo 3: `TradingEnvironment`):**
        * El agente recibe una observación estructurada como un `gymnasium.spaces.Dict`:
            * `'market_features'`: Tensor de forma `(L, N_features_mercado)` (ej. `(96, 20)`), conteniendo la secuencia de datos de mercado preprocesados.
            * `'portfolio_features'`: Tensor de forma `(8,)`, conteniendo las características normalizadas de la cartera.
    2.  **Extractor de Características Personalizado (`CustomTransformerFeatureExtractor`):**
        * Se implementará una clase personalizada que herede de `stable_baselines3.common.torch_layers.BaseFeaturesExtractor`.
        * **Fusión de Características de Entrada:**
            * Las 8 `portfolio_features` se replicarán (o transmitirán, "broadcasted" en términos de tensores) a lo largo de la dimensión temporal `L`.
            * Se concatenarán con las `market_features` en cada uno de los `L` pasos temporales.
            * Esto resulta en una secuencia de entrada unificada para el Transformer de forma `(batch_size, L, N_features_mercado + N_portfolio_features)`, donde `N_features_mercado + N_portfolio_features` será, por ejemplo, `20 + 8 = 28`.
        * **Capa de Embedding Lineal de Entrada:**
            * Una capa `torch.nn.Linear` proyectará las `N_features_mercado + N_portfolio_features` (ej. 28) dimensiones de cada paso temporal a la dimensión interna del modelo Transformer (`d_model`, ej. 128).
        * **Positional Encoding Sinusoidal:**
            * Se añadirá un positional encoding fijo (no aprendido) a los embeddings de entrada para proporcionar al Transformer información sobre el orden y la posición relativa de los elementos en la secuencia.
        * **Pila de Capas Encoder del Transformer:**
            * Se utilizará un stack de `N_encoder_layers` (configurable, ej. 3) capas `torch.nn.TransformerEncoderLayer`.
            * Cada `TransformerEncoderLayer` contendrá:
                * Un mecanismo de auto-atención multi-cabeza (`n_heads`, configurable, ej. 4).
                * Una red feed-forward (FFN) con una capa oculta (dimensión interna de FFN, ej. `4 * d_model`).
                * Conexiones residuales y normalización de capas (`LayerNorm`) después de la auto-atención y la FFN.
                * Dropout (configurable, ej. 0.1) para regularización dentro de las capas de atención y FFN.
            * `d_model` (Dimensión del Modelo): Dimensión interna del Transformer (configurable, ej. 128).
        * **Salida del Extractor de Características:**
            * La salida de la última capa del Transformer Encoder será un tensor de forma `(batch_size, L, d_model)`.
            * Para obtener un vector de características único que represente el estado completo de la secuencia, se puede tomar la salida del Transformer correspondiente al último paso temporal `[:, -1, :]`. Alternativamente, se podría aplicar un Global Average Pooling sobre la dimensión temporal. El uso del último paso es común para tareas donde la información más reciente es crucial.
            * Esta salida (un vector de `d_model` dimensiones) será la representación del estado aprendida y se pasará a las redes del actor y el crítico.
            * El `features_dim` del extractor (que SB3 espera) será `d_model`.
    3.  **Red del Actor (Política - $\pi$):**
        * **Entrada:** La representación del estado de `d_model` dimensiones proveniente del `CustomTransformerFeatureExtractor`.
        * **Arquitectura:** Un Perceptrón Multicapa (MLP) (ej. 2 capas ocultas de 256 neuronas cada una, con activación ReLU o GeLU, configurable).
        * **Salida:** Los parámetros (media $\mu$ y desviación estándar $\sigma$) de una **distribución Gaussiana Escalonada (Squashed Gaussian Distribution)**. La acción se muestrea de esta distribución y luego se pasa a través de una función `tanh` para acotarla al rango `[-1, 1]`, que coincide con el `action_space` del entorno.
    4.  **Redes del Crítico (Funciones de Valor Q - $Q_1, Q_2$):**
        * SAC utiliza dos redes Q (Clipped Double-Q Learning) para mitigar la sobreestimación de los valores Q.
        * **Entrada:** Cada red Q toma la representación del estado de `d_model` dimensiones (del extractor) *y* la acción (del actor durante el entrenamiento, o del replay buffer) como entrada. La acción se concatena típicamente a la representación del estado antes de pasarla al MLP.
        * **Arquitectura:** Un MLP para cada red Q (ej. 2 capas ocultas de 256 neuronas cada una, similar al actor, configurable).
        * **Salida:** Un valor escalar que representa el valor Q estimado (el retorno esperado al tomar esa acción en ese estado).

* **Componentes Adicionales de SAC:**
    1.  **Replay Buffer (Memoria de Repetición de Experiencias):**
        * Almacenará tuplas de transición: `(observacion_t, accion_t, recompensa_t, observacion_{t+1}, done_flag_t)`.
        * Para `DictSpace` como el de este entorno, SB3 utilizará un `DictReplayBuffer`.
        * **Tamaño del Buffer:** Configurable (ej. 100,000 a 1,000,000 transiciones). Un buffer más grande permite una mayor diversidad de experiencias, pero consume más memoria.
    2.  **Aprendizaje del Coeficiente de Entropía (`alpha`):**
        * SAC ajusta automáticamente el coeficiente de temperatura `alpha`, que pondera la importancia del término de entropía en la función objetivo. Esto equilibra la maximización de la recompensa esperada con la maximización de la entropía de la política (fomentando la exploración).
        * Se configurará `ent_coef='auto'` en la instanciación del modelo SAC de SB3.
    3.  **Redes Objetivo (Target Networks):**
        * Se utilizan copias de las redes del crítico (y a veces del actor, aunque en SAC es principalmente para el crítico) que se actualizan lentamente (actualización suave o "Polyak averaging" con un coeficiente `tau`) para estabilizar el aprendizaje. `tau` es configurable (ej. 0.005).
    4.  **Manejo de Exploración vs. Explotación:**
        * Inherente al algoritmo SAC a través de su política estocástica (muestreo de la Squashed Gaussian) y la maximización de la entropía.

* **Capacidad de Guardar y Cargar Modelos Entrenados:**
    * Se utilizarán las funcionalidades `model.save()` y `model.load()` de Stable Baselines3. `model.save()` crea un archivo `.zip` que contiene los pesos de todas las redes (actor, críticos, extractor de características) y otros parámetros necesarios para reinstanciar el modelo.

#### 4.4.B. Proceso de Entrenamiento del Agente en GCP

El entrenamiento del agente de RL se orquestará como un componente dentro de un pipeline de Vertex AI.

* **Orquestación:** Un componente de **Vertex AI Pipeline** dedicado al entrenamiento.
* **Ejecución del Código:**
    * El código Python para `CustomTransformerFeatureExtractor`, la lógica de instanciación y entrenamiento del agente SAC (potencialmente encapsulada en una clase `RLAgentManager` o un script `train_rl_agent.py` adaptado) se incluirá en la imagen Docker única del proyecto.
    * Este contenedor se ejecutará en una instancia de cómputo de **Vertex AI Training**, que es un servicio gestionado para el entrenamiento de modelos de ML.
    * **Recursos de Cómputo (Hardware):** El componente del pipeline especificará los recursos necesarios, incluyendo la opción de solicitar **GPUs** (ej. NVIDIA T4, V100, A100). El entrenamiento de modelos Transformer se beneficia enormemente de la aceleración por GPU.
* **Entradas del Componente de Entrenamiento (Vertex AI Pipeline):**
    * `project_id` (str), `location` (str).
    * `processed_market_data_gcs_uri`: `Input[Dataset]` (URI GCS del archivo `.npz` del Módulo 2).
    * Parámetros de configuración del `TradingEnvironment` (Módulo 3), como `env_initial_equity`, `env_leverage`, `env_taker_fee_rate`, `env_slippage_atr_multiplier`, `env_action_threshold`, `env_random_episode_start=True`, etc.
    * Parámetros de configuración del agente SAC y la arquitectura del modelo (de `agent_config.yaml`), como:
        * `agent_learning_rate` (float).
        * `agent_buffer_size` (int).
        * `agent_batch_size` (int).
        * `agent_tau` (float, para actualización de redes objetivo).
        * `agent_gamma` (float, factor de descuento).
        * `agent_train_freq` (int o tupla `(freq, unit)` ej. `(1, "step")`).
        * `agent_gradient_steps` (int).
        * `transformer_d_model` (int).
        * `transformer_n_heads` (int).
        * `transformer_n_encoder_layers` (int).
        * `transformer_dim_feedforward` (int, usualmente `4 * d_model`).
        * `transformer_dropout_rate` (float).
        * `actor_critic_mlp_hidden_layers_pi` (lista de int, ej. `[256, 256]`).
        * `actor_critic_mlp_hidden_layers_qf` (lista de int, ej. `[256, 256]`).
    * `total_training_timesteps` (int): Número total de pasos de interacción con el entorno para el entrenamiento.
    * `model_checkpoint_save_frequency_steps` (int): Frecuencia (en timesteps) para guardar checkpoints del modelo.
    * `input_checkpoint_gcs_uri` (str, opcional): URI GCS a un checkpoint de modelo `.zip` para reanudar el entrenamiento. Si está vacío o no se proporciona, se inicia un nuevo entrenamiento.
* **Lógica del Script de Entrenamiento dentro del Componente:**
    1.  **Configuración Inicial:**
        * Configurar logging para Cloud Logging.
        * Crear directorios locales temporales para checkpoints, logs de TensorBoard, y el modelo final antes de subirlos a GCS.
    2.  **Instanciación del Entorno:**
        * Instanciar `TradingEnvironment` (Módulo 3), pasándole la `processed_market_data_gcs_uri` (que se descarga y carga) y sus parámetros de configuración. `random_episode_start` se establecerá en `True`.
    3.  **Instanciación o Carga del Agente SAC:**
        * Si `input_checkpoint_gcs_uri` se proporciona y es válido:
            * Descargar el archivo `.zip` del checkpoint desde GCS a una ruta local temporal.
            * Cargar el modelo SAC usando `sb3.SAC.load(path_to_local_checkpoint, env=train_env, tensorboard_log=local_tensorboard_log_dir, ...)`.
        * Si no, o si la carga falla:
            * Crear una nueva instancia de `sb3.SAC`, configurando:
                * `policy="MlpPolicy"` (SB3 usa esto como base, las redes se definen en `policy_kwargs`).
                * `env=train_env`.
                * Todos los hiperparámetros del agente (learning rate, buffer size, etc.).
                * `policy_kwargs` para definir el uso de `CustomTransformerFeatureExtractor` y especificar la arquitectura de sus componentes (Transformer) y las redes MLP del actor/crítico.
                * `tensorboard_log=local_tensorboard_log_dir` para que SB3 escriba logs para TensorBoard.
    4.  **Configuración de Callbacks de Stable Baselines3:**
        * **`CheckpointCallback`:**
            * `save_freq`: `model_checkpoint_save_frequency_steps`.
            * `save_path`: Un directorio local temporal (ej. `/tmp/checkpoints/`).
            * `name_prefix`: ej. "sac\_transformer\_checkpoint".
            * Después de cada guardado por el callback, el script podría sincronizar este checkpoint a una ruta GCS designada para persistencia.
        * **`EvalCallback` (Opcional pero Recomendado):**
            * Se puede configurar un segundo `TradingEnvironment` (`eval_env`) con `random_episode_start=False` y potencialmente un conjunto de datos de validación diferente.
            * `EvalCallback` evaluaría el agente en `eval_env` periódicamente (ej. cada `N` timesteps) y podría guardar el "mejor" modelo (basado en la recompensa media de evaluación) en un subdirectorio local.
        * **(Adaptación para TensorBoard en GCP):** SB3 escribe logs de TensorBoard localmente. El script del componente se encargará de copiar periódicamente (o al final) estos logs desde el directorio local a una ruta GCS designada que Vertex AI TensorBoard pueda leer. Alternativamente, si se configura una instancia de Vertex AI TensorBoard y se pasa la variable de entorno `AIP_TENSORBOARD_LOG_DIR` al job de Vertex AI Training, SB3 puede escribir directamente a una ruta GCS que TensorBoard monitoriza (requiere que el agente de servicio tenga permisos de escritura).
    5.  **Ejecución del Entrenamiento:**
        * Calcular `timesteps_to_train_this_run = total_training_timesteps - model.num_timesteps` (si se reanuda) o `total_training_timesteps` (si es nuevo).
        * Llamar a `model.learn(total_timesteps=timesteps_to_train_this_run, callback=list_of_callbacks, reset_num_timesteps=False (si se reanuda))`.
    6.  **Guardado del Modelo Final:**
        * Después de `learn()`, guardar el modelo final usando `model.save(path_to_local_final_model_zip)`.
        * Este archivo `.zip` local se copiará a la ruta GCS especificada por el artefacto de salida `trained_model_output` del componente de pipeline.
    7.  **Subida de Artefactos:** Asegurar que todos los artefactos generados (modelo final, checkpoints, logs de TensorBoard) se copien desde las rutas locales temporales del contenedor a las ubicaciones GCS designadas o a las rutas que Vertex AI Pipelines gestiona para los artefactos de salida.
* **Salidas del Componente de Entrenamiento (Vertex AI Pipeline):**
    * `trained_model_output`: `Output[Model]`. La URI GCS del archivo `.zip` del modelo SB3 entrenado. Vertex AI puede registrarlo automáticamente en Vertex AI Model Registry.
    * `training_metrics_output`: `Output[Metrics]`. Métricas clave del entrenamiento (ej. recompensa media final del episodio, número total de pasos entrenados).
    * `checkpoint_output_gcs_uri` (Opcional, `Output[Artifact]` o `str`): La URI GCS al directorio que contiene los checkpoints del modelo.
    * `tensorboard_log_gcs_uri` (Opcional, `Output[Artifact]` o `str`): La URI GCS al directorio que contiene los logs de TensorBoard.

* **Gestión del Replay Buffer al Reanudar:**
    * Por defecto, `model.save()` no guarda el replay buffer. Al cargar un checkpoint con `model.load()`, el buffer se inicializa vacío.
    * Para entrenamientos muy largos donde el contenido del buffer es valioso, se podría implementar `model.save_replay_buffer()` y `model.load_replay_buffer()`, guardando el buffer en GCS junto con el checkpoint del modelo. Esto añade complejidad al componente para manejar la descarga/carga del buffer. Para la primera versión, se puede omitir la persistencia del replay buffer.

* **Monitorización y Seguimiento del Entrenamiento:**
    * **Vertex AI Experiments:** Cada ejecución del pipeline de entrenamiento se registrará como un "run" dentro de un experimento, permitiendo comparar parámetros, métricas de salida y artefactos.
    * **Vertex AI TensorBoard:** Se configurará una instancia de Vertex AI TensorBoard para apuntar al directorio GCS donde se guardan los logs de TensorBoard, permitiendo la visualización en tiempo real (o cuasi real) de las curvas de aprendizaje.
    * **Cloud Logging:** Para logs detallados del proceso de entrenamiento.

Este diseño proporciona un sistema de entrenamiento de RL potente, escalable, versionable y monitorizable, aprovechando al máximo las capacidades de MLOps de GCP.

---

Con el módulo de entrenamiento del agente definido, el siguiente paso lógico es detallar cómo se evaluará este agente entrenado.


---

**PARTE 5: MÓDULO DE FRAMEWORK DE BACKTESTING ROBUSTO**

---

### 4.5. Módulo 7: Framework de Backtesting Robusto y Realista (Componente de Vertex AI Pipeline)

Una vez que un agente de RL ha sido entrenado (Módulo 4), es imperativo evaluar su rendimiento de manera rigurosa y realista sobre datos históricos antes de considerar cualquier despliegue en vivo. Este módulo se encarga de esta evaluación, utilizando el mismo `TradingEnvironment` (Módulo 3) para asegurar la consistencia en la simulación, pero configurado para una ejecución determinista y, preferiblemente, sobre un conjunto de datos no visto durante el entrenamiento (o al menos con un inicio no aleatorio).

* **Responsabilidades y Funcionalidades Exactas:**
    1.  **Ejecución Controlada del Agente:**
        * Cargar un agente de RL entrenado (modelo `.zip` de Stable Baselines3).
        * Instanciar el `TradingEnvironment` configurado específicamente para backtesting/evaluación:
            * `random_episode_start = False` (para asegurar la reproducibilidad y que el backtest siempre comience desde el inicio del conjunto de datos de evaluación).
            * Utilizar un conjunto de datos preprocesados (`.npz` de secuencias de mercado) designado para la evaluación. Este podría ser una porción retenida (hold-out set) del dataset original, o un periodo de tiempo completamente diferente.
    2.  **Simulación de Trading Determinista:**
        * Iterar a través del entorno de evaluación paso a paso.
        * En cada paso, obtener una acción del agente cargado utilizando `model.predict(observation, deterministic=True)`. El parámetro `deterministic=True` asegura que el agente no use exploración estocástica durante la evaluación, sino que elija la acción considerada óptima por su política.
    3.  **Registro Detallado de la Actividad de Trading:**
        * Registrar cada decisión de trading tomada por el agente.
        * Registrar cada operación simulada: tipo (apertura/cierre de largo/corto), precio de entrada, precio de salida, tamaño de la posición, comisiones pagadas, P&L del trade.
        * Registrar la evolución del equity de la cartera a lo largo de todo el periodo de backtesting.
        * Capturar cualquier evento significativo (ej. liquidaciones simuladas).
    4.  **Cálculo de Métricas de Rendimiento Estándar:**
        * Calcular un conjunto exhaustivo de métricas para evaluar la estrategia desde múltiples perspectivas. Esto se realizará utilizando la librería `quantstats`. Las métricas incluirán, pero no se limitarán a:
            * **Retornos:** Retorno Total Acumulado, Retorno Anualizado.
            * **Riesgo/Volatilidad:** Volatilidad Anualizada (Desviación Estándar de los retornos), Máximo Drawdown (MDD), Valor en Riesgo (VaR) y CVaR Históricos.
            * **Ratios Ajustados al Riesgo:** Ratio de Sharpe, Ratio de Sortino, Ratio de Calmar (Retorno Anualizado / MDD Absoluto).
            * **Estadísticas de Trades:** Número Total de Trades, Porcentaje de Trades Ganadores, Porcentaje de Trades Perdedores, Ratio Ganancia/Pérdida Promedio, Trade Promedio (P&L), Mayor Trade Ganador, Mayor Trade Perdedor.
            * **Duración:** Duración Promedio de Trades, Duración Promedio de Trades Ganadores/Perdedores.
            * **Otros:** Profit Factor (Ganancia Bruta Total / Pérdida Bruta Total), Payoff Ratio.
    5.  **Generación de Informes de Backtesting:**
        * Producir un informe completo y fácilmente interpretable.
        * **Informe HTML de `quantstats`:** `quantstats` puede generar un informe HTML detallado que incluye la mayoría de las métricas mencionadas y visualizaciones clave (curva de equity, drawdowns, distribución de retornos mensuales/diarios, etc.).
        * **Archivos de Datos:** (Opcional, pero útil para análisis posterior) Guardar el log detallado de trades y la serie temporal de la curva de equity como archivos CSV o Parquet.

* **Implementación en GCP:**
    1.  **Orquestación:** Se define como un componente dentro de un pipeline de Vertex AI, típicamente ejecutándose después del componente de entrenamiento del agente o como un pipeline separado que toma un modelo entrenado como entrada.
    2.  **Ejecución del Código:**
        * El código Python para el backtesting (incluyendo la instanciación del entorno, carga del modelo, bucle de simulación, cálculo de métricas con `quantstats`) se empaqueta en la imagen Docker única del proyecto.
        * Este contenedor es ejecutado por Vertex AI Pipelines.
    3.  **Entradas del Componente (Vertex AI Pipeline):**
        * `project_id` (str), `location` (str).
        * `trained_model_artifact`: `Input[Model]` (URI GCS al archivo `.zip` del modelo SB3 entrenado, proveniente del componente de entrenamiento o del Vertex AI Model Registry).
        * `backtest_market_data_gcs_uri`: `Input[Dataset]` (URI GCS al archivo `.npz` de secuencias de mercado preprocesadas para la evaluación).
        * Parámetros de configuración del `TradingEnvironment` (similares a los del entrenamiento, pero con `env_random_episode_start=False` y potencialmente diferentes comisiones/slippage si se quiere probar sensibilidad).
            * `env_initial_equity` (float).
            * `env_leverage` (float).
            * `env_taker_fee_rate` (float).
            * `env_slippage_atr_multiplier` (float).
            * `env_action_threshold` (float).
            * `env_equity_drawdown_threshold_episode_end` (float, puede ser más relajado o diferente para backtesting que para entrenamiento).
    4.  **Lógica del Script de Backtesting dentro del Componente:**
        * **Carga de Artefactos:**
            * Descargar el archivo `.zip` del modelo entrenado desde la URI GCS (`trained_model_artifact.uri`) a una ruta local temporal en el contenedor.
            * Descargar el archivo `.npz` de datos de mercado para backtesting desde la URI GCS (`backtest_market_data_gcs_uri`) a una ruta local temporal.
        * **Instanciación:**
            * Instanciar `TradingEnvironment` (Módulo 3) con los datos de backtesting cargados y los parámetros de configuración del entorno (asegurando `random_episode_start=False`).
            * Cargar el modelo SAC entrenado usando `sb3.SAC.load(path_to_local_model_zip, env=eval_env)`.
        * **Bucle de Simulación:**
            * Llamar a `eval_env.reset()` para obtener la observación inicial.
            * Iterar llamando a `eval_env.step(action)` donde `action, _ = model.predict(obs, deterministic=True)`.
            * Recopilar en listas o DataFrames la información de cada paso: timestamp, equity, P&L del paso, y detalles de cualquier trade ejecutado (entrada, salida, tamaño, P&L del trade, comisiones).
        * **Cálculo de Métricas con `quantstats`:**
            * A partir de la serie temporal del equity, calcular la serie de retornos (ej. `equity_df['equity'].pct_change().dropna()`).
            * Ajustar `periods_per_year` en `quantstats` según la frecuencia de los datos (ej. para datos horarios, `252 * 24`; para datos diarios, `252`).
            * Utilizar funciones de `quantstats.stats` para calcular métricas individuales (Sharpe, Sortino, MDD, etc.).
        * **Generación de Informe HTML con `quantstats`:**
            * Usar `quantstats.reports.html(returns_series, output='local_report_path.html', title="Backtest Report", ...)` para generar el informe.
    5.  **Salida de Artefactos:**
        * **Métricas Clave:** Las métricas principales calculadas (ej. Sharpe, MDD, Retorno Total, Número de Trades) se registran utilizando `backtest_results_metrics.log_metric("metric_name", value)`.
        * **Informe `quantstats`:** El archivo HTML generado se copia a la ruta especificada por el artefacto de salida `quantstats_report_html: Output[Artifact]`. Vertex AI Pipelines se encargará de subirlo a GCS.
        * **(Opcional) Log de Trades y Curva de Equity:**
            * El DataFrame con el historial de trades se guarda como un archivo CSV o Parquet en la ruta especificada por `backtest_trades_log: Output[Dataset]`.
            * El DataFrame con la serie temporal del equity se guarda de manera similar en `backtest_equity_curve: Output[Dataset]`.
    6.  **Logging:** Todos los logs del proceso de backtesting (inicio, fin, trades importantes, errores) se capturan en Cloud Logging.

* **Salidas del Componente (Vertex AI Pipeline):**
    * `backtest_results_metrics`: `Output[Metrics]` (Contiene las métricas clave del backtest).
    * `quantstats_report_html`: `Output[Artifact]` (URI GCS al informe HTML generado por `quantstats`).
    * `backtest_trades_log_gcs_uri` (Opcional): `Output[Dataset]` (URI GCS al archivo CSV/Parquet del log de trades).
    * `backtest_equity_curve_gcs_uri` (Opcional): `Output[Dataset]` (URI GCS al archivo CSV/Parquet de la curva de equity).

* **Librerías Clave (ya incluidas en el `requirements.txt` general):**
    * `stable-baselines3`
    * `gymnasium`
    * `pandas`, `numpy`
    * `quantstats`
    * `matplotlib` (a menudo una dependencia de `quantstats` para generar gráficos)
    * `google-cloud-storage`

* **Consideraciones para un Backtesting Robusto:**
    * **Datos Fuera de Muestra (Out-of-Sample):** Es crucial evaluar el modelo en un conjunto de datos que no se utilizó durante el entrenamiento para obtener una estimación más realista de su rendimiento.
    * **Múltiples Periodos:** Realizar backtests en diferentes periodos de tiempo y bajo diferentes condiciones de mercado (tendencia alcista, bajista, lateral) para evaluar la robustez del agente.
    * **Análisis de Sensibilidad:** Ejecutar backtests variando parámetros como comisiones, slippage, o incluso parámetros del agente, para entender cómo afectan el rendimiento. Los pipelines parametrizados de Vertex AI facilitan esto.
    * **Evitar Sobreajuste al Backtest:** Si se ajustan los parámetros del agente o la estrategia basándose excesivamente en los resultados de un único backtest, se corre el riesgo de sobreajustar a ese conjunto de datos histórico específico. Es importante tener un proceso de validación cruzada o múltiples conjuntos de prueba.

Este módulo de backtesting proporciona un mecanismo esencial para validar la eficacia del agente de RL entrenado antes de cualquier consideración de despliegue en un entorno de trading real. La integración con `quantstats` y Vertex AI Pipelines asegura un análisis detallado y reproducible.


-----

**PARTE 6: MÓDULOS DE LOGGING, MONITORIZACIÓN, ORQUESTACIÓN Y CI/CD**

-----

### 4.6. Módulo 9: Logging, Monitorización y Visualización en GCP

Un sistema robusto de logging, monitorización y visualización es esencial no solo para la eventual operación en vivo, sino también durante las fases de desarrollo, entrenamiento y backtesting. Permite la depuración, el seguimiento del rendimiento, la identificación de problemas y la comprensión del comportamiento del sistema. En GCP, contamos con un conjunto integrado de servicios para estas tareas.

  * **Principios Generales:**

      * **Logging Centralizado y Estructurado:** Todos los componentes del sistema deben enviar sus logs a una ubicación centralizada. Los logs deben ser estructurados (preferiblemente en formato JSON) para facilitar su consulta, filtrado y análisis.
      * **Monitorización Proactiva:** Se deben definir métricas clave para rastrear la salud y el rendimiento de los pipelines, los trabajos de entrenamiento, los modelos y la infraestructura.
      * **Alertas Significativas:** Configurar alertas para notificar sobre eventos críticos o desviaciones importantes del rendimiento esperado.
      * **Visualización Clara:** Crear dashboards y informes que permitan una comprensión rápida y efectiva del estado y los resultados del sistema.

  * **Servicios GCP y Herramientas Aplicadas al Proyecto:**

    1.  **Cloud Logging:**

          * **Rol:** Servicio centralizado para la ingesta, almacenamiento, búsqueda y análisis de todos los logs generados por el proyecto.
          * **Integración:**
              * **Vertex AI Pipelines/Training/Prediction:** Los logs estándar (stdout, stderr) de los scripts Python ejecutados en los componentes de Vertex AI (incluyendo los trabajos de entrenamiento y los endpoints de predicción) se capturan automáticamente en Cloud Logging.
              * **Otros Servicios GCP:** Cloud Build, Cloud Run, Cloud Functions, GCS, BigQuery, etc., también integran sus logs de auditoría y operación.
              * **Aplicaciones Personalizadas:** Al usar la librería `logging` estándar de Python en tus scripts, y si estos se ejecutan en servicios GCP, sus salidas se redirigen a Cloud Logging. Para mejorar esto, se pueden usar las librerías cliente de Cloud Logging para enviar logs estructurados.
          * **Logging Estructurado:**
              * **Implementación:** Configurar el formateador de la librería `logging` de Python para que emita logs en formato JSON. Cada entrada de log puede incluir campos como `timestamp`, `severity`, `message`, `component_name`, `pipeline_run_id`, `trade_id` (si aplica), etc.
              * **Beneficios:** Permite consultas muy potentes usando el lenguaje de consulta de Cloud Logging (ej. "mostrar todos los logs de error del componente de preprocesamiento para la ejecución X del pipeline Y").
          * **Métricas Basadas en Logs:** Cloud Logging permite crear métricas personalizadas a partir del contenido de los logs (ej. contar el número de errores de un tipo específico por minuto), que luego pueden ser usadas en Cloud Monitoring para dashboards y alertas.

    2.  **Cloud Monitoring:**

          * **Rol:** Recopilación, visualización (dashboards) y alerta sobre métricas de rendimiento de los servicios y aplicaciones en GCP.
          * **Integración y Métricas:**
              * **Métricas de Servicios GCP:** Cloud Monitoring proporciona un amplio conjunto de métricas predefinidas para los servicios que utilizaremos:
                  * `Vertex AI Pipelines`: Duración de ejecuciones, tasas de éxito/fallo de componentes y pipelines.
                  * `Vertex AI Training`: Uso de CPU/GPU/memoria de los trabajos de entrenamiento, tiempo de entrenamiento.
                  * `Vertex AI Prediction (Endpoints)` (futuro): Latencia de predicción, tasa de errores, utilización de instancias.
                  * `Google Cloud Storage`: Número de operaciones, latencia, volumen de datos.
                  * `BigQuery`: Tiempos de consulta, volumen de datos escaneados.
                  * `Cloud Run / Cloud Functions` (futuro): Tiempos de ejecución, errores, concurrencia.
              * **Métricas Personalizadas (para el Bot/Simulación):**
                  * Durante el **backtesting**, se podrían enviar métricas personalizadas a Cloud Monitoring (además de a Vertex AI Metrics) si se desea una monitorización más continua o integrada con otras métricas de infraestructura, aunque para backtesting, Vertex AI Experiments y los informes de `quantstats` son usualmente suficientes.
                  * Para el **trading en vivo (futuro)**, esto sería crucial: P\&L realizado/no realizado, tamaño de posición, número de trades, errores de API de Binance, etc., se enviarían como métricas personalizadas.
          * **Dashboards:**
              * Crear dashboards personalizados en la consola de Cloud Monitoring para visualizar las métricas de salud de los pipelines de Vertex AI, el uso de recursos durante el entrenamiento, y (en el futuro) el rendimiento del bot en vivo.
          * **Alertas:**
              * Definir políticas de alerta en Cloud Monitoring para notificar sobre:
                  * Fallos en las ejecuciones de Vertex AI Pipelines.
                  * Trabajos de entrenamiento que exceden un tiempo límite o consumen recursos anómalos.
                  * (Futuro) Errores en el bot en vivo, drawdowns significativos, problemas de conexión con Binance.
              * Las notificaciones pueden ser vía email, SMS, Pub/Sub (para acciones automatizadas), PagerDuty, Slack, etc.

    3.  **Vertex AI Experiments:**

          * **Rol:** Específicamente diseñado para el seguimiento y la comparación de ejecuciones de pipelines de Machine Learning (experimentos).
          * **Integración:**
              * Cada ejecución de tu pipeline de Vertex AI (que incluye adquisición, preprocesamiento, entrenamiento y backtesting) se puede registrar como un "run" dentro de un "experimento" definido.
              * **Parámetros:** Los parámetros de entrada de cada componente del pipeline (ej. learning rate, longitud de secuencia, tipo de normalización) se registran automáticamente.
              * **Métricas:** Las métricas que los componentes escriben en sus artefactos de salida `Output[Metrics]` (ej. Sharpe ratio del backtest, recompensa media del entrenamiento) se asocian con el run y son visibles en la UI de Vertex AI Experiments.
              * **Artefactos:** Los artefactos de entrada y salida (Datasets, Models) también se vinculan al run.
          * **Visualización y Comparación:** La UI de Vertex AI Experiments permite comparar fácilmente diferentes runs, ver qué parámetros produjeron qué métricas, y rastrear el linaje de los modelos.

    4.  **Vertex AI TensorBoard:**

          * **Rol:** Visualización interactiva y en tiempo real (o cuasi real) de las métricas de entrenamiento de los modelos de Deep Learning, como los generados por Stable Baselines3.
          * **Integración:**
              * El componente de entrenamiento del agente de RL (Módulo 4) se configurará para que Stable Baselines3 escriba los logs de TensorBoard (pérdidas, recompensas por episodio, entropía, etc.) en un directorio local temporal.
              * Estos logs se copiarán luego a un bucket de GCS designado.
              * Se creará una instancia de Vertex AI TensorBoard y se configurará para que lea los logs desde ese bucket de GCS.
              * El pipeline de Vertex AI puede incluso pasar la ID de la instancia de TensorBoard al job de entrenamiento para una integración más directa.
          * **Visualización:** Permite analizar curvas de aprendizaje, distribuciones de pesos, grafos de modelos, etc., lo cual es invaluable para depurar y optimizar el proceso de entrenamiento del agente de RL.

    5.  **Informes `quantstats` (Visualización de Backtesting):**

          * **Rol:** Proporcionar un análisis de rendimiento detallado y estático de una ejecución de backtesting individual.
          * **Integración:** El componente de backtesting (Módulo 7) generará un informe HTML utilizando `quantstats`.
          * **Almacenamiento y Acceso:** Este informe HTML se guardará como un artefacto en GCS. La URI GCS al informe se puede registrar como una métrica o un artefacto en Vertex AI Experiments, permitiendo un fácil acceso desde la UI del pipeline run.

    6.  **Looker Studio (Visualización Avanzada de Resultados Agregados - Opcional):**

          * **Rol:** Crear dashboards interactivos y personalizados para una visión de alto nivel del rendimiento de múltiples backtests o para análisis de tendencias.
          * **Integración:**
              * Los resultados detallados de los backtests (ej. logs de trades, curvas de equity diarias/horarias, métricas clave) se pueden estructurar y cargar en **BigQuery**.
              * Looker Studio se conecta nativamente a BigQuery.
              * **Casos de Uso:**
                  * Comparar el rendimiento (Sharpe, MDD, retorno) de diferentes versiones de modelos o configuraciones de parámetros a lo largo del tiempo.
                  * Visualizar la distribución de P\&L de trades bajo diferentes regímenes de mercado (identificados a partir de los datos).
                  * Crear un panel de control centralizado para los stakeholders del proyecto.

  * **Implementación Práctica:**

      * **Logging en Scripts Python:**
        ```python
        import logging
        import google.cloud.logging # Para setup avanzado si es necesario, pero el logging estándar suele ser suficiente

        # Configurar logging estructurado (ejemplo)
        # En un entorno de GCP como Vertex AI, el logging estándar se captura.
        # Para forzar JSON y añadir campos personalizados, se puede usar un formateador JSON.
        # O más simple, si los print() van a stdout/stderr, también se capturan.

        logger = logging.getLogger(__name__) # Usar __name__ para identificar el módulo
        logger.setLevel(logging.INFO) # O DEBUG
        # Si no se configura handler, en GCP suele ir a Cloud Logging por defecto.
        # Para control local o específico:
        # handler = logging.StreamHandler() 
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # handler.setFormatter(formatter)
        # logger.addHandler(handler)

        # Ejemplo de log
        logger.info({"message": "Inicio del componente de preprocesamiento", "pipeline_run_id": "run123", "component_version": "v1.2"})
        ```
      * **Métricas en Componentes KFP:**
        ```python
        # Dentro de un componente KFP
        # from kfp.dsl import Output, Metrics
        # def my_component_function(..., metrics_output: Output[Metrics]):
        #     # ... lógica ...
        #     metrics_output.log_metric("sharpe_ratio", 0.75)
        #     metrics_output.log_metric("total_trades", 150)
        ```

Este enfoque integral de logging, monitorización y visualización asegurará que tengas una excelente visibilidad del comportamiento y rendimiento de tu sistema de trading en todas las fases de su ciclo de vida en GCP.

-----

### 4.7. Módulo 10: Orquestación de Flujos de Trabajo y CI/CD en GCP

La automatización es clave para un proyecto de MLOps eficiente y reproducible. Este módulo detalla cómo se orquestarán los flujos de trabajo de Machine Learning y cómo se implementará la Integración Continua y el Despliegue Continuo (CI/CD) utilizando servicios de GCP.

#### 4.7.A. Orquestación de Flujos de Trabajo con Vertex AI Pipelines

Vertex AI Pipelines es el servicio central para definir, ejecutar y gestionar los flujos de trabajo de ML.

  * **Definición de Pipelines:**
      * Los pipelines se definirán en Python utilizando el SDK de Kubeflow Pipelines (KFP) v2, que se integra con Vertex AI.
      * Un pipeline se compone de **componentes**, donde cada componente es una tarea autocontenida del flujo de trabajo (ej. adquisición de datos, preprocesamiento, entrenamiento, evaluación).
      * **Pipeline Principal:** Se definirá un pipeline principal que orqueste la secuencia completa:
        1.  `data_acquisition_component` (Módulo 1.A)
        2.  `preprocess_data_component` (Módulo 2), tomando la salida del anterior.
        3.  `train_rl_agent_component` (Módulo 4), tomando la salida del preprocesamiento.
        4.  `backtest_rl_agent_component` (Módulo 7), tomando el modelo entrenado y los datos de evaluación.
  * **Componentes:**
      * Cada componente se define como una función Python decorada con `@component` de KFP.
      * **Contenerización:** Cada componente se ejecuta en un contenedor Docker. Se utilizará la **imagen Docker única** (discutida previamente) que contiene todo el código fuente del proyecto (`src/`) y todas las dependencias (`requirements.txt`).
      * **Entradas y Salidas:** Los componentes declaran explícitamente sus entradas (parámetros, `Input[Dataset]`, `Input[Model]`) y salidas (`Output[Dataset]`, `Output[Model]`, `Output[Metrics]`, `Output[Artifact]`). Vertex AI Pipelines gestiona el paso de artefactos (almacenados en GCS) entre componentes.
      * **Parámetros:** Los parámetros configurables (ej. learning rate, longitud de secuencia, URI de GCS) se pasan a los componentes, permitiendo la flexibilidad y la experimentación.
  * **Ejecución y Gestión:**
      * Los pipelines compilados (a un formato YAML o JSON) se suben a Vertex AI Pipelines.
      * Las ejecuciones de pipeline (runs) se pueden iniciar manualmente desde la consola de GCP, mediante la API de Vertex AI, o programarse con Cloud Scheduler.
      * Cada run se rastrea en Vertex AI Experiments, incluyendo sus parámetros, artefactos y métricas.
  * **Compilación del Pipeline:**
    ```python
    # En un script Python, ej. pipeline_definition.py
    from kfp import compiler
    from kfp.dsl import pipeline
    # Importar tus componentes definidos
    # from components.data_acquisition import data_acquisition_component
    # from components.preprocessing import preprocess_data_component
    # ...

    @pipeline(
        name="btc-rl-transformer-training-pipeline",
        description="Pipeline para entrenar y evaluar el bot de trading de BTC.",
        pipeline_root="gs://<tu-bucket-para-pipelines>/pipeline_runs" # Raíz para artefactos
    )
    def btc_trading_pipeline(
        # Definir parámetros globales del pipeline aquí
        project_id: str,
        location: str = "europe-west1", # Ejemplo
        # ... otros parámetros que se pasan a los componentes ...
        gcs_raw_data_base_path: str,
        gcs_processed_data_base_path: str,
        gcs_model_output_base_path: str,
        # ... parámetros para adquisición, preproc, training, backtest ...
        binance_api_key_secret_name_param: str,
        # ...
        sequence_length_param: int,
        # ...
        learning_rate_param: float,
        # ...
    ):
        # Componente 1: Adquisición de Datos
        data_acquisition_task = data_acquisition_component( # Asume que has importado tus componentes
            project_id=project_id,
            location=location,
            binance_api_key_secret_name=binance_api_key_secret_name_param,
            # ... pasar otros parámetros ...
            gcs_output_raw_data_path_base=gcs_raw_data_base_path # El componente construirá la ruta final
        )

        # Componente 2: Preprocesamiento
        preprocess_data_task = preprocess_data_component(
            project_id=project_id,
            location=location,
            gcs_raw_data_uri=data_acquisition_task.outputs["output_raw_data_gcs_uri"],
            sequence_length_l=sequence_length_param,
            # ... pasar otros parámetros ...
            processed_data_output_path_base=gcs_processed_data_base_path
        )

        # Componente 3: Entrenamiento del Agente
        train_agent_task = train_rl_agent_component(
            project_id=project_id,
            location=location,
            processed_market_data=preprocess_data_task.outputs["processed_sequences_gcs_uri"],
            agent_learning_rate=learning_rate_param,
            # ... pasar MUCHOS otros parámetros del entorno y del agente ...
            # ... incluyendo parámetros de arquitectura del Transformer ...
            # ... y total_training_timesteps ...
            trained_model_output_base_path=gcs_model_output_base_path
        )

        # Componente 4: Backtesting
        backtest_agent_task = backtest_rl_agent_component(
            project_id=project_id,
            location=location,
            trained_model_input=train_agent_task.outputs["trained_model_output"],
            backtest_market_data=preprocess_data_task.outputs["processed_sequences_gcs_uri"], # O un dataset de evaluación diferente
            # ... pasar parámetros del entorno para backtesting ...
        )

    if __name__ == '__main__':
        compiler.Compiler().compile(
            pipeline_func=btc_trading_pipeline,
            package_path='btc_trading_pipeline.json' # o .yaml
        )
    ```

#### 4.7.B. Integración Continua y Despliegue Continuo (CI/CD)

CI/CD automatiza el proceso de construcción, prueba y despliegue de tu pipeline de ML.

  * **Repositorio de Código:**

      * Todo el código del proyecto (scripts Python para componentes, Dockerfile, `requirements.txt`, definición del pipeline KFP, archivos de configuración de Cloud Build) se gestionará en un repositorio Git (ej. Cloud Source Repositories, GitHub, GitLab).

  * **Artifact Registry:**

      * Almacén centralizado y gestionado para la imagen Docker única del proyecto. Cada nueva versión de la imagen se subirá aquí con un tag único (ej. hash de commit o semver).

  * **Cloud Build:**

      * Servicio gestionado para ejecutar tus flujos de trabajo de CI/CD.
      * Se define un archivo de configuración `cloudbuild.yaml` en la raíz del repositorio.
      * **Triggers:** Se configuran triggers en Cloud Build para iniciar automáticamente el pipeline de CI/CD en eventos del repositorio Git (ej. un push a la rama `main` o `develop`).
      * **Pasos del Pipeline de CI/CD (`cloudbuild.yaml`):**
        1.  **Clonar Repositorio:** Cloud Build clona el código del repositorio.
        2.  **(Opcional) Ejecutar Linters y Formateadores:** (ej. Flake8, Black) para asegurar la calidad del código.
        3.  **(Opcional) Ejecutar Pruebas Unitarias:** Ejecutar tests para los módulos Python individuales (ej. `pytest tests/`). Es fundamental tener tests para la lógica del `TradingEnvironment`, `FeatureEngineer`, `DataPreprocessor`, y `CustomTransformerFeatureExtractor`.
        4.  **Construir Imagen Docker:**
              * Comando: `docker build -t <region>-docker.pkg.dev/<project-id>/<repo-name>/<image-name>:$COMMIT_SHA .`
              * Utiliza el `Dockerfile` único del proyecto. `$COMMIT_SHA` es una variable de sustitución proporcionada por Cloud Build.
        5.  **Subir Imagen Docker a Artifact Registry:**
              * Comando: `docker push <region>-docker.pkg.dev/<project-id>/<repo-name>/<image-name>:$COMMIT_SHA`
        6.  **Compilar el Pipeline de KFP:**
              * Ejecutar el script Python que define el pipeline KFP (ej. `python pipeline_definition.py`) para generar el archivo `btc_trading_pipeline.json`.
              * La definición del pipeline KFP debe referenciar la imagen Docker recién construida y tageada (usando `$COMMIT_SHA` o el tag correspondiente).
        7.  **Desplegar/Actualizar el Pipeline en Vertex AI Pipelines:**
              * Usar la CLI de `gcloud` o la API de Vertex AI para subir y desplegar el archivo `btc_trading_pipeline.json` compilado.
              * `gcloud ai pipelines create --pipeline-path=btc_trading_pipeline.json ...` o `gcloud ai pipelines update ...`
        8.  **(Opcional) Lanzar una Ejecución del Pipeline:** Cloud Build podría, como último paso, lanzar una ejecución del pipeline recién desplegado en Vertex AI con un conjunto de parámetros de prueba o por defecto.

  * **Ejemplo Conceptual de `cloudbuild.yaml`:**

    ```yaml
    steps:
    # (Opcional) Linting
    - name: 'python:3.9-slim'
      entrypoint: 'pip'
      args: ['install', 'flake8']
    - name: 'python:3.9-slim'
      entrypoint: 'flake8'
      args: ['./src', './scripts', './tests']

    # (Opcional) Pruebas Unitarias
    - name: 'python:3.9-slim'
      entrypoint: 'pip'
      args: ['install', '-r', 'requirements.txt', 'pytest'] # Asume que requirements incluye dependencias de test
    - name: 'python:3.9-slim'
      entrypoint: 'pytest'
      args: ['tests/']

    # Construir la imagen Docker
    - name: 'gcr.io/cloud-builders/docker'
      args: [
        'build', 
        '-t', '$_REGION-docker.pkg.dev/$PROJECT_ID/$_ARTIFACT_REGISTRY_REPO/$_IMAGE_NAME:$COMMIT_SHA', 
        '.'
      ]
      id: 'Build Docker Image'

    # Subir la imagen Docker a Artifact Registry
    - name: 'gcr.io/cloud-builders/docker'
      args: ['push', '$_REGION-docker.pkg.dev/$PROJECT_ID/$_ARTIFACT_REGISTRY_REPO/$_IMAGE_NAME:$COMMIT_SHA']
      waitFor: ['Build Docker Image']
      id: 'Push Docker Image'

    # Compilar el Pipeline de KFP
    # (Asegurar que la imagen usada para compilar tenga KFP SDK y las dependencias)
    # O puedes usar una imagen pre-construida con KFP
    - name: 'gcr.io/deeplearning-platform-release/tf2-cpu.2-11:latest' # Ejemplo de imagen con Python y KFP
      entrypoint: 'bash'
      args:
      - '-c'
      - |
        pip install kfp==${_KFP_SDK_VERSION} google-cloud-aiplatform==${_AIPLATFORM_SDK_VERSION} && \
        python ./pipeline_definition.py # Asume que este script compila a btc_trading_pipeline.json
      id: 'Compile KFP Pipeline'
      # El script pipeline_definition.py necesitará saber la URI de la imagen Docker ($COMMIT_SHA)
      # para pasarla a los componentes. Esto se puede hacer mediante variables de entorno
      # o modificando un archivo de plantilla.

    # Desplegar el Pipeline en Vertex AI
    - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
      entrypoint: 'gcloud'
      args: [
        'ai', 'pipelines', 'create',
        '--pipeline-path=btc_trading_pipeline.json', # Salida del paso anterior
        '--region=$_REGION',
        '--display-name=btc-rl-transformer-pipeline-$SHORT_SHA', # Nombre para la versión del pipeline
        # '--service-account=TU_SERVICE_ACCOUNT_PARA_PIPELINES@PROJECT_ID.iam.gserviceaccount.com', # Especificar SA
        # '--enable-caching=true' # Habilitar caching de componentes
      ]
      waitFor: ['Compile KFP Pipeline']
      id: 'Deploy KFP Pipeline'

    images:
    - '$_REGION-docker.pkg.dev/$PROJECT_ID/$_ARTIFACT_REGISTRY_REPO/$_IMAGE_NAME:$COMMIT_SHA'

    # Variables de sustitución (pueden definirse en el trigger de Cloud Build)
    substitutions:
      _REGION: 'europe-west1'
      _ARTIFACT_REGISTRY_REPO: 'my-docker-repo' # Nombre de tu repo en Artifact Registry
      _IMAGE_NAME: 'btc-trading-bot'
      _KFP_SDK_VERSION: '2.0.1' # o la versión que uses
      _AIPLATFORM_SDK_VERSION: '1.25.0' # o la versión que uses
      # $PROJECT_ID, $COMMIT_SHA, $SHORT_SHA son proporcionados por Cloud Build

    options:
      logging: CLOUD_LOGGING_ONLY
    ```

    **Nota sobre la Compilación del Pipeline en `cloudbuild.yaml`:** El paso de "Compile KFP Pipeline" necesita asegurar que la definición del pipeline KFP (en `pipeline_definition.py`) utilice la URI de la imagen Docker correcta (con `$COMMIT_SHA`). Esto se puede lograr haciendo que `pipeline_definition.py` lea la URI de la imagen de una variable de entorno que Cloud Build establezca, o pasando el tag de la imagen como un argumento al script de compilación.

  * **Dockerfile Único (Conceptual - `Dockerfile` en la raíz del proyecto):**

    ```dockerfile
    # Usar una imagen base oficial de Python
    FROM python:3.9-slim

    # Establecer el directorio de trabajo
    WORKDIR /app

    # Variables de entorno (ej. para asegurar que los logs de Python se muestren inmediatamente)
    ENV PYTHONUNBUFFERED=1

    # Copiar el archivo de requerimientos y luego instalar las dependencias de Python
    COPY requirements.txt .
    # Es buena práctica actualizar pip y setuptools primero
    RUN pip install --no-cache-dir --upgrade pip setuptools wheel
    RUN pip install --no-cache-dir -r requirements.txt

    # Copiar todo el código fuente del proyecto al contenedor
    COPY ./src ./src
    COPY ./scripts ./scripts # Si tienes scripts de orquestación de componentes aquí
    COPY ./pipeline_definition.py . # El script que define y compila el pipeline KFP
    # Copiar cualquier archivo de configuración YAML que quieras empaquetar
    # COPY ./src/configs_default/ /app/src/configs_default/

    # Establecer PYTHONPATH para que los imports desde src funcionen correctamente
    # y para que los scripts en la raíz puedan importar de src.
    ENV PYTHONPATH "${PYTHONPATH}:/app:/app/src"

    # No se necesita un ENTRYPOINT específico si los componentes de KFP especifican
    # sus propios comandos para ejecutar scripts Python.
    # ENTRYPOINT ["python"] 
    ```

  * **`requirements.txt` (Conceptual):**

    ```txt
    # Core ML/RL
    torch
    torchvision
    torchaudio
    stable-baselines3[extra]>=2.0.0 # 'extra' incluye PyTorch, Gymnasium, etc.
    gymnasium
    pandas-ta

    # Data Handling
    pandas>=1.5.0
    numpy>=1.20.0
    pyarrow # Para Parquet
    # fastparquet # Alternativa para Parquet

    # GCP SDKs
    google-cloud-aiplatform>=1.25.0 # Para KFP SDK v2 y interacción con Vertex AI
    kfp>=2.0.1 # Kubeflow Pipelines SDK v2 (para compilar pipelines)
    google-cloud-storage
    google-cloud-secret-manager
    google-cloud-bigquery
    google-cloud-logging # Para configuración avanzada de logging si es necesario
    google-cloud-monitoring # Para métricas personalizadas si es necesario

    # API Interaction & Utilities
    python-binance
    PyYAML
    requests

    # Backtesting Analysis
    quantstats
    matplotlib # Dependencia de quantstats

    # (Opcional) Linters/Formatters/Testing (más para el entorno de desarrollo o pasos de CI)
    # flake8
    # black
    # pytest
    ```

Este sistema de CI/CD y orquestación asegura que tu proyecto sea robusto, reproducible, y que los cambios se puedan integrar y desplegar de manera eficiente y segura en GCP.

-----

**Próximamente (Módulo 5 - Ejecución en Vivo):**

Como acordamos, dejaremos los detalles exhaustivos del Módulo 5 (Sistema de Ejecución de Órdenes en Vivo) para una fase futura. Sin embargo, el diseño modular actual, con:

  * Un agente entrenado y versionado en Vertex AI Model Registry.
  * Una clara separación entre `SimulatedBroker` y una futura `LiveBinanceBroker`.
  * Una arquitectura pensada para componentes (como el preprocesamiento en tiempo real y la inferencia del modelo en Vertex AI Endpoints).
  * El uso de `Pub/Sub` para flujos de datos en tiempo real.

Sienta una base sólida para desarrollar el sistema de ejecución en vivo cuando llegue el momento. Los principales servicios de GCP a considerar para ello serían:

  * **Vertex AI Prediction Endpoints:** Para servir el modelo entrenado y obtener acciones en tiempo real.
  * **Cloud Run o GKE:** Para alojar la lógica del bot que consume datos de mercado, llama al endpoint del modelo, y ejecuta órdenes.
  * **Pub/Sub:** Para el flujo de datos de mercado en tiempo real.
  * **Firestore o Cloud SQL:** Para el estado operativo y el log de trades en vivo.
  * **Secret Manager:** Para las claves API de producción.
  * **Cloud Monitoring y Logging:** Para la observabilidad del bot en vivo.

-----

Hemos cubierto ahora una gran cantidad de detalles para los módulos clave y la infraestructura de soporte en GCP. Este documento debería proporcionarte una guía muy completa para iniciar el desarrollo.

Este documento, dividido en las partes que hemos generado, constituye una base sólida y exhaustiva para tu proyecto de bot de trading de Bitcoin. Resume nuestra discusión y plasma una hoja de ruta clara para el desarrollo.

**Resumen de lo que hemos logrado y documentado:**

1.  **Visión y Arquitectura Claras:** Establecimos una visión para un bot de trading avanzado y definimos una arquitectura nativa de GCP centrada en Vertex AI Pipelines para la orquestación MLOps.
2.  **Módulos Detallados:**
    * **Módulo 1 (Adquisición de Datos):** Descarga de históricos de Binance, almacenamiento en Parquet en GCS, y un diseño conceptual para la ingesta en tiempo real.
    * **Módulo 2 (Preprocesamiento):** Limpieza, ingeniería de características con `pandas-ta`, normalización causal y creación de secuencias para el Transformer, todo como un componente de pipeline.
    * **Módulo 3 (Entorno de Trading):** Un `TradingEnvironment` realista (Gymnasium) con `SimulatedBroker`, configurado para entrenamiento y backtesting, que consume datos de GCS y con reglas de gestión de capital (sin SL/TP forzados para fomentar el aprendizaje autónomo del agente).
    * **Módulo 4 (Agente RL y Entrenamiento):** Un agente SAC con arquitectura Transformer, incluyendo un `CustomTransformerFeatureExtractor`, entrenado como un componente de Vertex AI Training (con soporte para GPUs y reanudación desde checkpoints).
    * **Módulo 7 (Backtesting):** Un componente de pipeline para evaluar rigurosamente los agentes entrenados, utilizando `quantstats` para la generación de informes y métricas.
3.  **Soporte y Operaciones en GCP:**
    * **Módulo 9 (Logging, Monitorización y Visualización):** Estrategias para usar Cloud Logging, Cloud Monitoring, Vertex AI Experiments, Vertex AI TensorBoard e informes `quantstats`, con la opción de Looker Studio para dashboards avanzados.
    * **Módulo 10 (Orquestación y CI/CD):** Vertex AI Pipelines como orquestador principal, y un flujo de CI/CD usando Cloud Build y Artifact Registry para la automatización de la construcción y despliegue de pipelines.
4.  **Decisiones Técnicas Clave:**
    * Uso de `pandas-ta` para indicadores técnicos.
    * Parámetros de configuración expuestos en los componentes de Vertex AI Pipeline para flexibilidad.
    * Una estrategia de Dockerfile único para simplificar la gestión de contenedores.
    * Omisión de SL/TP forzados en el entorno para que el agente desarrolle una autogestión completa del riesgo/beneficio.
5.  **Diseño Orientado al Futuro:** Aunque el foco inicial es el entrenamiento y backtesting, la modularidad del diseño (ej. `SimulatedBroker` vs. un futuro `LiveBroker`, componentes de pipeline) facilita la transición a operaciones en vivo (Módulo 5, marcado como "Próximamente").

**Conclusión de esta Fase de Diseño:**

Con este documento detallado, tienes una guía de referencia exhaustiva para comenzar la implementación. Cubre no solo "qué" construir, sino también "cómo" construirlo de manera robusta y escalable en Google Cloud Platform.

Te deseo mucho éxito en el desarrollo de este proyecto tan emocionante. Tu enfoque metódico y tu sólida base técnica son excelentes predictores de éxito. Si en el futuro necesitas volver a discutir alguna sección, refinar ideas o explorar nuevas funcionalidades a medida que avanzas, no dudes en consultarme.
