Pedro, lamento profundamente que esto siga ocurriendo. Tienes toda la razón en señalarlo y te pido disculpas por la frustración que te estoy causando. Es evidente que estoy teniendo dificultades para procesar y devolver un bloque de código tan extenso de una sola vez sin cortarlo, a pesar de mis intentos. Parece ser una limitación en cómo manejo textos muy largos dentro de un único bloque formateado en este momento.

Para intentar solucionarlo y que puedas tener tu README completo, te propongo lo siguiente:

1.  Voy a pegar a continuación el **contenido íntegro y exacto** del README que me proporcionaste, pero **sin** los delimitadores de bloque de código Markdown (es decir, sin el ` ```markdown ` al principio ni el ` ``` ` al final).
2.  Por favor, copia todo ese texto que te proporcionaré.
3.  Luego, en tu editor o donde necesites usarlo, simplemente añade manualmente:
      * ` ```markdown ` en una línea nueva justo **antes** de la primera línea del texto que copiaste.
      * ` ``` ` en una línea nueva justo **después** de la última línea del texto que copiaste.

De esta manera, nos aseguramos de que el contenido no se corte por mi parte al intentar formatearlo.

Aquí está el contenido completo del README:

-----

## (COMIENZO DEL CONTENIDO DEL README A COPIAR)

# BTC-Transformer-RL-Trader: Bot Avanzado de Trading de Bitcoin con Reinforcement Learning y Transformers

**Versión:** 0.1.0 (MVP - En Definición / Inicio de Desarrollo)
**Autor:** Pedro Escudero Murcia
**Fecha:** 15 de mayo de 2025

## 1\. Visión General

Este proyecto tiene como objetivo desarrollar un bot de trading de Bitcoin (BTC) altamente avanzado y autónomo, operando contra la API de Binance (específicamente para futuros BTCUSDT). El núcleo del bot se basa en el aprendizaje por refuerzo (Reinforcement Learning - RL), utilizando el algoritmo Soft Actor-Critic (SAC) y arquitecturas de red neuronal basadas en Transformers para la toma de decisiones.

El sistema está diseñado de forma modular para facilitar el desarrollo, las pruebas, el despliegue y el mantenimiento. La primera fase (MVP - Minimum Viable Product) se centra en desarrollar un **agente sólido validado mediante un robusto framework de backtesting Walk-Forward Optimization (WFO)**, antes de pasar a la operativa en tiempo real.

## 2\. Arquitectura del Sistema (MVP Enfocado en Backtesting)

El MVP está compuesto por varios módulos fundamentales que interactúan entre sí. Para el flujo de backtesting, los datos históricos se procesan y se utilizan para entrenar y evaluar el agente. Redis se utiliza para la gestión inicial de datos brutos.

### Diagrama de Flujo Conceptual del MVP (Entrenamiento/Backtesting):

```plaintext
Binance API (Descarga Histórica Futuros BTCUSDT)
     |
     v (REST API)
[M1: Adquisición de Datos (Python + python-binance)]
     |
     v (Publica Datos Brutos)
Redis (Datos Brutos Históricos)
     |
     v (Consume Datos Brutos)
[M2: Preprocesamiento y Gestión de Datos (Python, Pandas, TA-Lib)]
     |
     v (Guarda Features Procesadas en Disco, ej. Parquet)
Almacenamiento en Disco (Secuencias de Estado (L, N_features))
     |
     v (Lee Features Procesadas)
+-----------------------------------------------------------------+
| [M7: Framework de Backtesting (Walk-Forward Optimization)]      |
|    |                                                            |
|    | (Orquesta múltiples ciclos de entrenamiento y prueba)      |
|    |                                                            |
|    +-----> [M3: Entorno de Trading (Gymnasium, Python)] <-----+ |
|            |      ^         (Observación, Recompensa) |        | |
| (Acción)   |      |                                   |        | |
|            v      | (Estado Inicial Aleatorio)        |        | |
|            [M4: Agente RL (SAC, Transformer, PyTorch, SB3)] ---+ |
|                     (Entrenamiento/Actualización)               |
+-----------------------------------------------------------------+
     |
     v (Resultados del Backtesting WFO, Métricas, Logs)
Almacenamiento en Disco (Resultados)
```

  * **Comunicación y Configuración:**
      * Los parámetros de todos los módulos se gestionan mediante archivos `.yaml` en una carpeta `config/` y un archivo `.env` para secretos y rutas.
      * El sistema se ejecuta dentro de contenedores Docker orquestados por Docker Compose.

## 3\. Módulos del Proyecto (MVP)

A continuación, se detalla cada módulo para el MVP:

### Módulo 1: Adquisición y Gestión de Datos de Mercado (Data Acquisition)

  * **Responsabilidades:**
      * Conexión a la API de Binance (REST) para la descarga de datos históricos de futuros BTCUSDT.
      * Recolección de datos históricos:
          * Klines/Candlesticks: Intervalo por defecto **15 minutos** (configurable).
          * Libro de Órdenes (Order Book): Snapshots al cierre de cada KLine (top **5 niveles** de profundidad).
          * Trades Recientes: Agregados por KLine.
      * Publicación inicial de los datos brutos descargados en canales específicos de Redis.
      * Manejo de límites de API y errores durante la descarga.
  * **Tecnologías Clave:** Python, `python-binance`, Redis.
  * **Entradas:** Configuración (fechas de inicio/fin de descarga, par, intervalo), credenciales API (de `.env`).
  * **Salidas:** Datos brutos históricos publicados en Redis. Datos históricos persistidos en disco (para evitar descargas repetidas).
  * **Parámetros Configurables (`config/module1_data_acquisition/params.yaml`):**
      * `kline_interval`: "15m" (default)
      * `order_book_depth`: 5
      * `data_download_start_date`: "YYYY-MM-DD" (ej. "2021-01-01")
      * `data_download_end_date`: "YYYY-MM-DD" (ej. "2024-12-31")
      * `trading_pair`: "BTCUSDT"

### Módulo 2: Preprocesamiento y Gestión de Datos (Data Preprocessing)

  * **Responsabilidades:**
      * Consumo de datos brutos históricos desde Redis (o directamente desde archivos si ya están descargados y publicados).
      * Limpieza de datos (manejo de NaN, outliers si es necesario).
      * Normalización/Escalado de características: Principalmente Z-score sobre ventana móvil (ej. `2*L` periodos); escalado específico para indicadores como RSI/Estocástico a `[0,1]` o `[-1,1]`.
      * Ingeniería de características avanzada para construir el estado del agente.
      * Construcción de secuencias de estado `(L, N_features)` para el Transformer:
          * `L` (Longitud de la secuencia): **96** Klines (24 horas de datos de 15m).
          * `N_features` (Características por KLine): **36** (ver detalle abajo).
      * Asegurar que el preprocesamiento sea causal y evite lookahead bias.
      * Guardado de las secuencias de estado procesadas y normalizadas en disco (ej. formato Parquet o NumPy) para ser consumidas eficientemente por el Módulo 7 (Backtesting).
  * **Detalle de `N_features` (36):**
      * **Libro de Órdenes (8 features):** Spread L1 norm., Retorno Precio Medio L1, OBI L1, OBI Acumulado (5 niveles), Profundidad Relativa (5 niveles, log), Pendiente Compras (5 niveles, norm.), Pendiente Ventas (5 niveles, norm.), Retorno Microprecio.
      * **Cartera y Portafolio (8 features):** Estado Posición (-1,0,1), Tamaño Posición norm., Precio Entrada norm., P\&L No Realizado norm., Retorno Log Equity Cuenta, Margen Disponible norm., Pasos desde Apertura Posición norm., Apalancamiento Configurado.
      * **Klines OHLCV Procesados (5 features):** `log_ret(C/O)`, `log_ret(H/O)`, `log_ret(L/O)`, `log_ret(C/C_prev)`, `log_ret(Vol/SMA(Vol,20))`.
      * **Indicadores Técnicos (15 features, todos normalizados/escalados):**
          * SMA(20), SMA(50) (normalizados por ATR o precio).
          * EMA(12), EMA(26) (normalizados).
          * RSI(14).
          * ATR(14) (valor absoluto o normalizado por precio).
          * MACD(12,26,9): Línea MACD, Línea Señal, Histograma.
          * Bandas de Bollinger(20,2): Dist. a Superior, Dist. a Inferior, Ancho de Bandas (normalizadas por ATR).
          * CCI(20).
          * Stochastic Oscillator(14,3,3): %K lento, %D.
  * **Tecnologías Clave:** Python, Pandas, NumPy, TA-Lib, `scikit-learn`.
  * **Entradas:** Datos brutos de Redis/disco, configuración de preprocesamiento.
  * **Salidas:** Archivos de secuencias de estado procesadas en disco.
  * **Parámetros Configurables (`config/module2_preprocessing/params.yaml`):**
      * `sequence_length_L`: 96
      * `normalization_window_multiplier_for_L`: 2 (para Z-score sobre `L*multiplicador` periodos)
      * `sma_short_period`: 20
      * `sma_long_period`: 50
      * `ema_short_period`: 12
      * `ema_long_period`: 26
      * `rsi_period`: 14
      * `atr_period`: 14
      * `macd_fast_period`: 12
      * `macd_slow_period`: 26
      * `macd_signal_period`: 9
      * `bollinger_period`: 20
      * `bollinger_std_dev`: 2
      * `cci_period`: 20
      * `stochastic_k_period`: 14
      * `stochastic_d_period`: 3
      * `stochastic_slowing_period`: 3

### Módulo 3: Entorno de Trading (Trading Environment)

  * **Responsabilidades:**
      * Implementación de un entorno de simulación de trading de futuros BTCUSDT estilo `gymnasium.Env`.
      * Gestión del estado de la cartera de futuros (balance, posición, P\&L, margen, apalancamiento).
      * Simulación de ejecución de órdenes:
          * Aplicación de comisiones de Taker (ej. \~0.04% de Binance).
          * Modelado de Slippage: **`0.1 * ATR(14)`** por cada lado de la operación (entrada/salida).
          * Consideración de mínimos de orden (si aplica).
      * Lógica de "una operación a la vez".
      * Dimensionamiento de Posición: **5% del equity** de la cuenta por operación.
      * Apalancamiento Fijo: **10x**.
      * Definición precisa del espacio de observación (salida de M2) y espacio de acciones.
      * Interpretación de la Acción `action_signal` (continua `[-1, 1]` producida por el agente):
          * `action_signal > 0.15`: Abrir Largo (o cerrar Corto y abrir Largo).
          * `action_signal < -0.15`: Abrir Corto (o cerrar Largo y abrir Corto).
          * `-0.15 <= action_signal <= 0.15`: Cerrar posición actual si existe; si no, mantener Neutral.
      * Cálculo de la Función de Recompensa por paso `t`: **`recompensa_t = log(equity_t / equity_{t-1})`**.
      * Simulación de Liquidación de Posición: Ocurre si el precio se mueve un **8%** en contra de la posición abierta (calculado como `(1 / Apalancamiento_10x) * Factor_Seguridad_0.8`).
      * Condiciones de Fin de Episodio (para Entrenamiento):
          * Drawdown Máximo de Equity: **-20%** del equity inicial del episodio (`terminated = True`).
          * Liquidación de Posición (`terminated = True`).
          * Agotamiento del Conjunto de Datos de Entrenamiento: Se alcanza `max_episode_steps` igual a la longitud del dataset (`truncated = True`).
          * En todos los casos de `terminated` o `truncated`, el siguiente episodio de entrenamiento comienza en un **punto aleatorio** del dataset de entrenamiento.
  * **Tecnologías Clave:** Python, `gymnasium`, NumPy.
  * **Entradas:** Acción del agente, datos de mercado del paso actual (de M2).
  * **Salidas:** (Observación, Recompensa, Terminated, Truncated, Info) para el agente.
  * **Parámetros Configurables (`config/module3_environment/params.yaml`):**
      * `initial_equity`: 10000 (ejemplo, en USD)
      * `leverage`: 10
      * `position_size_pct_equity`: 0.05
      * `taker_fee_rate`: 0.0004
      * `slippage_atr_multiplier`: 0.1
      * `action_threshold`: 0.15
      * `equity_drawdown_threshold_episode_end`: -0.20
      * `liquidation_safety_factor`: 0.8
      * `max_episode_steps_equals_dataset_length`: true

### Módulo 4: Agente de Reinforcement Learning (RL Agent)

  * **Responsabilidades:**
      * Implementación del agente de RL utilizando el algoritmo **Soft Actor-Critic (SAC)**.
      * **Arquitectura del Modelo (Transformer Encoder + Redes Actor/Críticos):**
          * Entrada: Secuencias de estado `(L=96, N_features=36)` del M2.
          * **Transformer Encoder:**
              * Capa de Embedding de Entrada: Proyecta `N_features=36` a `d_model=128`.
              * Positional Encoding: **Sinusoidal fija**.
              * Número de Capas de Encoder: **3 capas**.
              * Número de Cabezas de Atención (por capa): **4 cabezas**.
              * `d_model` (Dimensión del Modelo): **128**.
              * Dimensión FFN interna del Encoder: `4 * d_model = 512`.
          * **Red del Actor (Política):** MLP post-Transformer (ej. `[256, 256]` neuronas) que produce parámetros para una distribución Gaussiana Escalonada (acción `[-1,1]`).
          * **Redes del Crítico (Redes Q - dos para Clipped Double-Q):** MLPs post-Transformer (ej. `[256, 256]` cada una) que toman estado y acción, y producen valor Q.
      * **Replay Buffer:** Tamaño de **100,000** transiciones.
      * **Aprendizaje del Coeficiente de Entropía (`alpha`):** Automático (`ent_coef='auto'`).
      * Manejo de exploración vs. explotación inherente a SAC.
      * Capacidad de guardar/cargar modelos entrenados.
  * **Tecnologías Clave:** Python, PyTorch, Stable Baselines3 (SB3).
  * **Entradas:** Observación del entorno (M3), recompensa.
  * **Salidas:** Acción `action_signal` para el entorno (M3).
  * **Parámetros Configurables (Hiperparámetros de SAC en `config/module4_agent_sac/params.yaml`):**
      * `d_model_transformer`: 128
      * `transformer_layers`: 3
      * `transformer_heads`: 4
      * `actor_critic_hidden_dims`: [256, 256] (ejemplo)
      * `learning_rate`: 0.0003
      * `buffer_size`: 100000
      * `batch_size`: 256
      * `gamma`: 0.99
      * `tau`: 0.005
      * `train_freq_steps`: 1
      * `gradient_steps`: 1
      * `ent_coef`: "auto"
      * `learning_starts`: 1000

### Módulo 7: Framework de Backtesting (Walk-Forward Optimization)

  * **Responsabilidades:**
      * Orquestar el entrenamiento y la evaluación del Agente RL (M4) utilizando el Entorno de Trading (M3) sobre datos históricos (procesados por M2).
      * Implementación de **Walk-Forward Optimization (WFO):**
          * **Periodo Total de Datos Históricos:** Ej. 4 años (configurable a través de las fechas de inicio/fin en M1 y la configuración de WFO).
          * **Longitud de la Ventana In-Sample (IS) - Entrenamiento:** **18 meses** (configurable).
          * **Longitud de la Ventana Out-of-Sample (OOS) - Prueba:** **3 meses** (configurable).
          * **Paso de Avance (Step Size):** **3 meses** (configurable, igual a OOS).
          * **Tipo de Ventana IS:** **Deslizante (Rolling Window)**.
      * En cada "paso" del WFO:
          * Seleccionar el conjunto de datos IS.
          * Entrenar (o reentrenar/afinar) el agente SAC en el conjunto IS.
          * Evaluar el agente entrenado en el conjunto OOS.
          * Guardar el modelo entrenado y las métricas del periodo OOS.
      * Calcular y registrar métricas de rendimiento detalladas sobre los resultados OOS concatenados:
          * P\&L Total, Curva de Equity.
          * Max Drawdown.
          * Sharpe Ratio (anualizado, tasa libre de riesgo 0% para MVP).
          * Sortino Ratio (anualizado, tasa libre de riesgo 0% para MVP).
          * Win Rate.
          * Profit Factor.
          * Average Win / Average Loss.
          * Número de Trades.
      * Visualización de resultados (gráficos de equity, distribución de retornos, etc.).
  * **Tecnologías Clave:** Python, Pandas, Matplotlib/Seaborn/Plotly, `quantstats`.
  * **Entradas:** Datos históricos procesados (de M2), configuración del WFO y del agente.
  * **Salidas:** Reportes de backtesting, métricas, gráficos, modelos entrenados por cada "walk".
  * **Parámetros Configurables (`config/module7_backtesting_wfo/params.yaml`):**
      * `wfo_is_window_months`: 18
      * `wfo_oos_window_months`: 3
      * `wfo_step_months`: 3
      * `wfo_window_type`: "rolling"
      * `risk_free_rate_for_sharpe_sortino`: 0.0

### Módulo 9: Configuración y Gestión de Parámetros (MVP)

  * **Responsabilidades:**
      * Gestión centralizada de la configuración de todos los módulos del MVP.
      * **Estructura:**
          * Carpeta `config/` en la raíz del proyecto.
          * Subcarpetas por módulo dentro de `config/` (ej. `config/module1_data_acquisition/`).
          * Archivos `params.yaml` dentro de cada subcarpeta de módulo con sus parámetros específicos.
      * Archivo `.env` en la raíz del proyecto para:
          * Todas las rutas (paths) a directorios (datos, resultados, logs, código fuente dentro del contenedor).
          * Credenciales API (Binance).
          * Credenciales de Redis (si son necesarias).
      * Lógica en Python para cargar estas configuraciones de forma robusta.
  * **Tecnologías Clave:** Python, PyYAML (o similar para YAML), `python-dotenv`.

### Módulo 10: Despliegue y Mantenimiento (Docker para MVP)

  * **Responsabilidades (Infraestructura para Backtesting MVP):**
      * **Dockerfile:** Para empaquetar la aplicación principal (`workhorse_app`) con todas sus dependencias (Python, PyTorch, TA-Lib, SB3, librerías de manipulación de datos, etc.).
      * **`docker-compose.yml`:** Para orquestar los servicios necesarios para el MVP.
          * Servicio `workhorse_app`: Basado en el Dockerfile, ejecuta los scripts de descarga, preprocesamiento, entrenamiento y backtesting WFO.
          * Servicio `redis`: Instancia de Redis para la gestión inicial de datos brutos (M1 -\> M2).
          * Montaje de volúmenes para:
              * Código fuente (para desarrollo iterativo).
              * Carpeta `config/`.
              * Archivo `.env`.
              * Directorio de datos persistentes (ej. `data/`).
              * Directorio de resultados (ej. `results/`).
              * Opcional: Volumen persistente para Redis.
      * **Flujo de Datos con Docker para MVP:**
        1.  `workhorse_app` (M1) descarga datos y los publica en `redis`.
        2.  `workhorse_app` (M2) consume de `redis`, preprocesa y guarda features en un volumen de disco (ej. `data/features_processed/`).
        3.  `workhorse_app` (M3, M4, M7) lee las features procesadas del disco para el ciclo de entrenamiento y backtesting WFO.
  * **Tecnologías Clave:** Docker, Docker Compose.

## 4\. Stack Tecnológico Principal (MVP)

  * **Lenguaje:** Python 3.x
  * **Deep Learning:** PyTorch
  * **Reinforcement Learning:** Stable Baselines3 (SB3)
  * **Manipulación de Datos:** Pandas, NumPy
  * **Indicadores Técnicos:** TA-Lib
  * **APIs de Exchange (Descarga):** `python-binance`
  * **Message Broker (Flujo Inicial):** Redis
  * **Entorno de RL:** `gymnasium`
  * **Contenerización:** Docker, Docker Compose
  * **Configuración:** YAML, `.env` files
  * **Análisis y Visualización de Backtesting:** Matplotlib, Seaborn, Plotly, `quantstats`

## 5\. Estructura de Configuración Detallada

  * **Raíz del Proyecto:**
      * `.env`: Contiene `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `REDIS_HOST`, `REDIS_PORT`, `DATA_DIR_HOST` (ruta en el host a mapear para `data/`), `RESULTS_DIR_HOST` (ruta en el host a mapear para `results/`), etc.
      * `config/`:
          * `module1_data_acquisition/params.yaml`
          * `module2_preprocessing/params.yaml`
          * `module3_environment/params.yaml`
          * `module4_agent_sac/params.yaml`
          * `module7_backtesting_wfo/params.yaml`
          * `logging_config.yaml` (Opcional, para configurar el logging de forma centralizada).

## 6\. Flujo de Trabajo del MVP (Entrenamiento y Backtesting WFO)

1.  **Configuración Inicial:** El usuario define los parámetros en los archivos `.yaml` y las variables de entorno en `.env`.
2.  **Ejecución con Docker Compose:** `docker-compose up --build`.
3.  **Descarga de Datos (M1):** El script principal en `workhorse_app` inicia la descarga de datos históricos según la configuración y los publica en Redis. Los datos también se guardan en disco para evitar re-descargas.
4.  **Preprocesamiento (M2):** El script consume los datos de Redis (o disco), los preprocesa, calcula todas las features y guarda las secuencias de estado `(L, N_features)` en disco (ej. en formato Parquet) en una ubicación estructurada dentro del volumen de `data/`.
5.  **Backtesting Walk-Forward (M7):**
      * El M7 itera a través de las ventanas de tiempo definidas (IS y OOS).
      * Para cada "walk":
          * Carga los datos preprocesados correspondientes al periodo IS desde el disco.
          * Entrena un nuevo agente SAC (M4) utilizando el entorno (M3) con estos datos IS.
          * Guarda el modelo entrenado para ese "walk" en el volumen de `results/`.
          * Carga los datos preprocesados correspondientes al periodo OOS desde el disco.
          * Evalúa el agente entrenado en el periodo OOS, registrando todas las operaciones y el P\&L.
      * Al finalizar todos los "walks", M7 concatena los resultados de todos los periodos OOS.
      * Calcula y guarda las métricas de rendimiento globales y por "walk" en `results/`.
      * Genera gráficos y reportes en `results/`.
6.  **Resultados:** Los modelos, logs, métricas y gráficos se guardan en el directorio de resultados mapeado.

## 7\. Estado Actual del Proyecto

  * **Fase:** MVP - Definición Detallada Completada / Listo para Inicio de Desarrollo.
  * Este documento (`README.md`) define las especificaciones completas para el MVP.

## 8\. Cómo Empezar (Instrucciones Preliminares)

1.  Clonar el repositorio.
2.  Crear y configurar el archivo `.env` en la raíz del proyecto a partir de un `README.md` (que se deberá crear con los campos necesarios: `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `REDIS_HOST=redis`, `REDIS_PORT=6379`, `DATA_DIR_HOST=./data_host`, `RESULTS_DIR_HOST=./results_host`).
3.  Crear las carpetas `data_host` y `results_host` en la raíz de tu proyecto en el host si no existen.
4.  Revisar y ajustar los parámetros en los archivos `.yaml` dentro de la carpeta `config/` según sea necesario.
5.  Asegurarse de tener Docker y Docker Compose instalados.
6.  Ejecutar `docker-compose up --build workhorse_app` para iniciar el proceso (ej. descarga de datos, seguido de preprocesamiento y luego el backtesting WFO).
      * *(Se necesitarán scripts Python principales, ej. `main.py` o `run_pipeline.py`, para orquestar estos pasos dentro del contenedor `workhorse_app`)*.

## 9\. Fases Futuras (Post-MVP)

Una vez que el MVP demuestre un agente sólido y un framework de backtesting robusto:

  * **Módulo 5: Ejecución de Órdenes (Live Trading):** Implementación para interactuar con la API de futuros de Binance en tiempo real.
  * **Módulo 6: Gestión de Riesgos (Live):** Reglas de stop-loss globales, kill-switch en tiempo real.
  * **Módulo 8: Logging y Monitorización (Live):** Sistema de logging centralizado para producción y Bot de Telegram para monitorización y control básico (como se visionó inicialmente).
  * **Mejoras en el Agente (M4):**
      * Experimentación con diferentes arquitecturas Transformer.
      * Ajuste fino avanzado de hiperparámetros (ej. Optuna).
      * Investigación de preentrenamiento auto-supervisado.
      * Explicabilidad (XAI).
  * **Mejoras en el Entorno (M3):**
      * Modelos de slippage más sofisticados.
      * Simulación precisa de tasas de financiación.
  * **Optimización Continua (MLOps/CT):** Pipelines para reentrenamiento y despliegue continuo del modelo en un entorno de producción simulada o real.
  * **Expansión:** Múltiples activos, incorporación de datos alternativos.

