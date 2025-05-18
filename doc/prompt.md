## Mega-Prompt para Desarrollo del Proyecto en GitHub: Bot de Trading BTC Autónomo en GCP

**Título del Proyecto:** Bot de Trading Autónomo de BTC con RL y Transformers en GCP

**Objetivo General:**
Desarrollar un bot de trading de Bitcoin (BTC/USDT Futuros) altamente avanzado y autónomo, operando a través de la API de Binance. El sistema utilizará Reinforcement Learning (RL) con un agente basado en Transformers y estará diseñado desde cero para su ejecución, escalabilidad y optimización en Google Cloud Platform (GCP), enfocado inicialmente en un pipeline robusto de entrenamiento y backtesting.

**Referencia Principal:**
Este desarrollo debe seguir estrictamente las especificaciones y la arquitectura detalladas en el "Documento de Diseño Técnico: Bot de Trading BTC Autónomo en GCP con RL y Transformers v1.0" (referido en adelante como DDT). Se espera que el desarrollador consulte el DDT para todos los detalles de implementación de cada módulo, la arquitectura en GCP, las tecnologías seleccionadas y las decisiones de diseño.

**Principios Guía para el Desarrollo:**

1.  **Adherencia al DDT:** Todas las implementaciones deben seguir las decisiones y especificaciones del DDT. Cualquier desviación propuesta debe ser discutida y aprobada.
2.  **Modularidad:** Implementar cada módulo (Adquisición de Datos, Preprocesamiento, Entorno, Agente, Backtesting, etc.) como componentes de software bien definidos y cohesivos, tal como se describe en el DDT. Esto es crucial para la mantenibilidad, las pruebas y la futura transición a operaciones en vivo.
3.  **Código Limpio y Bien Documentado:** Escribir código Python claro, conciso, bien comentado y siguiendo las mejores prácticas (PEP 8). Incluir docstrings para todas las clases, métodos y funciones.
4.  **Nativo de GCP:** Diseñar e implementar cada componente pensando en su ejecución dentro del ecosistema GCP, especialmente Vertex AI Pipelines, GCS, BigQuery, Artifact Registry y Cloud Build.
5.  **Configurabilidad:** Hacer que el sistema sea altamente configurable a través de archivos YAML y parámetros de pipeline, como se especifica en el DDT (ej. parámetros para indicadores, arquitectura del modelo, hiperparámetros de entrenamiento, configuración del entorno).
6.  **Robustez y Manejo de Errores:** Implementar un manejo de errores robusto, logging detallado y mecanismos de reintento donde sea apropiado (especialmente en la adquisición de datos).
7.  **Automatización (MLOps):** El objetivo final de la fase inicial es un pipeline de Vertex AI completamente automatizado para el entrenamiento y la evaluación de modelos.
8.  **Pruebas:** Aunque no se detalla exhaustivamente en el DDT, se espera la implementación de pruebas unitarias para la lógica crítica de los módulos (ej. cálculos de `FeatureEngineer`, lógica del `TradingEnvironment`, `CustomTransformerFeatureExtractor`).
9.  **Seguridad:** Manejar de forma segura las credenciales (ej. claves API de Binance) utilizando Google Cloud Secret Manager.

**Estructura del Repositorio Sugerida:**

```
.
├── .github/                # Workflows de GitHub Actions (opcional, para CI/CD complementario a Cloud Build)
├── .gcloudignore           # Especifica archivos a ignorar por gcloud CLI
├── .gitignore
├── Dockerfile              # Dockerfile único para construir la imagen del proyecto
├── requirements.txt        # Dependencias de Python
├── cloudbuild.yaml         # Definición del pipeline de CI/CD para Cloud Build
├── pipeline_definition.py  # Script Python para definir y compilar el pipeline KFP v2
├── src/
│   ├── __init__.py
│   ├── configs_default/    # YAMLs con configuraciones por defecto (si se usa esa estrategia)
│   │   ├── acquisition_config_default.yaml
│   │   ├── preprocessing_config_default.yaml
│   │   ├── environment_config_default.yaml
│   │   └── agent_config_default.yaml
│   ├── data_acquisition/
│   │   ├── __init__.py
│   │   └── binance_downloader.py # Clase BinanceFuturesDownloader (Módulo 1)
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── feature_engineer.py   # Clase FeatureEngineer con pandas-ta (Módulo 2)
│   │   └── data_preprocessor.py  # Clase DataPreprocessor (Módulo 2)
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── trading_env.py        # Clase TradingEnvironment (Módulo 3)
│   │   └── simulated_broker.py   # Clase SimulatedBroker (Módulo 3)
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── custom_transformer_extractor.py # Clase CustomTransformerFeatureExtractor (Módulo 4)
│   │   └── rl_agent_manager.py   # (Opcional) Clase para gestionar creación/entrenamiento del agente
│   ├── backtesting/
│   │   ├── __init__.py
│   │   └── backtester.py         # Lógica para el bucle de backtesting y cálculo de métricas (Módulo 7)
│   ├── components/             # Scripts Python que actúan como puntos de entrada para los componentes KFP
│   │   ├── __init__.py
│   │   ├── run_data_acquisition.py
│   │   ├── run_preprocessing.py
│   │   ├── run_train_agent.py
│   │   └── run_backtest_agent.py
│   └── utils/
│       ├── __init__.py
│       └── (Opcional) config_manager_gcp.py # Si se necesita un gestor de config adaptado a GCP
│       └── (Opcional) gcp_utils.py          # Funciones de utilidad para interactuar con GCP
├── scripts/                # Scripts auxiliares o de orquestación local (si son necesarios para desarrollo)
├── tests/                  # Pruebas unitarias e de integración
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_environments.py
│   └── test_agent_components.py
└── README.md               # Este archivo, detallando el proyecto y cómo ejecutarlo.
```

**Tareas de Desarrollo Detalladas (por Módulo, referenciar DDT para especificaciones completas):**

**Módulo 1: Adquisición de Datos de Binance (DDT Sección 4.1.A)**

  * Implementar `BinanceFuturesDownloader` en `src/data_acquisition/binance_downloader.py`.
      * Conectar a API de Futuros de Binance, manejar autenticación (claves leídas como parámetros, provenientes de Secret Manager en GCP).
      * Descargar OHLCV para símbolo, intervalo y rango de fechas configurables.
      * Implementar paginación robusta y manejo de rate limits/errores con reintentos exponenciales.
      * Guardar datos en formato Parquet en GCS (la ruta GCS será un parámetro).
  * Crear `src/components/run_data_acquisition.py` como script de entrada para el componente KFP.
      * Este script parseará argumentos del pipeline, instanciará y ejecutará `BinanceFuturesDownloader`.

**Módulo 2: Preprocesamiento y Gestión de Datos (DDT Sección 4.2)**

  * Implementar `FeatureEngineer` en `src/preprocessing/feature_engineer.py` usando `pandas-ta`.
      * Calcular las 5 características OHLCV procesadas y las 15 características de indicadores técnicos especificadas. Prestar atención a los nombres de columnas generados por `pandas-ta` y mapearlos si es necesario.
  * Implementar `DataPreprocessor` en `src/preprocessing/data_preprocessor.py`.
      * Cargar datos Parquet crudos desde GCS.
      * Implementar limpieza de datos y manejo de NaNs en datos crudos (con `ffill_limit_for_nans`).
      * Utilizar `FeatureEngineer`.
      * Aplicar normalización/escalado causal a las 20 características finales (según configuración).
      * Construir secuencias 3D `(N_samples, L, N_features_mercado)`.
      * Manejar NaNs inducidos por lookbacks.
      * Guardar secuencias (`X_market`, `timestamps`) como archivo `.npz` en GCS.
  * Crear `src/components/run_preprocessing.py` como script de entrada para el componente KFP.

**Módulo 3: Entorno de Trading (Simulador) (DDT Sección 4.3)**

  * Implementar `SimulatedBroker` en `src/environments/simulated_broker.py`.
      * Lógica para calcular detalles de ejecución (comisiones, slippage basado en ATR, mínimos de orden), tamaño de posición, margen y precio de liquidación. Debe ser sin estado.
  * Implementar `TradingEnvironment` en `src/environments/trading_env.py`, heredando de `gymnasium.Env`.
      * `__init__`: Cargar datos de secuencias `.npz` desde GCS (pasado como URI), instanciar `SimulatedBroker`, definir `observation_space` (Dict con `market_features` y `portfolio_features` normalizadas) y `action_space` (Box continuo). Incluir parámetro `random_episode_start`.
      * `reset`: Reiniciar estado de cartera, seleccionar punto de inicio (aleatorio o fijo), devolver observación inicial e info.
      * `step`: Interpretar acción del agente (con umbral), usar `SimulatedBroker`, actualizar estado de cartera, calcular recompensa (`log(equity_t / equity_{t-1})`), simular liquidación, manejar fin de episodio (terminated/truncated), devolver `(obs, reward, terminated, truncated, info)`.
      * Implementar el cálculo y normalización de las 8 `portfolio_features`.
      * **NO implementar SL/TP forzados.**

**Módulo 4: Agente de RL y Entrenamiento (DDT Sección 4.4)**

  * Implementar `CustomTransformerFeatureExtractor` en `src/agent/custom_transformer_extractor.py`, heredando de `BaseFeaturesExtractor` de SB3.
      * Manejar `DictSpace` de observación.
      * Fusionar `market_features` y `portfolio_features` (replicando y concatenando portfolio\_features).
      * Implementar la arquitectura Transformer Encoder (Embedding Lineal -\> Positional Encoding -\> N Capas Encoder Transformer) como se especifica en el DDT (d\_model, n\_heads, n\_encoder\_layers, etc., configurables).
      * Devolver la representación del estado procesada (ej. salida del último paso del Transformer).
  * Crear `src/components/run_train_agent.py` como script de entrada para el componente KFP.
      * Parsear argumentos del pipeline (parámetros del entorno, del agente y del entrenamiento).
      * Instanciar `TradingEnvironment`.
      * Instanciar o cargar (desde checkpoint GCS) el agente SAC de SB3, configurando `policy_kwargs` para usar `CustomTransformerFeatureExtractor` y la arquitectura de redes MLP actor/crítico.
      * Configurar Callbacks de SB3: `CheckpointCallback` (guardando en GCS), `EvalCallback` (opcional), y adaptar logs para Vertex AI TensorBoard (escribiendo a GCS).
      * Ejecutar `model.learn()`, manejando la reanudación (`reset_num_timesteps=False`) y el cálculo de timesteps restantes.
      * Guardar el modelo final en GCS (y en el artefacto de salida del componente).

**Módulo 7: Framework de Backtesting (DDT Sección 4.5)**

  * Crear `src/components/run_backtest_agent.py` como script de entrada para el componente KFP.
      * Parsear argumentos del pipeline (URI del modelo entrenado, URI de datos de backtest, parámetros del entorno).
      * Instanciar `TradingEnvironment` con `random_episode_start=False`.
      * Cargar el modelo SAC entrenado desde GCS.
      * Ejecutar bucle de backtesting, obteniendo acciones con `model.predict(deterministic=True)`.
      * Registrar trades e historial de equity.
      * Calcular métricas de rendimiento y generar informe HTML con `quantstats`.
      * Guardar métricas y el informe `quantstats` como artefactos de salida del componente.

**Configuración General del Proyecto:**

  * Crear `Dockerfile` único siguiendo las especificaciones del DDT, incluyendo todas las dependencias de `requirements.txt`.
  * Crear `requirements.txt` con todas las librerías Python especificadas en el DDT.
  * Implementar la gestión de configuración: los componentes KFP deben aceptar parámetros para la configuración detallada, sobrescribiendo posibles defaults internos.
  * Utilizar `google-cloud-secret-manager` para el acceso a claves API de Binance en los componentes relevantes.

**Pipeline de Vertex AI (KFP):**

  * Implementar `pipeline_definition.py` para definir el pipeline KFP v2 que orquesta los componentes (Adquisición -\> Preprocesamiento -\> Entrenamiento -\> Backtesting).
      * Asegurar el correcto paso de artefactos (URIs GCS) y parámetros entre componentes.
      * El pipeline debe ser parametrizable.

**CI/CD:**

  * Implementar `cloudbuild.yaml` para automatizar:
      * Construcción de la imagen Docker.
      * Subida a Artifact Registry.
      * Compilación del pipeline KFP (asegurando que referencia la imagen correcta).
      * Despliegue del pipeline en Vertex AI Pipelines.

**Logging y Monitorización (DDT Sección 4.6):**

  * Asegurar el uso consistente de la librería `logging` de Python en todos los scripts, configurada para ser compatible con Cloud Logging (idealmente logs estructurados/JSON).
  * Los componentes KFP deben emitir métricas clave como `Output[Metrics]`.
  * El componente de entrenamiento debe generar logs para Vertex AI TensorBoard y guardarlos en GCS.

**Pruebas:**

  * Escribir pruebas unitarias (usando `pytest`) para la lógica crítica, especialmente para:
      * Cálculos de `FeatureEngineer`.
      * Normalización en `DataPreprocessor`.
      * Mecánicas clave del `TradingEnvironment` (ej. procesamiento de órdenes, cálculo de P\&L, recompensas, condiciones de fin de episodio).
      * Lógica de fusión de características y dimensiones en `CustomTransformerFeatureExtractor`.

**Documentación:**

  * Mantener este `README.md` actualizado con el progreso, la estructura del proyecto y las instrucciones para configurar y ejecutar los pipelines.
  * Asegurar docstrings y comentarios en el código.

**Foco Inicial:**
El desarrollo debe priorizar la creación de un pipeline de entrenamiento y backtesting completamente funcional en Vertex AI. Los módulos y funcionalidades orientados exclusivamente a trading en vivo (como el Módulo 5 detallado para ejecución) son "Próximamente" y no forman parte del alcance inicial de implementación, aunque el diseño modular debe facilitarlos.

Este prompt debe servir como una guía exhaustiva. Referirse constantemente al Documento de Diseño Técnico (DDT) para aclaraciones sobre cualquier especificación. Se espera un alto estándar de calidad de código, pruebas y adherencia a las mejores prácticas de MLOps en GCP.

-----