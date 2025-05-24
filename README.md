# Trading Bot con Reinforcement Learning para Futuros de Criptomonedas

> **⚠️ ADVERTENCIA IMPORTANTE ⚠️**
> 
> Este bot utiliza órdenes a MERCADO que se ejecutan inmediatamente al mejor precio disponible, lo que puede resultar en slippage significativo en mercados volátiles o con baja liquidez. 
> 
> Además, existe una discrepancia importante entre entrenamiento y operación en vivo: en entrenamiento se considera el "tiempo en posición" como característica, pero en modo real esta característica se establece como un valor constante (0.0). Esto puede causar que el rendimiento en vivo difiera considerablemente del backtesting.
> 
> **Utilice este software bajo su propia responsabilidad. Los resultados del backtesting no garantizan resultados similares en trading en vivo.**

## Descripción

Este proyecto implementa un agente de trading basado en Reinforcement Learning para operar en el mercado de futuros de criptomonedas. Utiliza el algoritmo SAC (Soft Actor-Critic) con una arquitectura de red neuronal basada en Transformers para procesar datos históricos del mercado y tomar decisiones de trading.

**Características Clave**: El bot combina análisis técnico avanzado con RL para tomar decisiones de trading basadas en características del mercado y del estado de la cartera. Utiliza un extractor de características personalizado basado en Transformers para capturar patrones temporales en datos OHLCV.

**Estado Actual del Proyecto**: El sistema está funcional tanto para entrenamiento como para operación en vivo a través de Binance Futures.

## Características Principales

- Descarga de datos históricos de Binance Futures API
- Preprocesamiento y cálculo de características técnicas de trading usando pandas-ta
- Simulación de un entorno de trading de futuros BTCUSDT con Gymnasium
- Agente de RL (Soft Actor-Critic) con extractor de características basado en Transformers
- Modelo de trading desplegable en Google Vertex AI para inferencia en tiempo real
- Modo de operación en vivo conectado a Binance Futures (real y testnet)
- Visualización de resultados y métricas de rendimiento

## Estructura del Proyecto

```
btcbot/
├── cloudbuild_cpu.yaml        # Build específico para imagen CPU
├── cloudbuild.yaml            # Build específico para imagen GPU
├── DEPLOYMENT.md              # Guía completa de despliegue en GKE
├── Dockerfile.cpu             # Imagen Docker para componentes CPU
├── Dockerfile.gpu             # Imagen Docker para entrenamiento con GPU
├── k8s/                       # Manifiestos de Kubernetes para GKE Autopilot
│   ├── configmap.yaml         # Variables de entorno no sensibles
│   ├── data-acquisition-cronjob.yaml  # Job programado de descarga
│   ├── data-preprocessing-job.yaml    # Job de preprocesamiento
│   ├── live-trader-deployment.yaml    # Deployment del bot 24/7
│   ├── model-training-job.yaml        # Job de entrenamiento con GPU
│   ├── namespace.yaml         # Namespace btcbot
│   └── pipeline-orchestrator.yaml     # Orquestador de pipeline secuencial
├── scripts/                   # Scripts ejecutables
│   ├── download_data.py       # Descarga datos históricos de Binance
│   ├── preprocess_data.py     # Preprocesa datos y extrae características
│   ├── train_rl_agent.py      # Entrena el agente de RL
│   ├── evaluate_rl_agent.py   # Evalúa el rendimiento del agente
│   ├── run_live_trader.py     # Ejecuta el bot de trading en vivo
│   └── test_binance_api.py    # Prueba la conexión con la API de Binance
└── src/                       # Código fuente
    ├── config.yaml            # Configuración centralizada
    ├── agent/                 # Implementación del agente RL
    ├── data/                  # Código para adquisición y preprocesamiento
    ├── environments/          # Entorno de simulación
    ├── live/                  # Módulos para trading en vivo
    └── utils/                 # Utilidades generales
```

## Arquitectura del Sistema

### Flujo de Datos y Operación

1. **Adquisición y Preprocesamiento de Datos**:
   - Descarga de datos históricos OHLCV desde Binance Futures
   - Cálculo de características técnicas (indicadores) y normalización
   - Almacenamiento en Google Cloud Storage

2. **Entrenamiento del Modelo**:
   - Entorno de simulación basado en Gymnasium
   - Entrenamiento del agente SAC con arquitectura Transformer
   - Guardado de modelos y checkpoints en Google Cloud Storage

3. **Despliegue del Modelo**:
   - Modelo cargado directamente: El modelo de RL se carga directamente en el script de trading en vivo para inferencia local.

4. **Trading en Vivo**:
   - Websocket para detección de cierre de velas
   - Preprocesamiento en tiempo real de datos de mercado
   - Construcción de características de cartera basadas en posición actual
   - Obtención de predicciones del modelo cargado localmente.
   - Ejecución de órdenes en Binance Futures

### Componentes Clave

- **PortfolioFeatureBuilder**: Construye características normalizadas del estado de la cartera
- **LiveDataProcessor**: Procesa datos OHLCV en tiempo real para extraer características
- **LiveWebsocketManager**: Detecta nuevas velas cerradas en tiempo real
- **LiveBinanceAPIManager**: Gestiona conexión con Binance para operaciones
- **LiveTrader**: Orquesta el flujo completo de trading en vivo

## Requisitos para la Nube y Despliegue

### Configuración de Google Cloud Platform

El proyecto está **completamente desplegado en Google Cloud Platform** usando:

- **Google Kubernetes Engine (GKE) Autopilot**: Orquestación y ejecución de todos los componentes
- **Google Cloud Storage (GCS)**: Almacenamiento para datos brutos, procesados y modelos
- **Google Cloud Secret Manager**: Gestión segura de credenciales de Binance
- **Google BigQuery**: Logging y análisis de entrenamiento y trading en vivo
- **Google Artifact Registry**: Almacenamiento de imágenes Docker
- **Cloud NAT**: IP estática para whitelist de Binance API

### Configuración del Proyecto

Los valores de configuración actuales del proyecto son:

- **GCP Project ID**: `lofty-complex-460416-r6`
- **GCS Bucket**: `lofty-complex-460416-r6`
- **Artifact Registry**: `lofty-complex-460416-r6-repo`
- **Región**: `europe-southwest1` (Madrid)
- **Clúster GKE**: `btcbot-autopilot-cluster`
- **IP Estática NAT**: `34.175.215.35` (para whitelist Binance)

### Arquitectura de Despliegue

```
┌─────────────────────────────────────────────────────────────┐
│                    GKE Autopilot Cluster                   │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │   CronJob Semanal   │  │    Live Trader Deployment  │  │
│  │  Data Acquisition   │  │       (24/7 Running)       │  │
│  └─────────────────────┘  └─────────────────────────────┘  │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │ Data Preprocessing  │  │   Model Training Job        │  │
│  │      Job (CPU)      │  │      (GPU - NVIDIA T4)      │  │
│  └─────────────────────┘  └─────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │           Pipeline Orchestrator                         ││
│  │    (Ejecuta: Acquisition → Preprocessing → Training)   ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    Cloud NAT Gateway                       │
│              IP Estática: 34.175.215.35                   │
│                (Para Binance Whitelist)                    │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                     Binance Futures API                    │
│                   (Trading & Websockets)                   │
└─────────────────────────────────────────────────────────────┘
```

## Requisitos Previos

- Python 3.9+
- Git
- Cuenta en Google Cloud Platform con facturación activada
- Cuenta en Binance Futures (real o testnet)

## Despliegue en GKE Autopilot

### 🚀 Despliegue Rápido

El proyecto está completamente containerizado y listo para desplegarse en Google Kubernetes Engine (GKE) Autopilot. Para una **guía completa de despliegue paso a paso**, consulta [`DEPLOYMENT.md`](./DEPLOYMENT.md).

### Verificación del Estado

Una vez desplegado, puedes verificar el estado de todos los componentes:

```bash
# Verificar pods en ejecución
kubectl get pods -n btcbot

# Ver logs del bot de trading en vivo
kubectl logs -f -n btcbot deployment/live-trader-deployment

# Verificar jobs programados
kubectl get cronjobs -n btcbot

# Ver estado general del sistema
kubectl get all -n btcbot
```

### Gestión del Sistema

```bash
# Ejecutar pipeline completo manualmente
kubectl apply -f k8s/pipeline-orchestrator.yaml

# Parar/reiniciar el bot de trading
kubectl scale deployment live-trader-deployment --replicas=0 -n btcbot  # Parar
kubectl scale deployment live-trader-deployment --replicas=1 -n btcbot  # Reiniciar

# Ejecutar adquisición de datos manual
kubectl create job data-acquisition-manual --from=cronjob/data-acquisition-cronjob -n btcbot

# Actualizar variables de configuración
kubectl edit configmap btcbot-env-vars -n btcbot
```

### Configuración de Binance API

**⚠️ IMPORTANTE**: Debes agregar la IP estática `34.175.215.35` a la whitelist de tu cuenta de Binance API para que el bot pueda operar correctamente.

## Instalación Local (Desarrollo)

1. **Clonar el repositorio**:
   ```
   git clone https://github.com/yourusername/btcbot.git
   cd btcbot
   ```

2. **Crear y activar un entorno virtual**:
   ```
   python3 -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```
   pip install -r requirements.txt
   ```

4. **Configurar Variables de Entorno**:
   
   Crear un archivo `.env` en la raíz del proyecto con el siguiente contenido:
   
   ```
   # Configuración de Google Cloud (OBLIGATORIO)
   GCP_PROJECT_ID="lofty-complex-460416-r6"
   GCS_BUCKET_NAME="lofty-complex-460416-r6"
   GCP_REGION="europe-southwest1"
   BIGQUERY_LOG_DATASET_ID="btcbot_logs"
   
   # Configuración de modo de trading (opcional, por defecto es TESTNET)
   LIVE_TRADING_MODE="false"  # Cambiar a "true" para trading real
   ```
   
   > **Nota**: Las credenciales de Binance deben estar almacenadas en Google Cloud Secret Manager como `BINANCE_API_KEY_FUTURES` y `BINANCE_API_SECRET_FUTURES` para el despliegue en producción, o en variables de entorno locales para desarrollo.

5. **Configurar Google Cloud**:

   ```bash
   # Iniciar sesión en Google Cloud
   gcloud auth login
   
   # Configurar credenciales de aplicación por defecto
   gcloud auth application-default login
   
   # Configurar proyecto (usa los valores reales del proyecto)
   gcloud config set project lofty-complex-460416-r6
   
   # Habilitar APIs necesarias
   gcloud services enable secretmanager.googleapis.com storage.googleapis.com container.googleapis.com
   
   # Crear bucket de GCS (si no existe)
   gsutil mb -p lofty-complex-460416-r6 -l europe-southwest1 gs://lofty-complex-460416-r6
   
   # Almacenar credenciales de Binance en Secret Manager
   echo -n "tu-api-key" | gcloud secrets create BINANCE_API_KEY_FUTURES --data-file=-
   echo -n "tu-api-secret" | gcloud secrets create BINANCE_API_SECRET_FUTURES --data-file=-
   ```

## Flujo de Trabajo

### 🔄 Pipeline Automatizado en GKE

El sistema está completamente automatizado y ejecuta los siguientes componentes en GKE Autopilot:

1. **📊 Adquisición de Datos (CronJob)**
   - Ejecuta automáticamente cada sábado a las 00:00 UTC
   - Descarga datos históricos de Binance Futures
   - Almacena en Google Cloud Storage

2. **🔧 Preprocesamiento de Datos (Job)**
   - Calcula indicadores técnicos
   - Normaliza características
   - Crea secuencias para entrenamiento

3. **🧠 Entrenamiento del Modelo (Job con GPU)**
   - Entrena el agente SAC con arquitectura Transformer
   - Utiliza GPU NVIDIA T4 en GKE Autopilot
   - Guarda modelos en GCS automáticamente

4. **📈 Trading en Vivo (Deployment 24/7)**
   - Bot de trading ejecutándose continuamente
   - Websocket en tiempo real con Binance
   - Decisiones automáticas con el modelo entrenado

### 🚀 Ejecución Manual del Pipeline

Para ejecutar todo el pipeline secuencialmente:

```bash
kubectl apply -f k8s/pipeline-orchestrator.yaml
kubectl logs -f -n btcbot job/pipeline-orchestrator
```

### 💻 Ejecución Local (Desarrollo)

> **Nota**: Para desarrollo local únicamente. En producción, usa el despliegue en GKE.

### 1. Descarga de Datos Históricos

Este paso descarga datos OHLCV de Binance Futures y los guarda en Google Cloud Storage:

```bash
python scripts/download_data.py
```

El script descargará datos para el símbolo y período definido en `config.yaml`. Por defecto es BTCUSDT con velas de 1 hora.

### 2. Preprocesamiento de Datos

Este paso calcula indicadores técnicos, normaliza características y crea secuencias:

```bash
python scripts/preprocess_data.py
```

Para procesar un archivo específico:
```bash
python scripts/preprocess_data.py --file BTCUSDT_FUTURES_1h_20200101_20250516.csv
```

Los datos procesados se guardan como arrays NumPy en GCS para su posterior uso en entrenamiento.

### 3. Entrenamiento del Modelo

Entrenar el agente de RL con el algoritmo SAC y la arquitectura de Transformers:

```bash
# Entrenamiento rápido (prueba)
python scripts/train_rl_agent.py --config src/config.yaml --timesteps 1000

# Entrenamiento completo
python scripts/train_rl_agent.py --config src/config.yaml --timesteps 1000000
```

Los modelos entrenados se guardan automáticamente en GCS, y también se guardan checkpoints durante el entrenamiento.

### 4. Evaluación del Modelo

Evalúa el rendimiento del modelo entrenado en datos no vistos:

```bash
python scripts/evaluate_rl_agent.py --config src/config.yaml --model-path models/sac_transformer_trading_agent_final_1000000_steps.zip --episodes 10
```

Esto generará gráficos y estadísticas de rendimiento del agente en el directorio `results/`.

### 5. Despliegue del Modelo (Opcional)

> **Nota**: Para el script principal `run_live_trader.py`, el despliegue de un servidor de inferencia separado como se describe a continuación ya no es el método estándar, ya que el modelo ahora se carga y utiliza directamente dentro del script. Las siguientes secciones sobre el servidor local y Vertex AI se mantienen como referencia para casos de uso alternativos donde se requiera un endpoint de predicción explícito (utilizando `serving/serve.py`).

#### 5.1 Servidor Local (para pruebas con `serving/serve.py`)

```bash
cd serving
python serve.py --model_path tu-bucket-nombre/models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000000_steps.zip
```

#### 5.2 Despliegue en Vertex AI (para `serving/serve.py`)

a) Construir la imagen del contenedor:
```bash
gcloud builds submit --config cloudbuild.yaml .
```

b) Desplegar en Vertex AI:
```bash
gcloud ai endpoints create --region=tu-region --display-name=btcbot-endpoint
gcloud ai models upload --container-image-uri=gcr.io/tu-proyecto/btcbot-serving:latest --region=tu-region --display-name=btcbot-model
gcloud ai endpoints deploy-model tu-id-endpoint --model=tu-id-modelo --region=tu-region
```

### 6. Trading en Vivo

**En GKE (Producción):**
```bash
# El bot ya está ejecutándose automáticamente
kubectl logs -f -n btcbot deployment/live-trader-deployment

# Para parar/reiniciar:
kubectl scale deployment live-trader-deployment --replicas=0 -n btcbot  # Parar
kubectl scale deployment live-trader-deployment --replicas=1 -n btcbot  # Reiniciar
```

**Local (Desarrollo):**
Asegúrese de que la ruta al modelo entrenado esté correctamente configurada en `src/config.yaml` bajo `agent.live_model_path`. Luego, ejecute:
```bash
python scripts/run_live_trader.py --config src/config.yaml
```

El bot comenzará a escuchar el cierre de nuevas velas, procesar datos en tiempo real y tomar decisiones de trading utilizando el modelo cargado localmente.

## Automatización y Orquestación MLOps con Cloud Build y GKE/Vertex AI

Esta sección describe cómo el pipeline de MLOps automatiza la construcción, el despliegue y la ejecución de los componentes del sistema BTCBot, utilizando Google Cloud Build, Google Kubernetes Engine (GKE) en modo Autopilot y Vertex AI Pipelines.

### Visión General

El archivo `cloudbuild.yaml` en la raíz del repositorio define un pipeline de Integración Continua y Despliegue Continuo (CI/CD) que automatiza los siguientes procesos clave:
1.  Construcción de imágenes Docker para la CPU y la GPU.
2.  Configuración de la infraestructura necesaria en GKE (cluster, namespace, secrets).
3.  Compilación del pipeline de Kubeflow (KFP) definido en Python.
4.  Envío y ejecución del pipeline de KFP en Vertex AI Pipelines para el reentrenamiento del modelo.
5.  Despliegue o actualización del bot de trading en vivo como un servicio continuo en GKE.

Este enfoque permite una gestión MLOps más robusta, facilitando la reproducibilidad, la automatización de reentrenamientos y el despliegue continuo de nuevas versiones del bot.

### Componentes de la Orquestación

-   **`cloudbuild.yaml`**:
    Es el corazón de la automatización CI/CD. Sus responsabilidades incluyen:
    -   Construir las imágenes Docker (`Dockerfile.cpu` y `Dockerfile.gpu`) y etiquetarlas.
    -   Pushear las imágenes a Google Artifact Registry.
    -   Asegurar que el cluster de GKE Autopilot esté creado y configurado.
    -   Crear el namespace de Kubernetes (`btcbot`) si no existe.
    -   Crear o actualizar el Secret de Kubernetes (`btcbot-env-vars`) con variables de entorno no sensibles, utilizando sustituciones de Cloud Build.
    -   Compilar el pipeline de KFP (`pipelines/trading_pipeline.py`) a su formato JSON.
    -   Enviar el pipeline compilado a Vertex AI Pipelines para su ejecución.
    -   Desplegar la última versión del bot de trading en vivo en GKE utilizando el manifiesto `k8s/live-trader-deployment.yaml`.

-   **Imágenes Docker**:
    -   `btcbot-cpu`: Imagen base utilizada para tareas que no requieren GPU, como la adquisición y preprocesamiento de datos, y la ejecución del bot de trading en vivo.
    -   `btcbot-gpu`: Imagen que incluye dependencias de CUDA/GPU, utilizada para el paso de entrenamiento del modelo en el pipeline de KFP.
    Ambas imágenes se almacenan en Artifact Registry.

-   **GKE Autopilot Cluster**:
    Cloud Build verifica o crea un cluster de GKE Autopilot llamado `btcbot-autopilot-cluster` en la región `europe-south1`. Este cluster gestiona la ejecución de los componentes del pipeline de KFP (como trabajos de Vertex AI) y el despliegue continuo del bot de trading en vivo.

-   **Kubernetes Secrets (`btcbot-env-vars`)**:
    El archivo `k8s/btcbot-env-secret.yaml.template` sirve como plantilla. Cloud Build crea un Secret en Kubernetes llamado `btcbot-env-vars` dentro del namespace `btcbot`. Este Secret almacena variables de entorno no sensibles (ej. `GCP_PROJECT_ID`, `GCS_BUCKET_NAME`, `BIGQUERY_LOG_DATASET_ID`, `USE_TESTNET`, `LOG_LEVEL`) que son necesarias para los scripts y aplicaciones. Los valores para estas variables se inyectan durante el proceso de Cloud Build mediante sustituciones.
    **Importante**: Las credenciales sensibles como las claves API de Binance NO se almacenan aquí; estas deben seguir siendo gestionadas a través de Google Cloud Secret Manager y accedidas directamente por la aplicación cuando sea necesario.

-   **Kubeflow Pipeline (`pipelines/trading_pipeline.py`)**:
    Define un pipeline de Machine Learning de tres pasos utilizando el SDK de KFP:
    1.  **Adquisición de Datos**: Ejecuta `scripts/download_data.py` en un contenedor CPU.
    2.  **Preprocesamiento de Datos**: Ejecuta `scripts/preprocess_data.py` en un contenedor CPU.
    3.  **Entrenamiento del Modelo**: Ejecuta `scripts/train_rl_agent.py` en un contenedor GPU, aprovechando el hardware acelerado para el entrenamiento.
    Este pipeline se compila a un archivo JSON y se envía a Vertex AI Pipelines, lo que permite ejecuciones programadas o activadas manualmente para el reentrenamiento del modelo.

-   **Deployment del Bot en Vivo (`k8s/live-trader-deployment.yaml`)**:
    Este manifiesto de Kubernetes describe el despliegue del bot de trading en vivo. Cloud Build lo utiliza para desplegar `scripts/run_live_trader.py` como un servicio continuo en el cluster de GKE. El Deployment asegura que una instancia del bot esté siempre en ejecución, utilizando la imagen CPU más reciente y consumiendo las variables de entorno del Secret `btcbot-env-vars`.

### Ejecución del Pipeline de Cloud Build

Para ejecutar el pipeline completo de MLOps definido en `cloudbuild.yaml`, se utiliza el siguiente comando `gcloud`:

```bash
gcloud builds submit . --config cloudbuild.yaml --substitutions=\
_GCS_BUCKET_NAME="tu-bucket-gcs-real",\
_SECRET_GCS_BUCKET_NAME="tu-bucket-gcs-real",\
_SECRET_BIGQUERY_LOG_DATASET_ID="tu_dataset_id_real",\
_SECRET_USE_TESTNET="false",\
_SECRET_LOG_LEVEL="INFO",\
_ARTIFACT_REGISTRY_REPO="btcbot-images",\
_GKE_CLUSTER_NAME="btcbot-autopilot-cluster",\
_KFP_PIPELINE_NAME="btcbot-mlops-pipeline",\
_VERTEX_AI_PIPELINES_RUNNER_SA="tu-sa-vertex-pipelines@tu-proyecto-id.iam.gserviceaccount.com"
# Añadir más sustituciones según sea necesario, como _GCP_REGION si es diferente al default.
# Asegúrate de que _GCS_BUCKET_NAME se usa para _VERTEX_PIPELINE_ROOT.
```

**Explicación de las Sustituciones**:
-   `_GCS_BUCKET_NAME`: (Requerido por `_VERTEX_PIPELINE_ROOT`) El bucket de GCS donde Vertex AI Pipelines almacenará los artefactos del pipeline.
-   `_SECRET_GCS_BUCKET_NAME`: (Requerido para el Secret `btcbot-env-vars`) El bucket de GCS que la aplicación usará.
-   `_SECRET_BIGQUERY_LOG_DATASET_ID`: (Requerido para el Secret `btcbot-env-vars`) El ID del dataset de BigQuery para los logs.
-   `_SECRET_USE_TESTNET`: (Requerido para el Secret `btcbot-env-vars`) Define si el bot opera en modo testnet (`true` o `false`).
-   `_SECRET_LOG_LEVEL`: (Requerido para el Secret `btcbot-env-vars`) Nivel de logging para la aplicación (ej. `INFO`, `DEBUG`).
-   `_ARTIFACT_REGISTRY_REPO`: Nombre del repositorio en Artifact Registry donde se almacenarán las imágenes Docker.
-   `_GKE_CLUSTER_NAME`: Nombre del cluster de GKE Autopilot.
-   `_KFP_PIPELINE_NAME`: Nombre para la ejecución del pipeline en Vertex AI.
-   `_VERTEX_AI_PIPELINES_RUNNER_SA`: La cuenta de servicio que Vertex AI Pipelines utilizará para ejecutar los componentes del pipeline. Esta cuenta de servicio necesita permisos para acceder a GCS, BigQuery, GKE (para crear trabajos si es necesario), y otros servicios de GCP.

Es fundamental que la cuenta de servicio de Cloud Build (`[PROJECT_NUMBER]@cloudbuild.gserviceaccount.com`) tenga los permisos necesarios para gestionar recursos en GCS, Artifact Registry, GKE, IAM (para roles en cuentas de servicio), y Vertex AI.

### Acceso a Variables de Entorno en los Componentes

-   **Pipeline de KFP / Vertex AI**: Los componentes del pipeline (definidos en `pipelines/trading_pipeline.py`) reciben parámetros como `gcp_project_id`, `gcs_bucket_name`, etc., directamente desde la definición del pipeline en Python. Estos valores se suministran al pipeline de Vertex AI en el paso `gcloud ai platform pipelines jobs submit` de Cloud Build, utilizando las sustituciones.
-   **Scripts dentro de Contenedores (Entrenamiento y Bot en Vivo)**: Los scripts como `train_rl_agent.py` o `run_live_trader.py`, cuando se ejecutan dentro de contenedores Docker (ya sea como parte de un trabajo de Vertex AI o como un Deployment en GKE), leen su configuración (ej. ID del proyecto, nombre del bucket) de variables de entorno. En GKE, estas variables de entorno son pobladas por Kubernetes a partir del Secret `btcbot-env-vars`. La aplicación accede a ellas usando `os.environ.get()` o a través del `ConfigManager`.

Este sistema de MLOps permite una gestión automatizada y centralizada del ciclo de vida del modelo de trading.

## Configuración del Sistema

### Configuración Centralizada

El archivo `src/config.yaml` contiene toda la configuración del sistema organizada en las siguientes secciones:

- **data_paths**: Rutas de GCS para datos brutos y procesados
- **binance_api**: Configuración de la API de Binance
- **data_acquisition_defaults**: Parámetros para descarga de datos (símbolo, intervalos)
- **preprocessing**: Configuración de indicadores técnicos y normalización
- **environment**: Parámetros para el entorno de simulación
- **agent**: Hiperparámetros del modelo SAC, arquitectura Transformer, ruta al modelo para trading en vivo (`live_model_path`) y configuración para el logging de entrenamiento a BigQuery (`bigquery_logging`).
- **live_trading**: Configuración específica para el trading en vivo (ej. configuración de delays). Los logs de trading en vivo se envían a BigQuery a la tabla `LiveTrading_{FECHA}` dentro del dataset especificado por la variable de entorno `BIGQUERY_LOG_DATASET_ID`.

### Administración de Credenciales y Datos

- **Datos**: Google Cloud Storage (GCS)
  - Datos brutos: `gs://tu-bucket-nombre/raw/`
  - Datos procesados: `gs://tu-bucket-nombre/processed/`
  - Modelos: `gs://tu-bucket-nombre/models/sac_transformer_trading_agent/`

- **Credenciales**: Google Cloud Secret Manager
  - `BINANCE_API_KEY_FUTURES`: Clave de API de Binance Futures
  - `BINANCE_API_SECRET_FUTURES`: Secreto de API de Binance Futures

## Componentes Principales del Trading en Vivo

### 1. Websocket Manager (`src/live/websocket_manager.py`)
Establece conexión WebSocket con Binance para detectar cierres de velas en tiempo real.

### 2. API Manager (`src/live/binance_api_manager.py`) 
Gestiona la comunicación con la API de Binance para obtener datos de cuenta y ejecutar órdenes.

### 3. Data Processor (`src/live/live_data_processor.py`)
Procesa los datos OHLCV en tiempo real, calcula indicadores técnicos y normaliza características.

### 4. Portfolio Feature Builder (`src/live/portfolio_feature_builder.py`)
Construye un vector de características que representa el estado actual de la cartera:
- Dirección de posición actual (-1, 0, 1)
- Tamaño de la posición normalizado
- PnL no realizado normalizado
- Pasos en la posición actual
- Precio de entrada vs precio actual
- Saldo disponible normalizado
- Margen de liquidación
- PnL total normalizado

### 5. Orquestador (`scripts/run_live_trader.py`)
Coordina todos los componentes, procesa nuevas velas, solicita predicciones al modelo y ejecuta decisiones de trading.

## Logging y Monitoreo

El sistema incluye un completo sistema de logging que registra:

- **Logs locales**: Archivos de texto detallados para depuración
- **Logs en Google Cloud Storage**: Registros CSV para análisis posterior
- **Logs en BigQuery**:
    - **Entrenamiento**: Durante el entrenamiento del agente (`scripts/train_rl_agent.py`), se registran datos detallados en tablas diarias con el formato `entrenamiento_{FECHA}` (ej. `entrenamiento_20231028`). Estas tablas contienen información por paso (estado, acción, recompensa), resúmenes de episodios y métricas de entrenamiento (pérdidas, tasa de aprendizaje). El dataset de BigQuery se especifica mediante la variable de entorno `BIGQUERY_LOG_DATASET_ID`. La configuración para este logging (ej. tamaño de batch) se encuentra en `src/config.yaml` bajo `agent.bigquery_logging`.
    - **Trading en Vivo**: Los eventos y decisiones del trading en vivo (`scripts/run_live_trader.py`) se registran en tablas diarias con el formato `LiveTrading_{FECHA}` (ej. `LiveTrading_20231028`) dentro del mismo dataset de BigQuery especificado por `BIGQUERY_LOG_DATASET_ID`. Esto permite un análisis detallado del rendimiento en tiempo real.
    - Los logs locales y en GCS (CSV) se mantienen para depuración y análisis complementarios.

## Tecnologías Utilizadas

- **PyTorch**: Framework para redes neuronales y extractor de características Transformer
- **Stable Baselines3**: Implementación del algoritmo SAC
- **Gymnasium**: Framework para el entorno de simulación
- **pandas-ta**: Biblioteca de indicadores técnicos para trading
- **python-binance**: Cliente API oficial de Binance
- **Google Cloud**:
    -   Google Cloud Storage (GCS): Para almacenamiento de datos y modelos.
    -   Google Cloud Secret Manager: Para gestión de credenciales.
    -   Google BigQuery: Para logging detallado de entrenamiento y trading en vivo.
    -   Vertex AI Pipelines: Para la orquestación y ejecución de pipelines de ML.
    -   Google Cloud Build: Para CI/CD y automatización de MLOps.
    -   Google Kubernetes Engine (GKE Autopilot): Para orquestación de contenedores (ej. bot en vivo).
- **Kubeflow Pipelines (KFP SDK)**: Para la definición de los pipelines de ML.
- **Docker**: Para la contenerización de aplicaciones.
- **Flask/Gunicorn**: Utilizado para `serving/serve.py` (opcional, para despliegues de API de predicción independientes).
- **Websockets**: Conexión en tiempo real con Binance

## Limitaciones y Consideraciones

- Los resultados en backtesting pueden diferir significativamente de los resultados en vivo
- El bot opera mejor en mercados con tendencias claras y volatilidad moderada
- Las estrategias basadas en RL pueden requerir reentrenamiento periódico
- El rendimiento del servidor de inferencia puede afectar los tiempos de ejecución

## Licencia

MIT
