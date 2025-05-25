Hola Copilot,

Necesito tu ayuda activa para desplegar mi proyecto de bot de trading de Python (BTCBot) en Google Kubernetes Engine (GKE) Autopilot. La idea es que me propongas los comandos `gcloud` y `kubectl`. **Conceptualiza que los ejecutas usando tus herramientas de terminal integradas.** Después de cada comando "ejecutado", evaluaremos el resultado (si lo ejecutas tú y puedes ver el output, o si yo lo ejecuto y te confirmo el éxito o te proporciono mensajes de error). Basándote en este resultado, me guiarás con los siguientes pasos, me ayudarás a depurar o generarás los manifiestos YAML necesarios.

**Resumen del Proyecto:**
* Es un bot de trading con un pipeline de datos (adquisición, preprocesamiento, entrenamiento) y un componente de trading en vivo 24/7.
* Las imágenes Docker (`btcbot-cpu`, `btcbot-gpu`) ya están construidas y disponibles en Google Artifact Registry. Te proporcionaré las rutas completas de las imágenes cuando las necesites (ej: `europe-southwest1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/YOUR_ARTIFACT_REGISTRY_REPO/btcbot-cpu:latest`).
* El proyecto utiliza servicios de GCP: GCS, Secret Manager y BigQuery. La **región GCP principal para este despliegue es Madrid (`europe-southwest1`)**.
* Los scripts principales se encuentran en el directorio `scripts/` (ej: `download_data.py`, `preprocess_data.py`, `train_rl_agent.py`, `run_live_trader.py`).
* La configuración se gestiona a través de `src/config.yaml` y un archivo `.env`.
* **IMPORTANTE**: El proyecto ahora usa un script orquestador único (`scripts/orchestrate_training.py`) que ejecuta toda la pipeline de entrenamiento en un solo job con acceso a GPU.

**Pasos de Despliegue (necesito que generes y "ejecutes" los comandos para esto):**

1.  **Configuración de GCP y GKE:**
    * Por favor, pregúntame por mi ID de Proyecto GCP (`YOUR_GCP_PROJECT_ID`), Nombre del Bucket GCS (`YOUR_GCS_BUCKET_NAME`), y el nombre del repositorio de Artifact Registry (`YOUR_ARTIFACT_REGISTRY_REPO`) si no puedes inferirlos de mis archivos de contexto (como `cloudbuild.yaml` o `README.md`). La región GCP que usaremos es `europe-southwest1` (Madrid).
    * **Clúster GKE Autopilot:**
        * Comando para verificar si mi clúster `btcbot-autopilot-cluster` (mencionado en `README.md`) existe y está configurado correctamente en `europe-southwest1`.
        * Si no existe, comandos para crear un nuevo clúster GKE Autopilot en `europe-southwest1`.

2.  **Workload Identity:**
    * Comandos para crear una Cuenta de Servicio (SA) de GCP (ej: `btcbot-gke-sa`).
    * Comandos para asignar los roles IAM necesarios a esta SA de GCP:
        * Google Cloud Storage: `roles/storage.objectAdmin` sobre el bucket específico (`gs://YOUR_GCS_BUCKET_NAME`).
        * Secret Manager: `roles/secretmanager.secretAccessor` sobre los secretos específicos de Binance (ej: `BINANCE_API_KEY_FUTURES`, `TESTNET_BINANCE_API_KEY_FUTURES`, etc., que están definidos como nombres de secretos en mi archivo `.env` y accedidos por `ConfigManager`).
        * BigQuery: `roles/bigquery.dataEditor` y `roles/bigquery.user` sobre el dataset de logs (el ID del dataset, ej: `BIGQUERY_LOG_DATASET_ID`, lo obtendremos de la configuración o variables de entorno).
    * Comandos para crear una Cuenta de Servicio (KSA) de Kubernetes (ej: `btcbot-ksa`) en el namespace `btcbot` (crea el namespace si no existe).
    * Comandos para vincular la KSA con la SA de GCP.

3.  **IP de Salida Persistente para el Bot en Vivo:**
    * Comandos para configurar una dirección IP estática para el tráfico de salida de los pods del bot de trading en vivo en `europe-southwest1`. Esto es para la lista blanca de la API de Binance. (Probablemente usando Cloud NAT en GKE Autopilot).

4.  **Manifiestos de Kubernetes y Despliegue:**
    Para cada uno de los siguientes, proporciónanos el manifiesto YAML. Lo guardaré en un archivo, y luego me indicarás el comando `kubectl apply -f <filename.yaml>`. Después de la "ejecución", confirmaremos el resultado.

    * **Namespace:** Un manifiesto para el namespace `btcbot` de Kubernetes si no se creó anteriormente.
    * **Variables de Entorno y Secretos:**
        * Mis aplicaciones leen variables de entorno para la configuración (`GCP_PROJECT_ID`, `GCS_BUCKET_NAME`, `BIGQUERY_LOG_DATASET_ID`, `LIVE_TRADING_MODE`, etc.). Mi `README.md` menciona un ConfigMap de Kubernetes `btcbot-env-vars` para variables no sensibles. Los secretos de Binance se acceden vía Secret Manager usando Workload Identity.
    * **Job Orquestrador de Pipeline (`pipeline-orchestrator.yaml`):**
        * Ejecuta `python scripts/orchestrate_training.py --phase full` usando la imagen `btcbot-gpu`.
        * Solicita recursos de GPU (ej: NVIDIA T4 o la más adecuada para Autopilot en `europe-southwest1`).
        * Usa Workload Identity (la KSA `btcbot-ksa`).
        * Los datos y modelos se almacenan en Google Cloud Storage (no se requiere almacenamiento persistente en Kubernetes).
        * Ejecuta toda la pipeline: descarga → preprocesamiento → entrenamiento → evaluación en un solo job.
    * **CronJob para Pipeline Programada (opcional):**
        * Para ejecutar la pipeline semanalmente, crear un CronJob que lance el job orquestrador.
        * Programación: Semanal, cada sábado a las 00:00 UTC.
    * **Despliegue del Bot de Trading en Vivo (`live-trader-deployment.yaml`):**
        * Ejecuta `python scripts/run_live_trader.py` (24/7) usando la imagen `btcbot-cpu`.
        * Usa Workload Identity (`btcbot-ksa`).
        * Enruta el tráfico de salida a través de la IP estática configurada en el paso 3.

**Características del Nuevo Enfoque:**
* **Orquestador Único**: Un solo job con GPU ejecuta toda la pipeline de entrenamiento.
* **Configuración Centralizada**: Toda la configuración (símbolos, timeframes, etc.) se gestiona a través de `config.yaml`.
* **Almacenamiento en GCS**: Los datos y modelos se almacenan directamente en Google Cloud Storage, sin necesidad de almacenamiento persistente en Kubernetes.
* **Flexibilidad**: El orquestador soporta diferentes fases (`data`, `preprocess`, `train`, `evaluate`, `full`).

**Flujo de Interacción:**
1.  Creas un comando `gcloud`, `kubectl` o un manifiesto YAML.
2.  Tú "ejecutas" el comando. Tu confirmas el resultado.
3.  Basándote en el resultado, proporcionas el siguiente comando, una corrección o el siguiente manifiesto.
4.  Continuamos este proceso hasta que todos los componentes estén desplegados.

Comencemos con la configuración o verificación del clúster GKE Autopilot en `europe-southwest1` (Madrid). Por favor, solicita los parámetros que necesites (como `YOUR_GCP_PROJECT_ID`, etc.) si no los puedes inferir del contexto de mis archivos.