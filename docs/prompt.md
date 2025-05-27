Hola Gemini,

Necesito tu ayuda activa para desplegar mi proyecto de bot de trading de Python (BTCBot) utilizando una nueva estrategia:
* **Vertex AI Training en `europe-west4` (ej. Países Bajos)** para el pipeline de entrenamiento completo, aprovechando la mejor disponibilidad de GPUs.
* **Cloud Run en `europe-southwest1` (Madrid)** para el bot de trading en vivo, buscando optimizar la latencia con la API de Binance.

La idea es que me propongas los comandos `gcloud`. **Conceptualiza que los ejecutas usando tus herramientas de terminal integradas.** Después de cada comando "ejecutado", evaluaremos el resultado (si lo ejecutas tú y puedes ver el output, o si yo lo ejecuto y te confirmo el éxito o te proporciono mensajes de error). Basándote en este resultado, me guiarás con los siguientes pasos o me ayudarás a depurar.

**Resumen del Proyecto (Nueva Estrategia Multiregión):**
* **Entrenamiento (`europe-west4`):** Un único script `scripts/orchestrate_training.py` ejecutará toda la pipeline (adquisición, preprocesamiento, entrenamiento del modelo y evaluación) como un **Custom Job en Vertex AI Training**.
* **Trading en Vivo (`europe-southwest1`):** El script `scripts/run_live_trader.py` se desplegará como un servicio en **Cloud Run** para operar 24/7.
* Las imágenes Docker (`btcbot-cpu`, `btcbot-gpu`) ya están construidas y disponibles en Google Artifact Registry (asumiremos que están en un repositorio en `europe-southwest1` pero accesibles desde `europe-west4`).
* El proyecto utiliza servicios de GCP: GCS, Secret Manager y BigQuery.
* La configuración se gestiona a través de `src/config.yaml` y un archivo `.env` para desarrollo local (en GCP se pasarán como variables de entorno o secretos).

**Pasos de Despliegue (necesito que generes y "ejecutes" los comandos para esto):**

### 0. Configuración Inicial y Verificación de Prerrequisitos
Antes de empezar, necesito que me confirmes o me ayudes a obtener la siguiente información:

* `YOUR_GCP_PROJECT_ID`: Tu ID de Proyecto GCP.
* `YOUR_GCS_BUCKET_NAME`: El nombre de tu bucket de GCS. **Recomendación:** Para esta estrategia, este bucket debería estar idealmente ubicado en `europe-west4` (Países Bajos) para optimizar el acceso durante el entrenamiento.
* `YOUR_ARTIFACT_REGISTRY_REPO`: El nombre de tu repositorio de Artifact Registry (ej: `btcbotrepo`). Podemos asumir que está en `europe-southwest1` (Madrid), pero las imágenes serán accesibles desde `europe-west4`.
* `YOUR_BIGQUERY_DATASET_ID`: El ID de tu dataset en BigQuery para los logs (ej: `btcbot_logs`). Deberás decidir la ubicación de este dataset (ej. `europe-west4`, `europe-southwest1`, o multiregión `EU`).
* `DOCKER_IMAGE_CPU`: Ruta completa a tu imagen Docker CPU en Artifact Registry (ej: `europe-southwest1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/YOUR_ARTIFACT_REGISTRY_REPO/btcbot-cpu:latest`).
* `DOCKER_IMAGE_GPU`: Ruta completa a tu imagen Docker GPU en Artifact Registry (ej: `europe-southwest1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/YOUR_ARTIFACT_REGISTRY_REPO/btcbot-gpu:latest`).

### 1. Cuenta de Servicio (Service Account) y Permisos IAM
Crearemos una única Cuenta de Servicio (SA) de GCP que será utilizada tanto por Vertex AI Training (en `europe-west4`) como por Cloud Run (en `europe-southwest1`).

* **Comandos para crear una SA de GCP** (ej: `btcbot-compute-sa@${YOUR_GCP_PROJECT_ID}.iam.gserviceaccount.com`).
* **Comandos para asignar los roles IAM necesarios a esta SA de GCP:**
    * **Google Cloud Storage**: `roles/storage.objectAdmin` sobre el bucket específico (`gs://YOUR_GCS_BUCKET_NAME`).
    * **Secret Manager**: `roles/secretmanager.secretAccessor` (especificando los secretos de Binance o acceso general según tu política de seguridad).
    * **BigQuery**: `roles/bigquery.dataEditor` y `roles/bigquery.user` sobre el proyecto o el dataset específico (`YOUR_BIGQUERY_DATASET_ID`).
    * **Vertex AI**: `roles/aiplatform.customCodeServiceAgent` y `roles/aiplatform.user`.
    * **Cloud Run**: La SA que usará el servicio de Cloud Run necesita estos permisos. Adicionalmente, la identidad de servicio de Cloud Run podría necesitar `roles/cloudnat.user`.
    * **Artifact Registry**: `roles/artifactregistry.reader`.
    * **(Opcional) Service Account User**: `roles/iam.serviceAccountUser` si fuera necesario.

### 2. Vertex AI Custom Training Job para el Pipeline de Entrenamiento (en `europe-west4`)
Este job ejecutará el script `scripts/orchestrate_training.py`.

* **Comando `gcloud ai custom-jobs create` para enviar el job de entrenamiento.**
    * `display-name`: (ej: `btcbot-training-pipeline-ew4`).
    * `region`: **`europe-west4`** (o la región de Países Bajos/Bélgica con GPUs).
    * `worker-pool-spec`:
        * `machine-type`: Un tipo de máquina compatible con GPU en `europe-west4` (ej: `a2-highgpu-1g` para A100, o `n1-standard-4` con T4 si A100 no es necesaria/disponible).
        * `accelerator-type`: (ej: `NVIDIA_A100_40GB` o `NVIDIA_TESLA_T4`).
        * `accelerator-count`: 1.
        * `replica-count`: 1.
        * `executor-image-uri` (o `container-image-uri`): Tu `DOCKER_IMAGE_GPU`.
        * `args`: `["python", "scripts/orchestrate_training.py", "--timesteps", "100000"]`.
    * `service-account`: El email de la SA creada en el paso 1.
    * Variables de entorno para el job: `GCP_PROJECT_ID`, `GCS_BUCKET_NAME` (que idealmente está en `europe-west4`), `BIGQUERY_LOG_DATASET_ID`.

### 3. Configuración de Red para IP de Salida Estática (para Cloud Run en `europe-southwest1`)
Esto es crucial para que la API de Binance pueda añadir la IP del bot a su lista blanca.

* **Comandos para crear un conector de Serverless VPC Access en `europe-southwest1`.**
    * Necesitará una red VPC y una subred en `europe-southwest1`.
* **Comandos para reservar una dirección IP estática** (ej: `btcbot-cloud-run-static-ip`) en `europe-southwest1`.
* **Comandos para crear un Cloud Router** en `europe-southwest1` en la misma red que el conector VPC.
* **Comandos para configurar Cloud NAT** usando el Cloud Router y la IP estática reservada, aplicado a la subred del conector VPC en `europe-southwest1`.

### 4. Despliegue del Bot de Trading en Vivo en Cloud Run (en `europe-southwest1`)
Desplegaremos el script `scripts/run_live_trader.py`.

* **Comando `gcloud run deploy` para el servicio del bot en vivo** (ej: `btcbot-live-trader`).
    * `image`: Tu `DOCKER_IMAGE_CPU`.
    * `region`: **`europe-southwest1`** (Madrid).
    * `service-account`: El email de la SA creada en el paso 1.
    * `set-env-vars`:
        * `GCP_PROJECT_ID=YOUR_GCP_PROJECT_ID`
        * `GCS_BUCKET_NAME=YOUR_GCS_BUCKET_NAME` (accediendo al bucket que podría estar en `europe-west4`)
        * `BIGQUERY_LOG_DATASET_ID=YOUR_BIGQUERY_DATASET_ID` (accediendo al dataset donde esté ubicado)
        * `LIVE_TRADING_MODE=true`
        * `PYTHONUNBUFFERED=1`.
    * `set-secrets`: Para montar las claves API de Binance desde Secret Manager.
    * `vpc-connector`: El nombre del conector Serverless VPC Access creado en `europe-southwest1`.
    * `vpc-egress`: `all-traffic`.
    * `cpu` y `memory`: (ej: 1 CPU, 2Gi memoria).
    * `concurrency`: Probablemente bajo.
    * `port`: A discutir cómo manejar los health checks de Cloud Run si `run_live_trader.py` no expone un puerto HTTP.

### 5. Programación de Entrenamientos (Opcional)
Si deseas reentrenar el modelo periódicamente:

* Podemos usar **Cloud Scheduler** (configurado, por ejemplo, en `europe-west1`) para activar el Vertex AI Custom Training Job en `europe-west4`.

**Flujo de Interacción:**
1.  Me propones un comando `gcloud`.
2.  Yo conceptualizo su ejecución y te confirmo el resultado esperado o te pido aclaraciones. Si lo ejecuto realmente y hay errores, te los proporciono.
3.  Basándote en el resultado, me proporcionas el siguiente comando o una corrección.
4.  Continuamos este proceso.

**Lecciones Aprendidas y Consideraciones (Adaptadas a la Estrategia Multiregión):**
1.  **Disponibilidad de GPU para Vertex AI Training en `europe-west4`**: Esta región suele tener mejor disponibilidad de GPUs (T4, A100) que Madrid.
2.  **Ubicación del Bucket GCS**: Idealmente en la misma región que el entrenamiento (`europe-west4`) para minimizar latencia y costos de E/S de datos.
3.  **Ubicación del Dataset BigQuery**: Considerar dónde se harán más escrituras/consultas (entrenamiento en `europe-west4` vs. bot en vivo en `europe-southwest1`) o usar multiregión `EU`.
4.  **Parámetros del Orquestador**: El script `scripts/orchestrate_training.py --timesteps 100000`.
5.  **Health Checks en Cloud Run**: Adaptar o encontrar una solución para `run_live_trader.py`.

Comencemos con la **información inicial del paso 0** y luego procederemos con la **creación de la Cuenta de Servicio y la asignación de permisos IAM (Paso 1)**.