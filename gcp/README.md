# Migración a Google Cloud Platform - BTCBot

Este directorio contiene los scripts necesarios para implementar el proyecto BTCBot en Google Cloud Platform (GCP), siguiendo las mejores prácticas de MLOps.

## Contenido del Directorio

- `common/` - Módulos comunes para los scripts
  - `config.py` - Configuración centralizada
  - `clients.py` - Clientes para servicios de GCP

- `01_setup_secrets.py` - Configura secretos en Secret Manager
- `02_setup_storage.py` - Crea buckets en Cloud Storage con versionado
- `03_setup_iam.py` - Configura IAM y cuentas de servicio
- `04_build_docker_image.py` - Construye y sube imágenes Docker a Artifact Registry
- `05_deploy_data_acquisition.py` - Despliega el servicio de adquisición de datos
- `06_launch_training_job.py` - Lanza un trabajo de entrenamiento en Vertex AI
- `07_create_training_pipeline.py` - Crea un pipeline de entrenamiento en Vertex AI
- `08_evaluate_model.py` - Evalúa un modelo desde Vertex AI Model Registry
- `09_deploy_model.py` - Despliega un modelo a un endpoint de Vertex AI

## Proceso de Migración

### 1. Configuración Inicial

Ejecuta los siguientes comandos para configurar la infraestructura base:

```bash
# Configurar secretos (reemplaza con tus propias claves API)
python gcp/01_setup_secrets.py --api_key YOUR_BINANCE_API_KEY --api_secret YOUR_BINANCE_API_SECRET

# Configurar buckets de almacenamiento
python gcp/02_setup_storage.py

# Configurar IAM y cuentas de servicio
python gcp/03_setup_iam.py
```

### 2. Construcción de Imágenes Docker

Construye las imágenes Docker necesarias:

```bash
# Construir imagen para entrenamiento
python gcp/04_build_docker_image.py --tag latest
```

### 3. Despliegue del Servicio de Adquisición de Datos

Despliega el servicio que descargará datos automáticamente:

```bash
python gcp/05_deploy_data_acquisition.py
```

### 4. Entrenamiento del Modelo

Tienes dos opciones para entrenar el modelo:

#### Opción 1: Trabajo de Entrenamiento Individual

```bash
python gcp/06_launch_training_job.py --machine_type n1-standard-8
```

#### Opción 2: Pipeline de Entrenamiento Completo

```bash
python gcp/07_create_training_pipeline.py --raw-data-bucket btcbot-raw-data-btcbot276299 --symbol BTCUSDT --timeframe 1h
```

### 5. Evaluación del Modelo

Una vez entrenado, puedes evaluar el modelo:

```bash
# Obtén el ID del modelo de la salida del trabajo de entrenamiento
MODEL_ID=projects/btcbot276299/locations/europe-southwest1/models/XXXXXX

# Evalúa el modelo con datos históricos
python gcp/08_evaluate_model.py --model_id $MODEL_ID --data_path /path/to/processed/data.npz
```

### 6. Despliegue del Modelo

Finalmente, puedes desplegar el modelo a un endpoint:

```bash
python gcp/09_deploy_model.py --model_id $MODEL_ID
```

## Notas Importantes

- Todos los scripts utilizan la configuración centralizada en `common/config.py`.
- Los buckets de Cloud Storage tienen el versionado habilitado para garantizar la trazabilidad.
- Los modelos se registran en Vertex AI Model Registry para un versionado formal.
- Los trabajos de entrenamiento y evaluación usan contenedores Docker personalizados.
- Asegúrate de tener permisos suficientes en tu proyecto GCP para ejecutar estos scripts.

## Monitorización y Logging

- Los logs de los servicios desplegados están disponibles en Cloud Logging.
- Las métricas del modelo se pueden visualizar en Vertex AI Experiments.
- Los resultados de evaluación se guardan en el bucket `btcbot-evaluation-results-btcbot276299`.

## Limpieza de Recursos

Para eliminar los recursos creados por estos scripts y evitar cargos innecesarios:

```bash
# Eliminar todos los recursos de GCP asociados con el proyecto BTCBot
# Esto incluye:
# - Endpoints de Vertex AI
# - Modelos desplegados
# - Jobs de entrenamiento
# - Servicios de Cloud Run
# - Buckets de almacenamiento
# - Secretos de Secret Manager
# - Cuentas de servicio personalizadas
# - Imágenes de Docker en Artifact Registry

python gcp/10_cleanup_resources.py --force
```

> **⚠️ ADVERTENCIA**: Este comando eliminará permanentemente todos los recursos relacionados con BTCBot en GCP. Úsalo con precaución y asegúrate de haber respaldado cualquier dato importante antes de ejecutarlo.

También puedes eliminar recursos específicos según sea necesario:

```bash
# Eliminar solo los endpoints de modelo
python gcp/10_cleanup_resources.py --resources endpoints

# Eliminar solo los buckets de almacenamiento
python gcp/10_cleanup_resources.py --resources storage

# Eliminar solo las imágenes de Docker
python gcp/10_cleanup_resources.py --resources docker_images
```

Para más información sobre las opciones disponibles:

```bash
python gcp/10_cleanup_resources.py --help
```
