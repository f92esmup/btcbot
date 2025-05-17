# Migración a Google Cloud Platform - BTCBot

Este directorio contiene los scripts necesarios para implementar el proyecto BTCBot en Google Cloud Platform (GCP), siguiendo las mejores prácticas de MLOps.

## Contenido del Directorio

- `common/` - Módulos comunes para los scripts
  - `config.py` - Configuración centralizada basada en variables de entorno
  - `clients.py` - Clientes para servicios de GCP

- `01_setup_secrets.py` - Configura secretos en Secret Manager
- `02_setup_storage.py` - Crea buckets en Cloud Storage con versionado
- `03_setup_iam.py` - Configura IAM y cuentas de servicio
- `04_build_docker_image.py` - Construye y sube imágenes Docker a Artifact Registry
- `06_create_training_pipeline.py` - Crea un pipeline de entrenamiento en Vertex AI (incluye adquisición de datos)
- `07_deploy_model.py` - Despliega un modelo a un endpoint de Vertex AI
- `08_cleanup_resources.py` - Limpia recursos de GCP (cuando sea necesario)

## Proceso de Migración

### 0. Configuración de Variables de Entorno

Este proyecto está diseñado para trabajar con variables de entorno. Copia el archivo `.env.example` a `.env` y personaliza las variables:

```bash
cp .env.example .env
# Editar .env con tus propios valores
```

Luego, carga las variables de entorno en tu sesión:

```bash
source .env
```

### 1. Configuración Inicial

Ejecuta los siguientes comandos para configurar la infraestructura base:

```bash
# Habilitar APIs necesarias de GCP
./gcp/enable_apis.sh

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
# Opcionalmente, añade --gpu para una imagen compatible con GPU
python gcp/04_build_docker_image.py --tag latest --gpu
```

### 3. Entrenamiento y Evaluación del Modelo con Pipeline Unificado

Utilizamos un pipeline de Vertex AI para orquestar todo el proceso (incluyendo la adquisición de datos):

```bash
# Variables de entorno para configurar el entrenamiento (opcional)
export AGENT_LEARNING_RATE=0.0005
export PIPELINE_TOTAL_TIMESTEPS=500000

# Ejecutar el pipeline con CPU
python gcp/06_create_training_pipeline.py \
    --symbol BTCUSDT \
    --timeframe 1h \
    --total-timesteps 500000 \
    --deploy-model \
    --min-sharpe-ratio 0.6

# Opcional: Ejecutar con GPU
python gcp/06_create_training_pipeline.py \
    --symbol BTCUSDT \
    --timeframe 1h \
    --total-timesteps 500000 \
    --use-gpu \
    --gpu-type NVIDIA_TESLA_T4 \
    --gpu-count 1 \
    --deploy-model \
    --min-sharpe-ratio 0.6
```

### 5. Despliegue Manual de Modelo (Opcional)

Si deseas desplegar manualmente un modelo específico:

```bash
# Obtén el ID del modelo de la salida del pipeline o desde la consola de Vertex AI
MODEL_ID=projects/btcbot276299/locations/europe-southwest1/models/XXXXXX

# Despliegue manual de un modelo específico
python gcp/07_deploy_model.py --model_id $MODEL_ID
```

### 6. Limpieza de Recursos (cuando sea necesario)

Para eliminar los recursos creados y evitar cargos innecesarios:

```bash
# Eliminar todos los recursos de GCP asociados con el proyecto
python gcp/08_cleanup_resources.py --force
```

## Configuración Centralizada y Uso de Variables de Entorno

Este proyecto está diseñado para funcionar con configuración centralizada y basada en variables de entorno siguiendo las mejores prácticas de MLOps:

1. **Prioridad de configuración**:
   - Variables de entorno (.env o exportadas en sesión)
   - Valores por defecto en `common/config.py`

2. **Centralización del acceso a config**:
   - Los scripts en `gcp/` usan `common.config` para acceder a valores como `PROJECT_ID`, `REGION`, etc.
   - El código del bot en `src/` usa `ConfigManager` que ahora prioriza variables de entorno

3. **Configuración específica del agente**:
   - Variables como `AGENT_LEARNING_RATE`, `AGENT_BUFFER_SIZE` se pueden configurar vía env vars

## Notas Importantes

- La configuración centralizada permite desplegar en múltiples entornos GCP cambiando únicamente las variables de entorno.
- Los buckets de Cloud Storage tienen el versionado habilitado para garantizar la trazabilidad.
- Los modelos se registran en Vertex AI Model Registry para un versionado formal.
- Los pipelines automatizan todo el proceso de MLOps desde el preprocesamiento hasta el despliegue condicional.

## Monitorización y Logging

- Los logs de los servicios desplegados están disponibles en Cloud Logging.
- Los pipeline de entrenamiento pueden verse en Vertex AI Pipelines.
- Las métricas del modelo se pueden visualizar en Vertex AI Experiments.
- Los resultados de evaluación se guardan en el bucket configurado en `EVALUATION_RESULTS_BUCKET`.

## Consideraciones de Seguridad

- Las credenciales de API de Binance se almacenan en Secret Manager.
- El acceso a secretos y recursos se gestiona mediante una cuenta de servicio con permisos específicos.
- La configuración sensible como claves API nunca debe incluirse directamente en el código fuente.
```

> **⚠️ ADVERTENCIA**: Este comando eliminará permanentemente todos los recursos relacionados con BTCBot en GCP. Úsalo con precaución y asegúrate de haber respaldado cualquier dato importante antes de ejecutarlo.

También puedes eliminar recursos específicos según sea necesario:

```bash
# Eliminar solo los endpoints de modelo
python gcp/08_cleanup_resources.py --resources endpoints

# Eliminar solo los buckets de almacenamiento
python gcp/08_cleanup_resources.py --resources storage

# Eliminar solo las imágenes de Docker
python gcp/08_cleanup_resources.py --resources docker_images
```

Para más información sobre las opciones disponibles:

```bash
python gcp/08_cleanup_resources.py --help
```
