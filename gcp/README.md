# Migración a Google Cloud Platform - BTCBot

Este directorio contiene los scripts necesarios para implementar el proyecto BTCBot en Google Cloud Platform (GCP) utilizando Kubeflow Pipelines (KFP) en Vertex AI, siguiendo las mejores prácticas de MLOps.

## Contenido del Directorio

- `common/` - Módulos comunes para los scripts
  - `config.py` - Configuración centralizada basada en variables de entorno
  - `clients.py` - Clientes para servicios de GCP

- `serving/` - Archivos para despliegue en producción
  - `server.py` - Servidor Flask para servir predicciones
  - `start.sh` - Script de inicio para el contenedor Docker

- Scripts de configuración:
  - `01_setup_secrets.py` - Configura secretos en Secret Manager
  - `02_setup_storage.py` - Crea buckets en Cloud Storage con versionado
  - `03_setup_iam.py` - Configura IAM y cuentas de servicio
  - `04_build_docker_image.py` - Construye y sube imágenes Docker a Artifact Registry

- Pipeline y despliegue:
  - `05_create_training_pipeline.py` - Define y ejecuta el pipeline de Kubeflow en Vertex AI
  - `06_deploy_model.py` - Despliega un modelo a un endpoint de Vertex AI
  - `07_cleanup_resources.py` - Limpia recursos de GCP

- Scripts utilitarios:
  - `deploy.sh` - Script para construir y desplegar imágenes Docker
  - `enable_apis.sh` - Habilita las APIs necesarias de GCP
  - `run_pipeline.sh` - Ejecuta el pipeline de entrenamiento con parámetros
  - `test_e2e_pipeline.sh` - Prueba end-to-end del pipeline completo

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

### 3. Entrenamiento y Evaluación con Kubeflow Pipelines

El proyecto ahora utiliza Kubeflow Pipelines (KFP) en Vertex AI para orquestar el proceso completo de entrenamiento, evaluación y despliegue:

```bash
# Método 1: Usar el script de conveniencia
./run_pipeline.sh --symbol BTCUSDT --timeframe 1h --lookback-window 96 \
  --total-timesteps 500000 --algorithm SAC --deploy --wait

# Método 2: Ejecutar directamente el script Python
python 05_create_training_pipeline.py \
    --symbol BTCUSDT \
    --timeframe 1h \
    --lookback_window 96 \
    --total_timesteps 500000 \
    --algorithm SAC \
    --deploy_model \
    --wait_for_completion
```

El pipeline completo incluye los siguientes componentes:

1. **Descarga de datos históricos** desde Binance
2. **Preprocesamiento de datos** con normalización y generación de features
3. **Entrenamiento del agente RL** utilizando algoritmos configurables
4. **Evaluación del modelo** con métricas de trading (Sharpe, Sortino, Drawdown, Win Rate)
5. **Despliegue condicional** si el modelo supera los umbrales de calidad

Para ejecutar una prueba rápida de toda la infraestructura:

```bash
./test_e2e_pipeline.sh
```

### 4. Despliegue Manual de Modelo (Opcional)

Si deseas desplegar manualmente un modelo específico:

```bash
# Obtén el ID del modelo de la salida del pipeline o desde la consola de Vertex AI
MODEL_ID=projects/btcbot276299/locations/europe-southwest1/models/XXXXXX

# Despliegue manual de un modelo específico
python 06_deploy_model.py --model_id $MODEL_ID
```

### 5. Limpieza de Recursos

Para eliminar los recursos creados y evitar cargos innecesarios:

```bash
# Eliminar todos los recursos de GCP asociados con el proyecto
python 07_cleanup_resources.py --force
```

## Configuración Centralizada y Uso de Variables de Entorno

Este proyecto está diseñado para funcionar con configuración centralizada y basada en variables de entorno siguiendo las mejores prácticas de MLOps:

1. **Prioridad de configuración**:
   - Variables de entorno (.env o exportadas en sesión)
   - Valores por defecto en `common/config.py`

2. **Centralización del acceso a config**:
   - Los scripts en `gcp/` usan `common.config` para acceder a valores como `PROJECT_ID`, `REGION`, etc.
   - Los componentes del pipeline acceden a la configuración a través de parámetros explícitos

3. **Configuración específica del agente**:
   - Parámetros como `learning_rate`, `buffer_size`, etc. se pasan explícitamente entre componentes

## Monitorización y Logging

- **Kubeflow Pipelines UI**: Visualización gráfica del progreso y resultados del pipeline
- **Vertex AI Model Registry**: Gestión de versiones de modelos y metadatos asociados
- **Cloud Logging**: Logs centralizados de todos los componentes
- **Artifact Storage**: Almacenamiento de artefactos (datos, modelos) en Cloud Storage
- **Evaluación automatizada**: Métricas de trading (Sharpe, Sortino, drawdown, win rate) para cada modelo

Para acceder a la interfaz de Kubeflow Pipelines:
1. Ve a Google Cloud Console > Vertex AI > Pipelines
2. Selecciona la ejecución del pipeline para ver su progreso y resultados
3. Cada componente del pipeline tiene sus propios logs y artefactos asociados

## Consideraciones de Seguridad

- **Gestión de secretos**: Credenciales de API de Binance almacenadas en Secret Manager
- **Principio de mínimo privilegio**: Cuentas de servicio con permisos específicos
- **Aislamiento**: Cada componente del pipeline opera en un entorno aislado
- **Configuración segura**: Valores sensibles nunca deben incluirse directamente en el código

## Arquitectura de Integración

![Arquitectura GCP](/docs/images/gcp_architecture.png)

La arquitectura utiliza principalmente:
- **Vertex AI Pipelines**: Para orquestación del flujo de trabajo
- **Cloud Storage**: Para almacenamiento de datos y modelos
- **Secret Manager**: Para gestión segura de credenciales
- **Container Registry**: Para imágenes Docker
- **Vertex AI Endpoints**: Para servir modelos en producción

## Nota Importante

> **⚠️ ADVERTENCIA**: El comando de limpieza eliminará permanentemente todos los recursos relacionados con BTCBot en GCP. Úsalo con precaución y asegúrate de haber respaldado cualquier dato importante antes de ejecutarlo.

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
