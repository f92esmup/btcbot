# Dependencias para la migración a GCP

Para poder utilizar la funcionalidad de migración a Google Cloud Platform, necesitas instalar las siguientes dependencias:

```bash
pip install google-cloud-storage google-cloud-aiplatform google-cloud-secretmanager google-cloud-artifact-registry google-api-python-client kfp
```

Estas bibliotecas son necesarias para los siguientes componentes:
- `google-cloud-storage`: Para interactuar con buckets de Cloud Storage
- `google-cloud-aiplatform`: Para utilizar Vertex AI (entrenamiento, modelado, etc.)
- `google-cloud-secretmanager`: Para gestionar secretos como las API keys
- `google-cloud-artifact-registry`: Para manejar imágenes Docker en Artifact Registry
- `google-api-python-client`: Para interactuar con varias APIs de Google (IAM, Cloud Run)
- `kfp`: Para crear y ejecutar pipelines de Kubeflow en Vertex AI

## Dependencias de Kubeflow Pipelines

El sistema utiliza Kubeflow Pipelines (KFP) para la orquestación de flujos de trabajo de ML. Los componentes específicos incluyen:

```bash
pip install kfp==2.7.0 google-cloud-aiplatform==1.38.1
```

KFP permite:
- Definir componentes modulares para cada paso del pipeline
- Conectar componentes en un grafo acíclico dirigido (DAG)
- Parametrizar la ejecución del pipeline
- Definir condiciones para la ejecución de ciertos componentes
- Pasar artefactos y datos entre componentes
- Visualizar el progreso y los resultados

## Imagen Docker para Componentes

Los componentes del pipeline se ejecutan en contenedores Docker. La imagen base incluye:

```dockerfile
# Dependencias básicas
python-binance==1.0.19
google-cloud-storage==2.14.0
google-cloud-secretmanager==2.19.0
google-cloud-aiplatform==1.38.1
fsspec==2023.12.2
gcsfs==2023.12.2

# Dependencias para ML/RL
pandas==2.0.3
pandas-ta==0.3.14b0
stable-baselines3==2.2.1
gymnasium==0.29.1
matplotlib==3.7.4
torch==2.1.2
tensorboard==2.16.2

# Dependencias para pipelines
kfp==2.7.0
```

## Requisitos adicionales

- Tener una cuenta de Google Cloud Platform con facturación habilitada
- Tener el SDK de Google Cloud (`gcloud`) instalado y configurado
- Tener habilitadas las APIs necesarias en tu proyecto:
  - Vertex AI API
  - Secret Manager API
  - Cloud Storage API
  - Artifact Registry API
  - IAM API
  - Resource Manager API

Para habilitar las APIs desde la línea de comandos:
```bash
# El script enable_apis.sh hace esto automáticamente
gcloud services enable aiplatform.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com
```

## Permisos IAM necesarios

Para ejecutar el pipeline completo, se necesitan los siguientes roles:

- `roles/aiplatform.user` - Para crear y ejecutar pipelines
- `roles/storage.admin` - Para gestionar buckets y objetos
- `roles/secretmanager.secretAccessor` - Para acceder a secretos
- `roles/artifactregistry.admin` - Para gestionar imágenes Docker

La cuenta de servicio creada por `03_setup_iam.py` tiene estos permisos correctamente configurados.
