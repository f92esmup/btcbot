# Dependencias para la migración a GCP

Para poder utilizar la funcionalidad de migración a Google Cloud Platform, necesitas instalar las siguientes dependencias:

```bash
pip install google-cloud-storage google-cloud-aiplatform google-cloud-secretmanager kfp
```

Estas bibliotecas son necesarias para los siguientes componentes:
- `google-cloud-storage`: Para interactuar con buckets de Cloud Storage
- `google-cloud-aiplatform`: Para utilizar Vertex AI (entrenamiento, modelado, etc.)
- `google-cloud-secretmanager`: Para gestionar secretos como las API keys
- `kfp`: Para crear y ejecutar pipelines de Vertex AI

## Requisitos adicionales

- Tener una cuenta de Google Cloud Platform con facturación habilitada
- Tener el SDK de Google Cloud (`gcloud`) instalado y configurado
- Tener habilitadas las APIs necesarias en tu proyecto:
  - Vertex AI API
  - Secret Manager API
  - Cloud Storage API
  - Cloud Functions API (opcional)
  - Cloud Run API (opcional)

Para habilitar las APIs desde la línea de comandos:
```bash
gcloud services enable aiplatform.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com \
  cloudfunctions.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com
```
