# Google Cloud Build - Pipelines de CI/CD para BTCBot

Este directorio contiene los archivos de configuración para los pipelines de CI/CD de Google Cloud Build para el proyecto BTCBot. Estos pipelines permiten automatizar el proceso de construcción de imágenes Docker y despliegue en Google Kubernetes Engine (GKE).

## Archivos Disponibles

- `cloudbuild_cpu.yaml`: Pipeline para construir y publicar la imagen Docker para CPU.
- `cloudbuild_full.yaml`: Pipeline completo que construye la imagen Docker y despliega tanto el servidor de inferencia como la aplicación de trading en GKE.
- `cloudbuild.yaml`: Pipeline original (mantiene compatibilidad).

## Requisitos Previos

Para utilizar estos pipelines, necesitas:

1. Una cuenta de Google Cloud con los siguientes servicios habilitados:
   - Google Cloud Build
   - Google Container Registry o Artifact Registry
   - Google Kubernetes Engine (GKE)
   - Google Cloud Storage

2. Un repositorio en Artifact Registry creado previamente
3. Un cluster de GKE creado previamente
4. Las APIs necesarias habilitadas en tu proyecto GCP:
   - cloudbuild.googleapis.com
   - containerregistry.googleapis.com
   - artifactregistry.googleapis.com
   - container.googleapis.com

5. Un bucket de GCS para los modelos (y opcionalmente otro para los logs de Cloud Build)

6. Permisos adecuados para el service account de Cloud Build:
   - roles/storage.admin
   - roles/container.developer
   - roles/artifactregistry.admin

## Variables de Configuración

Los archivos de Cloud Build utilizan variables de sustitución para personalizar el despliegue. Las principales variables son:

| Variable | Descripción | Valor actual |
|----------|-------------|--------------|
| `_ARTIFACT_REGISTRY` | URL del registry donde se publicarán las imágenes | `europe-southwest1-docker.pkg.dev` |
| `_REPOSITORY_NAME` | Nombre del repositorio en Artifact Registry | `lofty-complex-460416-r6-repo` |
| `_GKE_CLUSTER_NAME` | Nombre del cluster GKE | `btcbot-cluster` |
| `_GCP_REGION` | Región de GCP | `europe-southwest1` |
| `_GCS_BUCKET_NAME` | Nombre del bucket GCS | `lofty-complex-460416-r6` |
| `_LIVE_TRADING_MODE` | Modo de trading (TESTNET o REAL) | `TESTNET` |
| `_VERSION` | Versión de la imagen | `1.0.0` |
| `_NAMESPACE` | Namespace de Kubernetes donde se desplegará la aplicación | `btcbot` |

## Cómo Usar los Pipelines

### Construcción de Imagen CPU

Para construir solo la imagen Docker para CPU:

```bash
gcloud builds submit --config=gcp/cloudbuild/cloudbuild_cpu.yaml .
```

### Despliegue Completo

Para ejecutar el pipeline completo que construye la imagen y despliega la aplicación en GKE:

```bash
gcloud builds submit --config=gcp/cloudbuild/cloudbuild_full.yaml .
```

Con parámetros personalizados:

```bash
gcloud builds submit --config=gcp/cloudbuild/cloudbuild_full.yaml \
  --substitutions=_GCP_REGION="europe-southwest1",_GCS_BUCKET_NAME="mi-bucket",_LIVE_TRADING_MODE="TESTNET",_VERSION="1.1.0" .
```

## Flujo de Trabajo

El pipeline completo realiza las siguientes acciones en orden:

1. **Construcción de la imagen CPU**:
   - Utiliza el `Dockerfile.cpu` para construir la imagen
   - Etiqueta la imagen con la versión especificada y con "latest"

2. **Publicación en Artifact Registry**:
   - Publica la imagen etiquetada
   - Publica la imagen "latest"

3. **Preparación de archivos de Kubernetes**:
   - Genera el archivo de despliegue para el servidor de inferencia
   - Genera el archivo de despliegue para el trader
   - Reemplaza las variables en los templates con los valores configurados

4. **Conexión al cluster GKE**:
   - Obtiene las credenciales del cluster GKE

5. **Despliegue en GKE**:
   - Crea o actualiza el namespace
   - Crea o configura la cuenta de servicio de Kubernetes
   - Configura Workload Identity para acceso a GCS
   - Despliega el servidor de inferencia
   - Despliega el servicio interno del servidor
   - Despliega la aplicación de trading

6. **Verificación**:
   - Verifica que los despliegues se completen correctamente
   - Muestra información sobre los pods en ejecución

## Personalización para Otro Proyecto

Si deseas adaptar estos pipelines para otro proyecto, necesitarás modificar:

1. **Variables de sustitución** en los archivos de Cloud Build:
   - `_ARTIFACT_REGISTRY`: URL del registry (dependiendo de tu región)
   - `_REPOSITORY_NAME`: Nombre de tu repositorio de imágenes
   - `_GKE_CLUSTER_NAME`: Nombre de tu cluster GKE
   - `_GCP_REGION`: Tu región preferida de GCP
   - `_GCS_BUCKET_NAME`: Nombre de tu bucket para modelos y datos
   - `_PROJECT_ID`: (opcional) ID de tu proyecto GCP
   - `_NAMESPACE`: Namespace de Kubernetes para tu aplicación

2. **Rutas y nombres** en los archivos de despliegue:
   - Modifica las rutas de los modelos RL
   - Ajusta los nombres de los deployments si es necesario
   - Actualiza las referencias a los secretos según tus configuraciones

## Consideraciones de Seguridad

- Las credenciales de Binance se gestionan a través de Secret Manager
- El servicio de inferencia está configurado como interno (ClusterIP)
- Se utilizan liveness y readiness probes para verificar la salud de los servicios
- Se configuran límites de recursos para cada pod

## Solución de Problemas

Si encuentras problemas durante el despliegue:

1. Verifica los logs de Cloud Build en la consola de GCP
2. Comprueba el estado de los pods con `kubectl get pods -n btcbot`
3. Examina los logs de los contenedores con `kubectl logs -f [nombre-del-pod] -n btcbot`
4. Verifica la configuración de Workload Identity para acceso a GCS
