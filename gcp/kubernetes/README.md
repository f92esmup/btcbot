# Kubernetes - Configuración para BTCBot

Este directorio contiene los archivos de configuración y scripts para desplegar BTCBot en Google Kubernetes Engine (GKE). Incluye configuraciones tanto para el servidor de inferencia como para la aplicación de trading en vivo.

## Archivos Disponibles

- `btcbot-deployment.template.yaml`: Plantilla para el despliegue del trader con variables que serán sustituidas
- `btcbot-inference-deployment.yaml`: Configuración para desplegar el servidor de inferencia
- `btcbot-inference-service.yaml`: Servicio interno que expone el servidor de inferencia
- `btcbot-live-trader-deployment.yaml`: Configuración para desplegar el trader
- `deploy_to_k8s.sh`: Script para facilitar el despliegue manual a Kubernetes

## Requisitos Previos

Para utilizar estas configuraciones, necesitas:

1. Un cluster de GKE creado y configurado
2. La herramienta `kubectl` instalada y configurada para acceder a tu cluster
3. La imagen Docker de BTCBot publicada en Artifact Registry
4. Un bucket de GCS con el modelo RL entrenado
5. (Opcional) Secretos configurados en Google Secret Manager para las API keys de Binance

## Componentes Principales

### 1. Servidor de Inferencia

El servidor de inferencia es un componente que:
- Carga un modelo de RL desde GCS
- Expone un endpoint HTTP para hacer predicciones
- Responde a healthchecks para verificar su estado

**Archivos relevantes**:
- `btcbot-inference-deployment.yaml`
- `btcbot-inference-service.yaml`

### 2. Trader en Vivo

La aplicación de trading en vivo que:
- Se conecta a la API de Binance (testnet o real)
- Obtiene datos de mercado en tiempo real
- Envía datos al servidor de inferencia para obtener predicciones
- Ejecuta operaciones de trading basadas en esas predicciones

**Archivos relevantes**:
- `btcbot-deployment.template.yaml`
- `btcbot-live-trader-deployment.yaml`

## Configuración de Workload Identity

Para que los pods puedan acceder a recursos de GCP (como GCS o Secret Manager), utilizamos Workload Identity:

```bash
# 1. Crear una Kubernetes Service Account (KSA)
kubectl create namespace btcbot
kubectl create serviceaccount btcbot-inference-sa -n btcbot

# 2. Crear una IAM Service Account (GSA) que tenga los permisos necesarios
gcloud iam service-accounts create btcbot-gcs-sa --project=lofty-complex-460416-r6

# 3. Otorgar permisos a la GSA para acceder a GCS
gcloud projects add-iam-policy-binding lofty-complex-460416-r6 \
  --member="serviceAccount:btcbot-gcs-sa@lofty-complex-460416-r6.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"

# 4. Vincular la KSA con la GSA
gcloud iam service-accounts add-iam-policy-binding btcbot-gcs-sa@lofty-complex-460416-r6.iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:lofty-complex-460416-r6.svc.id.goog[btcbot/btcbot-inference-sa]"

# 5. Anotar la KSA con la GSA
kubectl annotate serviceaccount btcbot-inference-sa \
  iam.gke.io/gcp-service-account=btcbot-gcs-sa@lofty-complex-460416-r6.iam.gserviceaccount.com \
  -n btcbot
```

## Cómo Desplegar Manualmente

### Utilizando el Script

El script `deploy_to_k8s.sh` facilita el despliegue manual:

```bash
# Ejecutar con valores por defecto
./gcp/kubernetes/deploy_to_k8s.sh

# Con parámetros personalizados
./gcp/kubernetes/deploy_to_k8s.sh --project=mi-proyecto --bucket=mi-bucket --region=europe-west1 --mode=TESTNET
```

### Despliegue Paso a Paso

Alternativamente, puedes realizar el despliegue paso a paso:

1. **Conexión al Clúster GKE**:
   ```bash
   gcloud container clusters get-credentials btcbot-cluster --region europe-southwest1 --project lofty-complex-460416-r6
   ```

2. **Crear el Namespace (si no existe)**:
   ```bash
   kubectl create namespace btcbot
   ```

3. **Desplegar el Servidor de Inferencia**:
   ```bash
   kubectl apply -f gcp/kubernetes/btcbot-inference-deployment.yaml
   ```

4. **Desplegar el Servicio Interno**:
   ```bash
   kubectl apply -f gcp/kubernetes/btcbot-inference-service.yaml
   ```

5. **Desplegar el Trader**:
   ```bash
   kubectl apply -f gcp/kubernetes/btcbot-live-trader-deployment.yaml
   ```

## Verificación del Despliegue

Para verificar que todo está funcionando correctamente:

```bash
# Verificar los deployments
kubectl get deployments -n btcbot

# Verificar los pods
kubectl get pods -n btcbot

# Verificar el servicio interno
kubectl get services -n btcbot

# Ver logs del servidor de inferencia
kubectl logs -f -l app=btcbot-inference -n btcbot

# Ver logs del trader
kubectl logs -f -l app=btcbot-trader -n btcbot
```

## Prueba del Servicio Interno

Para verificar que el servicio de inferencia está respondiendo correctamente:

```bash
# Crear un pod temporal con curl para probar el servicio
kubectl run curl-test --image=curlimages/curl -n btcbot -i --tty --rm -- sh -c 'curl http://btcbot-inference-service:8080/health'
```

Deberías recibir una respuesta como: `{"status":"healthy"}`

## Personalización para Otro Proyecto

Para adaptar estas configuraciones a otro proyecto:

1. **Modificar las variables globales** en todos los archivos:
   - GCP_PROJECT_ID
   - GCS_BUCKET_NAME
   - GCP_REGION
   - URI de la imagen Docker
   - Rutas de los modelos en GCS

2. **Ajustar la configuración de recursos** según tus necesidades:
   - CPU y memoria para cada pod
   - Número de réplicas

3. **Actualizar las referencias a secretos** si usas distintas configuraciones en Secret Manager

## Referencia de Variables

Las principales variables usadas en los archivos de despliegue son:

| Variable | Descripción | Valor de ejemplo |
|----------|-------------|------------------|
| GCP_PROJECT_ID | ID del proyecto GCP | lofty-complex-460416-r6 |
| GCS_BUCKET_NAME | Nombre del bucket GCS | lofty-complex-460416-r6 |
| GCP_REGION | Región de GCP | europe-southwest1 |
| DOCKER_IMAGE | URI de la imagen Docker | europe-southwest1-docker.pkg.dev/lofty-complex-460416-r6/lofty-complex-460416-r6-repo/btcbot-cpu:latest |
| MODEL_PATH_GCS | Ruta al modelo en GCS | models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000_steps.zip |
| LIVE_TRADING_MODE | Modo de trading | TESTNET o REAL |

## Solución de Problemas

Si encuentras problemas con el despliegue:

1. **Problemas de acceso a GCS**:
   - Verifica la configuración de Workload Identity
   - Comprueba que la ruta del modelo sea correcta
   - Asegúrate de que el bucket existe y es accesible

2. **CrashLoopBackOff en los pods**:
   - Examina los logs con `kubectl logs`
   - Verifica que las variables de entorno sean correctas
   - Comprueba que la imagen Docker sea válida

3. **Problemas de comunicación entre servicios**:
   - Verifica que el servicio interno esté creado y tenga la IP correcta
   - Comprueba que los selectores del servicio coincidan con las etiquetas de los pods
   - Prueba la comunicación con el pod temporal curl

4. **Problemas de recursos**:
   - Verifica las cuotas y límites de tu cluster GKE
   - Ajusta las solicitudes de recursos si son demasiado altas
