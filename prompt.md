# Guía para desplegar un servicio de inferencia ML en GKE con comunicación interna

## Objetivo:
Desplegar un servicio de inferencia de Machine Learning en Google Kubernetes Engine (GKE) que cargue un modelo desde Google Cloud Storage (GCS) y exponga un endpoint interno para predicciones. Luego, configurar una aplicación de trading existente para que se comunique con este servicio.

## Prerrequisitos:
* Cuenta de Google Cloud y proyecto configurado
* `gcloud` CLI instalado y configurado
* `kubectl` instalado

## Variables del proyecto:
```bash
# Project configuration
export GCP_PROJECT_ID="lofty-complex-460416-r6"
export GCP_REGION="europe-southwest1"
export GCS_BUCKET_NAME="lofty-complex-460416-r6"
export CLUSTER_NAME="btcbot-cluster"

# Docker image and model path
export DOCKER_IMAGE="europe-southwest1-docker.pkg.dev/lofty-complex-460416-r6/lofty-complex-460416-r6-repo/btcbot-cpu:latest"
export MODEL_PATH="lofty-complex-460416-r6/models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000_steps.zip"

# Service accounts
export GCP_SA_NAME="btcbot-gcs-reader"
export K8S_SA_NAME="btcbot-inference-ksa"

# Binance API keys (for the live trader)
# Nota: Estos valores deben ser reemplazados con tus propias claves API
export TESTNET_BINANCE_API_KEY="tu_api_key_aqui"
export TESTNET_BINANCE_API_SECRET="tu_api_secret_aqui"
```

## Paso 1: Crear un nuevo clúster GKE Autopilot
```bash
gcloud container clusters create-auto ${CLUSTER_NAME} \
  --region=${GCP_REGION} \
  --project=${GCP_PROJECT_ID}
```

Verifica la creación del clúster:
```bash
gcloud container clusters list --project=${GCP_PROJECT_ID}
```

Configura `kubectl` para usar el nuevo clúster:
```bash
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --region=${GCP_REGION} \
  --project=${GCP_PROJECT_ID}
```

Verifica la conexión:
```bash
kubectl cluster-info
```

## Paso 2: Configurar permisos de acceso a GCS y Secret Manager

### Crear cuenta de servicio de GCP para acceso a GCS
```bash
# Crear una cuenta de servicio de GCP
gcloud iam service-accounts create ${GCP_SA_NAME} \
  --project=${GCP_PROJECT_ID} \
  --description="Service account for GCS access" \
  --display-name="BTC Bot GCS Reader"

# Asignar el rol storage.objectViewer para leer objetos de GCS
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:${GCP_SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"

# Asignar el rol secretmanager.secretAccessor para acceder a secretos
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:${GCP_SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Asignar el rol bigquery.dataEditor para escribir logs en BigQuery
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \
  --member="serviceAccount:${GCP_SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"
```

### Crear cuenta de servicio de Kubernetes
```bash
# Crear una cuenta de servicio de Kubernetes
kubectl create serviceaccount ${K8S_SA_NAME}
```

### Configurar Workload Identity para vincular las cuentas
```bash
# Habilitar Workload Identity en el clúster (si no está habilitado)
gcloud container clusters update ${CLUSTER_NAME} \
  --region=${GCP_REGION} \
  --workload-pool=${GCP_PROJECT_ID}.svc.id.goog

# Permitir que la cuenta de servicio de Kubernetes actúe como la cuenta de servicio de GCP
gcloud iam service-accounts add-iam-policy-binding \
  ${GCP_SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:${GCP_PROJECT_ID}.svc.id.goog[default/${K8S_SA_NAME}]"

# Anotar la cuenta de servicio de Kubernetes para usar Workload Identity
kubectl annotate serviceaccount ${K8S_SA_NAME} \
  iam.gke.io/gcp-service-account=${GCP_SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com
```

## Paso 3: Crear secretos para las claves API de Binance

```bash
# Crear secretos para las claves API (reemplaza con tus propias claves)
echo -n "${TESTNET_BINANCE_API_KEY}" > ./api-key.txt
echo -n "${TESTNET_BINANCE_API_SECRET}" > ./api-secret.txt

# Crear secretos en Kubernetes
kubectl create secret generic testnet-binance-api-key-futures --from-file=latest=./api-key.txt
kubectl create secret generic testnet-binance-api-secret-futures --from-file=latest=./api-secret.txt

# Limpiar archivos temporales
rm ./api-key.txt ./api-secret.txt
```

## Paso 4: Crear archivos de configuración de Kubernetes

### Archivo de servicio para la comunicación interna
Crea un archivo llamado `btcbot-inference-service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: btcbot-inference-service
  labels:
    app: btcbot-inference
spec:
  type: ClusterIP
  selector:
    app: btcbot-inference
  ports:
  - protocol: TCP
    port: 8080
    targetPort: 8080
```

### Archivo de despliegue para el servidor de inferencia
Crea un archivo llamado `btcbot-inference-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: btcbot-inference-server
  labels:
    app: btcbot-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: btcbot-inference
  template:
    metadata:
      labels:
        app: btcbot-inference
    spec:
      serviceAccountName: btcbot-inference-ksa
      containers:
      - name: inference-container
        image: europe-southwest1-docker.pkg.dev/lofty-complex-460416-r6/lofty-complex-460416-r6-repo/btcbot-cpu:latest
        ports:
        - containerPort: 8080
        env:
        - name: GCP_PROJECT_ID
          value: "lofty-complex-460416-r6"
        - name: GCS_BUCKET_NAME
          value: "lofty-complex-460416-r6"
        command: ["python"]
        args: ["serving/serve.py", "--model_path", "lofty-complex-460416-r6/models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000_steps.zip"]
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 120
          periodSeconds: 10
          failureThreshold: 3
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 180
          periodSeconds: 15
          failureThreshold: 3
```

### Archivo de despliegue para el trader en vivo
Crea un archivo llamado `btcbot-trader-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: btcbot-live-trader
  labels:
    app: btcbot-trader
spec:
  replicas: 1
  selector:
    matchLabels:
      app: btcbot-trader
  template:
    metadata:
      labels:
        app: btcbot-trader
    spec:
      serviceAccountName: btcbot-inference-ksa
      containers:
      - name: trader-container
        image: europe-southwest1-docker.pkg.dev/lofty-complex-460416-r6/lofty-complex-460416-r6-repo/btcbot-cpu:latest
        env:
        - name: GCP_PROJECT_ID
          value: "lofty-complex-460416-r6"
        - name: GCS_BUCKET_NAME
          value: "lofty-complex-460416-r6"
        - name: INFERENCE_SERVICE_URL
          value: "http://btcbot-inference-service:8080/predict"
        - name: USE_SECRET_MANAGER
          value: "false"
        - name: USE_VERTEX_AI
          value: "false"
        - name: TESTNET_BINANCE_API_KEY_FUTURES
          valueFrom:
            secretKeyRef:
              name: testnet-binance-api-key-futures
              key: latest
        - name: TESTNET_BINANCE_API_SECRET_FUTURES
          valueFrom:
            secretKeyRef:
              name: testnet-binance-api-secret-futures
              key: latest
        command: ["python"]
        args: ["scripts/run_live_trader.py", "--use_internal_service", "--internal_service_url", "http://btcbot-inference-service:8080/predict"]
        resources:
          requests:
            cpu: "300m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "1Gi"
```

## Paso 5: Aplicar las configuraciones de Kubernetes

```bash
# Aplicar las configuraciones
kubectl apply -f btcbot-inference-service.yaml
kubectl apply -f btcbot-inference-deployment.yaml
kubectl apply -f btcbot-trader-deployment.yaml
```

Verifica la creación de los recursos:
```bash
# Verificar servicios
kubectl get services

# Verificar despliegues
kubectl get deployments

# Verificar pods
kubectl get pods
```

## Paso 6: Verificar los logs y el funcionamiento

Espera a que los pods estén en estado "Running" y verifica los logs:

### Para el servidor de inferencia:
```bash
kubectl logs -f -l app=btcbot-inference
```

Deberías ver mensajes como:
```
INFO:serving.serve:Modelo cargado correctamente
INFO:gunicorn.info:Starting gunicorn 21.2.0
INFO:gunicorn.error:Listening at: http://0.0.0.0:8080 (123)
```

### Para el trader en vivo:
```bash
kubectl logs -f -l app=btcbot-trader
```

Deberías ver mensajes como:
```
INFO:LiveTrader:Usando endpoint interno de Kubernetes: http://btcbot-inference-service:8080/predict
INFO:LiveWebsocketManager:Conectado exitosamente al WebSocket
INFO:LiveTrader:Predicción recibida: -0.2470
```

## Solución de problemas comunes

### 1. Error de acceso a Secret Manager
Si ves un error como:
```
No se pudo obtener el secreto TESTNET_BINANCE_API_KEY_FUTURES de Secret Manager: 403 Permission denied
```

Asegúrate de:
- Haber creado correctamente los secretos en Kubernetes (paso 3)
- Que `USE_SECRET_MANAGER` esté configurado como "false" en el despliegue del trader
- Verificar que los nombres de los secretos coincidan exactamente

### 2. Error de conexión al endpoint de inferencia
Si ves un error como:
```
HTTP 401, Request is missing required authentication credential
```

Asegúrate de:
- Que el servicio `btcbot-inference-service` esté funcionando y accesible
- Haber configurado `--use_internal_service` y `--internal_service_url` en los argumentos del trader
- Que `USE_VERTEX_AI` esté configurado como "false" en el despliegue del trader

### 3. Error de carga del modelo desde GCS
Si ves un error como:
```
Error al cargar el modelo desde GCS: Access Denied
```

Asegúrate de:
- Que la configuración de Workload Identity esté correcta
- Que la cuenta de servicio de GCP tenga el rol `storage.objectViewer`
- Que la ruta del modelo en GCS sea correcta

### 4. Error al insertar datos en BigQuery
Si ves un error como:
```
403 Access Denied: Dataset lofty-complex-460416-r6:btcbot_logs: Permission bigquery.datasets.get denied
```

Asegúrate de:
- Que la cuenta de servicio de GCP tenga el rol `bigquery.dataEditor`
- Que el dataset y la tabla en BigQuery existan o tengan los permisos adecuados

## Conclusión

Si todos los pasos se han completado correctamente, ahora tienes:
1. Un servidor de inferencia ML ejecutándose en GKE que carga un modelo desde GCS
2. Un servicio interno de Kubernetes que expone este servidor al clúster
3. Una aplicación de trading que se comunica con el servidor de inferencia a través del servicio interno
4. Configuración de secretos para las claves API de Binance
5. Permisos adecuados para acceder a GCS y BigQuery

Este despliegue permite un escalado eficiente y una gestión centralizada de las aplicaciones, manteniendo la comunicación interna segura dentro del clúster de Kubernetes.