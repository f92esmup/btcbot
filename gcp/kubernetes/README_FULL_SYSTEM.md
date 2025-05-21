# BTCBot - Sistema Completo en GKE

Este documento detalla la arquitectura y operación del sistema BTCBot desplegado completamente en Google Kubernetes Engine (GKE).

## Arquitectura del Sistema

El sistema BTCBot se compone de tres componentes principales, todos ejecutándose en GKE:

1. **Entrenamiento del Modelo**:
   - Ejecutado como un `Job` de Kubernetes en GKE
   - Utiliza nodos con GPU (NVIDIA T4) para el entrenamiento
   - Escala a cero cuando no hay trabajos, optimizando costos
   - Guarda modelos entrenados en Google Cloud Storage

2. **Servicio de Inferencia**:
   - Ejecutado como un `Deployment` de Kubernetes en GKE
   - Carga modelos entrenados desde Google Cloud Storage
   - Expuesto internamente mediante un `Service` de tipo `ClusterIP`
   - Proporciona predicciones al trader en tiempo real

3. **Trader en Vivo**:
   - Ejecutado como un `Deployment` de Kubernetes en GKE
   - Conecta a la API de Binance Futures (modo TESTNET o REAL)
   - Obtiene datos del mercado en tiempo real
   - Solicita predicciones al servidor de inferencia interno
   - Ejecuta operaciones según las predicciones recibidas
   - Registra eventos de trading en BigQuery

## Flujo de Trabajo

1. **Entrenamiento de Modelos**:
   - Se inicia un job de entrenamiento con los parámetros adecuados
   - El modelo se entrena y se guarda en GCS
   - El trabajo finaliza y el pod se elimina automáticamente

2. **Actualización del Modelo de Inferencia**:
   - Se actualiza el modelo usado por el servidor de inferencia
   - El servidor se reinicia y carga el nuevo modelo

3. **Trading en Vivo**:
   - El trader se conecta a Binance y al servidor de inferencia
   - Opera continuamente según las predicciones del modelo

## Prerrequisitos

- Cluster GKE con nodos estándar y opcionalmente un node pool con GPU
- Cuenta de servicio con permisos para:
  - Google Cloud Storage (lectura/escritura)
  - Secret Manager (acceso a secretos)
  - BigQuery (inserción de datos)
- Bucket GCS para almacenar datos y modelos
- Secret Manager con las claves API de Binance

## Despliegue Inicial

Para desplegar todo el sistema:

### 1. Configurar el node pool con GPU (opcional)

```bash
# Definir variables
PROJECT_ID="lofty-complex-460416-r6"
CLUSTER_NAME="btcbot-cluster"
REGION="europe-southwest1"
ZONE="${REGION}-a"  # Asegúrate de que esta zona tenga GPUs T4 disponibles
NODE_POOL_NAME="gpu-training-pool"
MACHINE_TYPE="n1-standard-4"  # Tipo de máquina adecuado para GPUs
MIN_NODES=0  # Puede escalar a cero cuando no hay trabajos
MAX_NODES=2  # Máximo número de nodos con GPU

# Crear el node pool con GPU
gcloud container node-pools create ${NODE_POOL_NAME} \
  --cluster ${CLUSTER_NAME} \
  --project ${PROJECT_ID} \
  --region ${REGION} \
  --machine-type ${MACHINE_TYPE} \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --enable-autoscaling \
  --min-nodes ${MIN_NODES} \
  --max-nodes ${MAX_NODES} \
  --node-locations ${ZONE} \
  --node-labels=accelerator=nvidia-tesla-t4 \
  --node-taints=gpu-workload=true:NoSchedule
```

### 2. Desplegar el servidor de inferencia

```bash
# Aplicar el manifiesto de despliegue del servidor de inferencia
kubectl apply -f btcbot-inference-deployment.yaml

# Aplicar el manifiesto del servicio de inferencia
kubectl apply -f btcbot-inference-service.yaml

# Esperar a que el pod esté listo
kubectl wait --for=condition=ready pod -l app=btcbot-inference-server -n btcbot --timeout=300s
```

### 3. Desplegar el trader en vivo

```bash
# Aplicar el manifiesto de despliegue del trader
kubectl apply -f btcbot-live-trader-deployment.yaml

# Esperar a que el pod esté listo
kubectl wait --for=condition=ready pod -l app=btcbot-trader -n btcbot --timeout=300s
```

## Operaciones Comunes

### Iniciar un Trabajo de Entrenamiento

```bash
# Definir variables
JOB_NAME="btcbot-training-job-$(date +%Y%m%d-%H%M%S)"
TIMESTEPS=100000  # Número de pasos de entrenamiento
USE_GPU=false     # Cambiar a true para usar GPU

# Configurar recursos según si se usa GPU o no
if [ "$USE_GPU" == "true" ]; then
  # Configuración para GPU
  cat > training-job.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: btcbot
spec:
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      serviceAccountName: btcbot-inference-sa
      nodeSelector:
        cloud.google.com/gke-accelerator: "nvidia-tesla-t4"
        topology.kubernetes.io/zone: "europe-west4-a"
      containers:
      - name: training-container
        image: europe-southwest1-docker.pkg.dev/lofty-complex-460416-r6/lofty-complex-460416-r6-repo/btcbot-gpu:latest
        command: ["python", "scripts/train_rl_agent.py"]
        args:
          - "--config"
          - "src/config.yaml"
          - "--timesteps"
          - "${TIMESTEPS}"
        env:
        - name: GCP_PROJECT_ID
          value: "lofty-complex-460416-r6"
        - name: GCS_BUCKET_NAME
          value: "lofty-complex-460416-r6"
        - name: GCP_REGION
          value: "europe-southwest1"
        - name: NVIDIA_DRIVER_CAPABILITIES
          value: "compute,utility"
        - name: NVIDIA_VISIBLE_DEVICES
          value: "all"
        resources:
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
      restartPolicy: Never
  backoffLimit: 0
EOF
else
  # Configuración para CPU
  cat > training-job.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: btcbot
spec:
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      serviceAccountName: btcbot-inference-sa
      containers:
      - name: training-container
        image: europe-southwest1-docker.pkg.dev/lofty-complex-460416-r6/lofty-complex-460416-r6-repo/btcbot-cpu:latest
        command: ["python", "scripts/train_rl_agent.py"]
        args:
          - "--config"
          - "src/config.yaml"
          - "--timesteps"
          - "${TIMESTEPS}"
          - "--no-gpu"
        env:
        - name: GCP_PROJECT_ID
          value: "lofty-complex-460416-r6"
        - name: GCS_BUCKET_NAME
          value: "lofty-complex-460416-r6"
        - name: GCP_REGION
          value: "europe-southwest1"
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
          requests:
            memory: "2Gi"
            cpu: "1"
      restartPolicy: Never
  backoffLimit: 0
EOF
fi

# Aplicar el manifiesto
kubectl apply -f training-job.yaml

# Mostrar información de seguimiento
echo "Job de entrenamiento iniciado: ${JOB_NAME}"
echo "Para monitorear el progreso:"
echo "  kubectl get jobs -n btcbot ${JOB_NAME}"
echo "  kubectl get pods -n btcbot -l job-name=${JOB_NAME}"
echo "  kubectl logs -f -n btcbot -l job-name=${JOB_NAME}"

# Limpiar el archivo temporal
rm training-job.yaml
```

### Actualizar el Modelo en el Servidor de Inferencia

```bash
# Definir variables
MODEL_PATH="models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000_steps.zip"
FULL_GCS_PATH="gs://lofty-complex-460416-r6/${MODEL_PATH}"

# Crear patch para actualizar el despliegue con el nuevo modelo
cat > model-update-patch.yaml << EOF
spec:
  template:
    spec:
      containers:
      - name: inference-container
        env:
        - name: MODEL_PATH
          value: "${FULL_GCS_PATH}"
        command: ["python"]
        args: ["serving/serve.py", "--model_path", "\$(MODEL_PATH)"]
EOF

# Aplicar el patch
kubectl patch deployment btcbot-inference-server -n btcbot --patch "$(cat model-update-patch.yaml)"

# Esperar a que el despliegue se complete
kubectl rollout status deployment/btcbot-inference-server -n btcbot

# Verificar logs
POD_NAME=$(kubectl get pods -n btcbot -l app=btcbot-inference-server -o jsonpath="{.items[0].metadata.name}")
kubectl logs -n btcbot ${POD_NAME} | grep -i "modelo cargado"

# Limpiar archivo temporal
rm model-update-patch.yaml
```

### Monitoreo del Sistema

```bash
# Ver todos los pods
kubectl get pods -n btcbot

# Ver logs del servidor de inferencia
kubectl logs -f -n btcbot deployment/btcbot-inference-server

# Ver logs del trader
kubectl logs -f -n btcbot deployment/btcbot-live-trader

# Ver trabajos de entrenamiento
kubectl get jobs -n btcbot

# Ver logs de un trabajo de entrenamiento específico
kubectl logs -f -n btcbot -l job-name=NOMBRE_DEL_JOB
```

## Mantenimiento

### Reconstruir y Actualizar Imágenes Docker

1. Reconstruir imágenes:
```bash
gcloud builds submit --config=gcp/cloudbuild/cloudbuild.yaml
```

2. Reiniciar los despliegues para usar las nuevas imágenes:
```bash
kubectl rollout restart deployment/btcbot-inference-server -n btcbot
kubectl rollout restart deployment/btcbot-live-trader -n btcbot
```

### Limpieza de Trabajos Antiguos

Los trabajos de entrenamiento se eliminan automáticamente 1 hora después de completarse (`ttlSecondsAfterFinished: 3600`). Si necesitas limpiarlos manualmente:

```bash
kubectl delete jobs -n btcbot --field-selector status.successful=1
```
