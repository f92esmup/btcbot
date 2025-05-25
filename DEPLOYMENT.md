# BTCBot - Deployment en GKE Autopilot

## Resumen del Despliegue

### ✅ Configuración Completada

1. **Clúster GKE Autopilot**
   - Nombre: `btcbot-autopilot-cluster`
   - Región: `europe-southwest1`
   - Estado: ✅ Activo y funcionando

2. **Workload Identity**
   - Service Account GCP: `btcbot-gke-sa@lofty-complex-460416-r6.iam.gserviceaccount.com`
   - Service Account K8s: `btcbot-ksa` (namespace: `btcbot`)
   - Permisos IAM: ✅ Storage, Secret Manager, BigQuery

3. **IP Estática para Whitelist**
   - IP de salida: **34.175.215.35**
   - 🔸 **ACCIÓN REQUERIDA**: Agregar esta IP a la whitelist de Binance API

4. **Recursos Desplegados**
   - Namespace: `btcbot`
   - ConfigMap: `btcbot-env-vars`
   - CronJob: `data-acquisition-cronjob` (programado sábados 00:00 UTC)
   - Jobs: `data-preprocessing-job`, `model-training-job`
   - Deployment: `live-trader-deployment`

5. **Configuración del Modelo**
   - ✅ Modelo desplegado: `models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000_steps.zip`
   - ✅ Ruta configurada correctamente en `src/config.yaml` bajo `agent.live_model_path`

### 🔄 Estado Actual

- **Data Preprocessing Job**: ContainerCreating (descargando imagen)
- **Live Trader Deployment**: ✅ Running - Conectado a Binance TESTNET
- **Model Training Job**: Pending (esperando nodo con GPU)

### 📋 Próximos Pasos

1. **Monitorear el estado de los pods**:
   ```bash
   kubectl get pods -n btcbot
   kubectl logs -f -n btcbot deployment/live-trader-deployment
   ```

2. **Verificar logs de errores si los pods fallan**:
   ```bash
   kubectl describe pod -n btcbot <pod-name>
   kubectl logs -n btcbot <pod-name>
   ```

3. **Ejecutar pipeline completo secuencial**:
   ```bash
   kubectl apply -f k8s/pipeline-orchestrator.yaml
   kubectl logs -f -n btcbot job/pipeline-orchestrator
   ```

4. **Configurar whitelist en Binance**:
   - Agregar IP: `34.175.215.35` a la API de Binance Futures

### 🛠️ Comandos de Gestión

#### Verificar estado general
```bash
kubectl get all -n btcbot
```

#### Escalar el trading bot
```bash
kubectl scale deployment live-trader-deployment --replicas=0 -n btcbot  # Parar
kubectl scale deployment live-trader-deployment --replicas=1 -n btcbot  # Reiniciar
```

#### Ejecutar job manual de adquisición de datos
```bash
kubectl create job data-acquisition-manual --from=cronjob/data-acquisition-cronjob -n btcbot
```

#### Ver logs del trading en vivo
```bash
kubectl logs -f -n btcbot deployment/live-trader-deployment
```

### 🔧 Configuración de Variables

Las variables de entorno están en el ConfigMap `btcbot-env-vars`:
- `GCP_PROJECT_ID`: lofty-complex-460416-r6
- `GCS_BUCKET_NAME`: lofty-complex-460416-r6
- `GCP_REGION`: europe-southwest1
- `BIGQUERY_LOG_DATASET_ID`: btcbot_logs
- `LIVE_TRADING_MODE`: false (cambiar a "true" para trading real)

Para modificar:
```bash
kubectl edit configmap btcbot-env-vars -n btcbot
```

### 🚨 Notas Importantes

1. **GPU Availability**: El job de entrenamiento requiere GPU (NVIDIA T4). Si no hay disponibilidad inmediata, puede tardar o fallar.

2. **Imágenes Docker**: Asegúrate de que las imágenes estén disponibles en:
   - `europe-southwest1-docker.pkg.dev/lofty-complex-460416-r6/lofty-complex-460416-r6-repo/btcbot-cpu:latest`
   - `europe-southwest1-docker.pkg.dev/lofty-complex-460416-r6/lofty-complex-460416-r6-repo/btcbot-gpu:latest`

3. **Secretos de Binance**: Se acceden vía Secret Manager usando Workload Identity.

4. **Costos**: GKE Autopilot factura por recursos utilizados. Monitorea el uso.

5. **Ruta del Modelo**: El sistema está configurado para usar el modelo en la ruta:
   ```
   models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000_steps.zip
   ```
   Si necesitas usar un modelo diferente, modifica el valor de `agent.live_model_path` en `src/config.yaml`.

6. **Argumentos del Comando**: Para ejecutar el trader en vivo manualmente, usa:
   ```bash
   python scripts/run_live_trader.py --config_path src/config.yaml
   ```
   Asegúrate de usar `--config_path` (no `--config`) como argumento.

### 🔄 Pipeline de Datos

1. **Data Acquisition** → 2. **Preprocessing** → 3. **Model Training** → 4. **Live Trading**

Usa el `pipeline-orchestrator.yaml` para ejecutar el pipeline completo en secuencia.
