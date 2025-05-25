# BTCBot - Detalles de Despliegue en GKE

Este documento contiene la información detallada sobre el despliegue actual de BTCBot en Google Kubernetes Engine (GKE) Autopilot, para referencia y mantenimiento.

## Valores de Configuración

| Parámetro | Valor |
|-----------|-------|
| Proyecto GCP | `lofty-complex-460416-r6` |
| Región GCP | `europe-southwest1` (Madrid) |
| Bucket GCS | `lofty-complex-460416-r6` |
| Repositorio Artifact Registry | `lofty-complex-460416-r6-repo` |
| Clúster GKE | `btcbot-autopilot-cluster` |
| Namespace Kubernetes | `btcbot` |
| IP NAT Estática | `34.175.215.35` |
| Dataset BigQuery | `btcbot_logs` |

## Componentes Desplegados

### Cuentas de Servicio
- **GCP SA**: `btcbot-gke-sa@lofty-complex-460416-r6.iam.gserviceaccount.com`
- **Kubernetes SA**: `btcbot-ksa` (en namespace `btcbot`)

### Permisos IAM asignados
- `roles/storage.admin` - Para gestionar el bucket de GCS
- `roles/storage.objectAdmin` - Para gestionar objetos en el bucket
- `roles/secretmanager.secretAccessor` - Para acceder a secretos en Secret Manager
- `roles/bigquery.dataEditor` y `roles/bigquery.user` - Para escribir en BigQuery

### Imágenes Docker
- **CPU**: `europe-southwest1-docker.pkg.dev/lofty-complex-460416-r6/lofty-complex-460416-r6-repo/btcbot-cpu:latest`
- **GPU**: `europe-southwest1-docker.pkg.dev/lofty-complex-460416-r6/lofty-complex-460416-r6-repo/btcbot-gpu:latest`

### Componentes de Kubernetes

#### ConfigMap: `btcbot-env-vars`
- `GCP_PROJECT_ID`: `lofty-complex-460416-r6`
- `GCS_BUCKET_NAME`: `lofty-complex-460416-r6`
- `BIGQUERY_LOG_DATASET_ID`: `btcbot_logs`
- `GCP_REGION`: `europe-southwest1`
- `LIVE_TRADING_MODE`: `false` (sobreescrito a `true` en el deployment del trader)

#### Job Orquestador: `btcbot-pipeline-orchestrator`
- Script: `scripts/orchestrate_training.py --timesteps 100000`
- Recursos: 8-16 CPU, 16-32 GB RAM
- Imagen: Versión CPU (no hay GPUs disponibles en la región)
- Estado: Ejecución manual bajo demanda

#### CronJob: `btcbot-pipeline-scheduler`
- Programación: Sábados a las 00:00 UTC (`0 0 * * 6`)
- Script: `scripts/orchestrate_training.py --timesteps 100000`
- Recursos: 8-16 CPU, 16-32 GB RAM

#### Deployment: `btcbot-live-trader`
- Script: `scripts/run_live_trader.py`
- Recursos: 1-2 CPU, 2-4 GB RAM
- Sondas de salud: Verifica `/tmp/healthy` y `/tmp/ready`
- Modo: Live Trading (`LIVE_TRADING_MODE=true`)

## Configuración de Red

### Cloud NAT
- Router: `btcbot-nat-router`
- NAT Gateway: `btcbot-nat`
- IP Estática: `btcbot-nat-ip` (`34.175.215.35`)
- Configuración: Todo el tráfico saliente de los pods usa esta IP

## Notas importantes

1. **GPUs no disponibles**: En la región `europe-southwest1` (Madrid) no hay GPUs disponibles para GKE Autopilot. Si se requieren GPUs para entrenamiento, se recomienda:
   - Crear un clúster adicional en otra región europea que tenga GPUs (como `europe-west4`)
   - Modificar los manifiestos para usar GPU específicas (`nvidia-tesla-t4` o `nvidia-l4`)

2. **Sondas de salud**: El bot de trading en vivo requiere que existan los archivos `/tmp/healthy` y `/tmp/ready`
   para pasar las pruebas de liveness y readiness.

3. **Almacenamiento**: Todos los datos y modelos se almacenan en Google Cloud Storage, no hay almacenamiento persistente en Kubernetes.

4. **Variables de entorno**: Todas las variables de configuración están centralizadas en el ConfigMap `btcbot-env-vars`.

5. **Secretos**: Las credenciales de API de Binance se obtienen desde Secret Manager usando Workload Identity.

## Observaciones de rendimiento

- El bot de trading en vivo consume aproximadamente 1 CPU y 2 GB de RAM
- El entrenamiento sin GPU es significativamente más lento
- La configuración de Cloud NAT es esencial para que Binance reconozca la IP estática

Este documento se actualizó por última vez el 25 de mayo de 2025.
