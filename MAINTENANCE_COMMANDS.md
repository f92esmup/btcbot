# Comandos de Mantenimiento para BTCBot en GKE Autopilot

Este documento contiene los comandos más útiles para el mantenimiento y gestión de la infraestructura de BTCBot desplegada en Google Kubernetes Engine (GKE) Autopilot.

## Información General

```bash
# Ver todos los recursos en el namespace btcbot
kubectl get all -n btcbot

# Obtener información detallada del clúster
gcloud container clusters describe btcbot-autopilot-cluster --region=europe-southwest1 --project=lofty-complex-460416-r6
```

## Gestión de Pods

### Verificar estado y logs

```bash
# Listar todos los pods
kubectl get pods -n btcbot

# Ver logs del bot de trading en vivo
kubectl logs -f deployment/btcbot-live-trader -n btcbot

# Ver logs del job orquestador (reemplazar <pod_id> con el ID actual)
kubectl logs -f job/btcbot-pipeline-orchestrator -n btcbot
# o
kubectl logs $(kubectl get pods -n btcbot -l job-name=btcbot-pipeline-orchestrator -o jsonpath="{.items[0].metadata.name}") -n btcbot

# Obtener información detallada de un pod
kubectl describe pod <nombre_del_pod> -n btcbot
```

### Reiniciar componentes

```bash
# Reiniciar el deployment del bot de trading 
kubectl rollout restart deployment/btcbot-live-trader -n btcbot

# Eliminar y volver a crear el job orquestador
kubectl delete -f pipeline-orchestrator.yaml
kubectl apply -f pipeline-orchestrator.yaml

# Ejecutar manualmente el job de la pipeline (útil para pruebas sin esperar al CronJob)
kubectl create job --from=cronjob/btcbot-pipeline-scheduler manual-run-$(date +%s) -n btcbot
```

## Escalado

```bash
# Escalar el bot de trading a más réplicas (para redundancia)
kubectl scale deployment/btcbot-live-trader --replicas=2 -n btcbot

# Regresar a una sola réplica
kubectl scale deployment/btcbot-live-trader --replicas=1 -n btcbot
```

## Gestión de CronJob

```bash
# Listar CronJobs
kubectl get cronjobs -n btcbot

# Ver detalles del CronJob
kubectl describe cronjob/btcbot-pipeline-scheduler -n btcbot

# Suspender el CronJob temporalmente (ej: mantenimiento)
kubectl patch cronjobs btcbot-pipeline-scheduler -n btcbot -p '{"spec":{"suspend":true}}'

# Reactivar el CronJob
kubectl patch cronjobs btcbot-pipeline-scheduler -n btcbot -p '{"spec":{"suspend":false}}'

# Modificar la programación del CronJob (en este ejemplo, a cada domingo 2 AM UTC)
kubectl patch cronjob btcbot-pipeline-scheduler -n btcbot -p '{"spec":{"schedule":"0 2 * * 0"}}'
```

## Actualización de Imágenes

```bash
# Actualizar la imagen del bot de trading (cuando se publique una nueva versión)
kubectl set image deployment/btcbot-live-trader btcbot-live-trader=europe-southwest1-docker.pkg.dev/lofty-complex-460416-r6/lofty-complex-460416-r6-repo/btcbot-cpu:nueva_version -n btcbot

# Para el CronJob, se debe editar manualmente el archivo y volver a aplicarlo:
# Editar archivo: pipeline-cronjob.yaml
# Luego aplicar: kubectl apply -f pipeline-cronjob.yaml
```

## Gestión de ConfigMaps y Variables de Entorno

```bash
# Ver el ConfigMap actual
kubectl get configmap btcbot-env-vars -n btcbot -o yaml

# Editar el ConfigMap 
kubectl edit configmap btcbot-env-vars -n btcbot

# O crear un nuevo archivo y aplicarlo:
# kubectl create configmap btcbot-env-vars --from-literal=NUEVA_VARIABLE=valor -n btcbot --dry-run=client -o yaml > nuevo-configmap.yaml
# kubectl apply -f nuevo-configmap.yaml
```

## Depuración y Acceso

```bash
# Crear un pod temporal para depuración que use la misma cuenta de servicio
kubectl run debug-pod --image=europe-southwest1-docker.pkg.dev/lofty-complex-460416-r6/lofty-complex-460416-r6-repo/btcbot-cpu:latest -n btcbot --serviceaccount=btcbot-ksa --command -- sleep infinity

# Conectarse al pod de depuración
kubectl exec -it debug-pod -n btcbot -- /bin/bash

# Eliminar el pod de depuración cuando termine
kubectl delete pod debug-pod -n btcbot
```

## Supervisión de Recursos

```bash
# Ver el uso de recursos de los pods
kubectl top pods -n btcbot

# Ver el uso de recursos de los nodos
kubectl top nodes
```

## Eliminación completa (para recrear desde cero)

```bash
# Eliminar todos los recursos en el namespace (¡CUIDADO! Destruye todo)
kubectl delete namespace btcbot

# Si solo se quiere eliminar componentes específicos:
kubectl delete -f live-trader-deployment.yaml
kubectl delete -f pipeline-orchestrator.yaml
kubectl delete -f pipeline-cronjob.yaml

# Recrear el namespace
kubectl create namespace btcbot

# Volver a aplicar todos los manifiestos
kubectl apply -f live-trader-deployment.yaml
kubectl apply -f pipeline-orchestrator.yaml
kubectl apply -f pipeline-cronjob.yaml
```

## Gestión de Direcciones IP y Cloud NAT

```bash
# Verificar la dirección IP estática (para la lista blanca de Binance)
gcloud compute addresses describe btcbot-nat-ip --region=europe-southwest1 --project=lofty-complex-460416-r6

# Ver la configuración del router NAT
gcloud compute routers nats describe btcbot-nat --router=btcbot-nat-router --router-region=europe-southwest1 --project=lofty-complex-460416-r6
```

## Gestión de Workload Identity

```bash
# Verificar la configuración de la cuenta de servicio de Kubernetes
kubectl get serviceaccount btcbot-ksa -n btcbot -o yaml

# Verificar los permisos IAM de la cuenta de servicio de GCP
gcloud projects get-iam-policy lofty-complex-460416-r6 --format=json | grep -A 5 "btcbot-gke-sa"

# Verificar permisos sobre el bucket de GCS
gcloud storage buckets get-iam-policy gs://lofty-complex-460416-r6
```

---

**Importante:** Recuerda siempre documentar los cambios que realices en los recursos del clúster para mantener un registro y facilitar la resolución de problemas en caso de incidencias.
