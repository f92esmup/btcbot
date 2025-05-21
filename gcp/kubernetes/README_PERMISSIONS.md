# Asignando permisos a la cuenta de servicio para GCS

Para que el entrenamiento y el servidor de inferencia puedan leer y escribir archivos en Google Cloud Storage, es necesario asignar los permisos adecuados a la cuenta de servicio `btcbot-inference-sa`.

## Comandos para asignar permisos

```bash
# Establecer variables
PROJECT_ID="lofty-complex-460416-r6"
BUCKET_NAME="lofty-complex-460416-r6"
SERVICE_ACCOUNT="btcbot-inference-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Asignar el rol Storage Object Admin a la cuenta de servicio para el bucket
gcloud storage buckets add-iam-policy-binding gs://${BUCKET_NAME} \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectAdmin"

# También puedes asignar permisos a nivel de proyecto si es necesario
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/storage.objectAdmin"

# Verificar la asignación de permisos
gcloud storage buckets get-iam-policy gs://${BUCKET_NAME}
```

## Permisos necesarios

| Rol | Descripción | Uso |
|-----|-------------|-----|
| `roles/storage.objectAdmin` | Proporciona acceso completo para leer, escribir y eliminar objetos en GCS | Necesario para almacenar modelos entrenados y cargar datos |
| `roles/bigquery.dataEditor` | Proporciona acceso para insertar datos en BigQuery | Necesario para registrar eventos de trading |
| `roles/secretmanager.secretAccessor` | Proporciona acceso para leer secretos | Necesario para acceder a las claves API de Binance |

## Verificación de permisos

Después de asignar los permisos, puedes verificar el acceso ejecutando un job de prueba:

```bash
# Crear un job de Kubernetes para verificar el acceso a GCS
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: gcs-test-job
  namespace: btcbot
spec:
  ttlSecondsAfterFinished: 300
  template:
    spec:
      serviceAccountName: btcbot-inference-sa
      containers:
      - name: gsutil-test
        image: google/cloud-sdk:slim
        command:
        - "/bin/sh"
        - "-c"
        - "gsutil ls gs://${BUCKET_NAME} && echo 'Test successful!' || echo 'Access denied!'"
      restartPolicy: Never
  backoffLimit: 0
EOF

# Verificar los logs del job
kubectl logs -n btcbot -l job-name=gcs-test-job
```
