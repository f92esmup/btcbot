## **Manual de Despliegue y Mantenimiento: Bot de Trading en Cloud Run**

**Autor:** Pedro Escudero Murcia & Asistente Gemini
**Fecha:** 3 de Julio de 2025
**Objetivo:** Desplegar, gestionar y mantener el bot `live.py` como un worker persistente, seguro y resiliente en Google Cloud.

### **1. Filosofía de la Arquitectura**

Este sistema se basa en una arquitectura de **componentes especializados** donde cada parte del proceso (entrenamiento, ejecución en vivo) utiliza el servicio de Google Cloud más adecuado para su tarea.

  * **Entrenamiento (`train.py`):** Se ejecuta en **Vertex AI Training**, una plataforma serverless optimizada para trabajos de Machine Learning de uso intensivo y corta duración.
  * **Ejecución en Vivo (`live.py`):** Se despliega en **Cloud Run** como un **Worker Pool**. Esta es la solución moderna para ejecutar contenedores de larga duración (24/7) que no atienden peticiones HTTP, ofreciendo reinicio automático en caso de fallo sin necesidad de gestionar servidores.

### **2. Componentes de GCP Involucrados**

1.  **Cloud Build:** Construye la imagen Docker de nuestro bot.
2.  **Artifact Registry:** Almacena de forma segura nuestra imagen Docker.
3.  **IAM (Identity and Access Management):** Gestiona los permisos.
4.  **Secret Manager:** Almacena las credenciales (API keys, tokens).
5.  **Cloud Run:** Ejecuta nuestro bot `live.py`.
6.  **BigQuery:** Almacena los logs de trading para análisis futuros.
7.  **Cloud Logging:** Proporciona logs en tiempo real de la ejecución del bot.

-----

### **Fase 1: Configuración Inicial del Entorno (Realizar una sola vez)**

Estos pasos preparan tu proyecto de Google Cloud. Solo necesitas ejecutarlos una vez.

#### **1.1. Crear la Cuenta de Servicio Dedicada**

Crearemos una identidad específica para el bot, siguiendo el principio de menor privilegio.

```bash
gcloud iam service-accounts create btcbot-live-runner --display-name="Service Account for Live BTC Bot"
```

#### **1.2. Asignar Permisos a la Cuenta de Servicio**

Le damos a esta nueva cuenta los permisos que necesita para operar:

  * **Acceso a Secretos (API Keys de Binance, Telegram):**

    ```bash
    # Permiso para la API Key de Binance (Testnet)
    gcloud secrets add-iam-policy-binding TESTNET_BINANCE_API_KEY_FUTURES --member="serviceAccount:btcbot-live-runner@btcbot-2762.iam.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"

    # Permiso para el API Secret de Binance (Testnet)
    gcloud secrets add-iam-policy-binding TESTNET_BINANCE_API_SECRET_FUTURES --member="serviceAccount:btcbot-live-runner@btcbot-2762.iam.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"

    # Repetir para las claves de producción y de Telegram cuando sea necesario
    ```

  * **Acceso a BigQuery (para los logs de trading):**

    ```bash
    gcloud projects add-iam-policy-binding btcbot-2762 \
      --member="serviceAccount:btcbot-live-runner@btcbot-2762.iam.gserviceaccount.com" \
      --role="roles/bigquery.dataEditor"
    ```

#### **1.3. Asignarte Permiso para "Actuar Como" la Cuenta de Servicio**

Este paso es crucial. Te da permiso a ti, como usuario, para desplegar un servicio que *utiliza* la identidad del bot.

```bash
gcloud iam service-accounts add-iam-policy-binding \
  btcbot-live-runner@btcbot-2762.iam.gserviceaccount.com \
  --member="user:galletanosuco@gmail.com" \
  --role="roles/iam.serviceAccountUser"
```

-----

### **Fase 2: Ciclo de Despliegue del Bot**

Estos son los pasos que repetirás cada vez que quieras desplegar una nueva versión o cambiar la configuración.

#### **2.1. Construir y Subir la Imagen del Bot**

Usando Cloud Build, empaquetamos el código en una imagen Docker y la subimos a Artifact Registry.

```bash
# Este comando utiliza la configuración de tu archivo cloudbuild.yaml
gcloud builds submit --config cloudbuild.yaml .
```

*Nota: Para empezar, puedes usar tu imagen de entrenamiento existente. A largo plazo, se recomienda crear un `Dockerfile.live` optimizado y un `cloudbuild-live.yaml` específico.*

#### **2.2. Desplegar el Bot en Cloud Run**

Este es el comando final para lanzar (o actualizar) tu bot. Asegúrate de reemplazar `ID_DEL_RUN_A_USAR` con el modelo correcto.

```bash
gcloud beta run worker-pools deploy btcbot-live-worker \
  --image europe-southwest1-docker.pkg.dev/btcbot-2762/btcbotrepo/btcbot-train-gpu:latest \
  --service-account btcbot-live-runner@btcbot-2762.iam.gserviceaccount.com \
  --command="python" \
  --args="live.py,--run-id=ID_DEL_RUN_A_USAR,--mode=testnet" \
  --cpu=1 \
  --memory=512Mi \
  --region=europe-southwest1
```

  * **`worker-pools deploy`**: Comando específico para cargas de trabajo persistentes sin HTTP.
  * **`--service-account`**: Asigna la identidad segura que creamos.
  * **`--command` y `--args`**: Instrucciones para ejecutar `live.py` dentro del contenedor.

-----

### **Fase 3: Mantenimiento y Operaciones (Day-2)**

Una vez que el bot está en funcionamiento, necesitarás realizar estas tareas.

#### **3.1. Monitorizar los Logs en Tiempo Real**

La forma más rápida de ver lo que está haciendo tu bot.

```bash
gcloud beta run worker-pools logs tail btcbot-live-worker --region=europe-southwest1
```

#### **3.2. Actualizar el Bot a una Nueva Versión**

1.  Realiza los cambios en tu código.
2.  Construye la nueva imagen con `gcloud builds submit ...`. La nueva imagen se etiquetará como `:latest`.
3.  **Vuelve a ejecutar exactamente el mismo comando `gcloud beta run worker-pools deploy ...`**. Cloud Run detectará la nueva imagen `:latest` y creará una nueva revisión, redirigiendo la ejecución a ella sin tiempo de inactividad.

#### **3.3. Cambiar los Argumentos del Bot (ej. usar un nuevo `run-id`)**

No necesitas construir una nueva imagen. Simplemente modifica la bandera `--args` en el comando de despliegue y vuelve a ejecutarlo.

```bash
# Ejemplo para cambiar a un nuevo modelo
gcloud beta run worker-pools deploy btcbot-live-worker \
  --args="live.py,--run-id=NUEVO_ID_DE_MODELO,--mode=testnet" \
  --region=europe-southwest1
  # ... (el resto de banderas no necesitan repetirse si no cambian)
```

#### **3.4. Detener Temporalmente el Bot**

Si quieres pausar el bot sin eliminar la configuración, puedes escalar el worker pool a cero instancias.

```bash
gcloud beta run worker-pools update btcbot-live-worker --scaling=0 --region=europe-southwest1
```

Para reanudarlo, vuelve a ejecutar el comando de despliegue original.

#### **3.5. Eliminar Permanentemente el Bot**

Para borrar completamente el servicio de Cloud Run.

```bash
gcloud beta run worker-pools delete btcbot-live-worker --region=europe-southwest1
```
