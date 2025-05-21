Actúa como un experto en Google Kubernetes Engine (GKE) y Docker. Vamos a desplegar una aplicación de inferencia de modelos de Machine Learning (`serve.py`) en un clúster de GKE existente. Esta aplicación se comunicará internamente con otra aplicación ya desplegada en el mismo clúster (`run_live_trader.py`).

**Objetivo:**
Desplegar el script `serving/serve.py` del proyecto BTCBot como un servicio interno en GKE. Este servicio cargará un modelo de RL desde GCS y expondrá un endpoint `/predict`. Luego, configuraremos `scripts/run_live_trader.py` para que llame a este servicio interno en lugar de a Vertex AI.

**No quiero que generes scripts de despliegue automático ni que modifiques directamente mis archivos. Guíame paso a paso usando comandos de la CLI (`gcloud`, `kubectl`, `docker`). Yo ejecutaré los comandos en mi terminal y tú me ayudarás a verificar los resultados y a entender cada paso. Utiliza tu terminal de Copilot para mostrar ejemplos de cómo se verían los resultados de los comandos de verificación que me sugieras.**

**Contexto del Proyecto BTCBot:**
* **`serving/serve.py`**: Un servidor Flask/Gunicorn que carga un modelo de RL (entrenado con Stable Baselines3) desde GCS y expone los endpoints `/health` y `/predict`. Necesita las variables de entorno `GCP_PROJECT_ID` y `GCS_BUCKET_NAME` para funcionar, y un argumento `--model_path` que especifica la ruta completa en GCS al archivo del modelo (ej: `mi-bucket/models/mi-modelo.zip`).
* **`scripts/run_live_trader.py`**: La aplicación principal del bot de trading. Actualmente está configurada para llamar a un endpoint de Vertex AI, pero modificaremos esto para que llame al servicio interno de `serve.py` en GKE.
* **Imagen Docker**: Ya existe una imagen Docker para la CPU (`europe-southwest1-docker.pkg.dev/lofty-complex-460416-r6/lofty-complex-460416-r6-repo/btcbot-cpu:latest`) construida con `cloudbuild_cpu.yaml` que contiene todo el código del proyecto y sus dependencias. Esta imagen se usará para el servicio de inferencia.
* **Variables de Entorno Generales (que usaré en los YAML y comandos)**:
    * `GCP_PROJECT_ID`: "lofty-complex-460416-r6"
    * `GCS_BUCKET_NAME`: "lofty-complex-460416-r6" (Nota: El archivo `btcbot-deployment.yaml` usa "lofty-complex-460416-r6", mientras que `cloudbuild.yaml` y `README.md` mencionan "lofty-complex-460416-r6" o "bitcoin-460320_data". Usaremos "lofty-complex-460416-r6" para la carga del modelo desde GCS).
    * `GCP_REGION`: "europe-southwest1"
    * `DOCKER_IMAGE_INFERENCE`: "europe-southwest1-docker.pkg.dev/lofty-complex-460416-r6/lofty-complex-460416-r6-repo/btcbot-cpu:latest"
    * `MODEL_PATH_GCS`: "lofty-complex-460416-r6/models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000_steps.zip" (Ejemplo, ajustaré la ruta real del modelo guardado).

**Instrucciones Paso a Paso (CLI):**

Por favor, guíame a través de los siguientes pasos. Para cada paso, explícame qué hace el comando y qué debería esperar como resultado. Luego, indícame qué comando de `kubectl` o `gcloud` puedo usar para verificar que el paso se completó correctamente, y muéstrame un ejemplo de cómo se vería la salida de esa verificación en tu terminal de Copilot.

**Paso 1: Conexión al Clúster de GKE**
1.  Asegurarme de que `kubectl` está configurado para apuntar a mi clúster GKE correcto.
    * Comando: `gcloud container clusters get-credentials [NOMBRE_DEL_CLUSTER] --region [GCP_REGION] --project [GCP_PROJECT_ID]` (Ayúdame a verificar el nombre de mi clúster si es necesario, o asumamos que ya estoy conectado al correcto).
    * Verificación: `kubectl config current-context` y `kubectl cluster-info`.

**Paso 2: Crear el Archivo de `Deployment` para `serve.py`**
1.  Necesitamos un archivo YAML para el `Deployment`. Lo llamaremos `btcbot-inference-deployment.yaml`.
    * Pídele que te proporcione el contenido YAML. Debería incluir:
        * `apiVersion: apps/v1`, `kind: Deployment`
        * `metadata`: `name: btcbot-inference-server`, `labels: { app: btcbot-inference }`
        * `spec`:
            * `replicas: 1` (podemos empezar con 1)
            * `selector`: `matchLabels: { app: btcbot-inference }`
            * `template`:
                * `metadata`: `labels: { app: btcbot-inference }`
                * `spec`:
                    * (Opcional, si uso Workload Identity: `serviceAccountName: [MI_KSA_PARA_ACCESO_GCS]`)
                    * `containers`:
                        * `name: inference-container`
                        * `image: ${DOCKER_IMAGE_INFERENCE}`
                        * `ports: [{ containerPort: 8080 }]` (Gunicorn escuchará en este puerto)
                        * `env`:
                            * `GCP_PROJECT_ID: "${GCP_PROJECT_ID}"`
                            * `GCS_BUCKET_NAME: "${GCS_BUCKET_NAME_MODEL_LOAD}"` (donde está el modelo)
                        * `command: ["python", "serving/serve.py"]`
                        * `args: ["--model_path", "${MODEL_PATH_GCS}"]` (Asegúrate de que `serve.py` acepte este argumento para la ruta del modelo).
                        * `resources`: (Define solicitudes y límites razonables, ej: cpu: "500m", memory: "1Gi")
                        * `readinessProbe` y `livenessProbe` apuntando a `httpGet: { path: /health, port: 8080 }`, con `initialDelaySeconds` adecuados (ej. 60-120s para dar tiempo a que cargue el modelo).
2.  Una vez que tenga el contenido, lo guardaré en `btcbot-inference-deployment.yaml`.
3.  Aplicar el `Deployment`.
    * Comando: `kubectl apply -f btcbot-inference-deployment.yaml`
    * Verificación: `kubectl get deployments btcbot-inference-server` y `kubectl get pods -l app=btcbot-inference`. Esperamos ver el Pod creándose o en estado `Running`.
    * Verificación de logs del Pod: `kubectl logs -f [NOMBRE_DEL_POD_INFERENCE]` para ver si `serve.py` se inicia y carga el modelo correctamente.

**Paso 3: Crear el Archivo de `Service` para `serve.py`**
1.  Necesitamos un archivo YAML para el `Service`. Lo llamaremos `btcbot-inference-service.yaml`.
    * Pídele que te proporcione el contenido YAML. Debería incluir:
        * `apiVersion: v1`, `kind: Service`
        * `metadata`: `name: btcbot-inference-service`, `labels: { app: btcbot-inference }`
        * `spec`:
            * `type: ClusterIP` (ya que solo se accederá internamente)
            * `selector: { app: btcbot-inference }` (debe coincidir con las etiquetas de los Pods del Deployment)
            * `ports: [{ protocol: TCP, port: 8080, targetPort: 8080 }]`
2.  Una vez que tenga el contenido, lo guardaré en `btcbot-inference-service.yaml`.
3.  Aplicar el `Service`.
    * Comando: `kubectl apply -f btcbot-inference-service.yaml`
    * Verificación: `kubectl get services btcbot-inference-service`. Debería mostrar el servicio con un `CLUSTER-IP`.

**Paso 4: Probar la Comunicación Interna (Opcional pero recomendado)**
1.  Si es posible, desde un Pod temporal o desde el Pod de `run_live_trader.py` (si ya está desplegado y podemos acceder a su shell), intentar hacer un `curl` al servicio de inferencia.
    * Comando (ejemplo desde un Pod temporal con `curl`): `kubectl run curl-test --image=curlimages/curl -i --tty --rm -- sh` y luego dentro del shell del pod: `curl http://btcbot-inference-service:8080/health`
    * Esperamos una respuesta JSON `{"status": "healthy"}` (o similar, según `serve.py`).

**Paso 5: Modificar `run_live_trader.py` o su Configuración**
1.  Necesito cambiar la URL a la que `run_live_trader.py` envía las solicitudes de predicción.
    * Actualmente, `scripts/run_live_trader.py` usa `self.live_trading_config.get('vertex_ai_predict_url', "")`.
    * Opción 1: Modificar `src/config.yaml` para que `live_trading.vertex_ai_predict_url` apunte al servicio interno de GKE: `http://btcbot-inference-service:8080/predict`.
    * Opción 2: Añadir un nuevo parámetro de configuración en `config.yaml` o una nueva variable de entorno para la URL del servicio de inferencia interno y modificar `run_live_trader.py` para que use esta nueva configuración cuando no se use Vertex AI.
    * **Pregúntale cuál considera la mejor práctica para manejar esta URL.**
2.  Una vez modificado, si cambié `config.yaml`, este archivo está en el código fuente. Necesitaré reconstruir la imagen de `btcbot-live-trader` (la que usa `scripts/run_live_trader.py`) y actualizar el `Deployment` `btcbot-live-trader` en GKE. (Asumamos que ya sé cómo hacer esto o que me lo puedes recordar brevemente: `gcloud builds submit` y `kubectl rollout restart deployment btcbot-live-trader`).

**Paso 6: Verificar el Flujo Completo**
1.  Después de que `btcbot-live-trader` esté actualizado y ejecutándose, verificar sus logs para confirmar que está enviando predicciones al servicio interno `btcbot-inference-service` y recibiendo respuestas.
    * Comando: `kubectl logs -f -l app=btcbot-trader` (etiqueta de tu `btcbot-live-trader` deployment).
    * Y también: `kubectl logs -f -l app=btcbot-inference` para ver los logs de `serve.py` recibiendo las peticiones.

Por favor, para cada paso, proporciónale los comandos exactos de `kubectl` para aplicar y verificar, y el formato esperado de la salida de verificación. Si hay varios Pods, indícame cómo seleccionar uno específico para ver los logs si es necesario.