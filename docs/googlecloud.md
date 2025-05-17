# Guía de Ejecución de Scripts para la Implementación de BTCBot en GCP

Esta guía detalla el orden recomendado y el propósito de cada script en la carpeta `gcp/` para desplegar y gestionar tu proyecto BTCBot en Google Cloud Platform.

## Fase 1: Configuración Inicial de la Infraestructura (Ejecutar una vez)

Estos scripts preparan tu entorno de GCP. Generalmente, solo necesitas ejecutarlos una vez, o cuando haya cambios en la configuración fundamental.

1.  **`gcp/enable_apis.sh`**
    * **Propósito:** Activa todas las APIs de Google Cloud necesarias para el proyecto (Cloud Resource Manager, IAM, Storage, Secret Manager, Vertex AI, etc.).
    * **Cuándo ejecutar:** Al inicio de la configuración de tu proyecto en GCP.
    * **Comando Ejemplo:** `bash gcp/enable_apis.sh`
    * **Nota:** Después de este script, el `03_setup_iam.py` se ejecuta automáticamente por el `enable_apis.sh`. Si lo ejecutas por separado, asegúrate de que las APIs estén habilitadas.

2.  **`gcp/01_setup_secrets.py`**
    * **Propósito:** Configura de forma segura tus claves API de Binance en Google Secret Manager.
    * **Cuándo ejecutar:** Una vez, para almacenar tus credenciales. Repetir solo si necesitas actualizar las claves.
    * **Comando Ejemplo:** `python gcp/01_setup_secrets.py --api_key TU_BINANCE_API_KEY --api_secret TU_BINANCE_API_SECRET`

3.  **`gcp/02_setup_storage.py`**
    * **Propósito:** Crea los buckets necesarios en Google Cloud Storage (GCS) para datos crudos, datos procesados, staging de modelos y resultados de evaluación. Habilita el control de versiones en estos buckets.
    * **Cuándo ejecutar:** Una vez, para configurar el almacenamiento.
    * **Comando Ejemplo:** `python gcp/02_setup_storage.py`

4.  **`gcp/03_setup_iam.py`**
    * **Propósito:** Crea una cuenta de servicio dedicada para el proyecto BTCBot y le asigna los roles y permisos necesarios para interactuar con los servicios de GCP (Storage, Secret Manager, Vertex AI, Artifact Registry, etc.).
    * **Cuándo ejecutar:** Una vez, para configurar los permisos. El script `enable_apis.sh` ya lo invoca.
    * **Comando Ejemplo:** `python gcp/03_setup_iam.py` (si se ejecuta independientemente de `enable_apis.sh`).

## Fase 2: Empaquetado y Despliegue de Servicios Base

Estos scripts se encargan de empaquetar tu código y desplegar servicios fundamentales.

5.  **`gcp/04_build_docker_image.py`**
    * **Propósito:** Construye las imágenes Docker para tu código de entrenamiento (y opcionalmente para preprocesamiento si tienes un Dockerfile separado) usando Google Cloud Build. Las imágenes resultantes se almacenan en Artifact Registry.
    * **Cuándo ejecutar:** Cada vez que realices cambios significativos en tu código fuente (`src/`, `scripts/`), `requirements.txt`, o en los `Dockerfile` (`Dockerfile`, `Dockerfile.gpu`).
    * **Comando Ejemplo (para imagen GPU con tag 'latest'):** `python gcp/04_build_docker_image.py --gpu --tag latest`
    * **Comando Ejemplo (para imagen CPU con tag 'latest'):** `python gcp/04_build_docker_image.py --tag latest`

6.  **`gcp/05_deploy_data_acquisition.py`**
    * **Propósito:** Despliega un servicio en Cloud Run que ejecuta tu script de adquisición de datos (`scripts/download_data.py`). También configura un Cloud Scheduler para invocar este servicio periódicamente (actualmente, la configuración por defecto en el script es una vez al día para descargar datos con intervalo "1h").
    * **Cuándo ejecutar:** Una vez para configurar el servicio de adquisición de datos y su programación. Puedes reejecutarlo si cambias la lógica del servicio de adquisición o su programación.
    * **Comando Ejemplo:** `python gcp/05_deploy_data_acquisition.py`
    * **Nota:** Considera si la frecuencia y el intervalo fijados en este script son los adecuados para tu necesidad de actualización de datos (ver discusión previa sobre hacerlo dinámico o para descargas incrementales).

## Fase 3: Ciclo de Vida del Modelo (MLOps)

Aquí tienes dos enfoques principales: ejecutar trabajos individuales o un pipeline completo.

### Opción A: Flujo Manual/Individual (Bueno para Desarrollo y Pruebas)

7.  **`gcp/06_launch_training_job.py`**
    * **Propósito:** Lanza un trabajo de entrenamiento personalizado en Vertex AI utilizando una de las imágenes Docker construidas en el paso 5. Al finalizar, registra el modelo entrenado en Vertex AI Model Registry.
    * **Cuándo ejecutar:** Cuando quieras entrenar un modelo de forma individual, fuera de un pipeline completo.
    * **Comando Ejemplo (con GPU):** `python gcp/06_launch_training_job.py --use_gpu --gpu_type NVIDIA_TESLA_T4 --machine_type n1-standard-8 --image_tag latest`

8.  **`gcp/08_evaluate_model.py`**
    * **Propósito:** Descarga un modelo específico de Vertex AI Model Registry y (presumiblemente, según su estructura actual) lo evalúa localmente o en un entorno similar, registrando luego las métricas.
    * **Cuándo ejecutar:** Cuando quieras evaluar un modelo específico que ya está registrado, independientemente del ciclo de entrenamiento.
    * **Comando Ejemplo:** `python gcp/08_evaluate_model.py --model_id "projects/PROJECT_ID/locations/REGION/models/MODEL_NUMERIC_ID" --data_path "ruta/a/tus/datos_procesados.npz"`
    * **Nota:** Para una evaluación más integrada en MLOps, el pipeline (`07`) es preferible. Este script es más para evaluaciones ad-hoc.

9.  **`gcp/09_deploy_model.py`**
    * **Propósito:** Despliega un modelo específico desde Vertex AI Model Registry a un Vertex AI Endpoint, haciéndolo disponible para servir predicciones en tiempo real.
    * **Cuándo ejecutar:** Cuando tienes un modelo registrado y evaluado que consideras listo para ser usado para inferencia (ya sea para paper trading o una futura operativa en vivo).
    * **Comando Ejemplo:** `python gcp/09_deploy_model.py --model_id "projects/PROJECT_ID/locations/REGION/models/MODEL_NUMERIC_ID"`
    * **Importante:** Necesitarás un contenedor de predicción personalizado para que esto funcione correctamente con tu modelo SB3/PyTorch.

### Opción B: Flujo Automatizado con Pipeline (Práctica Recomendada para MLOps)

7.  **`gcp/07_create_training_pipeline.py`**
    * **Propósito:** Define y ejecuta un pipeline completo en Vertex AI Pipelines. Este pipeline orquesta varios pasos: preprocesamiento de datos, entrenamiento del modelo, registro del modelo, evaluación del modelo y, opcionalmente, despliegue condicional a un endpoint.
    * **Cuándo ejecutar:** Este es el script principal para ejecutar tu ciclo de MLOps de forma regular y automatizada cada vez que quieras reentrenar y potencialmente desplegar un nuevo modelo.
    * **Comando Ejemplo:** `python gcp/07_create_training_pipeline.py --raw-data-bucket "tu-bucket-de-datos-crudos" --symbol BTCUSDT --deploy-model`
    * **Nota:** Si usas este script, **generalmente no necesitas ejecutar `06` ni `08` por separado**, ya que el pipeline se encarga de esas funcionalidades.

## Fase 4: Limpieza (Cuando sea necesario)

10. **`gcp/10_cleanup_resources.py`**
    * **Propósito:** Elimina los recursos de GCP creados por los scripts anteriores para evitar costos innecesarios. Permite eliminar todos los recursos o categorías específicas.
    * **Cuándo ejecutar:** Cuando quieras desmantelar el entorno o partes de él. Úsalo con precaución.
    * **Comando Ejemplo (para eliminar todo, requiere confirmación):** `python gcp/10_cleanup_resources.py --force` (sin `--force` pedirá confirmación).
    * **Comando Ejemplo (para eliminar solo endpoints):** `python gcp/10_cleanup_resources.py --resources endpoints`

## Flujo de Trabajo Típico para MLOps:

1.  **Configuración Inicial:** Ejecutar scripts de la Fase 1 (una vez).
2.  **Desarrollo/Actualización de Código:**
    * Modificar `src/` o `scripts/`.
    * Reconstruir imagen Docker con `gcp/04_build_docker_image.py`.
3.  **Adquisición de Nuevos Datos:** El servicio desplegado por `gcp/05_deploy_data_acquisition.py` se encarga de esto automáticamente según su programación. Puedes invocarlo manualmente si es necesario.
4.  **Reentrenamiento y Evaluación del Modelo:**
    * Ejecutar el pipeline completo con `gcp/07_create_training_pipeline.py`. Esto preprocesará los datos más recientes, entrenará, registrará y evaluará el nuevo modelo.
5.  **Despliegue del Modelo (si no se hizo condicionalmente en el pipeline):**
    * Si el pipeline no incluyó el despliegue o quieres desplegar manualmente una versión específica, usa `gcp/09_deploy_model.py`.
6.  **Bot de Ejecución en Vivo (Componente a Desarrollar):**
    * Un script/servicio separado (desplegado en Cloud Run o GCE) que consume predicciones del endpoint de Vertex AI para operar.

Esta guía debería ayudarte a navegar el proceso de despliegue en GCP. ¡Mucha suerte!