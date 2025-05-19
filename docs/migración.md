**Guía Detallada de Tareas para la Transformación del Proyecto `btcbot` a una Arquitectura Nativa de GCP**

**Objetivo General:** Refactorizar y ampliar el proyecto `btcbot` para que utilice BigQuery para el almacenamiento de datos, Secret Manager para la gestión de secretos, y se construya mediante Cloud Build en dos imágenes Docker (CPU y GPU) distintas, preparándolo para el entrenamiento y despliegue en Vertex AI.

**Fase 1: Adaptación del Código para Servicios GCP y `pandas-ta`**

1.  **Reemplazo de `TA-Lib` por `Pandas-TA` (Ya discutido, pero reconfirmar):**

      * **Archivo:** `src/data/feature_engineering.py`.
      * **Acción:** Asegúrate de que toda la lógica de cálculo de indicadores técnicos use exclusivamente `pandas-ta`. Actualiza los nombres de columna si es necesario para que coincidan con `final_market_feature_columns` en `src/data/preprocessing_config.yaml`.

2.  **Actualizar `requirements.txt`:**

      * **Archivo:** `requirements.txt`.
      * **Acción:**
          * Elimina `ta-lib==0.6.3`.
          * Añade `google-cloud-bigquery`, `google-cloud-secret-manager`, `db-dtypes` (para `pandas` con BigQuery).
          * Confirma que `pandas_ta` está presente.
          * Mantén `torch==2.2.2` para la imagen GPU. Considera si la imagen CPU necesita `torch` o si puede usar una versión CPU más ligera o incluso no tenerlo si ciertos scripts no lo usan. Por simplicidad inicial, puedes incluirlo en ambos, pero para optimización, la imagen CPU podría omitir `torch` si los scripts que ejecute no lo requieren (ej. descarga de datos, preprocesamiento si es solo Pandas/NumPy).

3.  **Integración con Google Secret Manager (en `ConfigManager`):**

      * **Archivo:** `src/utils/config.py`.
      * **Acción:** Modifica `ConfigManager` para que pueda obtener secretos de Google Secret Manager.
        ```python
        import yaml
        import os
        from dotenv import load_dotenv
        from google.cloud import secretmanager # ¡Añadir import!

        class ConfigManager:
            _instance = None
            _secret_client = None # Cliente de Secret Manager

            def __new__(cls, config_path="src/config.yaml", env_path=".env", gcp_project_id: str = None):
                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
                    # Carga de .env (útil para desarrollo local fuera de GCP)
                    load_dotenv(dotenv_path=env_path)

                    # Carga de config.yaml
                    try:
                        with open(config_path, 'r') as f:
                            cls._instance.config = yaml.safe_load(f)
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Archivo de configuración {config_path} no encontrado.")
                    except yaml.YAMLError as e:
                        raise yaml.YAMLError(f"Error al parsear el archivo YAML {config_path}: {e}")

                    # Inicializar cliente de Secret Manager si estamos en GCP o se proporciona project_id
                    # En GCP (Cloud Run, Vertex AI, GCF), el project_id a menudo se infiere.
                    cls._instance.gcp_project_id = gcp_project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
                    if cls._instance.gcp_project_id:
                        try:
                            cls._secret_client = secretmanager.SecretManagerServiceClient()
                            # logger.info("Secret Manager client inicializado.")
                        except Exception as e:
                            # logger.warning(f"No se pudo inicializar Secret Manager client: {e}. Los secretos de GCP no estarán disponibles.")
                            cls._secret_client = None
                    else:
                        # logger.info("No se proporcionó GCP Project ID. Secret Manager no se inicializará.")
                        pass

                return cls._instance

            def get_env_variable(self, var_name: str, default=None):
                # Prioriza variables de entorno (útil para overrides o local)
                value = os.getenv(var_name)
                if value is not None:
                    return value
                return default

            def get_secret(self, secret_id: str, version_id: str = "latest", default: str = None) -> str:
                """Obtiene un secreto de Google Secret Manager."""
                if not self._secret_client or not self.gcp_project_id:
                    # logger.warning(f"Secret Manager client no disponible. No se puede obtener el secreto '{secret_id}'. Usando default.")
                    return self.get_env_variable(secret_id, default) # Fallback a variable de entorno

                secret_name = f"projects/{self.gcp_project_id}/secrets/{secret_id}/versions/{version_id}"
                try:
                    response = self._secret_client.access_secret_version(name=secret_name)
                    return response.payload.data.decode("UTF-8")
                except Exception as e:
                    # logger.warning(f"No se pudo acceder al secreto '{secret_id}' (versión '{version_id}') desde Secret Manager: {e}. Usando default o variable de entorno.")
                    return self.get_env_variable(secret_id, default) # Fallback

            def get_config_value(self, key_path: str, default=None):
                # ... (sin cambios, sigue cargando de config.yaml) ...
                try:
                    keys = key_path.split('.')
                    value = self.config
                    for key in keys:
                        value = value[key]
                    return value
                except KeyError:
                    return default
                except TypeError:
                     raise TypeError(f"Configuración no cargada. Imposible obtener '{key_path}'.")
        ```
      * **Uso:** En `BinanceFuturesDownloader`, en lugar de `get_env_variable('BINANCE_API_KEY_FUTURES')`, usarías `get_secret('BINANCE_API_KEY_FUTURES_SECRET_NAME')`. Deberás crear estos secretos en Secret Manager en GCP. Los nombres de los secretos serían por ejemplo `binance_api_key` y `binance_api_secret`.

4.  **Adaptación de Módulos de Datos para BigQuery:**

      * **`src/data/binance_futures_downloader.py`:**

          * **Acción:** Modifica `Workspace_historical_data` para escribir el DataFrame directamente en una tabla de BigQuery en lugar de un CSV.
              * Necesitarás definir un esquema para tu tabla de datos crudos en BigQuery (ej. `open_time` TIMESTAMP, `open` FLOAT64, `high` FLOAT64, `low` FLOAT64, `close` FLOAT64, `volume` FLOAT64).
              * Usa `pandas_gbq.to_gbq()` o el cliente `google-cloud-bigquery` para cargar el DataFrame.
            <!-- end list -->
            ```python
            # Dentro de BinanceFuturesDownloader
            # ...
            from google.cloud import bigquery # Añadir import
            # ...
            # En __init__ o un método de configuración de BQ:
            # self.bq_client = bigquery.Client()
            # self.bq_dataset_id = self.config_manager.get_config_value('gcp.bigquery.raw_dataset_id', 'market_data_raw')
            # self.bq_raw_table_id_prefix = self.config_manager.get_config_value('gcp.bigquery.raw_table_id_prefix', 'klines_')

            # ... en fetch_historical_data, después de crear y limpiar el DataFrame 'df':
            # output_table_id = f"{self.bq_raw_table_id_prefix}{symbol}_{interval}"
            # full_table_id = f"{self.gcp_project_id}.{self.bq_dataset_id}.{output_table_id}"
            #
            # # Configurar job para BigQuery
            # job_config = bigquery.LoadJobConfig(
            #     schema=[ # Define tu esquema aquí o usa autodetección si es simple
            #         bigquery.SchemaField("Open_Time", "TIMESTAMP"),
            #         bigquery.SchemaField("Open", "FLOAT64"),
            #         # ... otras columnas
            #     ],
            #     write_disposition="WRITE_APPEND", # o WRITE_TRUNCATE si quieres reemplazar
            # )
            #
            # try:
            #     job = self.bq_client.load_table_from_dataframe(df, full_table_id, job_config=job_config)
            #     job.result()  # Esperar a que el job termine
            #     logger.info(f"Datos para {symbol} ({len(df)} velas) cargados exitosamente en BigQuery: {full_table_id}")
            # except Exception as e:
            #     logger.error(f"Error al cargar datos a BigQuery {full_table_id}: {e}")
            ```
          * **Actualizar `src/config.yaml`** con configuraciones de BigQuery (ID de proyecto, ID de dataset, prefijos de tabla).
            ```yaml
            gcp:
              project_id: "tu-gcp-project-id" # Puede ser inferido en GCP
              bigquery:
                raw_dataset_id: "market_data_raw"
                processed_dataset_id: "market_data_processed"
                raw_table_id_prefix: "klines_"
                processed_table_id_prefix: "features_"
            # ...
            ```

      * **`src/data/preprocessor.py`:**

          * **Acción:**
              * Modifica `_load_and_prepare_base_df` para leer datos desde la tabla de BigQuery creada por el `downloader`.
                ```python
                # Dentro de DataPreprocessor, en _load_and_prepare_base_df
                # ...
                # raw_table_id = f"{self.gcfg.get_config_value('gcp.bigquery.raw_table_id_prefix')}{symbol}_{interval}" # Necesitarás pasar symbol e interval
                # full_raw_table_id = f"{self.gcfg.get_config_value('gcp.project_id')}.{self.gcfg.get_config_value('gcp.bigquery.raw_dataset_id')}.{raw_table_id}"
                # query = f"SELECT * FROM `{full_raw_table_id}` ORDER BY Open_Time" # Añadir filtros de fecha si es necesario
                #
                # try:
                #     df = self.bq_client.query(query).to_dataframe()
                #     df['Open_Time'] = pd.to_datetime(df['Open_Time'], utc=True)
                #     df.set_index('Open_Time', inplace=True)
                #     # ... resto de la preparación ...
                # except Exception as e:
                #    logger.error(f"Error leyendo datos desde BigQuery {full_raw_table_id}: {e}"); raise
                ```
              * Modifica `process_data` para guardar las secuencias procesadas (`X_sequences`, `ts_sequences`, y las `close_prices`, `atr_values` que guardabas) en una nueva tabla de BigQuery (o GCS si prefieres `.npz`, pero para BigQuery tendrías que "aplanar" las secuencias o guardarlas como BYTES/JSON).
                  * Guardar secuencias 3D en BigQuery es menos directo. Opciones:
                    1.  **Aplanar:** Cada fila de BigQuery representa un paso de tiempo de la secuencia. Esto hace la tabla muy larga.
                    2.  **ARRAY de STRUCTs:** Representar la secuencia como un ARRAY de STRUCTs en una columna.
                    3.  **Como String/Bytes JSON:** Serializar la secuencia NumPy como JSON y guardarla en una columna STRING/BYTES.
                    4.  **GCS para secuencias:** Guardar los archivos `.npz` en GCS como antes, y en BigQuery solo almacenar *metadatos* o referencias a estos archivos. **Esta es a menudo la opción más práctica para arrays NumPy grandes y estructurados.**
                  * **Por simplicidad inicial y alineado con tu código, sigamos guardando los `.npz` en GCS, pero el `preprocessor.py` podría ser ejecutado por un job de Dataflow leyendo de BigQuery (crudo) y escribiendo a GCS (procesado).**

      * **`src/environments/trading_env.py`:**

          * **Acción:** Modifica `_load_market_data` para cargar los archivos `.npz` de secuencias procesadas desde un bucket de **Google Cloud Storage (GCS)** en lugar del sistema de archivos local.
            ```python
            # Dentro de TradingEnvironment
            # ...
            from google.cloud import storage # Añadir import
            import io
            # ...
            # En __init__ o un método de configuración:
            # self.gcs_client = storage.Client()
            # self.gcs_bucket_name = self.config_manager.get_config_value('gcp.gcs.processed_bucket_name', 'tu-bucket-datos-procesados')

            # En _load_market_data:
            # ...
            # data_dir = self.config['processed_data_directory'] # Ahora sería un prefijo de GCS
            # file_identifier = self.config['processed_data_file_identifier']
            #
            # bucket = self.gcs_client.bucket(self.gcs_bucket_name)
            # # Lógica para encontrar el blob correcto en GCS, ej. el más reciente o uno específico
            # target_blob_name = f"{data_dir}/{nombre_del_archivo_npz}" # Construir el path completo en GCS
            # blob = bucket.blob(target_blob_name)
            #
            # try:
            #     in_memory_file = io.BytesIO()
            #     blob.download_to_file(in_memory_file)
            #     in_memory_file.seek(0) # Rebobinar al inicio del stream
            #     data = np.load(in_memory_file)
            #     # ... resto del procesamiento de 'data' como lo tienes ...
            # except Exception as e:
            #     logger.error(f"Error cargando datos desde GCS gs://{self.gcs_bucket_name}/{target_blob_name}: {e}"); raise
            ```
          * **Actualizar `src/config.yaml` y `src/environments/environment_config.yaml`** para incluir nombres de buckets de GCS.

**Fase 2: Creación de Dos `Dockerfile` (CPU y GPU)**

1.  **`Dockerfile.gpu` (para entrenamiento):**

      * **Acción:** Similar al Dockerfile discutido previamente.
          * Imagen Base: `pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime` (o la más adecuada).
          * Instala todas las dependencias de `requirements.txt`, incluyendo `google-cloud-secret-manager`, `google-cloud-bigquery`, `pandas-gbq`, `google-cloud-storage`.

2.  **`Dockerfile.cpu` (para tareas no intensivas como descarga, preprocesamiento si no usa torch, evaluación CPU):**

      * **Acción:**
          * Imagen Base: `python:3.9-slim` (o la versión que uses).
          * Instala dependencias de `requirements.txt`, **pero puedes excluir `torch` si los scripts que esta imagen ejecutará no lo necesitan**, o instalar la versión CPU de PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`). Esto reducirá significativamente el tamaño de la imagen.
          * Incluye `google-cloud-secret-manager`, `google-cloud-bigquery`, `pandas-gbq`, `google-cloud-storage`.

3.  **`.dockerignore`:**

      * **Acción:** Crear un archivo `.dockerignore` como se discutió anteriormente para excluir archivos innecesarios.

**Fase 3: Configuración Avanzada de `cloudbuild.yaml` para Múltiples Imágenes**

1.  **Archivo `cloudbuild.yaml`:**
      * **Acción:** Modifica para construir y etiquetar ambas imágenes.
        ```yaml
        # cloudbuild.yaml
        steps:
        - name: 'gcr.io/cloud-builders/docker'
          id: 'build-gpu-image'
          args:
          - 'build'
          - '-t'
          - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO_NAME}/btcbot-gpu:${COMMIT_SHA}'
          - '-t'
          - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO_NAME}/btcbot-gpu:latest'
          - '.'
          - '-f' # Especificar el Dockerfile
          - 'Dockerfile.gpu'

        - name: 'gcr.io/cloud-builders/docker'
          id: 'build-cpu-image'
          args:
          - 'build'
          - '-t'
          - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO_NAME}/btcbot-cpu:${COMMIT_SHA}'
          - '-t'
          - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO_NAME}/btcbot-cpu:latest'
          - '.'
          - '-f' # Especificar el Dockerfile
          - 'Dockerfile.cpu'

        images:
        - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO_NAME}/btcbot-gpu:${COMMIT_SHA}'
        - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO_NAME}/btcbot-gpu:latest'
        - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO_NAME}/btcbot-cpu:${COMMIT_SHA}'
        - '${_REGION}-docker.pkg.dev/${PROJECT_ID}/${_REPO_NAME}/btcbot-cpu:latest'

        # Sustituciones (puedes definirlas aquí o pasarlas con gcloud builds submit)
        substitutions:
          _REGION: 'europe-west1' # Tu región de Artifact Registry
          _REPO_NAME: 'btcbot-images' # Tu nombre de repositorio en Artifact Registry
          # PROJECT_ID se infiere automáticamente por Cloud Build

        options:
          logging: CLOUD_LOGGING_ONLY
          # machineType: 'E2_HIGHCPU_8' # Para construcciones más rápidas
        ```

**Fase 4: Flujo de Trabajo y Scripts**

1.  **Scripts en `scripts/`:**

      * **`download_data.py`:** Se ejecutará utilizando la imagen CPU. Leerá configuración y secretos, descargará datos y los guardará en BigQuery.
      * **`preprocess_data.py`:**
          * Si el preprocesamiento es ligero (solo Pandas/NumPy), puede usar la imagen CPU. Leerá datos crudos de BigQuery y guardará los `.npz` procesados en GCS.
          * Si el preprocesamiento llegara a usar PyTorch para algo (poco probable para este caso), necesitaría la imagen GPU.
      * **`train_rl_agent.py`:** Se ejecutará con la imagen GPU en Vertex AI Training. Leerá datos procesados de GCS, guardará modelos en GCS.
      * **`evaluate_rl_agent.py`:** Puede usar la imagen CPU (si la predicción no es muy pesada o si `args.no_gpu` está activo) o la GPU. Leerá modelos de GCS y datos de GCS.
      * **`test_environment.py`:** Para pruebas rápidas, puede usar la imagen CPU.

2.  **Ejecución en GCP:**

      * **Descarga:** Podrías usar Cloud Scheduler para invocar una Cloud Function o un servicio de Cloud Run (con la imagen CPU) que ejecute `download_data.py`.
      * **Preprocesamiento:** Similarmente, Cloud Scheduler + Cloud Function/Run (imagen CPU) o un trabajo de Dataflow si se vuelve muy pesado.
      * **Entrenamiento:** Trabajos personalizados de Vertex AI Training (imagen GPU).
      * **Evaluación:** Trabajos personalizados de Vertex AI Training (imagen CPU o GPU).

**Resumen de Cambios Críticos en Código:**

  * `ConfigManager`: Añadir carga desde Secret Manager.
  * `BinanceFuturesDownloader`: Escribir a BigQuery, leer secretos.
  * `DataPreprocessor`: Leer de BigQuery (datos crudos), escribir a GCS (datos procesados .npz).
  * `TradingEnvironment`: Leer de GCS (datos procesados .npz).
  * `FeatureEngineer`: Usar `pandas-ta`.
  * `requirements.txt`: Actualizar con librerías de GCP.
  * Añadir `Dockerfile.cpu`, `Dockerfile.gpu`, `.dockerignore`, `cloudbuild.yaml`.

Este es un plan más ambicioso y te acerca mucho a una solución robusta en GCP. Como no puedes construir localmente, la depuración del Dockerfile se hará a través de iteraciones con Cloud Build, revisando los logs de construcción cuidadosamente.

**Mi recomendación para el orden de implementación:**

1.  **Modificaciones de Código Base:** `pandas-ta`, `requirements.txt`.
2.  **Modificaciones para GCP Services:** `ConfigManager` (Secret Manager), luego los módulos de datos para BigQuery y GCS. Empieza por hacer que la escritura/lectura funcione con los clientes de GCP.
3.  **Creación de los Dockerfiles:** `Dockerfile.gpu` primero, luego `Dockerfile.cpu`.
4.  **Configuración de `cloudbuild.yaml`** y primeras construcciones.

Esto es un trabajo considerable. Tómalo paso a paso, probando cada integración con GCP a medida que avanzas (ej. primero asegúrate de que puedes escribir/leer una tabla simple en BigQuery desde un script de Python local con las credenciales adecuadas antes de integrarlo en el bot).
