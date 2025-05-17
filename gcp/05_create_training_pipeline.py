#!/usr/bin/env python
"""
Script para crear y ejecutar un pipeline de entrenamiento en Vertex AI.
Este pipeline completo integra las funcionalidades de los scripts 06_launch_training_job.py y 08_evaluate_model.py
en un único flujo de trabajo automatizado y dinámico.

Este pipeline incluye los siguientes componentes:
1. Adquisición de datos - Descarga datos históricos de Binance
2. Preprocesamiento de datos - Prepara los datos históricos para el entrenamiento
3. Entrenamiento del modelo - Entrena el agente de RL con opciones para CPU o GPU
4. Registro del modelo en Vertex AI Model Registry - Registra el modelo con metad        output_dataset=acquire_data_task.outputs["output_dataset"]
    )
    
    # Paso 3: Entrenamiento del modelocompletos
5. Evaluación del modelo - Calcula métricas detalladas de rendimiento del agente
6. (Opcional) Despliegue condicional - Despliega el modelo solo si cumple criterios de calidad

    print(f"\n📊 Resumen de configuración del pipeline:")
        print(f"- Símbolo: {args.symbol}")
        print(f"- Timeframe: {args.timeframe}")
        print(f"- Lookback window: {args.lookback_window}")
        print(f"- Pasos de entrenamiento: {total_timesteps:,}")
        print(f"- Hardware de entrenamiento: {'🖥️ GPU ' + args.gpu_type + ' x' + str(args.gpu_count) if args.use_gpu else '💻 CPU'}")
        print(f"- Episodios de evaluación: {args.num_eval_episodes}")
        print(f"- Despliegue automático: {'✅ Sí' if args.deploy_model else '❌ No'}")ísticas dinámicas:
- Soporte para entrenamiento con GPU (configurable por tipo y cantidad)
- Múltiples métricas de evaluación (Sharpe, Sortino, Drawdown, Win Rate)
- Criterios flexibles para despliegue automático
- Opciones de hardware para el despliegue (CPU/GPU)
- Control de tráfico para despliegues canary
- Registro detallado de metadatos en cada paso
"""
import os
import argparse
import sys
from datetime import datetime

# Asegúrate de instalar estas dependencias antes de ejecutar
# pip install kfp google-cloud-aiplatform
try:
    import kfp
    from kfp.v2 import compiler
    from kfp.v2.dsl import pipeline, component, Input, Output, Artifact, Model, Dataset, Condition
    from google.cloud import aiplatform
except ImportError:
    print("Error: Se requieren las bibliotecas 'kfp' y 'google-cloud-aiplatform'")
    print("Instala las dependencias con: pip install kfp google-cloud-aiplatform")
    sys.exit(1)

from common.config import (
    PROJECT_ID, REGION, PROCESSED_DATA_BUCKET, MODELS_STAGING_BUCKET,
    EVALUATION_RESULTS_BUCKET, TRAINING_IMAGE_NAME, PREPROCESSING_IMAGE_NAME,
    RAW_DATA_BUCKET, TRAINING_JOB_NAME_PREFIX, PIPELINE_IMAGE_NAME,
    PIPELINE_NAME, PIPELINE_ROOT_BUCKET
)
from common.clients import get_aiplatform_client

# Definimos los componentes del pipeline utilizando los contenedores Docker personalizados

@component(
    base_image=PIPELINE_IMAGE_NAME,
    packages_to_install=["google-cloud-storage", "google-cloud-secretmanager", "python-binance"]
)
def acquire_data_component(
    project_id: str,
    raw_data_bucket: str,
    api_key_secret_name: str,
    api_secret_secret_name: str,
    output_dataset: Output[Dataset],
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start_date: str = "2020-01-01",
    output_gcs_prefix: str = None
):
    """
    Componente para descargar datos históricos de Binance y guardarlos en GCS.
    Este componente sustituye al servicio de adquisición de datos que antes
    se ejecutaba en Cloud Run.
    """
    import os
    import logging
    import sys
    from datetime import datetime

    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("acquire-data-component")
    
    # Añadir la ruta de la aplicación al PYTHONPATH
    sys.path.append("/app")
    
    # Importar el descargador de datos
    from src.data.binance_futures_downloader import BinanceFuturesDownloader
    
    logger.info(f"Iniciando descarga de datos para {symbol}, intervalo {interval}, desde {start_date}")
    
    # Crear el cliente de descarga
    downloader = BinanceFuturesDownloader(
        project_id=project_id,
        api_key_secret_name=api_key_secret_name,
        api_secret_secret_name=api_secret_secret_name,
        bucket_name=raw_data_bucket
    )
    
    # Descargar datos
    logger.info("Iniciando descarga de datos históricos...")
    output_uri = downloader.download_historical_data(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        gcs_prefix=output_gcs_prefix
    )
    
    logger.info(f"Datos descargados exitosamente en: {output_uri}")
    
    # Guardar la referencia del dataset para los siguientes componentes
    output_dataset.uri = output_uri
    output_dataset.metadata = {
        "symbol": symbol,
        "interval": interval,
        "start_date": start_date,
        "downloaded_at": datetime.now().isoformat()
    }
    
    return output_uri


@component(
    base_image=PREPROCESSING_IMAGE_NAME,
    packages_to_install=["google-cloud-storage"]
)
def preprocess_data(
    project_id: str,
    raw_data_bucket: str,
    processed_data_bucket: str,
    output_dataset: Output[Dataset],
    input_dataset: Input[Dataset] = None,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    lookback_window: int = 96
):
    """
    Componente para preprocesar datos descargados de Binance.
    """
    import os
    import tempfile
    from google.cloud import storage
    
    # Crear cliente de Storage
    storage_client = storage.Client(project=project_id)
    
    # Si se proporciona un dataset de entrada, obtener su URI
    input_uri = None
    if input_dataset and hasattr(input_dataset, 'uri') and input_dataset.uri:
        input_uri = input_dataset.uri
    
    # Descargar los archivos CSV más recientes del bucket de datos raw
    raw_bucket = storage_client.bucket(raw_data_bucket)
    
    # Buscar el archivo más reciente que coincida con el patrón o usar el proporcionado
    if input_uri:
        # Extraer el nombre del blob del URI de GCS
        bucket_name = input_uri.replace("gs://", "").split("/")[0]
        blob_name = input_uri.replace(f"gs://{bucket_name}/", "")
        if bucket_name == raw_data_bucket:
            latest_blob = raw_bucket.blob(blob_name)
        else:
            # Si está en un bucket diferente, listar por patrón
            blobs = list(raw_bucket.list_blobs(prefix=f"{symbol}_{timeframe}"))
            if not blobs:
                raise ValueError(f"No se encontraron archivos para {symbol}_{timeframe}")
            latest_blob = max(blobs, key=lambda x: x.name)
    else:
        # Buscar el archivo más reciente que coincida con el patrón
        blobs = list(raw_bucket.list_blobs(prefix=f"{symbol}_{timeframe}"))
        if not blobs:
            raise ValueError(f"No se encontraron archivos para {symbol}_{timeframe}")
        latest_blob = max(blobs, key=lambda x: x.name)
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    local_file_path = os.path.join(temp_dir, latest_blob.name)
    
    # Descargar archivo
    latest_blob.download_to_filename(local_file_path)
    
    # Preprocesar datos (aquí utilizaríamos tu lógica de preprocessor.py)
    # Por simplicidad, asumimos que el script de preprocesamiento está en el contenedor
    import sys
    sys.path.append("/app")
    from src.data.preprocessor import DataPreprocessor
    
    preprocessor = DataPreprocessor(config_path="/app/src/data/preprocessing_config.yaml")
    output_file = preprocessor.process_file(local_file_path, lookback_window=lookback_window)
    
    # Subir el resultado al bucket de datos procesados
    processed_bucket = storage_client.bucket(processed_data_bucket)
    output_blob_name = os.path.basename(output_file)
    blob = processed_bucket.blob(output_blob_name)
    blob.upload_from_filename(output_file)
    
    # Guardar la ubicación del dataset para el siguiente componente
    output_uri = f"gs://{processed_data_bucket}/{output_blob_name}"
    
    # Metadata para el componente de salida
    output_dataset.uri = output_uri
    output_dataset.metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "lookback_window": lookback_window
    }


@pipeline(
    name=TRAINING_JOB_NAME_PREFIX,
    description="Pipeline de entrenamiento completo para el agente de trading",
    pipeline_root=f"gs://{MODELS_STAGING_BUCKET}/pipelines"
)
def crypto_trading_pipeline(
    # Parámetros para descarga de datos
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    start_date: str = "2020-01-01",
    end_date: str = None,
    
    # Parámetros para el procesamiento de datos
    sequence_length: int = 96,
    norm_window_multiplier: int = 2,
    use_float32: bool = True,
    
    # Parámetros para el entrenamiento
    initial_equity: float = 10000.0,
    leverage: int = 1,
    position_size: float = 0.2,
    stop_loss: float = None,
    take_profit: float = None,
    trading_fees: float = 0.0004,
    
    # Parámetros del agente RL
    algorithm: str = "SAC",
    learning_rate: float = 0.0003,
    buffer_size: int = 100000,
    batch_size: int = 256,
    gamma: float = 0.99,
    total_timesteps: int = 500000,
    
    # Parámetros de evaluación y despliegue
    num_eval_episodes: int = 10,
    success_threshold_sharpe: float = 0.5,
    success_threshold_drawdown: float = 0.2,
    success_threshold_winrate: float = 0.5,
    auto_deploy: bool = False,
    
    # Secretos de Binance (nombres en Secret Manager)
    api_key_secret_name: str = "binance-api-key",
    api_secret_secret_name: str = "binance-api-secret"
):
    """
    Pipeline completo para entrenar un agente de trading con RL.
    
    Pasos:
    1. Descargar datos históricos de Binance
    2. Preprocesar datos para training y evaluación
    3. Entrenar modelo RL
    4. Evaluar modelo con métricas detalladas
    5. Desplegar modelo condicionalmente (si auto_deploy=True y supera umbrales de calidad)
    """
    # Crear instancias de los componentes
    download_comp = create_download_component()
    preprocess_comp = create_preprocess_component()
    train_comp = create_train_component()
    evaluate_comp = create_evaluation_component()
    deploy_comp = create_deploy_model_component()
    
    # Timestamp para identificar esta ejecución
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Paso 1: Descarga de datos históricos
    download_task = download_comp(
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        project_id=PROJECT_ID,
        raw_data_bucket=RAW_DATA_BUCKET,
        api_key_secret_name=api_key_secret_name,
        api_secret_secret_name=api_secret_secret_name
    )
    
    # Paso 2: Preprocesamiento de datos para training y evaluación
    train_data_output = f"{symbol}_{interval}_train_{timestamp}"
    eval_data_output = f"{symbol}_{interval}_eval_{timestamp}"
    
    # Procesar datos de entrenamiento (80% de los datos)
    preprocess_train_task = preprocess_comp(
        input_file_gcs=download_task.outputs["output_gcs_uri"],
        project_id=PROJECT_ID,
        raw_data_bucket=RAW_DATA_BUCKET,
        processed_data_bucket=PROCESSED_DATA_BUCKET,
        sequence_length=sequence_length,
        norm_window_multiplier=norm_window_multiplier,
        use_float32=use_float32,
        output_gcs_path=train_data_output,
        extra_metadata={"data_split": "train", "train_test_ratio": 0.8}
    )
    
    # Procesar datos de evaluación (20% restante)
    preprocess_eval_task = preprocess_comp(
        input_file_gcs=download_task.outputs["output_gcs_uri"],
        project_id=PROJECT_ID,
        raw_data_bucket=RAW_DATA_BUCKET,
        processed_data_bucket=PROCESSED_DATA_BUCKET,
        sequence_length=sequence_length,
        norm_window_multiplier=norm_window_multiplier,
        use_float32=use_float32,
        output_gcs_path=eval_data_output,
        extra_metadata={"data_split": "eval", "train_test_ratio": 0.2}
    )
    
    # Paso 3: Entrenar modelo
    model_output_path = f"{symbol}_{interval}_model_{timestamp}"
    
    train_task = train_comp(
        input_data_gcs=preprocess_train_task.outputs["output_npz_path"],
        output_model_gcs=f"{MODELS_STAGING_BUCKET}/{model_output_path}",
        project_id=PROJECT_ID,
        sequence_length=sequence_length,
        initial_equity=initial_equity,
        leverage=leverage,
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trading_fees=trading_fees,
        algorithm=algorithm,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        total_timesteps=total_timesteps,
        models_bucket=MODELS_STAGING_BUCKET
    )
    
    # Paso 4: Evaluar modelo
    evaluate_task = evaluate_comp(
        model_gcs_path=train_task.outputs["model_path"],
        test_data_gcs=preprocess_eval_task.outputs["output_npz_path"],
        project_id=PROJECT_ID,
        num_episodes=num_eval_episodes,
        sequence_length=sequence_length,
        initial_equity=initial_equity,
        leverage=leverage,
        position_size=position_size,
        trading_fees=trading_fees,
        evaluation_bucket=EVALUATION_RESULTS_BUCKET,
        success_threshold_sharpe=success_threshold_sharpe,
        success_threshold_drawdown=success_threshold_drawdown,
        success_threshold_winrate=success_threshold_winrate
    )
    
    # Paso 5: Desplegar modelo si cumple criterios de calidad (opcional)
    with Condition(
        auto_deploy == True,
        name="auto_deploy_condition"
    ) as deploy_condition:
        with Condition(
            evaluate_task.outputs["deploy_recommendation"] == True,
            name="quality_check_condition"
        ) as quality_condition:
            deploy_task = deploy_comp(
                model_gcs_path=train_task.outputs["serving_model_path"],
                project_id=PROJECT_ID,
                region=REGION,
                model_name=f"{symbol}_{interval}_agent_{timestamp}",
                model_display_name=f"Trading Agent {symbol} {interval}",
                machine_type="n1-standard-2"
            )
    
    # Outputs del pipeline
    return {
        "model_path": train_task.outputs["model_path"],
        "evaluation_metrics": evaluate_task.outputs["metrics"],
        "raw_data_path": download_task.outputs["output_gcs_uri"],
        "train_data_path": preprocess_train_task.outputs["output_npz_path"],
        "eval_data_path": preprocess_eval_task.outputs["output_npz_path"]
    }

# Definición de los componentes específicos para el pipeline
def create_download_component():
    """Creates and returns the data download component for the pipeline."""
    @component(
        base_image=PIPELINE_IMAGE_NAME,
        packages_to_install=["google-cloud-storage", "google-cloud-secretmanager", "python-binance"]
    )
    def download_data(
        symbol: str,
        interval: str,
        start_date: str,
        project_id: str,
        raw_data_bucket: str,
        end_date: str = None,
        api_key_secret_name: str = "binance-api-key",
        api_secret_secret_name: str = "binance-api-secret",
        output_gcs_uri: Output[str] = None
    ):
        """
        Component to download historical data from Binance and save it to GCS.
        """
        import os
        import sys
        import logging
        from datetime import datetime
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("download-component")
        
        # Add app to python path
        sys.path.append("/app")
        
        # Import the Binance downloader
        try:
            from src.data.binance_futures_downloader import BinanceFuturesDownloader
            logger.info("Imported BinanceFuturesDownloader successfully")
        except ImportError as e:
            logger.error(f"Failed to import BinanceFuturesDownloader: {e}")
            raise
        
        try:
            from google.cloud import secretmanager
            
            # Access Binance API credentials from Secret Manager
            client = secretmanager.SecretManagerServiceClient()
            
            api_key_name = f"projects/{project_id}/secrets/{api_key_secret_name}/versions/latest"
            api_secret_name = f"projects/{project_id}/secrets/{api_secret_secret_name}/versions/latest"
            
            api_key_response = client.access_secret_version(request={"name": api_key_name})
            api_secret_response = client.access_secret_version(request={"name": api_secret_name})
            
            api_key = api_key_response.payload.data.decode("UTF-8")
            api_secret = api_secret_response.payload.data.decode("UTF-8")
            
            # Create the Binance downloader
            downloader = BinanceFuturesDownloader(
                api_key=api_key,
                api_secret=api_secret,
                gcs_bucket=raw_data_bucket,
                project_id=project_id
            )
            
            # Download historical data
            logger.info(f"Downloading {symbol} {interval} data from {start_date} to {end_date or 'now'}")
            file_uri = downloader.fetch_historical_data(
                symbol=symbol,
                interval=interval,
                start_date_str=start_date,
                end_date_str=end_date,
                save_to_gcs=True
            )
            
            if output_gcs_uri:
                output_gcs_uri.value = file_uri
                
            logger.info(f"Data downloaded successfully to {file_uri}")
            return file_uri
            
        except Exception as e:
            logger.error(f"Error in download component: {e}")
            raise
    
    return download_data

def create_preprocess_component():
    """Creates and returns the data preprocessing component for the pipeline."""
    @component(
        base_image=PIPELINE_IMAGE_NAME,
        packages_to_install=["google-cloud-storage", "pandas", "numpy", "scikit-learn"]
    )
    def preprocess_data(
        input_file_gcs: str,
        project_id: str,
        raw_data_bucket: str,
        processed_data_bucket: str, 
        sequence_length: int = 96,
        norm_window_multiplier: int = 2,
        use_float32: bool = True,
        output_gcs_path: str = None,
        extra_metadata: dict = None,
        output_npz_path: Output[str] = None
    ):
        """
        Component to preprocess downloaded data for the RL agent.
        """
        import os
        import sys
        import logging
        import tempfile
        from datetime import datetime
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("preprocess-component")
        
        # Add app to python path
        sys.path.append("/app")
        
        try:
            from src.data.preprocessor import DataPreprocessor
            from google.cloud import storage
            
            # Create GCS client
            storage_client = storage.Client(project=project_id)
            
            # Parse file path
            if input_file_gcs.startswith("gs://"):
                bucket_name = input_file_gcs.replace("gs://", "").split("/")[0]
                blob_path = "/".join(input_file_gcs.replace(f"gs://{bucket_name}/", "").split("/"))
            else:
                # Assume it's a direct blob path in the raw data bucket
                bucket_name = raw_data_bucket
                blob_path = input_file_gcs
            
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            input_file_local = os.path.join(temp_dir, os.path.basename(blob_path))
            
            # Download file
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            logger.info(f"Downloading {blob_path} to {input_file_local}")
            blob.download_to_filename(input_file_local)
            
            # Instantiate preprocessor
            preprocessor = DataPreprocessor()
            
            # Generate output filename
            if not output_gcs_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_base = os.path.basename(input_file_local).split(".")[0]
                output_filename_base = f"{filename_base}_seq{sequence_length}_{timestamp}"
            else:
                output_filename_base = output_gcs_path
            
            # Process data
            logger.info(f"Processing data with sequence length {sequence_length}")
            output_file_local = preprocessor.process_data(
                input_file_local,
                sequence_length=sequence_length,
                output_filename_base=output_filename_base,
                norm_window_multiplier=norm_window_multiplier,
                use_float32=use_float32,
                extra_metadata=extra_metadata
            )
            
            # Upload processed file to GCS
            processed_bucket = storage_client.bucket(processed_data_bucket)
            output_blob_name = os.path.basename(output_file_local)
            blob = processed_bucket.blob(output_blob_name)
            logger.info(f"Uploading processed data to gs://{processed_data_bucket}/{output_blob_name}")
            blob.upload_from_filename(output_file_local)
            
            output_uri = f"gs://{processed_data_bucket}/{output_blob_name}"
            
            if output_npz_path:
                output_npz_path.value = output_uri
                
            logger.info(f"Data preprocessed successfully and saved to {output_uri}")
            return output_uri
            
        except Exception as e:
            logger.error(f"Error in preprocess component: {e}")
            raise
    
    return preprocess_data

def create_train_component():
    """Creates and returns the model training component for the pipeline."""
    @component(
        base_image=PIPELINE_IMAGE_NAME,
        packages_to_install=["google-cloud-storage", "numpy", "tensorflow", "stable-baselines3", "gym"]
    )
    def train_model(
        input_data_gcs: str,
        output_model_gcs: str,
        project_id: str,
        sequence_length: int = 96,
        initial_equity: float = 10000.0,
        leverage: int = 1,
        position_size: float = 0.2,
        stop_loss: float = None,
        take_profit: float = None,
        trading_fees: float = 0.0004,
        algorithm: str = "SAC",
        learning_rate: float = 0.0003,
        buffer_size: int = 100000,
        batch_size: int = 256,
        gamma: float = 0.99,
        total_timesteps: int = 500000,
        models_bucket: str = None,
        model_path: Output[str] = None,
        serving_model_path: Output[str] = None
    ):
        """
        Component to train an RL agent for trading.
        """
        import os
        import sys
        import logging
        import tempfile
        from datetime import datetime
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("train-component")
        
        # Add app to python path
        sys.path.append("/app")
        
        try:
            from src.agent.rl_agent_manager import RLAgentManager
            from src.environments.trading_env import TradingEnvironment
            from google.cloud import storage
            
            # Create GCS client
            storage_client = storage.Client(project=project_id)
            
            # Parse file path
            if input_data_gcs.startswith("gs://"):
                bucket_name = input_data_gcs.replace("gs://", "").split("/")[0]
                blob_path = "/".join(input_data_gcs.replace(f"gs://{bucket_name}/", "").split("/"))
            else:
                raise ValueError(f"Invalid GCS path: {input_data_gcs}")
            
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            input_file_local = os.path.join(temp_dir, os.path.basename(blob_path))
            
            # Download file
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            logger.info(f"Downloading {blob_path} to {input_file_local}")
            blob.download_to_filename(input_file_local)
            
            # Create environment
            logger.info("Creating trading environment")
            env = TradingEnvironment(
                data_path=input_file_local,
                sequence_length=sequence_length,
                initial_balance=initial_equity,
                trading_fee=trading_fees,
                position_size=position_size,
                leverage=leverage,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Create RL agent
            logger.info(f"Creating {algorithm} agent")
            agent_manager = RLAgentManager(algorithm=algorithm)
            
            # Train agent
            logger.info(f"Training agent for {total_timesteps:,} timesteps")
            agent = agent_manager.train(
                env=env, 
                total_timesteps=total_timesteps,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                batch_size=batch_size,
                gamma=gamma
            )
            
            # Save model locally
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{algorithm}_model_{timestamp}"
            local_model_path = os.path.join(temp_dir, model_name)
            logger.info(f"Saving model to {local_model_path}")
            agent.save(local_model_path)
            
            # Save model to GCS
            if not models_bucket and not output_model_gcs:
                raise ValueError("Either models_bucket or output_model_gcs must be provided")
            
            if output_model_gcs:
                if output_model_gcs.startswith("gs://"):
                    gcs_bucket_name = output_model_gcs.replace("gs://", "").split("/")[0]
                    gcs_path_prefix = "/".join(output_model_gcs.replace(f"gs://{gcs_bucket_name}/", "").split("/"))
                else:
                    gcs_bucket_name = models_bucket
                    gcs_path_prefix = output_model_gcs
            else:
                gcs_bucket_name = models_bucket
                gcs_path_prefix = model_name
            
            gcs_model_path = f"{gcs_path_prefix}/{model_name}.zip"
            
            # Upload model to GCS
            model_bucket = storage_client.bucket(gcs_bucket_name)
            model_blob = model_bucket.blob(gcs_model_path)
            model_blob.upload_from_filename(f"{local_model_path}.zip")
            
            # Create path for serving model
            serving_path = f"gs://{gcs_bucket_name}/{gcs_path_prefix}/serving"
            
            # Export model for serving
            agent_manager.export_model_for_serving(agent, temp_dir, serving_path, storage_client)
            
            # Set outputs
            output_model_uri = f"gs://{gcs_bucket_name}/{gcs_model_path}"
            if model_path:
                model_path.value = output_model_uri
            
            if serving_model_path:
                serving_model_path.value = serving_path
                
            logger.info(f"Model trained successfully and saved to {output_model_uri}")
            logger.info(f"Serving model exported to {serving_path}")
            
            return output_model_uri
            
        except Exception as e:
            logger.error(f"Error in training component: {e}")
            raise
    
    return train_model

def create_evaluation_component():
    """Creates and returns the model evaluation component for the pipeline."""
    @component(
        base_image=PIPELINE_IMAGE_NAME,
        packages_to_install=["google-cloud-storage", "numpy", "pandas", "tensorflow", "stable-baselines3", "gym", "matplotlib"]
    )
    def evaluate_model(
        model_gcs_path: str,
        test_data_gcs: str,
        project_id: str,
        num_episodes: int = 10,
        sequence_length: int = 96,
        initial_equity: float = 10000.0,
        leverage: int = 1,
        position_size: float = 0.2,
        trading_fees: float = 0.0004,
        evaluation_bucket: str = None,
        success_threshold_sharpe: float = 0.5,
        success_threshold_drawdown: float = 0.2,
        success_threshold_winrate: float = 0.5,
        metrics: Output[dict] = None,
        deploy_recommendation: Output[bool] = None
    ):
        """
        Component to evaluate a trained RL agent.
        """
        import os
        import sys
        import json
        import logging
        import tempfile
        import numpy as np
        import pandas as pd
        from datetime import datetime
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("evaluate-component")
        
        # Add app to python path
        sys.path.append("/app")
        
        try:
            from src.agent.rl_agent_manager import RLAgentManager
            from src.environments.trading_env import TradingEnvironment
            from google.cloud import storage
            
            # Create GCS client
            storage_client = storage.Client(project=project_id)
            
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            
            # Helper function to download from GCS
            def download_from_gcs(gcs_path, local_filename):
                if gcs_path.startswith("gs://"):
                    bucket_name = gcs_path.replace("gs://", "").split("/")[0]
                    blob_path = "/".join(gcs_path.replace(f"gs://{bucket_name}/", "").split("/"))
                else:
                    raise ValueError(f"Invalid GCS path: {gcs_path}")
                
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                local_path = os.path.join(temp_dir, local_filename)
                logger.info(f"Downloading {blob_path} to {local_path}")
                blob.download_to_filename(local_path)
                return local_path
            
            # Download test data
            test_data_local = download_from_gcs(
                test_data_gcs, 
                os.path.basename(test_data_gcs.split("/")[-1])
            )
            
            # Download model
            model_local = download_from_gcs(
                model_gcs_path,
                os.path.basename(model_gcs_path.split("/")[-1])
            )
            
            # Create environment
            logger.info("Creating evaluation environment")
            env = TradingEnvironment(
                data_path=test_data_local,
                sequence_length=sequence_length,
                initial_balance=initial_equity,
                trading_fee=trading_fees,
                position_size=position_size,
                leverage=leverage
            )
            
            # Load model
            logger.info("Loading model")
            agent_manager = RLAgentManager()
            agent = agent_manager.load_model(model_local)
            
            # Evaluate model
            logger.info(f"Evaluating model over {num_episodes} episodes")
            evaluation_results = agent_manager.evaluate(
                agent=agent,
                env=env,
                n_episodes=num_episodes
            )
            
            # Calculate key metrics
            sharpe_ratio = evaluation_results.get('sharpe_ratio', 0)
            sortino_ratio = evaluation_results.get('sortino_ratio', 0)
            max_drawdown = evaluation_results.get('max_drawdown', 1)
            win_rate = evaluation_results.get('win_rate', 0)
            
            # Determine if model meets deployment criteria
            meets_criteria = (
                sharpe_ratio >= success_threshold_sharpe and
                max_drawdown <= success_threshold_drawdown and
                win_rate >= success_threshold_winrate
            )
            
            # Save evaluation results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"evaluation_results_{timestamp}.json"
            local_results_path = os.path.join(temp_dir, results_filename)
            
            with open(local_results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            # Upload results to GCS
            if evaluation_bucket:
                results_bucket = storage_client.bucket(evaluation_bucket)
                results_blob = results_bucket.blob(results_filename)
                results_blob.upload_from_filename(local_results_path)
                logger.info(f"Evaluation results saved to gs://{evaluation_bucket}/{results_filename}")
            
            # Set outputs
            if metrics:
                metrics.value = evaluation_results
            
            if deploy_recommendation:
                deploy_recommendation.value = meets_criteria
            
            logger.info(f"Evaluation completed with metrics: Sharpe={sharpe_ratio:.2f}, Sortino={sortino_ratio:.2f}, " 
                        f"Max Drawdown={max_drawdown:.2f}, Win Rate={win_rate:.2f}")
            logger.info(f"Deployment recommendation: {'Yes' if meets_criteria else 'No'}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in evaluation component: {e}")
            raise
    
    return evaluate_model

def create_deploy_model_component():
    """Creates and returns the model deployment component for the pipeline."""
    @component(
        base_image=PIPELINE_IMAGE_NAME,
        packages_to_install=["google-cloud-storage", "google-cloud-aiplatform"]
    )
    def deploy_model(
        model_gcs_path: str,
        project_id: str,
        region: str,
        model_name: str,
        model_display_name: str = None,
        machine_type: str = "n1-standard-2",
        min_replica_count: int = 1,
        max_replica_count: int = 1,
        endpoint_id: Output[str] = None
    ):
        """
        Component to deploy a trained model to Vertex AI.
        """
        import os
        import sys
        import logging
        from datetime import datetime
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("deploy-component")
        
        try:
            from google.cloud import aiplatform
            
            # Initialize Vertex AI
            logger.info(f"Initializing Vertex AI client for project {project_id} in {region}")
            aiplatform.init(project=project_id, location=region)
            
            # Get or create an endpoint
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{model_display_name}-endpoint"',
                order_by='create_time desc',
                project=project_id, 
                location=region
            )
            
            if endpoints:
                endpoint = endpoints[0]
                logger.info(f"Using existing endpoint: {endpoint.name}")
            else:
                logger.info(f"Creating new endpoint for {model_display_name}")
                endpoint = aiplatform.Endpoint.create(
                    display_name=f"{model_display_name}-endpoint",
                    project=project_id,
                    location=region
                )
            
            # Upload and deploy the model
            logger.info(f"Uploading model from {model_gcs_path}")
            
            model = aiplatform.Model.upload(
                display_name=model_display_name,
                artifact_uri=model_gcs_path,
                serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest",
                project=project_id,
                location=region
            )
            
            logger.info(f"Deploying model {model.display_name} to endpoint {endpoint.name}")
            
            deployment = model.deploy(
                endpoint=endpoint,
                machine_type=machine_type,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
                project=project_id,
                location=region
            )
            
            # Set output
            if endpoint_id:
                endpoint_id.value = endpoint.name
            
            logger.info(f"Model deployed successfully to endpoint {endpoint.name}")
            return endpoint.name
            
        except Exception as e:
            logger.error(f"Error in deployment component: {e}")
            raise
    
    return deploy_model

# Main execution function
def run_pipeline(args):
    """Run the Vertex AI Pipeline with the specified arguments."""
    # Initialize the Vertex AI client
    aiplatform_client = get_aiplatform_client()
    
    # Compile the pipeline
    pipeline_filename = f"{PIPELINE_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    compiler.Compiler().compile(
        pipeline_func=crypto_trading_pipeline,
        package_path=pipeline_filename
    )
    
    # Create a pipeline job from the compiled pipeline
    pipeline_job = aiplatform_client.PipelineJob.create(
        display_name=PIPELINE_NAME,
        template_path=pipeline_filename,
        pipeline_root=PIPELINE_ROOT_BUCKET,
        parameter_values={
            "symbol": args.symbol,
            "interval": args.timeframe,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "sequence_length": args.lookback_window,
            "total_timesteps": args.total_timesteps,
            "num_eval_episodes": args.num_eval_episodes,
            "auto_deploy": args.deploy_model,
            "api_key_secret_name": args.api_key_secret_name,
            "api_secret_secret_name": args.api_secret_secret_name
        },
        enable_caching=args.enable_caching
    )
    
    # Submit the pipeline job
    pipeline_job.submit()
    print(f"Pipeline job '{pipeline_job.display_name}' submitted successfully.")
    print(f"You can monitor the pipeline at:")
    print(f"https://console.cloud.google.com/vertex-ai/locations/{REGION}/pipelines/runs/{pipeline_job.name}")
    
    if args.wait_for_completion:
        print("Waiting for pipeline completion...")
        pipeline_job.wait()
        print(f"Pipeline job completed with state: {pipeline_job.state}")

if __name__ == "__main__":
    """
    Script principal para crear y ejecutar un pipeline de entrenamiento en Vertex AI.
    
    Ejemplo de uso:
    python 05_create_training_pipeline.py --symbol BTCUSDT --timeframe 1h --lookback_window 96 --total_timesteps 200000
    """
    parser = argparse.ArgumentParser(description="Crea y ejecuta un pipeline de entrenamiento para el agente de trading")
    
    # Parámetros para la descarga de datos
    parser.add_argument("--symbol", default="BTCUSDT", help="Símbolo de criptomoneda a utilizar")
    parser.add_argument("--timeframe", default="1h", help="Intervalo de tiempo para los datos (ej. 1h, 4h, 1d)")
    parser.add_argument("--start_date", default="2020-01-01", help="Fecha de inicio para datos históricos (YYYY-MM-DD)")
    parser.add_argument("--end_date", default=None, help="Fecha fin para datos históricos (YYYY-MM-DD), opcional")
    
    # Parámetros para el procesamiento
    parser.add_argument("--lookback_window", type=int, default=96, help="Tamaño de la ventana de secuencia para el agente")
    
    # Parámetros para el entrenamiento
    parser.add_argument("--total_timesteps", type=int, default=500000, help="Pasos totales de entrenamiento")
    parser.add_argument("--algorithm", default="SAC", choices=["SAC", "PPO", "TD3"], help="Algoritmo RL a utilizar")
    
    # Parámetros para la evaluación
    parser.add_argument("--num_eval_episodes", type=int, default=10, help="Número de episodios para evaluación")
    
    # Parámetros para el despliegue
    parser.add_argument("--deploy_model", action="store_true", help="Desplegar modelo automáticamente si cumple criterios")
    
    # Parámetros de ejecución del pipeline
    parser.add_argument("--enable_caching", action="store_true", help="Habilitar caché para evitar repetir pasos")
    parser.add_argument("--wait_for_completion", action="store_true", help="Esperar a que el pipeline termine")
    
    # Parámetros de secretos
    parser.add_argument("--api_key_secret_name", default="binance-api-key", help="Nombre del secreto para API Key de Binance")
    parser.add_argument("--api_secret_secret_name", default="binance-api-secret", help="Nombre del secreto para API Secret de Binance")
    
    args = parser.parse_args()
    
    # Ejecutar el pipeline
    try:
        run_pipeline(args)
    except Exception as e:
        print(f"Error al ejecutar el pipeline: {e}")
        sys.exit(1)
