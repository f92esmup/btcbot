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
    from kfp.v2.dsl import pipeline, component, Input, Output, Artifact, Model, Dataset
    from google.cloud import aiplatform
except ImportError:
    print("Error: Se requieren las bibliotecas 'kfp' y 'google-cloud-aiplatform'")
    print("Instala las dependencias con: pip install kfp google-cloud-aiplatform")
    sys.exit(1)

from common.config import (
    PROJECT_ID, REGION, PROCESSED_DATA_BUCKET, MODELS_STAGING_BUCKET,
    EVALUATION_RESULTS_BUCKET, TRAINING_IMAGE_NAME, PREPROCESSING_IMAGE_NAME,
    RAW_DATA_BUCKET, TRAINING_JOB_NAME_PREFIX, PIPELINE_IMAGE_NAME
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
    from src.data.binance_futures_downloader_cloud import BinanceFuturesDownloaderCloud
    
    logger.info(f"Iniciando descarga de datos para {symbol}, intervalo {interval}, desde {start_date}")
    
    # Crear el cliente de descarga
    downloader = BinanceFuturesDownloaderCloud(
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


@component(
    base_image=TRAINING_IMAGE_NAME,
    packages_to_install=["google-cloud-storage", "stable-baselines3", "gymnasium", "tensorflow"]
)
def train_model(
    project_id: str,
    processed_data_uri: Input[Dataset],
    model_staging_bucket: str,
    input_dataset: Input[Dataset],
    output_model: Output[Model],
    total_timesteps: int = 1000000,
    use_gpu: bool = False,
    gpu_type: str = "NVIDIA_TESLA_T4",
    gpu_count: int = 1
):
    """
    Componente para entrenar el modelo de RL.
    """
    import os
    import tempfile
    from google.cloud import storage
    import uuid
    
    # Crear un ID único para este entrenamiento
    training_id = uuid.uuid4().hex[:8]
    
    # Crear cliente de Storage
    storage_client = storage.Client(project=project_id)
    
    # Descargar el dataset procesado - obtenemos la URI del input_dataset si es posible
    if hasattr(input_dataset, 'uri') and input_dataset.uri:
        uri_to_use = input_dataset.uri
    else:
        uri_to_use = processed_data_uri
        
    filename = os.path.basename(uri_to_use)
    local_file_path = os.path.join(tempfile.mkdtemp(), filename)
    
    bucket_name = uri_to_use.replace("gs://", "").split("/")[0]
    blob_name = uri_to_use.replace(f"gs://{bucket_name}/", "")
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_file_path)
    
    # Entrenar el modelo utilizando el script train_rl_agent.py
    import sys
    sys.path.append("/app")
    from src.agent.rl_agent_manager import RLAgentManager
    from src.environments.trading_env import TradingEnv
    
    # Crear el entorno y el agente
    env = TradingEnv(data_path=local_file_path, config_path="/app/src/environments/environment_config.yaml")
    
    # Configuración del agente con información de hardware
    agent_config = {
        "use_gpu": use_gpu,
        "gpu_type": gpu_type if use_gpu else None,
        "gpu_count": gpu_count if use_gpu else 0
    }
    
    agent_manager = RLAgentManager(config_path="/app/src/agent/agent_config.yaml", **agent_config)
    
    # Entrenar el modelo
    trained_model = agent_manager.train(env, total_timesteps=total_timesteps)
    
    # Guardar el modelo localmente
    model_filename = f"sac_transformer_trading_agent_{training_id}.zip"
    model_path = os.path.join(tempfile.mkdtemp(), model_filename)
    trained_model.save(model_path)
    
    # Subir el modelo al bucket de staging
    model_gcs_path = f"gs://{model_staging_bucket}/{model_filename}"
    staging_bucket = storage_client.bucket(model_staging_bucket)
    blob = staging_bucket.blob(model_filename)
    blob.upload_from_filename(model_path)
    
    # Metadata para el componente de salida
    output_model.uri = model_gcs_path
    output_model.metadata = {
        "framework": "stable-baselines3",
        "algorithm": "SAC",
        "training_steps": total_timesteps,
        "training_id": training_id,
        "use_gpu": use_gpu,
        "gpu_type": gpu_type if use_gpu else "none",
        "gpu_count": gpu_count if use_gpu else 0
    }


@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform"]
)
def register_model(
    project_id: str,
    region: str,
    model_artifact_uri: Input[Model],
    model_display_name: str,
    model_description: str,
    input_model: Input[Model],
    registered_model: Output[Model],
    serving_container_image_uri: str = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    labels: dict = None
):
    """
    Componente para registrar el modelo en Vertex AI Model Registry.
    """
    from google.cloud import aiplatform
    import json
    
    # Inicializar el cliente de Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Obtener el URI del modelo desde el artefacto de entrada
    uri = model_artifact_uri.uri if hasattr(model_artifact_uri, 'uri') else str(model_artifact_uri)
    
    # Obtener metadatos del modelo de entrada
    metadata = input_model.metadata.copy() if hasattr(input_model, "metadata") else {}
    
    # Crear etiquetas predeterminadas si no se proporcionan
    if labels is None:
        labels = {}
    
    # Añadir información básica a las etiquetas
    labels.update({
        "framework": metadata.get("framework", "stable-baselines3"),
        "algorithm": metadata.get("algorithm", "SAC"),
        "created_by": "vertex_pipeline"
    })
    
    # Registrar el modelo
    registered_model_obj = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=uri,
        serving_container_image_uri=serving_container_image_uri,
        description=model_description,
        labels=labels,
        metadata=metadata,
        sync=True
    )
    
    # Guardar el ID del modelo registrado
    registered_model.uri = uri
    registered_model.metadata = {
        "model_id": registered_model_obj.resource_name,
        "model_display_name": model_display_name,
        "framework": metadata.get("framework", "stable-baselines3"),
        "algorithm": metadata.get("algorithm", "SAC"),
        "training_steps": metadata.get("training_steps", 0),
        "training_id": metadata.get("training_id", ""),
        "use_gpu": metadata.get("use_gpu", False),
        "gpu_type": metadata.get("gpu_type", "none"),
        "gpu_count": metadata.get("gpu_count", 0),
        "registry_uri": f"https://console.cloud.google.com/vertex-ai/models/{registered_model_obj.name}?project={project_id}"
    }


@component(
    base_image=TRAINING_IMAGE_NAME,
    packages_to_install=["google-cloud-storage", "stable-baselines3", "gymnasium", "tensorflow"]
)
def evaluate_model(
    project_id: str,
    processed_data_uri: Input[Dataset],
    model_uri: Input[Model],
    evaluation_bucket: str,
    input_model: Input[Model],
    evaluation_metrics: Output[Artifact],
    num_episodes: int = 10
):
    """
    Componente para evaluar el modelo entrenado.
    """
    import os
    import json
    import tempfile
    from google.cloud import storage
    import uuid
    from datetime import datetime
    
    # Crear un ID único para esta evaluación
    eval_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear cliente de Storage
    storage_client = storage.Client(project=project_id)
    
    # Extraer URIs de los artefactos de entrada
    data_uri = processed_data_uri.uri if hasattr(processed_data_uri, 'uri') else str(processed_data_uri)
    model_uri_to_use = model_uri.uri if hasattr(model_uri, 'uri') else str(model_uri)
    
    # Descargar el dataset procesado
    data_filename = os.path.basename(data_uri)
    data_local_path = os.path.join(tempfile.mkdtemp(), data_filename)
    
    data_bucket_name = data_uri.replace("gs://", "").split("/")[0]
    data_blob_name = data_uri.replace(f"gs://{data_bucket_name}/", "")
    
    data_bucket = storage_client.bucket(data_bucket_name)
    data_blob = data_bucket.blob(data_blob_name)
    data_blob.download_to_filename(data_local_path)
    
    # Descargar el modelo
    model_filename = os.path.basename(model_uri_to_use)
    model_local_path = os.path.join(tempfile.mkdtemp(), model_filename)
    
    model_bucket_name = model_uri_to_use.replace("gs://", "").split("/")[0]
    model_blob_name = model_uri_to_use.replace(f"gs://{model_bucket_name}/", "")
    
    model_bucket = storage_client.bucket(model_bucket_name)
    model_blob = model_bucket.blob(model_blob_name)
    model_blob.download_to_filename(model_local_path)
    
    # Evaluar el modelo
    import sys
    sys.path.append("/app")
    from src.agent.rl_agent_manager import RLAgentManager
    from src.environments.trading_env import TradingEnv
    
    # Crear el entorno de evaluación con modo explícito de evaluación
    eval_env = TradingEnv(data_path=data_local_path, config_path="/app/src/environments/environment_config.yaml", mode="eval")
    
    # Cargar el modelo entrenado
    agent_manager = RLAgentManager(config_path="/app/src/agent/agent_config.yaml")
    model = agent_manager.load(model_local_path)
    
    # Evaluar el modelo con el número de episodios especificado
    evaluation_results = agent_manager.evaluate(model, eval_env, num_episodes=num_episodes)
    
    # Añadir información adicional a los resultados
    evaluation_results["evaluation_timestamp"] = timestamp
    evaluation_results["model_id"] = input_model.metadata.get("model_id", "")
    evaluation_results["training_id"] = input_model.metadata.get("training_id", "")
    
    # Guardar los resultados completos de evaluación
    eval_results_path = os.path.join(tempfile.mkdtemp(), f"evaluation_results_{eval_id}_{timestamp}.json")
    with open(eval_results_path, "w") as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Subir los resultados al bucket de evaluación
    eval_bucket = storage_client.bucket(evaluation_bucket)
    eval_blob_name = f"evaluation_results_{eval_id}_{timestamp}.json"
    eval_blob = eval_bucket.blob(eval_blob_name)
    eval_blob.upload_from_filename(eval_results_path)
    
    # Metadata para el componente de salida
    evaluation_uri = f"gs://{evaluation_bucket}/{eval_blob_name}"
    evaluation_metrics.uri = evaluation_uri
    evaluation_metrics.metadata = {
        "mean_reward": evaluation_results.get("mean_reward", 0),
        "std_reward": evaluation_results.get("std_reward", 0),
        "max_drawdown": evaluation_results.get("max_drawdown", 0),
        "sharpe_ratio": evaluation_results.get("sharpe_ratio", 0),
        "sortino_ratio": evaluation_results.get("sortino_ratio", 0),
        "win_rate": evaluation_results.get("win_rate", 0),
        "model_id": input_model.metadata.get("model_id", ""),
        "training_id": input_model.metadata.get("training_id", ""),
        "evaluation_id": eval_id,
        "evaluation_timestamp": timestamp,
        "num_episodes": num_episodes,
        "results_file": eval_blob_name
    }


@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform"]
)
def conditional_deployment(
    project_id: str,
    region: str,
    model_id: Input[Model],
    evaluation_metrics: Input[Artifact],
    deployed_model: Output[Model],
    min_sharpe_ratio: float = 0.5,
    min_sortino_ratio: float = 0.75,
    max_drawdown_threshold: float = -0.2,
    min_win_rate: float = 0.5,
    machine_type: str = "n1-standard-2",
    accelerator_type: str = None,
    accelerator_count: int = 0,
    traffic_percentage: int = 0,
    deploy_all: bool = False
):
    """
    Componente para desplegar condicionalmente el modelo si cumple con los criterios.
    """
    from google.cloud import aiplatform
    import json
    
    # Inicializar el cliente de Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Obtener métricas de evaluación
    sharpe_ratio = evaluation_metrics.metadata.get("sharpe_ratio", 0)
    sortino_ratio = evaluation_metrics.metadata.get("sortino_ratio", 0)
    max_drawdown = evaluation_metrics.metadata.get("max_drawdown", -1)
    win_rate = evaluation_metrics.metadata.get("win_rate", 0)
    
    # Verificar si el modelo cumple con los criterios
    meets_sharpe = sharpe_ratio >= min_sharpe_ratio
    meets_sortino = sortino_ratio >= min_sortino_ratio
    meets_drawdown = max_drawdown >= max_drawdown_threshold
    meets_win_rate = win_rate >= min_win_rate
    
    # Preparar registro de criterios
    deployment_criteria = {
        "sharpe_ratio": {"value": sharpe_ratio, "threshold": min_sharpe_ratio, "passed": meets_sharpe},
        "sortino_ratio": {"value": sortino_ratio, "threshold": min_sortino_ratio, "passed": meets_sortino},
        "max_drawdown": {"value": max_drawdown, "threshold": max_drawdown_threshold, "passed": meets_drawdown},
        "win_rate": {"value": win_rate, "threshold": min_win_rate, "passed": meets_win_rate}
    }
    
    # Determinar si se debe desplegar
    should_deploy = deploy_all or (meets_sharpe and meets_sortino and meets_drawdown and meets_win_rate)
    
    # Preparar configuración de acelerador (GPU) si se proporciona
    machine_spec = {"machine_type": machine_type}
    if accelerator_type and accelerator_count > 0:
        machine_spec["accelerator_type"] = accelerator_type
        machine_spec["accelerator_count"] = accelerator_count
    
    if should_deploy:
        # Obtener el modelo
        model = aiplatform.Model(model_id_str)
        
        # Crear un endpoint o usar uno existente
        endpoint_name = f"btcbot-agent-endpoint-{model.display_name.lower().replace('_', '-')}"
        
        try:
            # Intentar obtener un endpoint existente
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_name}"',
                order_by="create_time desc"
            )
            
            if endpoints:
                endpoint = endpoints[0]
                print(f"Usando endpoint existente: {endpoint.resource_name}")
            else:
                # Crear nuevo endpoint
                endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
                print(f"Creado nuevo endpoint: {endpoint.resource_name}")
            
            # Desplegar el modelo al endpoint
            deployment = model.deploy(
                endpoint=endpoint,
                deployed_model_display_name=model.display_name,
                machine_type=machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count,
                traffic_percentage=traffic_percentage,
                deploy_request_timeout=1800,
                sync=True
            )
            
            # Guardar información del despliegue
            deployed_model.uri = model.uri
            deployed_model.metadata = {
                "model_id": model_id,
                "endpoint_id": endpoint.resource_name,
                "endpoint_name": endpoint_name,
                "deployment_criteria": deployment_criteria,
                "evaluation_metrics": dict(evaluation_metrics.metadata),
                "deployed": True,
                "machine_type": machine_type,
                "accelerator_type": accelerator_type or "none",
                "accelerator_count": accelerator_count,
                "traffic_percentage": traffic_percentage
            }
            
        except Exception as e:
            # Registrar error pero no fallar el pipeline
            print(f"Error al desplegar modelo: {e}")
            deployed_model.uri = model.uri
            deployed_model.metadata = {
                "model_id": model_id,
                "deployment_criteria": deployment_criteria,
                "evaluation_metrics": dict(evaluation_metrics.metadata),
                "deployed": False,
                "error": str(e)
            }
    else:
        # No desplegar el modelo
        deployed_model.uri = ""
        deployed_model.metadata = {
            "model_id": model_id,
            "deployment_criteria": deployment_criteria,
            "evaluation_metrics": dict(evaluation_metrics.metadata),
            "deployed": False,
            "reason": "No cumple con los criterios de despliegue"
        }


# Definimos el pipeline completo
@pipeline(
    name="btcbot-training-pipeline",
    description="Pipeline completo para entrenar y registrar un modelo de RL para trading de BTC"
)
def btcbot_training_pipeline(
    project_id: str = PROJECT_ID,
    region: str = REGION,
    raw_data_bucket: str = RAW_DATA_BUCKET,
    processed_data_bucket: str = PROCESSED_DATA_BUCKET,
    model_staging_bucket: str = MODELS_STAGING_BUCKET,
    evaluation_bucket: str = EVALUATION_RESULTS_BUCKET,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    lookback_window: int = 96,
    total_timesteps: int = 1000000,
    num_eval_episodes: int = 10,
    use_gpu: bool = False,
    gpu_type: str = "NVIDIA_TESLA_T4",
    gpu_count: int = 1,
    min_sharpe_ratio: float = 0.5,
    min_sortino_ratio: float = 0.75,
    max_drawdown_threshold: float = -0.2,
    min_win_rate: float = 0.5,
    api_key_secret_name: str = "binance-api-key",
    api_secret_secret_name: str = "binance-api-secret",
    start_date: str = "2020-01-01",
    deploy_model: bool = False,
    deployment_machine_type: str = "n1-standard-2",
    deployment_accelerator_type: str = None,
    deployment_accelerator_count: int = 0,
    deployment_traffic_percentage: int = 0,
    serving_container_image_uri: str = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
):
    # Paso 1: Adquisición de datos
    acquire_data_task = acquire_data_component(
        project_id=project_id,
        raw_data_bucket=raw_data_bucket,
        api_key_secret_name=api_key_secret_name,
        api_secret_secret_name=api_secret_secret_name,
        symbol=symbol,
        interval=timeframe,
        start_date=start_date
    )
    
    # Paso 2: Preprocesamiento de datos
    preprocess_task = preprocess_data(
        project_id=project_id,
        raw_data_bucket=raw_data_bucket,
        processed_data_bucket=processed_data_bucket,
        symbol=symbol,
        timeframe=timeframe,
        lookback_window=lookback_window,
        input_dataset=acquire_data_task.outputs["output_dataset"]
    )
    
    
    # Paso 3: Entrenamiento del modelo
    train_task = train_model(
        project_id=project_id,
        processed_data_uri=preprocess_task.outputs["output_dataset"],
        model_staging_bucket=model_staging_bucket,
        total_timesteps=total_timesteps,
        input_dataset=preprocess_task.outputs["output_dataset"],
        use_gpu=use_gpu,
        gpu_type=gpu_type,
        gpu_count=gpu_count
    )
    
    # Paso 4: Registro del modelo en Vertex AI Model Registry
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_display_name = f"btcbot_trading_agent_{symbol}_{timeframe}_{timestamp}"
    model_description = f"Modelo de RL para trading de {symbol} con timeframe {timeframe}, {lookback_window} periodos de lookback y {total_timesteps} pasos de entrenamiento"
    
    register_task = register_model(
        project_id=project_id,
        region=region,
        model_artifact_uri=train_task.outputs["output_model"],
        model_display_name=model_display_name,
        model_description=model_description,
        input_model=train_task.outputs["output_model"],
        serving_container_image_uri=serving_container_image_uri,
        labels={
            "symbol": symbol,
            "timeframe": timeframe,
            "lookback_window": str(lookback_window),
            "total_timesteps": str(total_timesteps),
            "use_gpu": str(use_gpu).lower()
        }
    )
    
    # Paso 5: Evaluación del modelo
    evaluate_task = evaluate_model(
        project_id=project_id,
        processed_data_uri=preprocess_task.outputs["output_dataset"],
        model_uri=train_task.outputs["output_model"],
        evaluation_bucket=evaluation_bucket,
        input_model=register_task.outputs["registered_model"],
        num_episodes=num_eval_episodes
    )
    
    # Paso 6 (Opcional): Despliegue condicional
    if deploy_model:
        conditional_deployment(
            project_id=project_id,
            region=region,
            model_id=register_task.outputs["registered_model"],
            evaluation_metrics=evaluate_task.outputs["evaluation_metrics"],
            min_sharpe_ratio=min_sharpe_ratio,
            min_sortino_ratio=min_sortino_ratio,
            max_drawdown_threshold=max_drawdown_threshold,
            min_win_rate=min_win_rate,
            machine_type=deployment_machine_type,
            accelerator_type=deployment_accelerator_type,
            accelerator_count=deployment_accelerator_count,
            traffic_percentage=deployment_traffic_percentage,
            deploy_all=True  # Simplifying this parameter for now
        )


def main():
    """Función principal para compilar y ejecutar el pipeline."""
    parser = argparse.ArgumentParser(description="Crea y ejecuta un pipeline de entrenamiento en Vertex AI")
    
    # Opciones básicas
    parser.add_argument("--compile-only", action="store_true", 
                       help="Solo compilar el pipeline sin ejecutarlo")
    parser.add_argument("--raw-data-bucket", type=str, default=RAW_DATA_BUCKET, 
                       help=f"Bucket para datos raw (default: {RAW_DATA_BUCKET})")
    
    # Parámetros de datos
    data_group = parser.add_argument_group('Configuración de datos')
    data_group.add_argument("--symbol", type=str, default="BTCUSDT", 
                         help="Símbolo a procesar (default: BTCUSDT)")
    data_group.add_argument("--timeframe", type=str, default="1h", 
                         help="Timeframe a procesar (default: 1h)")
    data_group.add_argument("--lookback-window", type=int, default=96, 
                         help="Ventana de lookback (default: 96)")
    
    # Parámetros de entrenamiento
    training_group = parser.add_argument_group('Configuración de entrenamiento')
    training_group.add_argument("--total-timesteps", type=int, default=1000000, 
                             help="Pasos totales de entrenamiento (default: 1000000)")
    training_group.add_argument("--use-gpu", action="store_true", 
                             help="Utilizar GPU para el entrenamiento (IMPORTANTE: activar solo si se necesita)")
    training_group.add_argument("--gpu-type", type=str, default="NVIDIA_TESLA_T4", 
                             choices=["NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100", "NVIDIA_TESLA_P100", "NVIDIA_TESLA_P4", "NVIDIA_TESLA_K80"],
                             help="Tipo de GPU a utilizar (default: NVIDIA_TESLA_T4)")
    training_group.add_argument("--gpu-count", type=int, default=1, choices=range(1, 9),
                             help="Número de GPUs a utilizar, 1-8 (default: 1)")
    
    # Parámetros de evaluación
    eval_group = parser.add_argument_group('Configuración de evaluación')
    eval_group.add_argument("--num-eval-episodes", type=int, default=10, 
                         help="Número de episodios para evaluación (default: 10)")
    
    # Parámetros de despliegue
    deploy_group = parser.add_argument_group('Configuración de despliegue (opcional)')
    deploy_group.add_argument("--deploy-model", action="store_true", 
                           help="Desplegar el modelo si la evaluación es satisfactoria (default: False)")
    deploy_group.add_argument("--min-sharpe-ratio", type=float, default=0.5, 
                           help="Sharpe ratio mínimo para despliegue (default: 0.5)")
    deploy_group.add_argument("--min-sortino-ratio", type=float, default=0.75, 
                           help="Sortino ratio mínimo para despliegue (default: 0.75)")
    deploy_group.add_argument("--max-drawdown-threshold", type=float, default=-0.2, 
                           help="Drawdown máximo permitido para despliegue (default: -0.2)")
    deploy_group.add_argument("--min-win-rate", type=float, default=0.5, 
                           help="Win rate mínimo para despliegue (default: 0.5)")
    deploy_group.add_argument("--deployment-machine-type", type=str, default="n1-standard-2", 
                           help="Tipo de máquina para despliegue (default: n1-standard-2)")
    deploy_group.add_argument("--deployment-use-gpu", action="store_true", 
                           help="Utilizar GPU para el despliegue (IMPORTANTE: activar solo si se necesita)")
    deploy_group.add_argument("--deployment-gpu-type", type=str, default="NVIDIA_TESLA_T4",
                           choices=["NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100", "NVIDIA_TESLA_P100", "NVIDIA_TESLA_P4", "NVIDIA_TESLA_K80"],
                           help="Tipo de GPU a utilizar para despliegue (default: NVIDIA_TESLA_T4)")
    deploy_group.add_argument("--deployment-gpu-count", type=int, default=1, choices=range(1, 9),
                           help="Número de GPUs a utilizar para despliegue, 1-8 (default: 1)")
    deploy_group.add_argument("--deployment-traffic", type=int, default=0, 
                           help="Porcentaje de tráfico a dirigir al nuevo modelo desplegado, 0-100 (default: 0)")
    
    # Parámetros avanzados
    advanced_group = parser.add_argument_group('Configuración avanzada')
    advanced_group.add_argument("--serving-container", type=str, 
                             default="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
                             help="Imagen contenedora para servir el modelo (default: sklearn-cpu.1-0)")
    
    args = parser.parse_args()
    args.total_timesteps_provided = "--total-timesteps" in " ".join(sys.argv)
    
    # Advertencia sobre GPU
    if args.use_gpu:
        print(f"\n⚠️  ATENCIÓN: Has activado el uso de GPU ({args.gpu_type} x{args.gpu_count}) para el entrenamiento.")
        print("   Esto puede aumentar significativamente los costos. Asegúrate de que sea necesario.\n")
    
    if args.deploy_model and args.deployment_use_gpu:
        print(f"\n⚠️  ATENCIÓN: Has activado el uso de GPU ({args.deployment_gpu_type} x{args.deployment_gpu_count}) para el despliegue.")
        print("   Esto puede aumentar significativamente los costos mensuales. Asegúrate de que sea necesario.\n")
    
    # Ruta para el archivo del pipeline compilado
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_path = f"btcbot_training_pipeline_{timestamp}.json"
    
    # Compilar el pipeline
    compiler.Compiler().compile(
        pipeline_func=btcbot_training_pipeline,
        package_path=pipeline_path
    )
    
    print(f"Pipeline compilado en: {pipeline_path}")
    
    if not args.compile_only:
        # Inicializar cliente de Vertex AI
        aiplatform.init(project=PROJECT_ID, location=REGION)
        
        # Configurar aceleradores para despliegue si se requiere
        deployment_accelerator_type = args.deployment_gpu_type if args.deployment_use_gpu else None
        deployment_accelerator_count = args.deployment_gpu_count if args.deployment_use_gpu else 0
        
        # Obtener valores de variables de entorno para parámetros de RL si están definidos
        agent_learning_rate = os.getenv("AGENT_LEARNING_RATE")
        agent_buffer_size = os.getenv("AGENT_BUFFER_SIZE")
        agent_batch_size = os.getenv("AGENT_BATCH_SIZE")
        agent_gamma = os.getenv("AGENT_GAMMA")
        pipeline_total_timesteps = os.getenv("PIPELINE_TOTAL_TIMESTEPS")
        
        # Usar valores de línea de comandos o variables de entorno
        if pipeline_total_timesteps and not args.total_timesteps_provided:
            total_timesteps = int(pipeline_total_timesteps)
            print(f"Usando PIPELINE_TOTAL_TIMESTEPS de variables de entorno: {total_timesteps}")
        else:
            total_timesteps = args.total_timesteps
        
        # Parámetros del pipeline
        pipeline_params = {
            "project_id": PROJECT_ID,
            "region": REGION,
            "raw_data_bucket": args.raw_data_bucket or RAW_DATA_BUCKET,
            "processed_data_bucket": PROCESSED_DATA_BUCKET,  # Usar valor de config.py
            "model_staging_bucket": MODELS_STAGING_BUCKET,   # Usar valor de config.py
            "evaluation_bucket": EVALUATION_RESULTS_BUCKET,  # Usar valor de config.py
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "lookback_window": args.lookback_window,
            "total_timesteps": total_timesteps,
            "num_eval_episodes": args.num_eval_episodes,
            "use_gpu": args.use_gpu,
            "gpu_type": args.gpu_type,
            "gpu_count": args.gpu_count,
            "deploy_model": args.deploy_model,
            "min_sharpe_ratio": args.min_sharpe_ratio,
            "min_sortino_ratio": args.min_sortino_ratio,
            "max_drawdown_threshold": args.max_drawdown_threshold,
            "min_win_rate": args.min_win_rate,
            "deployment_machine_type": args.deployment_machine_type,
            "deployment_accelerator_type": deployment_accelerator_type,
            "deployment_accelerator_count": deployment_accelerator_count,
            "deployment_traffic_percentage": args.deployment_traffic,
            "serving_container_image_uri": args.serving_container
        }
        
        # Ejecutar el pipeline
        job_name = f"{TRAINING_JOB_NAME_PREFIX}-{args.symbol}-{args.timeframe}-{timestamp}"
        job = aiplatform.PipelineJob(
            display_name=job_name,
            template_path=pipeline_path,
            pipeline_root=f"gs://{MODELS_STAGING_BUCKET}/pipeline_root",
            parameter_values=pipeline_params
        )
        
        job.run(sync=True)
        print(f"Pipeline ejecutado con ID: {job.resource_name}")
        print(f"Ver en la consola: https://console.cloud.google.com/vertex-ai/pipelines/runs/{job.name}?project={PROJECT_ID}")
        
        # Mostrar resumen de configuración
        print("\n📊 Resumen de configuración del pipeline:")
        print(f"- Símbolo: {args.symbol}")
        print(f"- Timeframe: {args.timeframe}")
        print(f"- Lookback window: {args.lookback_window}")
        print(f"- Pasos de entrenamiento: {args.total_timesteps:,}")
        print(f"- Hardware de entrenamiento: {'🖥️ GPU ' + args.gpu_type + ' x' + str(args.gpu_count) if args.use_gpu else '💻 CPU'}")
        print(f"- Episodios de evaluación: {args.num_eval_episodes}")
        print(f"- Despliegue automático: {'✅ Sí' if args.deploy_model else '❌ No'}")
        if args.deploy_model:
            print(f"  - Criterios mínimos: Sharpe>{args.min_sharpe_ratio}, Sortino>{args.min_sortino_ratio}, DrawDown>{args.max_drawdown_threshold}, WinRate>{args.min_win_rate}")
            print(f"  - Hardware para despliegue: {args.deployment_machine_type} " + 
                 (f"con GPU {args.deployment_gpu_type} x{args.deployment_gpu_count}" if args.deployment_use_gpu else "sin GPU"))
            print(f"  - Tráfico asignado: {args.deployment_traffic}%")
if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                 BTCBOT - PIPELINE DE ENTRENAMIENTO EN VERTEX AI            ║
╠════════════════════════════════════════════════════════════════════════════╣
║ NOTA IMPORTANTE:                                                           ║
║ - Este script integra las funcionalidades de los scripts 06 y 08           ║
║ - Utiliza valores predeterminados de common/config.py cuando es posible    ║
║ - El uso de GPU es opcional pero requiere activación explícita             ║
║ - Para ver todas las opciones disponibles, use --help                      ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
    main()
