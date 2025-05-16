#!/usr/bin/env python
"""
Script para crear y ejecutar un pipeline de entrenamiento en Vertex AI.
Este pipeline incluye los siguientes componentes:
1. Preprocesamiento de datos
2. Entrenamiento del modelo
3. Registro del modelo en Vertex AI Model Registry
4. Evaluación del modelo
5. (Opcional) Despliegue condicional
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
    EVALUATION_RESULTS_BUCKET, TRAINING_IMAGE_NAME, PREPROCESSING_IMAGE_NAME
)
from common.clients import get_aiplatform_client

# Definimos los componentes del pipeline utilizando los contenedores Docker personalizados

@component(
    base_image=PREPROCESSING_IMAGE_NAME,
    packages_to_install=["google-cloud-storage"]
)
def preprocess_data(
    project_id: str,
    raw_data_bucket: str,
    processed_data_bucket: str,
    output_dataset: Output[Dataset],
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
    
    # Descargar los archivos CSV más recientes del bucket de datos raw
    raw_bucket = storage_client.bucket(raw_data_bucket)
    
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
    processed_data_uri: str,
    model_staging_bucket: str,
    input_dataset: Input[Dataset],
    output_model: Output[Model],
    total_timesteps: int = 1000000
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
    
    # Descargar el dataset procesado
    filename = os.path.basename(processed_data_uri)
    local_file_path = os.path.join(tempfile.mkdtemp(), filename)
    
    bucket_name = processed_data_uri.replace("gs://", "").split("/")[0]
    blob_name = processed_data_uri.replace(f"gs://{bucket_name}/", "")
    
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
    agent_manager = RLAgentManager(config_path="/app/src/agent/agent_config.yaml")
    
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
        "training_id": training_id
    }


@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform"]
)
def register_model(
    project_id: str,
    region: str,
    model_artifact_uri: str,
    model_display_name: str,
    model_description: str,
    input_model: Input[Model],
    registered_model: Output[Model]
):
    """
    Componente para registrar el modelo en Vertex AI Model Registry.
    """
    from google.cloud import aiplatform
    
    # Inicializar el cliente de Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Registrar el modelo
    registered_model_obj = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_artifact_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",  # Placeholder, se debe usar una imagen compatible con tu modelo
        description=model_description,
        metadata={
            "framework": input_model.metadata["framework"],
            "algorithm": input_model.metadata["algorithm"],
            "training_steps": input_model.metadata["training_steps"],
            "training_id": input_model.metadata["training_id"]
        }
    )
    
    # Guardar el ID del modelo registrado
    registered_model.uri = model_artifact_uri
    registered_model.metadata = {
        "model_id": registered_model_obj.resource_name,
        "model_display_name": model_display_name,
        "framework": input_model.metadata["framework"],
        "algorithm": input_model.metadata["algorithm"],
        "training_steps": input_model.metadata["training_steps"],
        "training_id": input_model.metadata["training_id"]
    }


@component(
    base_image=TRAINING_IMAGE_NAME,
    packages_to_install=["google-cloud-storage", "stable-baselines3", "gymnasium", "tensorflow"]
)
def evaluate_model(
    project_id: str,
    processed_data_uri: str,
    model_uri: str,
    evaluation_bucket: str,
    input_model: Input[Model],
    evaluation_metrics: Output[Artifact]
):
    """
    Componente para evaluar el modelo entrenado.
    """
    import os
    import json
    import tempfile
    from google.cloud import storage
    import uuid
    
    # Crear un ID único para esta evaluación
    eval_id = uuid.uuid4().hex[:8]
    
    # Crear cliente de Storage
    storage_client = storage.Client(project=project_id)
    
    # Descargar el dataset procesado
    data_filename = os.path.basename(processed_data_uri)
    data_local_path = os.path.join(tempfile.mkdtemp(), data_filename)
    
    data_bucket_name = processed_data_uri.replace("gs://", "").split("/")[0]
    data_blob_name = processed_data_uri.replace(f"gs://{data_bucket_name}/", "")
    
    data_bucket = storage_client.bucket(data_bucket_name)
    data_blob = data_bucket.blob(data_blob_name)
    data_blob.download_to_filename(data_local_path)
    
    # Descargar el modelo
    model_filename = os.path.basename(model_uri)
    model_local_path = os.path.join(tempfile.mkdtemp(), model_filename)
    
    model_bucket_name = model_uri.replace("gs://", "").split("/")[0]
    model_blob_name = model_uri.replace(f"gs://{model_bucket_name}/", "")
    
    model_bucket = storage_client.bucket(model_bucket_name)
    model_blob = model_bucket.blob(model_blob_name)
    model_blob.download_to_filename(model_local_path)
    
    # Evaluar el modelo
    import sys
    sys.path.append("/app")
    from src.agent.rl_agent_manager import RLAgentManager
    from src.environments.trading_env import TradingEnv
    
    # Crear el entorno de evaluación
    eval_env = TradingEnv(data_path=data_local_path, config_path="/app/src/environments/environment_config.yaml", mode="eval")
    
    # Cargar el modelo entrenado
    agent_manager = RLAgentManager(config_path="/app/src/agent/agent_config.yaml")
    model = agent_manager.load(model_local_path)
    
    # Evaluar el modelo
    evaluation_results = agent_manager.evaluate(model, eval_env, num_episodes=10)
    
    # Guardar los resultados de evaluación
    eval_results_path = os.path.join(tempfile.mkdtemp(), f"evaluation_results_{eval_id}.json")
    with open(eval_results_path, "w") as f:
        json.dump(evaluation_results, f)
    
    # Subir los resultados al bucket de evaluación
    eval_bucket = storage_client.bucket(evaluation_bucket)
    eval_blob = eval_bucket.blob(f"evaluation_results_{eval_id}.json")
    eval_blob.upload_from_filename(eval_results_path)
    
    # Metadata para el componente de salida
    evaluation_uri = f"gs://{evaluation_bucket}/evaluation_results_{eval_id}.json"
    evaluation_metrics.uri = evaluation_uri
    evaluation_metrics.metadata = {
        "mean_reward": evaluation_results.get("mean_reward", 0),
        "std_reward": evaluation_results.get("std_reward", 0),
        "max_drawdown": evaluation_results.get("max_drawdown", 0),
        "sharpe_ratio": evaluation_results.get("sharpe_ratio", 0),
        "model_id": input_model.metadata.get("model_id", ""),
        "training_id": input_model.metadata.get("training_id", "")
    }


@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform"]
)
def conditional_deployment(
    project_id: str,
    region: str,
    model_id: str,
    evaluation_metrics: Input[Artifact],
    deployed_model: Output[Model],
    min_sharpe_ratio: float = 0.5
):
    """
    Componente para desplegar condicionalmente el modelo si cumple con los criterios.
    """
    from google.cloud import aiplatform
    
    # Inicializar el cliente de Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Verificar si el modelo cumple con los criterios
    sharpe_ratio = evaluation_metrics.metadata.get("sharpe_ratio", 0)
    
    if sharpe_ratio >= min_sharpe_ratio:
        # Obtener el modelo
        model = aiplatform.Model(model_id)
        
        # Desplegar el modelo a un endpoint
        endpoint = model.deploy(
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=1,
            deploy_request_timeout=1800,
            sync=True
        )
        
        # Guardar información del despliegue
        deployed_model.uri = model.uri
        deployed_model.metadata = {
            "model_id": model_id,
            "endpoint_id": endpoint.resource_name,
            "sharpe_ratio": sharpe_ratio,
            "deployed": True
        }
    else:
        # No desplegar el modelo
        deployed_model.uri = ""
        deployed_model.metadata = {
            "model_id": model_id,
            "sharpe_ratio": sharpe_ratio,
            "deployed": False,
            "reason": f"Sharpe ratio {sharpe_ratio} below threshold {min_sharpe_ratio}"
        }


# Definimos el pipeline completo
@pipeline(
    name="btcbot-training-pipeline",
    description="Pipeline completo para entrenar y registrar un modelo de RL para trading de BTC"
)
def btcbot_training_pipeline(
    project_id: str = PROJECT_ID,
    region: str = REGION,
    raw_data_bucket: str = "",
    processed_data_bucket: str = PROCESSED_DATA_BUCKET,
    model_staging_bucket: str = MODELS_STAGING_BUCKET,
    evaluation_bucket: str = EVALUATION_RESULTS_BUCKET,
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    lookback_window: int = 96,
    total_timesteps: int = 1000000,
    min_sharpe_ratio: float = 0.5,
    deploy_model: bool = False
):
    # Paso 1: Preprocesamiento de datos
    preprocess_task = preprocess_data(
        project_id=project_id,
        raw_data_bucket=raw_data_bucket,
        processed_data_bucket=processed_data_bucket,
        symbol=symbol,
        timeframe=timeframe,
        lookback_window=lookback_window
    )
    
    # Paso 2: Entrenamiento del modelo
    train_task = train_model(
        project_id=project_id,
        processed_data_uri=preprocess_task.outputs["output_dataset"].uri,
        model_staging_bucket=model_staging_bucket,
        total_timesteps=total_timesteps,
        input_dataset=preprocess_task.outputs["output_dataset"]
    )
    
    # Paso 3: Registro del modelo en Vertex AI Model Registry
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_display_name = f"btcbot_trading_agent_{timestamp}"
    model_description = f"Modelo de RL para trading de {symbol} con {lookback_window} periodos de lookback"
    
    register_task = register_model(
        project_id=project_id,
        region=region,
        model_artifact_uri=train_task.outputs["output_model"].uri,
        model_display_name=model_display_name,
        model_description=model_description,
        input_model=train_task.outputs["output_model"]
    )
    
    # Paso 4: Evaluación del modelo
    evaluate_task = evaluate_model(
        project_id=project_id,
        processed_data_uri=preprocess_task.outputs["output_dataset"].uri,
        model_uri=train_task.outputs["output_model"].uri,
        evaluation_bucket=evaluation_bucket,
        input_model=register_task.outputs["registered_model"]
    )
    
    # Paso 5 (Opcional): Despliegue condicional
    if deploy_model:
        conditional_deployment(
            project_id=project_id,
            region=region,
            model_id=register_task.outputs["registered_model"].metadata["model_id"],
            evaluation_metrics=evaluate_task.outputs["evaluation_metrics"],
            min_sharpe_ratio=min_sharpe_ratio
        )


def main():
    """Función principal para compilar y ejecutar el pipeline."""
    parser = argparse.ArgumentParser(description="Crea y ejecuta un pipeline de entrenamiento en Vertex AI")
    parser.add_argument("--compile-only", action="store_true", help="Solo compilar el pipeline sin ejecutarlo")
    parser.add_argument("--raw-data-bucket", type=str, help="Bucket para datos raw", required=True)
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Símbolo a procesar")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe a procesar")
    parser.add_argument("--lookback-window", type=int, default=96, help="Ventana de lookback")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="Pasos totales de entrenamiento")
    parser.add_argument("--deploy-model", action="store_true", help="Desplegar el modelo si la evaluación es satisfactoria")
    parser.add_argument("--min-sharpe-ratio", type=float, default=0.5, help="Sharpe ratio mínimo para despliegue")
    args = parser.parse_args()
    
    # Ruta para el archivo del pipeline compilado
    pipeline_path = "btcbot_training_pipeline.json"
    
    # Compilar el pipeline
    compiler.Compiler().compile(
        pipeline_func=btcbot_training_pipeline,
        package_path=pipeline_path
    )
    
    print(f"Pipeline compilado en: {pipeline_path}")
    
    if not args.compile_only:
        # Inicializar cliente de Vertex AI
        aiplatform.init(project=PROJECT_ID, location=REGION)
        
        # Parámetros del pipeline
        pipeline_params = {
            "project_id": PROJECT_ID,
            "region": REGION,
            "raw_data_bucket": args.raw_data_bucket,
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "lookback_window": args.lookback_window,
            "total_timesteps": args.total_timesteps,
            "deploy_model": args.deploy_model,
            "min_sharpe_ratio": args.min_sharpe_ratio
        }
        
        # Ejecutar el pipeline
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job = aiplatform.PipelineJob(
            display_name=f"btcbot-training-{timestamp}",
            template_path=pipeline_path,
            pipeline_root=f"gs://{MODELS_STAGING_BUCKET}/pipeline_root",
            parameter_values=pipeline_params
        )
        
        job.run(sync=True)
        print(f"Pipeline ejecutado con ID: {job.resource_name}")


if __name__ == "__main__":
    main()
