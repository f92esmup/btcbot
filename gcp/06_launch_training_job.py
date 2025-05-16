"""
Script para lanzar un trabajo de entrenamiento en Vertex AI y registrar el modelo.
"""
import argparse
import os
import time
import uuid
from google.cloud import aiplatform
from common import config, clients

def launch_training_job(training_image_uri, output_dir, job_name_suffix=None, machine_type="n1-standard-4"):
    """
    Lanza un trabajo de entrenamiento personalizado en Vertex AI.
    
    Args:
        training_image_uri: URI de la imagen Docker para entrenamiento.
        output_dir: Directorio en GCS para guardar los artefactos del modelo.
        job_name_suffix: Sufijo opcional para el nombre del trabajo.
        machine_type: Tipo de máquina para el entrenamiento.
        
    Returns:
        CustomJobRunOp: El objeto del trabajo personalizado.
    """
    # Inicializar el cliente de AI Platform
    aiplatform_client = clients.get_aiplatform_client()
    
    # Generar un nombre para el trabajo
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    suffix = job_name_suffix or str(uuid.uuid4())[:8]
    job_name = f"{config.TRAINING_JOB_NAME_PREFIX}_{timestamp}_{suffix}"
    
    # Configurar el contenedor personalizado
    container_spec = {
        "image_uri": training_image_uri,
        "command": [],
        "args": [],
        "env": [
            {"name": "AIP_MODEL_DIR", "value": output_dir},
            {"name": "PYTHONPATH", "value": "/app"}
        ]
    }
    
    # Configurar el trabajo personalizado
    job = aiplatform.CustomJob(
        display_name=job_name,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": machine_type,
                    "accelerator_type": "ACCELERATOR_TYPE_UNSPECIFIED",
                    "accelerator_count": 0
                },
                "replica_count": 1,
                "container_spec": container_spec
            }
        ],
        base_output_dir=output_dir
    )
    
    # Lanzar el trabajo y esperar a que termine
    job.run(sync=True, service_account=config.SERVICE_ACCOUNT_EMAIL)
    
    print(f"Trabajo de entrenamiento {job_name} completado.")
    return job

def register_model(model_artifacts_uri, model_name, model_description=None, framework="custom"):
    """
    Registra un modelo en Vertex AI Model Registry.
    
    Args:
        model_artifacts_uri: URI en GCS a los artefactos del modelo.
        model_name: Nombre para el modelo.
        model_description: Descripción opcional para el modelo.
        framework: Marco de ML utilizado (ej: "custom", "tensorflow", etc).
        
    Returns:
        Model: El objeto del modelo registrado.
    """
    # Inicializar el cliente de AI Platform
    aiplatform_client = clients.get_aiplatform_client()
    
    # Configurar el modelo
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model_artifacts_uri,
        serving_container_image_uri=f"us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",  # Contenedor por defecto
        description=model_description or f"Modelo entrenado: {model_name}",
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_environment_variables={},
        sync=True
    )
    
    print(f"Modelo registrado en Vertex AI Model Registry con ID: {model.resource_name}")
    return model

def main(job_name_suffix=None, machine_type="n1-standard-4", image_tag="latest"):
    """
    Función principal para lanzar un trabajo de entrenamiento y registrar el modelo.
    
    Args:
        job_name_suffix: Sufijo opcional para el nombre del trabajo.
        machine_type: Tipo de máquina para el entrenamiento.
        image_tag: Etiqueta de la imagen Docker a utilizar.
    """
    # Construir el URI de la imagen Docker
    training_image_uri = f"{config.TRAINING_IMAGE_NAME}:{image_tag}"
    
    # Generar la ruta de salida en GCS
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_dir = f"gs://{config.MODELS_STAGING_BUCKET}/training_{timestamp}"
    
    try:
        # Lanzar el trabajo de entrenamiento
        job = launch_training_job(
            training_image_uri=training_image_uri,
            output_dir=output_dir,
            job_name_suffix=job_name_suffix,
            machine_type=machine_type
        )
        
        # Registrar el modelo en Vertex AI Model Registry
        model_artifacts_uri = f"{output_dir}/model"
        model_name = f"btcbot_trading_agent_{timestamp}"
        
        model = register_model(
            model_artifacts_uri=model_artifacts_uri,
            model_name=model_name,
            model_description=f"Agente de trading entrenado el {timestamp}",
            framework="custom"
        )
        
        print(f"Modelo disponible en: {model.resource_name}")
        print(f"Artefactos del modelo en: {model_artifacts_uri}")
        
        # Imprimir la URL para acceder al modelo en la consola
        print(f"Ver en la consola: https://console.cloud.google.com/vertex-ai/models/{model.name}?project={config.PROJECT_ID}")
        
    except Exception as e:
        print(f"Error al ejecutar el trabajo de entrenamiento: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lanzar trabajo de entrenamiento en Vertex AI")
    parser.add_argument("--job_suffix", help="Sufijo opcional para el nombre del trabajo")
    parser.add_argument("--machine_type", default="n1-standard-4", help="Tipo de máquina para el entrenamiento")
    parser.add_argument("--image_tag", default="latest", help="Etiqueta de la imagen Docker a utilizar")
    
    args = parser.parse_args()
    main(args.job_suffix, args.machine_type, args.image_tag)
