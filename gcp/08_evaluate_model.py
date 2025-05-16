#!/usr/bin/env python
"""
Script para evaluar un modelo registrado en Vertex AI Model Registry.
"""
import argparse
import os
import time
import tempfile
from datetime import datetime
from google.cloud import aiplatform
from google.cloud import storage
import json

from common import config, clients

def download_model(model_id, output_dir):
    """
    Descarga un modelo desde Vertex AI Model Registry.
    
    Args:
        model_id: ID del modelo en Vertex AI.
        output_dir: Directorio donde guardar el modelo descargado.
        
    Returns:
        str: Ruta local al modelo descargado.
    """
    # Inicializar el cliente de AI Platform
    aiplatform_client = clients.get_aiplatform_client()
    
    # Obtener el modelo
    model = aiplatform.Model(model_id)
    
    # Obtener la URI de los artefactos del modelo
    artifact_uri = model.artifact_uri
    
    if not artifact_uri.startswith("gs://"):
        raise ValueError(f"URI de artefactos no válida: {artifact_uri}")
    
    # Extraer bucket y ruta
    bucket_name = artifact_uri.replace("gs://", "").split("/")[0]
    prefix = artifact_uri.replace(f"gs://{bucket_name}/", "")
    
    # Inicializar cliente de Storage
    storage_client = storage.Client(project=config.PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    
    # Listar archivos en la ruta de artefactos
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    # Crear directorio local si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Descargar cada archivo
    model_files = []
    for blob in blobs:
        # Obtener la ruta relativa del archivo
        rel_path = blob.name[len(prefix):].lstrip("/")
        if rel_path:
            local_file_path = os.path.join(output_dir, rel_path)
            
            # Crear subdirectorios si es necesario
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Descargar el archivo
            blob.download_to_filename(local_file_path)
            model_files.append(local_file_path)
            print(f"Descargado: {blob.name} -> {local_file_path}")
    
    # Buscar el archivo zip del modelo
    model_zip = None
    for file in model_files:
        if file.endswith(".zip"):
            model_zip = file
            break
    
    if not model_zip:
        raise ValueError("No se encontró un archivo .zip del modelo en los artefactos")
    
    return model_zip

def evaluate_model(model_path, data_path, num_episodes=10):
    """
    Evalúa un modelo de RL para trading.
    
    Args:
        model_path: Ruta al archivo del modelo.
        data_path: Ruta al archivo de datos para evaluación.
        num_episodes: Número de episodios para evaluar.
        
    Returns:
        dict: Resultados de la evaluación.
    """
    # Importar los módulos necesarios
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.agent.rl_agent_manager import RLAgentManager
    from src.environments.trading_env import TradingEnv
    
    # Crear el entorno de evaluación
    eval_env = TradingEnv(data_path=data_path, mode="eval")
    
    # Cargar el modelo
    agent_manager = RLAgentManager()
    model = agent_manager.load(model_path)
    
    # Evaluar el modelo
    return agent_manager.evaluate(model, eval_env, num_episodes=num_episodes)

def log_evaluation_results(model_id, evaluation_results):
    """
    Guarda los resultados de evaluación en GCS y actualiza metadatos del modelo.
    
    Args:
        model_id: ID del modelo en Vertex AI.
        evaluation_results: Resultados de la evaluación.
    """
    # Inicializar el cliente de Storage
    storage_client = storage.Client(project=config.PROJECT_ID)
    bucket = storage_client.bucket(config.EVALUATION_RESULTS_BUCKET)
    
    # Generar un nombre para el archivo de resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_{model_id.split('/')[-1]}_{timestamp}.json"
    
    # Guardar los resultados en un archivo local temporal
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        json.dump(evaluation_results, temp_file, indent=2)
        temp_file_path = temp_file.name
    
    # Subir el archivo a GCS
    blob = bucket.blob(filename)
    blob.upload_from_filename(temp_file_path)
    
    # Eliminar el archivo temporal
    os.unlink(temp_file_path)
    
    print(f"Resultados de evaluación guardados en: gs://{config.EVALUATION_RESULTS_BUCKET}/{filename}")
    
    # Actualizar metadatos del modelo
    try:
        # Inicializar el cliente de AI Platform
        aiplatform_client = clients.get_aiplatform_client()
        
        # Obtener el modelo
        model = aiplatform.Model(model_id)
        
        # Actualizar metadatos con los resultados de evaluación
        model.update(
            metadata_schema_uri=None,  # No cambiar el esquema
            metadata={
                "evaluation": {
                    "timestamp": timestamp,
                    "mean_reward": evaluation_results.get("mean_reward", 0),
                    "std_reward": evaluation_results.get("std_reward", 0),
                    "max_drawdown": evaluation_results.get("max_drawdown", 0),
                    "sharpe_ratio": evaluation_results.get("sharpe_ratio", 0),
                    "results_uri": f"gs://{config.EVALUATION_RESULTS_BUCKET}/{filename}"
                }
            }
        )
        
        print(f"Metadatos del modelo actualizados con resultados de evaluación")
    except Exception as e:
        print(f"Error al actualizar metadatos del modelo: {e}")

def main():
    """Función principal para evaluar un modelo desde Vertex AI Model Registry."""
    parser = argparse.ArgumentParser(description="Evaluar modelo desde Vertex AI Model Registry")
    parser.add_argument("--model_id", required=True, help="ID completo del modelo en Vertex AI")
    parser.add_argument("--data_path", required=True, help="Ruta al archivo de datos para evaluación")
    parser.add_argument("--num_episodes", type=int, default=10, help="Número de episodios para evaluar")
    
    args = parser.parse_args()
    
    # Crear directorio temporal para descargar el modelo
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Descargar el modelo
            print(f"Descargando modelo {args.model_id}...")
            model_path = download_model(args.model_id, temp_dir)
            
            # Evaluar el modelo
            print(f"Evaluando modelo con {args.num_episodes} episodios...")
            evaluation_results = evaluate_model(model_path, args.data_path, args.num_episodes)
            
            # Imprimir resultados
            print("\nResultados de la evaluación:")
            for key, value in evaluation_results.items():
                print(f"{key}: {value}")
            
            # Guardar resultados y actualizar metadatos
            log_evaluation_results(args.model_id, evaluation_results)
            
            print("\nEvaluación completada exitosamente.")
        except Exception as e:
            print(f"Error durante la evaluación: {e}")
            raise

if __name__ == "__main__":
    main()
