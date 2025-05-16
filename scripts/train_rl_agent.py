#!/usr/bin/env python3
"""
Script para entrenar un agente de RL utilizando el algoritmo SAC
con la arquitectura personalizada basada en Transformer.
Compatibilidad con entrenamiento en Vertex AI.
"""

import os
import argparse
import logging
import uuid
from pathlib import Path
from datetime import datetime

# Importaciones locales
from src.agent.rl_agent_manager import RLAgentManager
from src.utils.config import ConfigManager

# Intentar importar las bibliotecas de GCP
try:
    from google.cloud import storage
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TrainRLAgent")


def parse_arguments():
    """
    Parsea los argumentos de la línea de comandos.
    
    Returns:
        Argumentos parseados
    """
    parser = argparse.ArgumentParser(description="Entrena un agente de RL para trading")
    
    parser.add_argument(
        "--agent-config",
        type=str,
        default="src/agent/agent_config.yaml",
        help="Ruta al archivo de configuración del agente"
    )
    
    parser.add_argument(
        "--env-config",
        type=str,
        default="src/environments/environment_config.yaml",
        help="Ruta al archivo de configuración del entorno"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Número total de pasos de entrenamiento (si se omite, se usa el valor del config)"
    )
    
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Ruta a un modelo guardado para continuar el entrenamiento"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Desactivar el uso de GPU incluso si está disponible"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio para guardar el modelo entrenado (usa AIP_MODEL_DIR si está disponible en Vertex AI)"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Ruta al archivo de datos preprocesados"
    )
    
    return parser.parse_args()


def save_model_to_gcs(model, model_local_path, gcs_uri=None):
    """
    Guarda el modelo en GCS si se está ejecutando en Vertex AI.
    
    Args:
        model: Modelo entrenado para guardar.
        model_local_path: Ruta local donde se guarda el modelo.
        gcs_uri: URI de GCS donde guardar el modelo.
    
    Returns:
        str: URI completo donde se guardó el modelo.
    """
    if not GCP_AVAILABLE or not gcs_uri:
        # Guardar localmente si no podemos guardar en GCS
        model.save(model_local_path)
        logger.info(f"Modelo guardado localmente en: {model_local_path}")
        return model_local_path
    
    try:
        # Primero guardar localmente
        model.save(model_local_path)
        logger.info(f"Modelo guardado localmente en: {model_local_path}")
        
        # Luego subir a GCS
        if gcs_uri.startswith("gs://"):
            bucket_name = gcs_uri.replace("gs://", "").split("/")[0]
            blob_prefix = "/".join(gcs_uri.replace(f"gs://{bucket_name}/", "").split("/"))
            
            # Asegurarse de que el prefijo termine con /
            if blob_prefix and not blob_prefix.endswith("/"):
                blob_prefix += "/"
            
            filename = os.path.basename(model_local_path)
            blob_name = f"{blob_prefix}{filename}"
            
            # Subir a GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            blob.upload_from_filename(model_local_path)
            full_gcs_uri = f"gs://{bucket_name}/{blob_name}"
            logger.info(f"Modelo subido a GCS: {full_gcs_uri}")
            
            return full_gcs_uri
        else:
            logger.warning(f"URI GCS no válido: {gcs_uri}. El modelo solo se guardó localmente.")
            return model_local_path
    except Exception as e:
        logger.error(f"Error al guardar el modelo en GCS: {e}")
        return model_local_path


def main():
    """
    Función principal para entrenar un agente de RL.
    """
    # Parsear argumentos
    args = parse_arguments()
    
    # Cargar la configuración del agente
    config_manager = ConfigManager(config_path=args.agent_config)
    agent_config = config_manager.config
    
    # Actualizar la configuración si se solicita no usar GPU
    if args.no_gpu:
        agent_config["use_gpu"] = False
        logger.info("Uso de GPU desactivado por argumento de línea de comandos")
    
    # Crear el administrador del agente con la configuración actualizada
    agent_manager = RLAgentManager(config_path=args.agent_config)
    
    # Si se desactivó la GPU por argumento, aplicar la configuración al administrador
    if args.no_gpu:
        agent_manager.config["use_gpu"] = False
        agent_manager.device = "cpu"
    
    # Configurar el agente
    should_load_model = args.load_model is not None
    agent_manager.setup_agent(
        env_config_path=args.env_config,
        load_model=should_load_model,
        model_path=args.load_model,
        data_path=args.data_path
    )
    
    # Entrenar el agente
    trained_model = agent_manager.train_agent(total_timesteps=args.timesteps)
    
    # Determinar el directorio de salida
    output_dir = args.output_dir
    
    # Verificar si estamos en Vertex AI
    if os.environ.get("AIP_MODEL_DIR") and not output_dir:
        output_dir = os.environ.get("AIP_MODEL_DIR")
        logger.info(f"Detectado entorno de Vertex AI. Usando directorio de salida: {output_dir}")
    
    # Guardar el modelo
    if output_dir:
        # Generar nombre de archivo para el modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"sac_transformer_trading_agent_{timestamp}_{unique_id}.zip"
        
        if output_dir.startswith("gs://"):
            # Para GCS, guardamos en un directorio temporal primero
            local_temp_path = os.path.join("/tmp", filename)
            gcs_uri = save_model_to_gcs(trained_model, local_temp_path, output_dir)
            logger.info(f"Modelo guardado en GCS: {gcs_uri}")
        else:
            # Para guardado local
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, filename)
            trained_model.save(model_path)
            logger.info(f"Modelo guardado en: {model_path}")
    else:
        # Si no se especificó un directorio, guardar en la carpeta models por defecto
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(models_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(models_dir, f"sac_transformer_trading_agent_{timestamp}.zip")
        trained_model.save(model_path)
        logger.info(f"Modelo guardado en: {model_path}")
    
    logger.info("Entrenamiento completado.")


if __name__ == "__main__":
    main()
