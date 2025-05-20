"""
Optimizaciones para el servidor de inferencia de BTC Trading Agent.

Este archivo proporciona clases y funciones optimizadas para inferencia.
"""

import os
import numpy as np
import torch
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import tempfile
from google.cloud import storage

# Importaciones de Stable Baselines
from stable_baselines3 import SAC

logger = logging.getLogger("inference_utils")

class InferenceOptimizer:
    """
    Clase que proporciona métodos para optimizar la inferencia del agente RL.
    """
    
    @staticmethod
    def load_model_for_inference(model_path: str, config_path: str = "src/config.yaml"):
        """
        Carga el modelo entrenado desde GCS de forma optimizada para inferencia.
        
        Args:
            model_path (str): Ruta completa en GCS donde se encuentra el modelo
            config_path (str): Ruta al archivo de configuración
            
        Returns:
            RLAgentManager con el modelo cargado para inferencia
        """
        # Importar RLAgentManager aquí para evitar importaciones circulares
        from src.agent.rl_agent_manager import RLAgentManager
        
        # Inicializar el administrador del agente
        agent_manager = RLAgentManager(config_path=config_path)
        
        # Crear un entorno minimal solo para la carga del modelo
        minimal_env = InferenceOptimizer.create_minimal_env(agent_manager.config_path)
        
        # Cargar el modelo desde GCS
        logger.info(f"Cargando modelo desde GCS: {model_path}")
        InferenceOptimizer.load_model_from_gcs(agent_manager, minimal_env, model_path)
        
        return agent_manager
    
    @staticmethod
    def create_minimal_env(config_path: str) -> gym.Env:
        """
        Crea un entorno mínimo para poder cargar el modelo sin cargar datos históricos.
        
        Args:
            config_path: Ruta al archivo de configuración
            
        Returns:
            Un entorno mínimo compatible con el modelo
        """
        # Utilizar una clase de entorno simplificada que es compatible con el modelo
        # pero no carga datos históricos
        from src.environments.trading_env import TradingEnvironment
        
        # Modificar la clase para evitar la carga de datos
        original_load_market_data = TradingEnvironment._load_market_data
        
        # Reemplazar temporalmente el método de carga de datos
        def minimal_load_market_data(self) -> Tuple[np.ndarray, list]:
            # Configuración para datos mínimos que satisfagan al modelo
            config = self.config_manager.get_preprocessing_config()
            sequence_length = config.get('sequence_length_L', 96)
            num_features = len(config.get('final_market_feature_columns', []))
            if num_features == 0:
                num_features = 15  # Valor por defecto si no se encuentra en config
                
            # Crear datos sintéticos mínimos
            logger.info(f"Creando datos sintéticos mínimos para inferencia: ({1}, {sequence_length}, {num_features})")
            return np.zeros((1, sequence_length, num_features), dtype=np.float32), ["feature_"+str(i) for i in range(num_features)]
        
        # Reemplazar temporalmente el método
        TradingEnvironment._load_market_data = minimal_load_market_data
        
        # Crear el entorno con el método modificado
        env = TradingEnvironment(config_path=config_path)
        
        # Restaurar el método original para no afectar otras partes del código
        TradingEnvironment._load_market_data = original_load_market_data
        
        return env
    
    @staticmethod
    def load_model_from_gcs(agent_manager, env: gym.Env, model_path: str):
        """
        Carga un modelo directamente desde GCS.
        
        Args:
            agent_manager: Instancia del RLAgentManager
            env: Entorno compatible con el modelo
            model_path: Ruta en GCS al modelo
        """
        # Establecer el entorno en el agent_manager
        agent_manager.env = env
        
        # Configurar el dispositivo
        agent_manager.device = agent_manager._setup_device()
        
        # Separar la ruta del bucket si es necesario
        if model_path.startswith("gs://"):
            model_path = model_path[5:]  # Quitar el prefijo gs://
            
        # Separar nombre del bucket y ruta del objeto
        parts = model_path.split('/', 1)
        bucket_name = parts[0]
        blob_name = parts[1] if len(parts) > 1 else ""
        
        # Inicializar cliente de GCS
        storage_client = storage.Client(project=agent_manager.config_manager.gcp_project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Crear archivo temporal para el modelo
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
            temp_path = temp_file.name
            
        # Descargar el modelo al archivo temporal
        logger.info(f"Descargando modelo desde gs://{bucket_name}/{blob_name}")
        blob.download_to_filename(temp_path)
        logger.info(f"Modelo descargado exitosamente a: {temp_path}")
        
        # Cargar el modelo
        agent_manager.model = SAC.load(temp_path, env=agent_manager.env, device=agent_manager.device)
        logger.info(f"Modelo cargado exitosamente en dispositivo: {agent_manager.device}")
        
        # Limpiar archivo temporal
        os.remove(temp_path)
        
        return agent_manager.model
