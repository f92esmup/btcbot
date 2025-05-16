"""
Administrador del Agente de Reinforcement Learning para el bot de trading.
Se encarga de la creación, entrenamiento, guardado y carga del agente SAC.
"""

import os
import yaml
import gymnasium as gym
import numpy as np
from typing import Dict, Union, Optional, Any, Tuple
import logging
from pathlib import Path
import torch

# Importaciones de Stable Baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Importaciones locales
from src.agent.custom_transformer_extractor import CustomTransformerFeatureExtractor
from src.environments.trading_env import TradingEnvironment
from src.utils.config import ConfigManager

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RLAgentManager")


class RLAgentManager:
    """
    Clase que gestiona el ciclo de vida del agente de RL:
    - Creación/configuración del agente
    - Entrenamiento
    - Guardado/carga de modelos
    - Interacción con el entorno para predicciones
    """
    
    def __init__(self, config_path: str = "src/agent/agent_config.yaml"):
        """
        Inicializa el administrador del agente de RL.
        
        Args:
            config_path: Ruta al archivo de configuración YAML del agente
        """
        self.config_path = config_path
        config_manager_instance = ConfigManager(config_path=config_path)
        self.config = config_manager_instance.config
        self.model = None
        self.env = None
        self.eval_env = None
        
        # Detectar y configurar el dispositivo (GPU o CPU)
        self.device = self._setup_device()
        
        # Crear directorios para guardar modelos si no existen
        save_path_prefix = self.config.get("save_path_prefix", "models/sac_transformer_trading_agent")
        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
        
    def _setup_device(self) -> str:
        """
        Detecta y configura el dispositivo para entrenamiento (GPU o CPU).
        
        Returns:
            Nombre del dispositivo ("cuda", "mps" o "cpu")
        """
        use_gpu = self.config.get("use_gpu", True)
        
        if not use_gpu:
            logger.info("Uso de GPU desactivado por configuración. Usando CPU.")
            return "cpu"
        
        # Verificar disponibilidad de CUDA (NVIDIA)
        if torch.cuda.is_available():
            device = "cuda"
            num_gpus = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU CUDA disponible: {gpu_name} (Total: {num_gpus})")
        # Verificar disponibilidad de MPS (Apple M1/M2)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("GPU MPS disponible (Apple Silicon)")
        else:
            device = "cpu"
            logger.info("No se detectó GPU. Usando CPU.")
        
        return device
        
    def setup_environment(self, 
                          env_config_path: str = "src/environments/environment_config.yaml",
                          is_eval: bool = False) -> TradingEnvironment:
        """
        Configura y crea una instancia del entorno de trading.
        
        Args:
            env_config_path: Ruta al archivo de configuración del entorno
            is_eval: Si se está configurando un entorno para evaluación
            
        Returns:
            Instancia del entorno de trading configurado
        """
        # Cargar configuración del entorno pero sin pasarla directamente al constructor
        # En lugar de eso, simplemente pasamos la ruta al archivo de config
        
        # Determinar el modo de renderización
        render_mode = None
        if is_eval:
            # Si es entorno de evaluación, se podría configurar el modo de renderización
            # render_mode = 'human'  # Descomentar esta línea si quieres renderización en evaluación
            pass
        
        # Crear instancia del entorno pasando solo los parámetros que acepta
        env = TradingEnvironment(config_path=env_config_path, render_mode=render_mode)
        
        # Envolver con Monitor para seguimiento de recompensas y otra telemetría
        log_dir = "logs/"
        os.makedirs(log_dir, exist_ok=True)
        env_name = "eval_env" if is_eval else "train_env"
        
        return Monitor(env, log_dir + env_name)
        
    def setup_agent(self,
                    env: Optional[gym.Env] = None,
                    env_config_path: str = "src/environments/environment_config.yaml",
                    load_model: bool = False,
                    model_path: Optional[str] = None) -> SAC:
        """
        Configura y crea una instancia del agente SAC con arquitectura personalizada.
        
        Args:
            env: Entorno de gymnasium (opcional, si ya está creado)
            env_config_path: Ruta al archivo de configuración del entorno (si env no se proporciona)
            load_model: Si se debe cargar un modelo existente
            model_path: Ruta al modelo guardado (si load_model es True)
            
        Returns:
            Instancia del modelo SAC configurado
        """
        # Si no se proporciona un entorno, crearlo
        if env is None:
            self.env = self.setup_environment(env_config_path)
        else:
            self.env = env
            
        # Configurar el entorno de evaluación
        self.eval_env = self.setup_environment(env_config_path, is_eval=True)
        
        # Si se debe cargar un modelo existente
        if load_model:
            if model_path is None:
                model_path = self.config.get("model_path_to_load")
                
            if model_path is None:
                raise ValueError("Se solicitó cargar un modelo pero no se especificó la ruta")
                
            logger.info(f"Cargando modelo desde: {model_path}")
            self.model = SAC.load(model_path, env=self.env)
            return self.model
            
        # --- Configuración de hiperparámetros del algoritmo SAC ---
        sac_params = {
            "learning_rate": self.config.get("learning_rate", 0.0003),
            "buffer_size": self.config.get("buffer_size", 100000),
            "learning_starts": self.config.get("learning_starts", 10000),
            "batch_size": self.config.get("batch_size", 256),
            "tau": self.config.get("tau", 0.005),
            "gamma": self.config.get("gamma", 0.99),
            "train_freq": self.config.get("train_freq", 1),
            "gradient_steps": self.config.get("gradient_steps", 1),
            "ent_coef": self.config.get("ent_coef", "auto"),
            "use_sde": self.config.get("use_sde", False),
            "verbose": 1  # Para mostrar información durante el entrenamiento
        }
        
        # --- Configuración de la arquitectura de la política ---
        policy_kwargs = self.config.get("policy_kwargs", {})
        
        # Asegurar que la clase del extractor de características esté definida
        if "features_extractor_class" in policy_kwargs and isinstance(policy_kwargs["features_extractor_class"], str):
            # Si es una cadena, convertirla a la referencia real a la clase
            if policy_kwargs["features_extractor_class"] == "CustomTransformerFeatureExtractor":
                policy_kwargs["features_extractor_class"] = CustomTransformerFeatureExtractor
        else:
            policy_kwargs["features_extractor_class"] = CustomTransformerFeatureExtractor
        
        sac_params["policy_kwargs"] = policy_kwargs
        
        # Agregar configuración del dispositivo a los parámetros
        sac_params["device"] = self.device
        
        # Crear el modelo SAC
        logger.info(f"Creando modelo SAC con los siguientes parámetros: {sac_params}")
        logger.info(f"Usando dispositivo: {self.device}")
        self.model = SAC("MultiInputPolicy", self.env, **sac_params)
        
        return self.model
    
    def train_agent(self, 
                    total_timesteps: Optional[int] = None,
                    callbacks: list = None) -> SAC:
        """
        Entrena al agente por el número especificado de pasos.
        
        Args:
            total_timesteps: Número total de pasos de entrenamiento
            callbacks: Lista de callbacks para usar durante el entrenamiento
            
        Returns:
            El modelo SAC entrenado
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido configurado. Llame a setup_agent primero.")
            
        if total_timesteps is None:
            total_timesteps = self.config.get("total_training_timesteps", 1000000)
            
        # Configurar callbacks por defecto si no se proporcionan
        if callbacks is None:
            save_freq = self.config.get("save_frequency_steps", 50000)
            save_path = self.config.get("save_path_prefix", "models/sac_transformer_trading_agent")
            
            # Callback para guardar el modelo periódicamente
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=os.path.dirname(save_path),
                name_prefix=os.path.basename(save_path),
                save_replay_buffer=True,
                save_vecnormalize=True
            )
            
            # Callback para evaluación
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=os.path.dirname(save_path) + "/best_model",
                log_path="./logs/eval/",
                eval_freq=save_freq,
                n_eval_episodes=10,
                deterministic=True
            )
            
            callbacks = [checkpoint_callback, eval_callback]
            
        # Configurar la ruta de logs
        log_path = "logs/training/"
        os.makedirs(log_path, exist_ok=True)
        logger_sb3 = configure(log_path, ["stdout", "csv", "tensorboard"])
        self.model.set_logger(logger_sb3)
        
        # Entrenar el modelo
        logger.info(f"Iniciando entrenamiento por {total_timesteps} pasos")
        self.model.learn(total_timesteps=total_timesteps, callback=callbacks)
        
        # Guardar el modelo final
        final_save_path = f"{self.config.get('save_path_prefix')}_final_{total_timesteps}_steps.zip"
        self.model.save(final_save_path)
        logger.info(f"Entrenamiento completado. Modelo guardado en {final_save_path}")
        
        return self.model
    
    def predict_action(self, 
                       observation: Dict[str, np.ndarray], 
                       deterministic: bool = True) -> np.ndarray:
        """
        Predice una acción basada en la observación actual.
        
        Args:
            observation: Diccionario con las características del mercado y del portafolio
            deterministic: Si la predicción debe ser determinística (sin exploración)
            
        Returns:
            La acción predicha como array numpy
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido configurado. Llame a setup_agent primero.")
            
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Guarda el modelo en la ruta especificada.
        
        Args:
            path: Ruta donde guardar el modelo (opcional)
            
        Returns:
            La ruta donde se guardó el modelo
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar. Llame a setup_agent primero.")
            
        if path is None:
            # Generar un nombre basado en la configuración
            path = f"{self.config.get('save_path_prefix', 'models/sac_transformer_trading_agent')}.zip"
            
        self.model.save(path)
        logger.info(f"Modelo guardado en: {path}")
        return path
    
    def load_model(self, path: str) -> SAC:
        """
        Carga un modelo guardado.
        
        Args:
            path: Ruta al modelo guardado
            
        Returns:
            El modelo cargado
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el archivo del modelo en: {path}")
        
        # Cargar el modelo especificando el dispositivo
        self.model = SAC.load(path, env=self.env, device=self.device)
        logger.info(f"Modelo cargado desde: {path} en dispositivo: {self.device}")
        return self.model
