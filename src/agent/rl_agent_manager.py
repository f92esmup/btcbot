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
import tempfile
import time

# Importaciones de Stable Baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Importaciones locales
from src.agent.custom_transformer_extractor import CustomTransformerFeatureExtractor
from src.environments.trading_env import TradingEnvironment
from src.utils.config import ConfigManager
from src.utils.gcs_utils import upload_model_to_gcs, download_model_from_gcs

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
    
    def __init__(self, config_path: str = "src/config.yaml"):
        """
        Inicializa el administrador del agente de RL.
        
        Args:
            config_path: Ruta al archivo de configuración centralizada YAML
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path=config_path)
        self.config = self.config_manager.get_agent_config()
        self.model = None
        self.env = None
        self.eval_env = None
        
        # Obtener nombre del bucket de GCS
        self.gcs_bucket_name = self.config_manager.get_env_variable("GCS_BUCKET_NAME")
        if not self.gcs_bucket_name:
            raise ValueError("Error: GCS_BUCKET_NAME no está configurado en las variables de entorno. Es obligatorio para el almacenamiento de modelos.")
        
        logger.info(f"Bucket de GCS configurado: {self.gcs_bucket_name}")
        
        # Detectar y configurar el dispositivo (GPU o CPU)
        self.device = self._setup_device()
        
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
                          is_eval: bool = False) -> TradingEnvironment:
        """
        Configura y crea una instancia del entorno de trading.
        
        Args:
            is_eval: Si se está configurando un entorno para evaluación
            
        Returns:
            Instancia del entorno de trading configurado
        """
        # Determinar el modo de renderización
        render_mode = None
        if is_eval:
            # Si es entorno de evaluación, se podría configurar el modo de renderización
            # render_mode = 'human'  # Descomentar esta línea si quieres renderización en evaluación
            pass
        
        # Crear instancia del entorno pasando la ruta a la configuración centralizada
        env = TradingEnvironment(config_path=self.config_path, render_mode=render_mode)
        
        # Envolver con Monitor para seguimiento de recompensas y otra telemetría
        log_dir = "logs/"
        os.makedirs(log_dir, exist_ok=True)
        env_name = "eval_env" if is_eval else "train_env"
        
        return Monitor(env, log_dir + env_name)
        
    def setup_agent(self,
                    env: Optional[gym.Env] = None,
                    load_model: bool = False,
                    model_path: Optional[str] = None) -> SAC:
        """
        Configura y crea una instancia del agente SAC con arquitectura personalizada.
        
        Args:
            env: Entorno de gymnasium (opcional, si ya está creado)
            load_model: Si se debe cargar un modelo existente
            model_path: Ruta al modelo guardado en GCS (si load_model es True)
            
        Returns:
            Instancia del modelo SAC configurado
        """
        # Si no se proporciona un entorno, crearlo
        if env is None:
            self.env = self.setup_environment()
        else:
            self.env = env
            
        # Configurar el entorno de evaluación
        self.eval_env = self.setup_environment(is_eval=True)
        
        # Si se debe cargar un modelo existente
        if load_model:
            if model_path is None:
                # Si no se especifica ruta, intentar usar la configurada
                model_path = self.config.get("model_path_to_load")
                
            if model_path is None:
                raise ValueError("Se solicitó cargar un modelo pero no se especificó la ruta")
                
            logger.info(f"Cargando modelo desde GCS: {model_path}")
            self.model = self.load_model(model_path)
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
            
            # Crear un directorio temporal para guardar los checkpoints localmente antes de subirlos a GCS
            temp_save_dir = tempfile.mkdtemp(prefix="model_checkpoints_")
            logger.info(f"Creando directorio temporal para checkpoints: {temp_save_dir}")
            
            # Callback personalizado para guardar checkpoints y subirlos a GCS
            class GCSCheckpointCallback(CheckpointCallback):
                def __init__(self, save_freq, save_path, gcs_bucket, gcs_path_prefix, *args, **kwargs):
                    super().__init__(save_freq=save_freq, save_path=save_path, *args, **kwargs)
                    self.gcs_bucket = gcs_bucket
                    self.gcs_path_prefix = gcs_path_prefix
                
                def _on_step(self):
                    if self.n_calls % self.save_freq == 0:
                        # Primero guardamos localmente (implementación de la clase padre)
                        super()._on_step()
                        
                        # Luego subimos a GCS el archivo más reciente guardado
                        if hasattr(self, "last_save_path") and self.last_save_path:
                            # Construir la ruta en GCS
                            checkpoint_filename = os.path.basename(self.last_save_path)
                            gcs_model_path = f"{self.gcs_bucket}/{self.gcs_path_prefix}/{checkpoint_filename}"
                            
                            # Subir a GCS
                            try:
                                gcs_url = upload_model_to_gcs(self.last_save_path, gcs_model_path)
                                logger.info(f"Checkpoint subido a GCS: {gcs_url}")
                            except Exception as e:
                                logger.error(f"Error al subir checkpoint a GCS: {str(e)}")
                    
                    return True
            
            # Clase personalizada para guardar el mejor modelo en GCS
            class GCSEvalCallback(EvalCallback):
                def __init__(self, eval_env, gcs_bucket, gcs_path_prefix, *args, **kwargs):
                    # Crear un directorio temporal para guardar el mejor modelo
                    temp_best_dir = os.path.join(temp_save_dir, "best_model")
                    os.makedirs(temp_best_dir, exist_ok=True)
                    
                    super().__init__(eval_env, best_model_save_path=temp_best_dir, *args, **kwargs)
                    self.gcs_bucket = gcs_bucket
                    self.gcs_path_prefix = gcs_path_prefix
                
                def _on_step(self):
                    # Primero ejecutamos la lógica de evaluación de la clase padre
                    result = super()._on_step()
                    
                    # Si se guardó un nuevo mejor modelo, lo subimos a GCS
                    if self.best_model_save_path is not None and self.best_mean_reward > 0:
                        # Construir la ruta al mejor modelo (coincide con lo que hace la clase base)
                        best_model_path = os.path.join(self.best_model_save_path, "best_model")
                        
                        # Verificar si el archivo existe antes de intentar subirlo
                        if os.path.exists(f"{best_model_path}.zip"):
                            # Construir la ruta en GCS
                            gcs_model_path = f"{self.gcs_bucket}/{self.gcs_path_prefix}/best_model/best_model.zip"
                            
                            # Subir a GCS
                            try:
                                gcs_url = upload_model_to_gcs(f"{best_model_path}.zip", gcs_model_path)
                                logger.info(f"Mejor modelo subido a GCS: {gcs_url}")
                            except Exception as e:
                                logger.error(f"Error al subir mejor modelo a GCS: {str(e)}")
                    
                    return result
            
            # Obtener prefijo de ruta para modelos en GCS
            gcs_path_prefix = self.config.get("gcs_models_path_prefix", "models/sac_transformer_trading_agent")
            
            # Instanciar los callbacks personalizados
            checkpoint_callback = GCSCheckpointCallback(
                save_freq=save_freq,
                save_path=temp_save_dir,
                name_prefix="checkpoint",
                save_replay_buffer=True,
                save_vecnormalize=True,
                gcs_bucket=self.gcs_bucket_name,
                gcs_path_prefix=gcs_path_prefix
            )
            
            # Callback para evaluación y guardado del mejor modelo
            eval_callback = GCSEvalCallback(
                self.eval_env,
                log_path="./logs/eval/",
                eval_freq=save_freq,
                n_eval_episodes=10,
                deterministic=True,
                gcs_bucket=self.gcs_bucket_name,
                gcs_path_prefix=gcs_path_prefix
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
        
        # Guardar el modelo final localmente primero
        temp_final_model = tempfile.mktemp(suffix=".zip")
        self.model.save(temp_final_model)
        
        # Subir modelo final a GCS
        gcs_path_prefix = self.config.get("gcs_models_path_prefix", "models/sac_transformer_trading_agent")
        final_model_name = f"sac_transformer_trading_agent_final_{total_timesteps}_steps.zip"
        gcs_final_path = f"{self.gcs_bucket_name}/{gcs_path_prefix}/{final_model_name}"
        
        try:
            gcs_url = upload_model_to_gcs(temp_final_model, gcs_final_path)
            logger.info(f"Entrenamiento completado. Modelo final guardado en GCS: {gcs_url}")
            # Eliminar el archivo temporal
            os.remove(temp_final_model)
        except Exception as e:
            logger.error(f"Error al subir modelo final a GCS: {str(e)}")
            logger.info(f"Modelo final guardado localmente en: {temp_final_model}")
        
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
    
    def save_model(self, gcs_path: Optional[str] = None) -> str:
        """
        Guarda el modelo en Google Cloud Storage.
        
        Args:
            gcs_path: Ruta en GCS donde guardar el modelo (opcional, sin gs://)
                     Formato: "path/to/model.zip" (se añade el bucket automáticamente)
            
        Returns:
            La URL completa del modelo en GCS (gs://bucket/path/to/model.zip)
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar. Llame a setup_agent primero.")
            
        # Guardar primero en un archivo temporal
        temp_model_path = tempfile.mktemp(suffix=".zip")
        self.model.save(temp_model_path)
        logger.info(f"Modelo guardado temporalmente en: {temp_model_path}")
        
        # Determinar la ruta en GCS
        if gcs_path is None:
            # Generar un nombre basado en la configuración
            gcs_path_prefix = self.config.get("gcs_models_path_prefix", "models/sac_transformer_trading_agent")
            gcs_path = f"{gcs_path_prefix}/model_{int(time.time())}.zip"
            
        # Ruta completa en GCS incluyendo el bucket
        full_gcs_path = f"{self.gcs_bucket_name}/{gcs_path}"
        
        # Subir a GCS
        try:
            gcs_url = upload_model_to_gcs(temp_model_path, full_gcs_path)
            # Eliminar el archivo temporal
            os.remove(temp_model_path)
            logger.info(f"Modelo subido exitosamente a GCS: {gcs_url}")
            return gcs_url
        except Exception as e:
            logger.error(f"Error al subir modelo a GCS: {str(e)}")
            logger.info(f"El modelo permanece guardado localmente en: {temp_model_path}")
            raise
    
    def load_model(self, gcs_path: str) -> SAC:
        """
        Carga un modelo guardado desde Google Cloud Storage.
        
        Args:
            gcs_path: Ruta al modelo en GCS (sin gs://)
                     Formato: "path/to/model.zip" o "bucket_name/path/to/model.zip"
            
        Returns:
            El modelo cargado
        """
        # Si la ruta no incluye el bucket, añadirlo
        if not '/' in gcs_path or not gcs_path.startswith(self.gcs_bucket_name):
            full_gcs_path = f"{self.gcs_bucket_name}/{gcs_path}"
        else:
            full_gcs_path = gcs_path
        
        # Descargar desde GCS
        try:
            local_model_path = download_model_from_gcs(full_gcs_path)
        except Exception as e:
            logger.error(f"Error al descargar modelo desde GCS: {str(e)}")
            raise FileNotFoundError(f"No se pudo descargar el modelo desde GCS: {str(e)}")
        
        # Cargar el modelo especificando el dispositivo
        self.model = SAC.load(local_model_path, env=self.env, device=self.device)
        logger.info(f"Modelo cargado desde GCS: gs://{full_gcs_path} en dispositivo: {self.device}")
        
        # Eliminar el archivo local después de cargarlo
        os.remove(local_model_path)
        
        return self.model
