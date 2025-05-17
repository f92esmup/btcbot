"""
Administrador del Agente de Reinforcement Learning para el bot de trading.
Versión optimizada para GCP, utilizando variables de entorno para configuración.
"""

import os
import fsspec
import gymnasium as gym
import numpy as np
import json
import logging
import tempfile
from typing import Dict, Union, Optional, Any, Tuple
from pathlib import Path
import torch
from google.cloud import storage

# Importaciones de Stable Baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Importaciones locales
from src.agent.custom_transformer_extractor import CustomTransformerFeatureExtractor
from src.environments.trading_env_cloud import TradingEnvironmentCloud

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RLAgentManagerCloud")


class RLAgentManagerCloud:
    """
    Clase que gestiona el ciclo de vida del agente de RL optimizada para la nube:
    - Creación/configuración del agente con parámetros directos
    - Entrenamiento con guardado en GCS
    - Carga de modelos desde GCS
    - Evaluación y métricas
    """
    
    def __init__(self, 
                 project_id: str,
                 algorithm: str = "SAC",
                 learning_rate: float = 0.0003,
                 buffer_size: int = 100000,
                 learning_starts: int = 10000,
                 batch_size: int = 256,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 train_freq: int = 1,
                 gradient_steps: int = 1,
                 ent_coef: Union[str, float] = "auto",
                 use_sde: bool = False,
                 policy_kwargs_dict: Optional[Dict] = None,
                 device: str = "auto",
                 models_bucket: str = None):
        """
        Inicializa el administrador del agente de RL con parámetros explícitos.
        
        Args:
            project_id: ID del proyecto de GCP
            algorithm: Algoritmo de RL a usar (por ahora, solo "SAC")
            learning_rate: Tasa de aprendizaje para el optimizador
            buffer_size: Tamaño del buffer de experiencia
            learning_starts: Pasos antes de comenzar a entrenar
            batch_size: Tamaño del batch para entrenar
            tau: Coeficiente para actualizaciones suaves
            gamma: Factor de descuento
            train_freq: Frecuencia de entrenamiento (pasos)
            gradient_steps: Pasos de gradiente por actualización
            ent_coef: Coeficiente de entropía ('auto' o valor)
            use_sde: Si usar exploración dependiente del estado
            policy_kwargs_dict: Parámetros de la arquitectura de la política
            device: Dispositivo ('auto', 'cuda', 'cpu', 'mps')
            models_bucket: Bucket para guardar modelos
        """
        self.project_id = project_id
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.ent_coef = ent_coef
        self.use_sde = use_sde
        self.models_bucket = models_bucket
        
        # Gestionar dispositivo para entrenamiento
        self.device = device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Inicializar cliente de GCS si se proporciona un bucket
        self.storage_client = None
        if models_bucket:
            self.storage_client = storage.Client(project=project_id)
        
        # Configuración para el extractor de características
        if policy_kwargs_dict is None:
            # Configuración predeterminada para el Transformer
            self.policy_kwargs = {
                "features_extractor_class": CustomTransformerFeatureExtractor,
                "features_extractor_kwargs": {
                    "market_features_key": "market_features",
                    "portfolio_features_key": "portfolio_features",
                    "features_in_transformer": 28,  # 20 (mercado) + 8 (cartera)
                    "d_model": 128,
                    "n_heads": 4,
                    "n_encoder_layers": 3,
                    "dim_feedforward": 512,
                    "dropout_rate": 0.1
                },
                "net_arch": {
                    "pi": [256, 256],  # Actor
                    "qf": [256, 256]   # Crítico
                }
            }
        else:
            self.policy_kwargs = policy_kwargs_dict
            # Asegurarse de que se usa la clase correcta para el extractor
            if "features_extractor_class" not in self.policy_kwargs:
                self.policy_kwargs["features_extractor_class"] = CustomTransformerFeatureExtractor
        
        # El modelo y entorno se inicializarán más tarde
        self.model = None
        self.env = None
        self.eval_env = None
    
    def setup_agent(self, 
                    sequence_length_L: int = 96,
                    initial_equity: float = 10000.0,
                    leverage: int = 1,
                    position_size_percentage: float = 0.2,
                    stop_loss_percentage: Optional[float] = None,
                    take_profit_percentage: Optional[float] = None,
                    trading_fees: float = 0.0004,
                    slippage: float = 0.0001,
                    data_gcs_path: str = None,
                    random_start: bool = True):
        """
        Configura el agente y los entornos de entrenamiento y evaluación.
        
        Args:
            sequence_length_L: Longitud de la secuencia para el Transformer
            initial_equity: Saldo inicial para trading
            leverage: Apalancamiento a usar
            position_size_percentage: Porcentaje de equity para posiciones
            stop_loss_percentage: Porcentaje de stop loss (None = desactivado)
            take_profit_percentage: Porcentaje de take profit (None = desactivado)
            trading_fees: Comisiones por trade (porcentaje)
            slippage: Deslizamiento por trade (porcentaje)
            data_gcs_path: Ruta a los datos procesados en GCS
            random_start: Si es True, iniciar en posiciones aleatorias durante entrenamiento
        """
        if data_gcs_path is None:
            raise ValueError("Se requiere la ruta a los datos procesados (data_gcs_path)")
        
        logger.info(f"Configurando agente con datos de: {data_gcs_path}")
        
        # Crear entorno de entrenamiento principal
        self.env = TradingEnvironmentCloud(
            sequence_length_L=sequence_length_L,
            initial_equity=initial_equity,
            leverage=leverage,
            position_size_percentage=position_size_percentage,
            stop_loss_percentage=stop_loss_percentage,
            take_profit_percentage=take_profit_percentage,
            trading_fees=trading_fees,
            slippage=slippage,
            data_gcs_path=data_gcs_path,
            use_reward_scaling=True
        )
        
        # Opcionalmente, envolver el entorno con Monitor para registrar estadísticas
        log_dir = "/tmp/sb3_trading_logs"
        os.makedirs(log_dir, exist_ok=True)
        self.env = Monitor(self.env, log_dir)
        
        # Crear un entorno separado para evaluación
        self.eval_env = TradingEnvironmentCloud(
            sequence_length_L=sequence_length_L,
            initial_equity=initial_equity,
            leverage=leverage,
            position_size_percentage=position_size_percentage,
            stop_loss_percentage=stop_loss_percentage,
            take_profit_percentage=take_profit_percentage,
            trading_fees=trading_fees,
            slippage=slippage,
            data_gcs_path=data_gcs_path,
            use_reward_scaling=True
        )
        
        # Crear modelo SAC 
        if self.algorithm == "SAC":
            self.model = SAC(
                "MultiInputPolicy",
                self.env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                learning_starts=self.learning_starts,
                batch_size=self.batch_size,
                tau=self.tau,
                gamma=self.gamma,
                train_freq=self.train_freq,
                gradient_steps=self.gradient_steps,
                ent_coef=self.ent_coef,
                use_sde=self.use_sde,
                policy_kwargs=self.policy_kwargs,
                device=self.device,
                verbose=1
            )
            logger.info(f"Modelo SAC creado con política Transformer y device={self.device}")
        else:
            raise ValueError(f"Algoritmo no soportado: {self.algorithm}")
    
    def train_agent(self, 
                    total_timesteps: int,
                    eval_freq: int = 10000,
                    save_freq: int = 50000,
                    n_eval_episodes: int = 5,
                    save_path_gcs: str = None,
                    tensorboard_log: str = "/tmp/tensorboard_logs",
                    custom_callbacks: list = None) -> Dict[str, Any]:
        """
        Entrena el agente por un número específico de pasos.
        
        Args:
            total_timesteps: Número total de pasos de entrenamiento
            eval_freq: Frecuencia para evaluación durante entrenamiento
            save_freq: Frecuencia para guardar checkpoints
            n_eval_episodes: Número de episodios para evaluación
            save_path_gcs: Ruta GCS para guardar el modelo final y checkpoints
            tensorboard_log: Ruta para logs de TensorBoard
            custom_callbacks: Lista de callbacks personalizados adicionales
        
        Returns:
            Diccionario con estadísticas de entrenamiento
        """
        if self.model is None:
            raise ValueError("El modelo no ha sido inicializado. Llama a setup_agent() primero.")
        
        # Preparar rutas para guardado (local temporal y GCS)
        temp_save_dir = "/tmp/model_checkpoints"
        os.makedirs(temp_save_dir, exist_ok=True)
        
        # Configurar logger de TensorBoard
        os.makedirs(tensorboard_log, exist_ok=True)
        
        # Crear callback para monitorear valores específicos de trading
        class TradingMetricsCallback(CheckpointCallback):
            """Callback personalizado para monitorear métricas de trading."""
            
            def __init__(self, save_freq, save_path, name_prefix="sac_trading", verbose=1):
                super().__init__(save_freq, save_path, name_prefix, verbose)
                self.episode_rewards = []
                self.episode_lengths = []
                self.episode_win_rates = []
                self.episode_max_drawdowns = []
                self.episode_sharpe_ratios = []
                self.equity_curves = []
                
            def _on_step(self) -> bool:
                result = super()._on_step()
                
                # Registrar métricas de trading si el episodio ha terminado
                infos = self.locals['infos']
                if len(infos) > 0 and all([('terminal_observation' in info) for info in infos]):
                    # Extraer métricas de trading del último episodio
                    for info in infos:
                        if 'episode' in info:
                            ep_rew = info['episode'].r
                            ep_len = info['episode'].l
                            self.episode_rewards.append(ep_rew)
                            self.episode_lengths.append(ep_len)
                        
                        # Extraer métricas específicas de trading si están disponibles
                        if 'episode_stats' in info:
                            stats = info['episode_stats']
                            if 'win_rate' in stats:
                                self.episode_win_rates.append(stats['win_rate'])
                            if 'max_drawdown' in stats:
                                self.episode_max_drawdowns.append(stats['max_drawdown'])
                            if 'sharpe_ratio' in stats:
                                self.episode_sharpe_ratios.append(stats['sharpe_ratio'])
                            if 'equity_curve' in stats:
                                self.equity_curves.append(stats['equity_curve'])
                
                return result
        
        # Callbacks para guardar y evaluar el modelo
        checkpoint_callback = TradingMetricsCallback(
            save_freq=save_freq,
            save_path=temp_save_dir,
            name_prefix="sac_trading",
            verbose=1
        )
        
        # Callback para evaluación durante entrenamiento
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=temp_save_dir,
            log_path=temp_save_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        # Combinar todos los callbacks
        callbacks = [checkpoint_callback, eval_callback]
        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        
        # Configurar logging avanzado
        new_logger = configure(tensorboard_log, ["stdout", "tensorboard"])
        self.model.set_logger(new_logger)
        
        # Entrenar el modelo
        logger.info(f"Iniciando entrenamiento por {total_timesteps} timesteps")
        start_time = time.time()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=1000,
            tb_log_name="SAC_trading"
        )
        training_time = time.time() - start_time
        
        # Guardar el modelo final localmente primero
        final_model_local_path = os.path.join(temp_save_dir, "final_model.zip")
        self.model.save(final_model_local_path)
        logger.info(f"Modelo final guardado localmente en: {final_model_local_path}")
        
        # Si se proporcionó una ruta GCS, subir el modelo final y checkpoints
        if save_path_gcs and self.storage_client:
            try:
                # Subir modelo final
                self._upload_to_gcs(final_model_local_path, save_path_gcs)
                
                # Subir checkpoints
                for filename in os.listdir(temp_save_dir):
                    if filename.startswith("sac_trading_") and filename.endswith(".zip"):
                        local_path = os.path.join(temp_save_dir, filename)
                        checkpoint_steps = filename.replace("sac_trading_", "").replace(".zip", "")
                        gcs_checkpoint_path = f"{save_path_gcs}/checkpoints/checkpoint_{checkpoint_steps}.zip"
                        self._upload_to_gcs(local_path, gcs_checkpoint_path)
                
                # Subir logs de TensorBoard si existen
                if os.path.exists(tensorboard_log):
                    for root, dirs, files in os.walk(tensorboard_log):
                        for file in files:
                            local_path = os.path.join(root, file)
                            relative_path = os.path.relpath(local_path, tensorboard_log)
                            gcs_log_path = f"{save_path_gcs}/tensorboard_logs/{relative_path}"
                            self._upload_to_gcs(local_path, gcs_log_path)
                
                logger.info(f"Modelo, checkpoints y logs subidos a: {save_path_gcs}")
            except Exception as e:
                logger.error(f"Error subiendo modelos a GCS: {e}")
        
        # Calcular estadísticas de entrenamiento
        training_stats = {
            "model_path": save_path_gcs if save_path_gcs else final_model_local_path,
            "total_timesteps": total_timesteps,
            "training_time_seconds": training_time,
            "final_eval_reward": eval_callback.last_mean_reward,
            "best_eval_reward": eval_callback.best_mean_reward
        }
        
        # Añadir estadísticas de episodios si están disponibles
        if hasattr(checkpoint_callback, 'episode_rewards') and len(checkpoint_callback.episode_rewards) > 0:
            training_stats.update({
                "mean_episode_reward": float(np.mean(checkpoint_callback.episode_rewards[-100:])),
                "mean_episode_length": float(np.mean(checkpoint_callback.episode_lengths[-100:])),
                "total_episodes": len(checkpoint_callback.episode_rewards)
            })
            
            # Añadir métricas específicas de trading si están disponibles
            if hasattr(checkpoint_callback, 'episode_win_rates') and len(checkpoint_callback.episode_win_rates) > 0:
                training_stats.update({
                    "mean_win_rate": float(np.mean(checkpoint_callback.episode_win_rates[-100:])),
                    "mean_max_drawdown": float(np.mean(checkpoint_callback.episode_max_drawdowns[-100:])),
                    "mean_sharpe_ratio": float(np.mean(checkpoint_callback.episode_sharpe_ratios[-100:]))
                })
        
        logger.info(f"Entrenamiento completado en {training_time:.2f} segundos")
        logger.info(f"Recompensa media final: {training_stats.get('mean_episode_reward', 'N/A')}")
        
        return training_stats
    
    def _upload_to_gcs(self, local_path: str, gcs_path: str):
        """
        Sube un archivo local a GCS.
        
        Args:
            local_path: Ruta local del archivo
            gcs_path: Ruta completa en GCS (gs://bucket/path/to/file)
        """
        if not gcs_path.startswith("gs://"):
            raise ValueError(f"La ruta GCS debe empezar con 'gs://': {gcs_path}")
        
        # Extraer bucket y blob de la URI de GCS
        gcs_path = gcs_path.replace("gs://", "")
        bucket_name = gcs_path.split("/")[0]
        blob_name = "/".join(gcs_path.split("/")[1:])
        
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(local_path)
        logger.info(f"Archivo subido: {local_path} → gs://{bucket_name}/{blob_name}")
    
    def evaluate_agent(self, 
                       n_eval_episodes: int = 10, 
                       deterministic: bool = True) -> Dict[str, Any]:
        """
        Evalúa el agente en el entorno de evaluación.
        
        Args:
            n_eval_episodes: Número de episodios para evaluación
            deterministic: Si usar política determinista
        
        Returns:
            Diccionario con métricas de evaluación
        """
        if self.model is None or self.eval_env is None:
            raise ValueError("Modelo o entorno de evaluación no inicializados")
        
        logger.info(f"Evaluando modelo por {n_eval_episodes} episodios")
        
        all_episode_returns = []
        all_episode_lengths = []
        all_final_equities = []
        all_sharpe_ratios = []
        all_max_drawdowns = []
        all_win_rates = []
        
        for i in range(n_eval_episodes):
            logger.info(f"Episodio de evaluación {i+1}/{n_eval_episodes}")
            
            obs, info = self.eval_env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                episode_length += 1
            
            # Obtener estadísticas del episodio
            episode_stats = self.eval_env.get_episode_stats()
            
            all_episode_returns.append(episode_reward)
            all_episode_lengths.append(episode_length)
            all_final_equities.append(info['equity'])
            all_sharpe_ratios.append(episode_stats['sharpe_ratio'])
            all_max_drawdowns.append(episode_stats['max_drawdown'])
            all_win_rates.append(episode_stats['win_rate'])
            
            logger.info(f"Episodio {i+1}: Reward={episode_reward:.2f}, "
                       f"Equity final=${info['equity']:.2f}, "
                       f"Sharpe={episode_stats['sharpe_ratio']:.2f}")
        
        # Calcular métricas promedio
        avg_return = np.mean(all_episode_returns)
        avg_final_equity = np.mean(all_final_equities)
        avg_sharpe = np.mean(all_sharpe_ratios)
        avg_max_dd = np.mean(all_max_drawdowns)
        avg_win_rate = np.mean(all_win_rates)
        
        # Estadísticas para informe
        eval_metrics = {
            "avg_episode_return": float(avg_return),
            "avg_final_equity": float(avg_final_equity),
            "avg_sharpe_ratio": float(avg_sharpe),
            "avg_max_drawdown": float(avg_max_dd),
            "avg_win_rate": float(avg_win_rate),
            "equity_change_pct": float((avg_final_equity / self.eval_env.initial_equity - 1) * 100),
            "n_episodes": n_eval_episodes
        }
        
        logger.info(f"Evaluación completada: Return={avg_return:.2f}, "
                   f"Equity=${avg_final_equity:.2f} ({eval_metrics['equity_change_pct']:.2f}%), "
                   f"Sharpe={avg_sharpe:.2f}, MaxDD={avg_max_dd*100:.2f}%, "
                   f"WinRate={avg_win_rate*100:.2f}%")
        
        return eval_metrics
    
    def save_model(self, gcs_path: str):
        """
        Guarda el modelo en GCS.
        
        Args:
            gcs_path: Ruta completa en GCS donde guardar el modelo
        """
        if self.model is None:
            raise ValueError("No hay modelo para guardar")
        
        # Guardar primero en una ubicación local temporal
        temp_file = "/tmp/model_tmp.zip"
        self.model.save(temp_file)
        
        # Subir a GCS
        self._upload_to_gcs(temp_file, gcs_path)
        
        return gcs_path
    
    def load_model(self, model_path: str):
        """
        Carga un modelo desde una ubicación local o GCS.
        
        Args:
            model_path: Ruta al modelo (local o GCS)
        """
        if self.env is None:
            raise ValueError("Entorno no inicializado. Llama a setup_agent() primero.")
        
        try:
            if model_path.startswith("gs://"):
                # Descargar de GCS a ubicación temporal
                temp_file = "/tmp/model_from_gcs.zip"
                
                # Extraer bucket y blob de la URI de GCS
                gcs_path = model_path.replace("gs://", "")
                bucket_name = gcs_path.split("/")[0]
                blob_name = "/".join(gcs_path.split("/")[1:])
                
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                
                blob.download_to_filename(temp_file)
                logger.info(f"Modelo descargado desde GCS: {model_path} → {temp_file}")
                
                # Cargar desde la ubicación temporal
                self.model = SAC.load(temp_file, env=self.env, device=self.device)
            else:
                # Cargar directamente desde archivo local
                self.model = SAC.load(model_path, env=self.env, device=self.device)
            
            logger.info(f"Modelo cargado desde: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo desde {model_path}: {e}")
            return False
