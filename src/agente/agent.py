"""
Agente SAC (Soft Actor-Critic) con arquitectura Transformer para trading de futuros.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path
import pickle
import logging
import tempfile
import os

from .networks import ActorNetwork, CriticNetwork
from .replay_buffer import ReplayBuffer
from ..configuration.config import config

logger = logging.getLogger(__name__)


class TransformerSACAgent:
    """
    Agente SAC con arquitectura Transformer para procesar secuencias de mercado.
    
    Implementa el algoritmo Soft Actor-Critic con:
    - Actor que produce distribución de política
    - Dos críticos para reducir sobreestimación
    - Redes objetivo para estabilidad
    - Parámetro de temperatura alpha aprendible
    """
    
    def __init__(
        self,
        observation_space_shape: Tuple[int, ...],
        action_space_shape: Tuple[int, ...],
        market_features: int,
        portfolio_features: int,
        sequence_length: int,
        device: Optional[torch.device] = None,
        config_override: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializa el agente SAC.
        
        Args:
            observation_space_shape: Forma del espacio de observación
            action_space_shape: Forma del espacio de acción  
            market_features: Número de características de mercado por paso
            portfolio_features: Número de características del portfolio
            sequence_length: Longitud de la secuencia (ventana)
            device: Dispositivo de cómputo
            config_override: Configuración opcional para sobrescribir
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Parámetros del espacio
        self.observation_space_shape = observation_space_shape
        self.action_space_shape = action_space_shape
        self.market_features = market_features
        self.portfolio_features = portfolio_features
        self.sequence_length = sequence_length
        self.action_dim = action_space_shape[0]
        
        # Configuración
        self.config = config_override or self._load_config_from_yaml()
        
        # Hiperparámetros
        self.gamma = self.config['gamma']
        self.tau = self.config['tau']
        self.batch_size = self.config['batch_size']
        self.learning_frequency = self.config['learning_frequency']
        self.update_target_frequency = self.config['update_target_frequency']
        
        # Contadores de pasos
        self.total_steps = 0
        self.learning_steps = 0
        
        # Inicializar redes
        self._init_networks()
        
        # Inicializar optimizadores
        self._init_optimizers()
        
        # Inicializar replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.config['replay_buffer_size'],
            observation_shape=observation_space_shape,
            action_dim=self.action_dim
        )
        
        # Parámetro de temperatura alpha (entropía)
        self.learn_alpha = self.config['learn_alpha']
        
        # Manejar target_entropy: puede ser 'auto' o un valor numérico
        target_entropy_config = self.config['target_entropy']
        if target_entropy_config == 'auto':
            # Calcular automáticamente como -dim_action
            self.target_entropy = -float(self.action_dim)
            logger.info(f"Target entropy calculado automáticamente: {self.target_entropy} (= -dim_action = -{self.action_dim})")
        else:
            # Usar valor especificado en configuración
            self.target_entropy = float(target_entropy_config)
            logger.info(f"Target entropy configurado manualmente: {self.target_entropy}")
        
        if self.learn_alpha:
            self.log_alpha = torch.tensor(
                self.config['initial_log_alpha'], 
                dtype=torch.float32, 
                requires_grad=True, 
                device=self.device
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config['alpha_learning_rate'])
        else:
            self.log_alpha = torch.tensor(
                self.config['initial_log_alpha'], 
                dtype=torch.float32, 
                device=self.device
            )
        
        logger.info(f"TransformerSACAgent inicializado en {self.device}")
        logger.info(f"  - Observación shape: {observation_space_shape}")
        logger.info(f"  - Acción dim: {self.action_dim}")
        logger.info(f"  - Market features: {market_features}")
        logger.info(f"  - Portfolio features: {portfolio_features}")
        logger.info(f"  - Sequence length: {sequence_length}")
        logger.info(f"  - Alpha aprendible: {self.learn_alpha}")
        logger.info(f"  - Target entropy: {self.target_entropy}")
    
    def _load_config_from_yaml(self) -> Dict[str, Any]:
        """Carga configuración desde config.yaml."""
        return {
            'gamma': config.gamma,
            'tau': config.tau,
            'batch_size': config.batch_size,
            'replay_buffer_size': config.replay_buffer_size,
            'actor_learning_rate': config.actor_learning_rate,
            'critic_learning_rate': config.critic_learning_rate,
            'alpha_learning_rate': config.alpha_learning_rate,
            'learn_alpha': config.learn_alpha,
            'target_entropy': config.target_entropy,
            'initial_log_alpha': config.initial_log_alpha,
            'learning_frequency': config.learning_frequency,
            'update_target_frequency': config.update_target_frequency,
            'transformer_config': {
                'd_model': config.d_model,
                'n_head': config.n_head,
                'num_encoder_layers': config.num_encoder_layers,
                'dim_feedforward': config.dim_feedforward,
                'dropout_rate': config.dropout_rate
            },
            'mlp_hidden_dims': config.hidden_dims
        }
    
    def _init_networks(self) -> None:
        """Inicializa todas las redes neuronales."""
        transformer_config = self.config['transformer_config']
        mlp_hidden_dims = self.config['mlp_hidden_dims']
        
        # Red del Actor
        self.actor = ActorNetwork(
            market_features=self.market_features,
            portfolio_features=self.portfolio_features,
            transformer_config=transformer_config,
            mlp_hidden_dims=mlp_hidden_dims,
            action_dim=self.action_dim,
            max_seq_len=self.sequence_length
        ).to(self.device)
        
        # Redes de los Críticos (2 redes)
        self.critic_1 = CriticNetwork(
            market_features=self.market_features,
            portfolio_features=self.portfolio_features,
            action_dim=self.action_dim,
            transformer_config=transformer_config,
            mlp_hidden_dims=mlp_hidden_dims,
            max_seq_len=self.sequence_length
        ).to(self.device)
        
        self.critic_2 = CriticNetwork(
            market_features=self.market_features,
            portfolio_features=self.portfolio_features,
            action_dim=self.action_dim,
            transformer_config=transformer_config,
            mlp_hidden_dims=mlp_hidden_dims,
            max_seq_len=self.sequence_length
        ).to(self.device)
        
        # Redes objetivo (copias de los críticos)
        self.critic_target_1 = CriticNetwork(
            market_features=self.market_features,
            portfolio_features=self.portfolio_features,
            action_dim=self.action_dim,
            transformer_config=transformer_config,
            mlp_hidden_dims=mlp_hidden_dims,
            max_seq_len=self.sequence_length
        ).to(self.device)
        
        self.critic_target_2 = CriticNetwork(
            market_features=self.market_features,
            portfolio_features=self.portfolio_features,
            action_dim=self.action_dim,
            transformer_config=transformer_config,
            mlp_hidden_dims=mlp_hidden_dims,
            max_seq_len=self.sequence_length
        ).to(self.device)
        
        # Copiar pesos a las redes objetivo
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        
        # Congelar redes objetivo
        for param in self.critic_target_1.parameters():
            param.requires_grad = False
        for param in self.critic_target_2.parameters():
            param.requires_grad = False
    
    def _init_optimizers(self) -> None:
        """Inicializa los optimizadores."""
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=self.config['actor_learning_rate']
        )
        
        self.critic_1_optimizer = optim.Adam(
            self.critic_1.parameters(), 
            lr=self.config['critic_learning_rate']
        )
        
        self.critic_2_optimizer = optim.Adam(
            self.critic_2.parameters(), 
            lr=self.config['critic_learning_rate']
        )
    
    @property
    def alpha(self) -> torch.Tensor:
        """Parámetro de temperatura actual."""
        return torch.exp(self.log_alpha)
    
    def _parse_observation(self, observation: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Separa la observación en datos de mercado y portfolio.
        
        Args:
            observation: Observación completa del entorno
            
        Returns:
            Tupla (market_data, portfolio_data)
        """
        # La observación tiene la forma: [ventana_mercado_flattened, portfolio_features]
        market_data_size = self.sequence_length * self.market_features
        
        # Separar componentes
        market_flat = observation[:market_data_size]
        portfolio_data = observation[market_data_size:]
        
        # Reshape market data para Transformer
        market_data = market_flat.reshape(self.sequence_length, self.market_features)
        
        return (
            torch.FloatTensor(market_data).unsqueeze(0).to(self.device),  # (1, seq_len, features)
            torch.FloatTensor(portfolio_data).unsqueeze(0).to(self.device)  # (1, portfolio_features)
        )
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Selecciona una acción basada en la observación.
        
        Args:
            observation: Observación actual del entorno
            deterministic: Si True, usa la media de la política (evaluación)
            
        Returns:
            Acción seleccionada
        """
        market_data, portfolio_data = self._parse_observation(observation)
        
        with torch.no_grad():
            if deterministic:
                # Para evaluación: usar la media de la distribución
                mean, _ = self.actor(market_data, portfolio_data)
                action = torch.tanh(mean)
            else:
                # Para entrenamiento: muestrear de la distribución
                action, _ = self.actor.sample(market_data, portfolio_data)
        
        return action.cpu().numpy().flatten()
    
    def store_transition(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool
    ) -> None:
        """
        Almacena una transición en el replay buffer.
        """
        self.replay_buffer.add(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated
        )
    
    def can_learn(self) -> bool:
        """Verifica si el agente puede aprender."""
        return self.replay_buffer.can_sample(self.batch_size)
    
    def learn(self) -> Optional[Dict[str, float]]:
        """
        Realiza un paso de aprendizaje SAC.
        
        Returns:
            Diccionario con métricas de entrenamiento si se realizó aprendizaje
        """
        if not self.can_learn():
            return None
        
        self.total_steps += 1
        
        # Verificar frecuencia de aprendizaje
        if self.total_steps % self.learning_frequency != 0:
            return None
        
        # Muestrear batch del replay buffer
        observations, actions, rewards, next_observations, terminated, truncated = self.replay_buffer.sample(
            self.batch_size, self.device
        )
        
        # Parsear observaciones
        batch_market_data = []
        batch_portfolio_data = []
        batch_next_market_data = []
        batch_next_portfolio_data = []
        
        for i in range(self.batch_size):
            market, portfolio = self._parse_observation(observations[i].cpu().numpy())
            next_market, next_portfolio = self._parse_observation(next_observations[i].cpu().numpy())
            
            batch_market_data.append(market.squeeze(0))
            batch_portfolio_data.append(portfolio.squeeze(0))
            batch_next_market_data.append(next_market.squeeze(0))
            batch_next_portfolio_data.append(next_portfolio.squeeze(0))
        
        market_data = torch.stack(batch_market_data)
        portfolio_data = torch.stack(batch_portfolio_data)
        next_market_data = torch.stack(batch_next_market_data)
        next_portfolio_data = torch.stack(batch_next_portfolio_data)
        
        # Máscaras para episodios finalizados
        done_mask = terminated | truncated
        
        # Normalización de recompensas para estabilidad del crítico
        scaled_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        with torch.no_grad():
            # Siguiente acción y log_prob usando la política actual
            next_actions, next_log_probs = self.actor.sample(next_market_data, next_portfolio_data)
            
            # Q-valores objetivo
            target_q1 = self.critic_target_1(next_market_data, next_portfolio_data, next_actions)
            target_q2 = self.critic_target_2(next_market_data, next_portfolio_data, next_actions)
            
            # Tomar el mínimo para reducir sobreestimación
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            
            # Calcular targets
            q_targets = scaled_rewards.unsqueeze(1) + self.gamma * (1 - done_mask.float().unsqueeze(1)) * target_q
        
        # Actualizar críticos
        current_q1 = self.critic_1(market_data, portfolio_data, actions)
        current_q2 = self.critic_2(market_data, portfolio_data, actions)
        
        critic_1_loss = F.mse_loss(current_q1, q_targets)
        critic_2_loss = F.mse_loss(current_q2, q_targets)
        
        # Optimizar crítico 1
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        
        # Optimizar crítico 2
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # Actualizar actor
        new_actions, log_probs = self.actor.sample(market_data, portfolio_data)
        
        q1_new = self.critic_1(market_data, portfolio_data, new_actions)
        q2_new = self.critic_2(market_data, portfolio_data, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Actualizar alpha si es aprendible
        alpha_loss = torch.tensor(0.0)
        if self.learn_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Actualizar redes objetivo
        self.learning_steps += 1
        if self.learning_steps % self.update_target_frequency == 0:
            self._soft_update_target_networks()
        
        # Métricas de entrenamiento
        metrics = {
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'mean_q1': current_q1.mean().item(),
            'mean_q2': current_q2.mean().item(),
            'mean_log_prob': log_probs.mean().item(),
            'buffer_size': self.replay_buffer.size,
            'learning_steps': self.learning_steps
        }
        
        return metrics
    
    def _soft_update_target_networks(self) -> None:
        """Actualización suave de las redes objetivo."""
        for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_models(self, filepath_prefix: str) -> None:
        """
        Guarda todos los modelos y estados del agente.
        
        Args:
            filepath_prefix: Prefijo para los archivos de guardado
        """
        save_path = Path(filepath_prefix).parent
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar redes principales
        torch.save(self.actor.state_dict(), f"{filepath_prefix}_actor.pth")
        torch.save(self.critic_1.state_dict(), f"{filepath_prefix}_critic_1.pth")
        torch.save(self.critic_2.state_dict(), f"{filepath_prefix}_critic_2.pth")
        torch.save(self.critic_target_1.state_dict(), f"{filepath_prefix}_critic_target_1.pth")
        torch.save(self.critic_target_2.state_dict(), f"{filepath_prefix}_critic_target_2.pth")
        
        # Guardar optimizadores
        torch.save(self.actor_optimizer.state_dict(), f"{filepath_prefix}_actor_optimizer.pth")
        torch.save(self.critic_1_optimizer.state_dict(), f"{filepath_prefix}_critic_1_optimizer.pth")
        torch.save(self.critic_2_optimizer.state_dict(), f"{filepath_prefix}_critic_2_optimizer.pth")
        
        # Guardar alpha y su optimizador
        torch.save(self.log_alpha, f"{filepath_prefix}_log_alpha.pth")
        if self.learn_alpha:
            torch.save(self.alpha_optimizer.state_dict(), f"{filepath_prefix}_alpha_optimizer.pth")
        
        # Guardar metadatos
        metadata = {
            'total_steps': self.total_steps,
            'learning_steps': self.learning_steps,
            'config': self.config,
            'observation_space_shape': self.observation_space_shape,
            'action_space_shape': self.action_space_shape,
            'market_features': self.market_features,
            'portfolio_features': self.portfolio_features,
            'sequence_length': self.sequence_length
        }
        
        with open(f"{filepath_prefix}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Modelos guardados en {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str) -> None:
        """
        Carga todos los modelos y estados del agente.
        
        Args:
            filepath_prefix: Prefijo de los archivos a cargar
        """
        # Cargar metadatos
        with open(f"{filepath_prefix}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.total_steps = metadata['total_steps']
        self.learning_steps = metadata['learning_steps']
        
        # Cargar redes principales
        self.actor.load_state_dict(torch.load(f"{filepath_prefix}_actor.pth", map_location=self.device))
        self.critic_1.load_state_dict(torch.load(f"{filepath_prefix}_critic_1.pth", map_location=self.device))
        self.critic_2.load_state_dict(torch.load(f"{filepath_prefix}_critic_2.pth", map_location=self.device))
        self.critic_target_1.load_state_dict(torch.load(f"{filepath_prefix}_critic_target_1.pth", map_location=self.device))
        self.critic_target_2.load_state_dict(torch.load(f"{filepath_prefix}_critic_target_2.pth", map_location=self.device))
        
        # Cargar optimizadores
        self.actor_optimizer.load_state_dict(torch.load(f"{filepath_prefix}_actor_optimizer.pth", map_location=self.device))
        self.critic_1_optimizer.load_state_dict(torch.load(f"{filepath_prefix}_critic_1_optimizer.pth", map_location=self.device))
        self.critic_2_optimizer.load_state_dict(torch.load(f"{filepath_prefix}_critic_2_optimizer.pth", map_location=self.device))
        
        # Cargar alpha
        self.log_alpha = torch.load(f"{filepath_prefix}_log_alpha.pth", map_location=self.device)
        if self.learn_alpha:
            self.alpha_optimizer.load_state_dict(torch.load(f"{filepath_prefix}_alpha_optimizer.pth", map_location=self.device))
        
        logger.info(f"Modelos cargados desde {filepath_prefix}")
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Guarda el modelo usando el modo de storage configurado.
        Wrapper que maneja tanto almacenamiento local como GCS.
        
        Args:
            filepath: Ruta del archivo donde guardar (Path o string)
                     En GCS debe incluir el run_id y subcarpeta (ej: "{run_id}/best_model/model.pth")
        """
        filepath_str = str(filepath)
        
        # Verificar modo de storage
        if config.storage_mode == "gcp":
            # Modo GCS: usar archivos temporales y subir a GCS
            from ..configuration.gcs_utils import gcs_utils
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generar prefijo temporal para archivos - usar nombre consistente
                local_temp_file_prefix = os.path.join(temp_dir, "agent_model_components")
                
                # Guardar modelos localmente en directorio temporal
                self.save_models(local_temp_file_prefix)
                
                # Determinar directorio de destino en GCS usando el filepath proporcionado
                gcs_target_directory = Path(filepath).parent
                
                # Lista de archivos que se generan en save_models
                model_files = [
                    f"{local_temp_file_prefix}_actor.pth",
                    f"{local_temp_file_prefix}_critic_1.pth", 
                    f"{local_temp_file_prefix}_critic_2.pth",
                    f"{local_temp_file_prefix}_critic_target_1.pth",
                    f"{local_temp_file_prefix}_critic_target_2.pth",
                    f"{local_temp_file_prefix}_actor_optimizer.pth",
                    f"{local_temp_file_prefix}_critic_1_optimizer.pth",
                    f"{local_temp_file_prefix}_critic_2_optimizer.pth",
                    f"{local_temp_file_prefix}_log_alpha.pth",
                    f"{local_temp_file_prefix}_metadata.pkl"
                ]
                
                # Agregar archivo del optimizador de alpha si aplica
                if self.learn_alpha:
                    model_files.append(f"{local_temp_file_prefix}_alpha_optimizer.pth")
                
                # Subir cada archivo a GCS
                success_count = 0
                for local_file in model_files:
                    if os.path.exists(local_file):
                        # Construir el nombre del archivo en GCS
                        local_file_path_obj = Path(local_file)
                        gcs_blob_name = str(gcs_target_directory / local_file_path_obj.name)
                        
                        if gcs_utils.upload_file_to_gcs(local_file, gcs_blob_name):
                            success_count += 1
                        else:
                            logger.error(f"Error al subir {local_file} a GCS como {gcs_blob_name}")
                
                if success_count == len([f for f in model_files if os.path.exists(f)]):
                    logger.info(f"Modelo guardado exitosamente en GCS: {gcs_target_directory}")
                else:
                    logger.error(f"Error al guardar modelo en GCS. Solo {success_count} de {len(model_files)} archivos subidos.")
        else:
            # Modo local: usar save_models directamente
            # Remover extensión .pth si existe para que save_models funcione correctamente
            filepath_prefix = str(filepath).replace('.pth', '')
            self.save_models(filepath_prefix)
    
    def load(self, filepath: Union[str, Path]) -> None:
        """
        Carga el modelo usando el modo de storage configurado.
        Wrapper que maneja tanto almacenamiento local como GCS.
        
        Args:
            filepath: Ruta del archivo desde donde cargar (Path o string)
                     En GCS debe incluir el run_id y subcarpeta (ej: "{run_id}/best_model/model.pth")
        """
        filepath_str = str(filepath)
        
        # Verificar modo de storage  
        if config.storage_mode == "gcp":
            # Modo GCS: descargar archivos de GCS a directorio temporal
            from ..configuration.gcs_utils import gcs_utils
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generar prefijo temporal para archivos - usar nombre consistente con save
                temp_prefix = os.path.join(temp_dir, "agent_model_components")
                
                # Determinar directorio fuente en GCS usando el filepath proporcionado
                gcs_source_directory = Path(filepath).parent
                
                # Lista de archivos que necesitamos descargar (nombres consistentes con save)
                component_files = [
                    "agent_model_components_actor.pth",
                    "agent_model_components_critic_1.pth",
                    "agent_model_components_critic_2.pth", 
                    "agent_model_components_critic_target_1.pth",
                    "agent_model_components_critic_target_2.pth",
                    "agent_model_components_actor_optimizer.pth",
                    "agent_model_components_critic_1_optimizer.pth",
                    "agent_model_components_critic_2_optimizer.pth",
                    "agent_model_components_log_alpha.pth",
                    "agent_model_components_metadata.pkl"
                ]
                
                # Agregar archivo del optimizador de alpha si aplica
                if self.learn_alpha:
                    component_files.append("agent_model_components_alpha_optimizer.pth")
                
                # Descargar cada archivo de GCS
                success_count = 0
                for component_filename in component_files:
                    gcs_blob_name = str(gcs_source_directory / component_filename)
                    local_temp_file = os.path.join(temp_dir, component_filename)
                    
                    # Verificar si el archivo existe en GCS antes de intentar descargarlo
                    if gcs_utils.file_exists_in_gcs(gcs_blob_name):
                        if gcs_utils.download_file_from_gcs(gcs_blob_name, local_temp_file):
                            success_count += 1
                        else:
                            logger.error(f"Error al descargar {gcs_blob_name} de GCS a {local_temp_file}")
                    else:
                        # Para archivos opcionales como alpha_optimizer, no es un error si no existe
                        if component_filename == "agent_model_components_alpha_optimizer.pth" and not self.learn_alpha:
                            continue
                        logger.error(f"Archivo {gcs_blob_name} no existe en GCS")
                
                if success_count >= len(component_files) - (0 if self.learn_alpha else 1):  # Ajustar por alpha_optimizer opcional
                    # Cargar modelos desde archivos temporales
                    self.load_models(temp_prefix)
                    logger.info(f"Modelo cargado exitosamente desde GCS: {gcs_source_directory}")
                else:
                    logger.error(f"Error al cargar modelo desde GCS. Solo {success_count} de {len(component_files)} archivos descargados.")
                    raise FileNotFoundError(f"No se pudo cargar el modelo completo desde GCS: {gcs_source_directory}")
        else:
            # Modo local: usar load_models directamente
            # Remover extensión .pth si existe para que load_models funcione correctamente  
            filepath_prefix = str(filepath).replace('.pth', '')
            self.load_models(filepath_prefix)
    
    def eval_mode(self) -> None:
        """
        Pone todas las redes neuronales en modo evaluación.
        Desactiva dropout y pone batch normalization en modo evaluación.
        """
        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.critic_target_1.eval()
        self.critic_target_2.eval()
    
    def train_mode(self) -> None:
        """
        Pone todas las redes neuronales en modo entrenamiento.
        Activa dropout y pone batch normalization en modo entrenamiento.
        """
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
        # Las redes objetivo no necesitan estar en modo entrenamiento
        # self.critic_target_1.train()
        # self.critic_target_2.train()

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del agente.
        
        Returns:
            Diccionario con estadísticas
        """
        buffer_stats = self.replay_buffer.get_stats()
        
        agent_stats = {
            'total_steps': self.total_steps,
            'learning_steps': self.learning_steps,
            'alpha': self.alpha.item(),
            'device': str(self.device),
            'can_learn': self.can_learn()
        }
        
        return {**agent_stats, 'buffer': buffer_stats}
