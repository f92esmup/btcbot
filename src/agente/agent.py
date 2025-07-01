"""
Agente SAC (Soft Actor-Critic) con arquitectura Transformer para trading de futuros.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from pathlib import Path
import pickle
import logging
import tempfile
import os

from .networks import ActorNetwork, CriticNetwork

logger = logging.getLogger(__name__)


class TransformerSACAgent:
    """
    Agente SAC que funciona tanto en modo de un solo nodo como en modo distribuido (con DDP).
    
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
        config_override: Dict[str, Any],  # Now mandatory
        device: Optional[torch.device] = None,
        is_distributed: bool = False  # Flag para activar el modo distribuido
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
            config_override: Configuración requerida para el agente
            is_distributed: Flag para activar el modo distribuido con DDP
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_distributed = is_distributed
        
        # Parámetros del espacio
        self.observation_space_shape = observation_space_shape
        self.action_space_shape = action_space_shape
        self.market_features = market_features
        self.portfolio_features = portfolio_features
        self.sequence_length = sequence_length
        self.action_dim = action_space_shape[0]
        
        # Configuración - ahora exclusivamente del parámetro config_override
        self.config = config_override
        
        # Extraer sub-configuraciones para claridad
        sac_params = self.config.get('hiperparametros_sac', {})
        
        # Hiperparámetros
        self.gamma = sac_params['gamma']
        self.tau = sac_params['tau']
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
        
        # Parámetro de temperatura alpha (entropía)
        self.learn_alpha = sac_params['learn_alpha']
        
        # Manejar target_entropy: puede ser 'auto' o un valor numérico
        target_entropy_config = sac_params['target_entropy']
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
                sac_params['initial_log_alpha'], 
                dtype=torch.float32, 
                requires_grad=True, 
                device=self.device
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=sac_params['alpha_learning_rate'])
        else:
            self.log_alpha = torch.tensor(
                sac_params['initial_log_alpha'], 
                dtype=torch.float32, 
                device=self.device
            )
        
        # Inicializar GradScaler para Automatic Mixed Precision (AMP)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        
        logger.info(f"TransformerSACAgent inicializado en {self.device}. Modo distribuido: {self.is_distributed}")
        logger.info(f"  - Observación shape: {observation_space_shape}")
        logger.info(f"  - Acción dim: {self.action_dim}")
        logger.info(f"  - Market features: {market_features}")
        logger.info(f"  - Portfolio features: {portfolio_features}")
        logger.info(f"  - Sequence length: {sequence_length}")
        logger.info(f"  - Alpha aprendible: {self.learn_alpha}")
        logger.info(f"  - Target entropy: {self.target_entropy}")
        logger.info(f"  - AMP habilitado: {self.device.type == 'cuda'}")
    
    def _init_networks(self) -> None:
        """Inicializa las redes y las envuelve para DDP si está en modo distribuido."""
        transformer_config = self.config['transformer']
        mlp_hidden_dims = self.config['mlp_heads']['hidden_dims']
        
        # Red del Actor
        self.actor = ActorNetwork(
            market_features=self.market_features,
            portfolio_features=self.portfolio_features,
            transformer_config=transformer_config,
            mlp_hidden_dims=mlp_hidden_dims,
            action_dim=self.action_dim,
            agent_config=self.config  # Pasar la configuración completa
        ).to(self.device)
        logger.info("✅ ActorNetwork creado.")

        # Redes de los Críticos
        self.critic_1 = CriticNetwork(
            market_features=self.market_features,
            portfolio_features=self.portfolio_features,
            action_dim=self.action_dim,
            transformer_config=transformer_config,
            mlp_hidden_dims=mlp_hidden_dims,
            agent_config=self.config  # Pasar la configuración completa
        ).to(self.device)
        
        self.critic_2 = CriticNetwork(
            market_features=self.market_features,
            portfolio_features=self.portfolio_features,
            action_dim=self.action_dim,
            transformer_config=transformer_config,
            mlp_hidden_dims=mlp_hidden_dims,
            agent_config=self.config  # Pasar la configuración completa
        ).to(self.device)
        logger.info("✅ CriticNetworks creados.")

        # Redes objetivo
        self.critic_target_1 = CriticNetwork(
            market_features=self.market_features,
            portfolio_features=self.portfolio_features,
            action_dim=self.action_dim,
            transformer_config=transformer_config,
            mlp_hidden_dims=mlp_hidden_dims,
            agent_config=self.config  # Pasar la configuración completa
        ).to(self.device)
        
        self.critic_target_2 = CriticNetwork(
            market_features=self.market_features,
            portfolio_features=self.portfolio_features,
            action_dim=self.action_dim,
            transformer_config=transformer_config,
            mlp_hidden_dims=mlp_hidden_dims,
            agent_config=self.config  # Pasar la configuración completa
        ).to(self.device)
        logger.info("✅ Critic Target Networks creados (sin JIT).")

        # Envolver con DDP si está en modo distribuido
        if self.is_distributed:
            logger.info(f"Envolviendo modelos con DistributedDataParallel en el dispositivo {self.device}.")
            if self.device.type == 'cuda':
                # Para dispositivos CUDA, especificar device_ids
                device_ids = [self.device.index] if self.device.index is not None else None
                self.actor = DDP(self.actor, device_ids=device_ids)
                self.critic_1 = DDP(self.critic_1, device_ids=device_ids)
                self.critic_2 = DDP(self.critic_2, device_ids=device_ids)
            else:
                # Para CPU, no especificar device_ids
                self.actor = DDP(self.actor)
                self.critic_1 = DDP(self.critic_1)
                self.critic_2 = DDP(self.critic_2)

        # Copiar pesos a las redes objetivo
        critic1_model = self._get_critic_model(self.critic_1)
        critic2_model = self._get_critic_model(self.critic_2)
        
        self.critic_target_1.load_state_dict(critic1_model.state_dict())
        self.critic_target_2.load_state_dict(critic2_model.state_dict())
        
        # Congelar redes objetivo
        for param in self.critic_target_1.parameters():
            param.requires_grad = False
        for param in self.critic_target_2.parameters():
            param.requires_grad = False
    
    def _init_optimizers(self) -> None:
        """Inicializa los optimizadores."""
        sac_params = self.config.get('hiperparametros_sac', {})
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=sac_params['actor_learning_rate']
        )
        
        self.critic_1_optimizer = optim.Adam(
            self.critic_1.parameters(), 
            lr=sac_params['critic_learning_rate']
        )
        
        self.critic_2_optimizer = optim.Adam(
            self.critic_2.parameters(), 
            lr=sac_params['critic_learning_rate']
        )
    
    @property
    def alpha(self) -> torch.Tensor:
        """Valor actual del parámetro de temperatura alpha."""
        return self.log_alpha.exp()
    
    def _get_actor_model(self):
        """Obtiene el modelo actor subyacente (sin envoltura DDP)."""
        # Verificar si la red específica tiene el atributo 'module' (está envuelta con DDP)
        if hasattr(self.actor, 'module'):
            return self.actor.module
        else:
            return self.actor
    
    def _get_critic_model(self, critic_network):
        """Obtiene el modelo crítico subyacente (sin envoltura DDP)."""
        # Verificar si la red específica tiene el atributo 'module' (está envuelta con DDP)
        if hasattr(critic_network, 'module'):
            return critic_network.module
        else:
            return critic_network
    
    def select_action(self, market_data: torch.Tensor, portfolio_data: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        """
        Selecciona una acción basada en los datos de mercado y portfolio.
        
        Args:
            market_data: Tensor con datos de mercado pre-procesados
            portfolio_data: Tensor con datos de portfolio pre-procesados
            deterministic: Si True, usa la media de la política (evaluación)
            
        Returns:
            Acción seleccionada
        """
        with torch.no_grad():
            actor_model = self._get_actor_model()
            mean, _ = actor_model(market_data, portfolio_data)
            
            if deterministic:
                action = torch.tanh(mean)
            else:
                # Para entrenamiento, muestreamos desde la distribución que se crea aquí
                mean, log_std = actor_model(market_data, portfolio_data)
                std = torch.exp(log_std)
                normal = torch.distributions.Normal(mean, std)
                x_t = normal.rsample()  # Reparameterization trick
                action = torch.tanh(x_t)
        
        return action.cpu().numpy().flatten()

    def sample_action(self, market_data: torch.Tensor, portfolio_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Muestrea una acción de la política y calcula su log_prob.
        Esto es usado durante el paso de aprendizaje.
        """
        actor_model = self._get_actor_model()
        mean, log_std = actor_model(market_data, portfolio_data)
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Para permitir backprop
        action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6) # Corrección de Jacobiano para tanh
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob
    
    def learn(
        self,
        market_data: torch.Tensor,
        portfolio_data: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_market_data: torch.Tensor,
        next_portfolio_data: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor
    ) -> Optional[Dict[str, float]]:
        """
        Realiza un paso de aprendizaje SAC.
        
        Args:
            market_data: Tensor con datos de mercado del batch
            portfolio_data: Tensor con datos de portfolio del batch
            actions: Tensor con acciones del batch
            rewards: Tensor con recompensas del batch
            next_market_data: Tensor con siguientes datos de mercado del batch
            next_portfolio_data: Tensor con siguientes datos de portfolio del batch
            terminated: Tensor con flags de terminación del batch
            truncated: Tensor con flags de truncamiento del batch
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        self.total_steps += 1
        
        # Verificar frecuencia de aprendizaje
        if self.total_steps % self.learning_frequency != 0:
            return None
        
        # Máscaras para episodios finalizados
        done_mask = terminated | truncated
        
        # Normalización de recompensas para estabilidad del crítico
        scaled_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        with torch.no_grad():
            # Obtener el modelo actor subyacente
            actor_model = self._get_actor_model()
            
            # Siguiente acción y log_prob usando la política actual
            next_actions, next_log_probs = self.sample_action(next_market_data, next_portfolio_data)
            
            # Q-valores objetivo (las redes objetivo nunca están envueltas en DDP)
            target_q1 = self.critic_target_1(next_market_data, next_portfolio_data, next_actions)
            target_q2 = self.critic_target_2(next_market_data, next_portfolio_data, next_actions)
            
            # Tomar el mínimo para reducir sobreestimación
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            
            # Calcular targets
            q_targets = scaled_rewards.unsqueeze(1) + self.gamma * (1 - done_mask.float().unsqueeze(1)) * target_q
        
        # Actualizar críticos
        with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
            # Obtener los modelos críticos subyacentes
            critic_1_model = self._get_critic_model(self.critic_1)
            critic_2_model = self._get_critic_model(self.critic_2)
            
            current_q1 = critic_1_model(market_data, portfolio_data, actions)
            current_q2 = critic_2_model(market_data, portfolio_data, actions)
            
            critic_1_loss = F.mse_loss(current_q1, q_targets)
            critic_2_loss = F.mse_loss(current_q2, q_targets)
        
        # Optimizar crítico 1
        self.critic_1_optimizer.zero_grad()
        self.scaler.scale(critic_1_loss).backward()
        self.scaler.step(self.critic_1_optimizer)
        
        # Optimizar crítico 2
        self.critic_2_optimizer.zero_grad()
        self.scaler.scale(critic_2_loss).backward()
        self.scaler.step(self.critic_2_optimizer)
        
        # Actualizar actor
        with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
            # Obtener los modelos subyacentes
            actor_model = self._get_actor_model()
            new_actions, log_probs = self.sample_action(market_data, portfolio_data)
            
            # Obtener los modelos críticos subyacentes
            critic_1_model = self._get_critic_model(self.critic_1)
            critic_2_model = self._get_critic_model(self.critic_2)
            
            q1_new = critic_1_model(market_data, portfolio_data, new_actions)
            q2_new = critic_2_model(market_data, portfolio_data, new_actions)
            q_new = torch.min(q1_new, q2_new)
            
            actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optimizer)
        
        # Actualizar alpha si es aprendible
        alpha_loss = torch.tensor(0.0)
        if self.learn_alpha:
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            self.scaler.scale(alpha_loss).backward()
            self.scaler.step(self.alpha_optimizer)
        
        # Actualizar el scaler después de todos los pasos de optimización
        self.scaler.update()
        
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
            'learning_steps': self.learning_steps
        }
        
        return metrics
    
    def _soft_update_target_networks(self) -> None:
        """Actualización suave de las redes objetivo."""
        critic1_model = self._get_critic_model(self.critic_1)
        critic2_model = self._get_critic_model(self.critic_2)
        
        for target_param, param in zip(self.critic_target_1.parameters(), critic1_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target_2.parameters(), critic2_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
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
        agent_stats = {
            'total_steps': self.total_steps,
            'learning_steps': self.learning_steps,
            'alpha': self.alpha.item(),
            'device': str(self.device)
        }
        
        return agent_stats
