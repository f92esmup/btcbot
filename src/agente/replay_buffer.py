"""
Replay Buffer para almacenar y muestrear experiencias del agente SAC.
"""

import numpy as np
import torch
from typing import Tuple, Optional
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Buffer de repetición para almacenar transiciones de experiencia del agente.
    
    Almacena tuplas (observación, acción, recompensa, siguiente_observación, terminated, truncated)
    y proporciona muestreo aleatorio para el entrenamiento.
    """
    
    def __init__(self, capacity: int, observation_shape: Tuple[int, ...], action_dim: int):
        """
        Inicializa el replay buffer.
        
        Args:
            capacity: Capacidad máxima del buffer
            observation_shape: Forma de las observaciones
            action_dim: Dimensión del espacio de acción
        """
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        
        # Usar arrays numpy para eficiencia de memoria
        self.observations = np.zeros((capacity,) + observation_shape, dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity,) + observation_shape, dtype=np.float32)
        self.terminated_flags = np.zeros(capacity, dtype=bool)
        self.truncated_flags = np.zeros(capacity, dtype=bool)
        
        self.size = 0
        self.ptr = 0
        
        logger.info(f"ReplayBuffer inicializado con capacidad {capacity}")
        logger.info(f"  - Observación shape: {observation_shape}")
        logger.info(f"  - Acción dim: {action_dim}")
    
    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
        truncated: bool
    ) -> None:
        """
        Añade una nueva transición al buffer.
        
        Args:
            observation: Observación actual
            action: Acción tomada
            reward: Recompensa recibida
            next_observation: Siguiente observación
            terminated: Si el episodio terminó por condición terminal
            truncated: Si el episodio terminó por truncamiento
        """
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_observation
        self.terminated_flags[self.ptr] = terminated
        self.truncated_flags[self.ptr] = truncated
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def __len__(self) -> int:
        """
        Retorna el número actual de elementos en el buffer.
        
        Returns:
            Número de elementos almacenados
        """
        return self.size
    
    def sample(self, batch_size: int, device: torch.device = torch.device('cpu')) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Muestrea un batch aleatorio de transiciones.
        
        Args:
            batch_size: Tamaño del batch a muestrear
            device: Dispositivo donde colocar los tensores
            
        Returns:
            Tupla con (observations, actions, rewards, next_observations, terminated, truncated)
        """
        if self.size < batch_size:
            raise ValueError(f"No hay suficientes muestras en el buffer. Size: {self.size}, Requested: {batch_size}")
        
        # Muestreo aleatorio
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Verificar si se está usando CUDA para transferencias no bloqueantes
        is_cuda = device.type == 'cuda'
        
        # Convertir a tensors de PyTorch con transferencias no bloqueantes
        observations = torch.from_numpy(self.observations[indices]).to(device, non_blocking=is_cuda)
        actions = torch.from_numpy(self.actions[indices]).to(device, non_blocking=is_cuda)
        rewards = torch.from_numpy(self.rewards[indices]).to(device, non_blocking=is_cuda)
        next_observations = torch.from_numpy(self.next_observations[indices]).to(device, non_blocking=is_cuda)
        terminated = torch.from_numpy(self.terminated_flags[indices]).to(device, non_blocking=is_cuda)
        truncated = torch.from_numpy(self.truncated_flags[indices]).to(device, non_blocking=is_cuda)
        
        return observations, actions, rewards, next_observations, terminated, truncated
    
    def can_sample(self, batch_size: int) -> bool:
        """
        Verifica si el buffer puede proporcionar un batch del tamaño solicitado.
        
        Args:
            batch_size: Tamaño del batch deseado
            
        Returns:
            True si hay suficientes muestras, False en caso contrario
        """
        return self.size >= batch_size
    
    def clear(self) -> None:
        """Limpia completamente el buffer."""
        self.size = 0
        self.ptr = 0
        logger.info("ReplayBuffer limpiado")
    
    def get_fill_percentage(self) -> float:
        """
        Obtiene el porcentaje de llenado del buffer.
        
        Returns:
            Porcentaje de llenado (0.0 a 1.0)
        """
        return self.size / self.capacity
    
    def get_stats(self) -> dict:
        """
        Obtiene estadísticas del buffer.
        
        Returns:
            Diccionario con estadísticas
        """
        if self.size == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'fill_percentage': 0.0,
                'ptr': self.ptr
            }
        
        # Calcular estadísticas de las recompensas
        rewards_slice = self.rewards[:self.size]
        
        return {
            'size': self.size,
            'capacity': self.capacity,
            'fill_percentage': self.get_fill_percentage(),
            'ptr': self.ptr,
            'reward_mean': float(np.mean(rewards_slice)),
            'reward_std': float(np.std(rewards_slice)),
            'reward_min': float(np.min(rewards_slice)),
            'reward_max': float(np.max(rewards_slice)),
            'terminated_ratio': float(np.mean(self.terminated_flags[:self.size])),
            'truncated_ratio': float(np.mean(self.truncated_flags[:self.size]))
        }
