"""
Replay Buffer para almacenar y muestrear experiencias del agente SAC.
"""

import numpy as np
import torch
from typing import Tuple, Optional
from collections import deque
import random
import logging
from .sumtree import SumTree

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


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer using SumTree for efficient sampling.
    
    This buffer implements prioritized experience replay as described in the paper
    "Prioritized Experience Replay" by Schaul et al. It uses a SumTree data structure
    for efficient O(log N) sampling based on TD-error priorities.
    """
    
    def __init__(
        self, 
        capacity: int, 
        observation_shape: Tuple[int, ...], 
        action_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4
    ):
        """
        Initialize the Prioritized Replay Buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
            observation_shape: Shape of observations
            action_dim: Dimension of action space
            alpha: Priority exponent (0 = uniform random, 1 = fully prioritized)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        """
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.alpha = alpha
        self.beta = beta
        
        # Small constants for numerical stability
        self.epsilon = 0.01  # Small value added to TD errors for priority calculation
        self.max_priority = 1.0  # Initial maximum priority for new experiences
        
        # Initialize SumTree for priority management
        self.tree = SumTree(capacity)
        
        # Numpy arrays for storing experience data (similar to standard ReplayBuffer)
        self.observations = np.zeros((capacity,) + observation_shape, dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity,) + observation_shape, dtype=np.float32)
        self.terminated_flags = np.zeros(capacity, dtype=bool)
        self.truncated_flags = np.zeros(capacity, dtype=bool)
        
        self.size = 0
        self.ptr = 0
        
        logger.info(f"PrioritizedReplayBuffer inicializado con capacidad {capacity}")
        logger.info(f"  - Observación shape: {observation_shape}")
        logger.info(f"  - Acción dim: {action_dim}")
        logger.info(f"  - Alpha: {alpha}, Beta: {beta}")
    
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
        Add a new transition to the buffer with maximum priority.
        
        New experiences are assigned the current maximum priority to ensure
        they get sampled at least once. The actual priority will be updated
        after the first learning step.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            terminated: Whether episode terminated naturally
            truncated: Whether episode was truncated
        """
        # Store transition data in numpy arrays
        self.observations[self.ptr] = observation
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_observation
        self.terminated_flags[self.ptr] = terminated
        self.truncated_flags[self.ptr] = truncated
        
        # Add to SumTree with maximum priority
        # Use current buffer position as data_index
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, self.ptr)
        
        # Update buffer pointers
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, device: torch.device = torch.device('cpu')) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor
    ]:
        """
        Sample a prioritized batch of transitions.
        
        This method implements stratified sampling where the priority range is divided
        into batch_size segments, and one sample is drawn from each segment.
        
        Args:
            batch_size: Size of the batch to sample
            device: Device to place the tensors on
            
        Returns:
            Tuple containing:
                - observations: Batch of observations
                - actions: Batch of actions  
                - rewards: Batch of rewards
                - next_observations: Batch of next observations
                - terminated: Batch of terminated flags
                - truncated: Batch of truncated flags
                - tree_indices: Tree indices for priority updates
                - is_weights: Importance sampling weights
        """
        if self.size < batch_size:
            raise ValueError(f"No hay suficientes muestras en el buffer. Size: {self.size}, Requested: {batch_size}")
        
        # Get total priority from SumTree
        total_priority = self.tree.total_priority
        
        if total_priority <= 0:
            raise ValueError("Total priority is zero or negative")
        
        # Calculate priority segments for stratified sampling
        segment_size = total_priority / batch_size
        
        # Storage for sampled data
        data_indices = []
        tree_indices = []
        priorities = []
        
        # Sample one transition from each segment
        for i in range(batch_size):
            # Calculate segment boundaries
            segment_start = i * segment_size
            segment_end = (i + 1) * segment_size
            
            # Sample uniformly within the segment
            sample_value = np.random.uniform(segment_start, segment_end)
            
            # Get transition from SumTree
            tree_idx, priority, data_idx = self.tree.get(sample_value)
            
            data_indices.append(data_idx)
            tree_indices.append(tree_idx)
            priorities.append(priority)
        
        # Convert to numpy arrays
        data_indices = np.array(data_indices)
        tree_indices = np.array(tree_indices)
        priorities = np.array(priorities)
        
        # Retrieve transitions from numpy arrays
        observations = self.observations[data_indices]
        actions = self.actions[data_indices]
        rewards = self.rewards[data_indices]
        next_observations = self.next_observations[data_indices]
        terminated = self.terminated_flags[data_indices]
        truncated = self.truncated_flags[data_indices]
        
        # Calculate importance sampling weights
        # Weight = (buffer_size * priority / total_priority) ** -beta
        sampling_probabilities = priorities / total_priority
        is_weights = (self.size * sampling_probabilities) ** (-self.beta)
        
        # Normalize weights by maximum weight in batch for stability
        max_weight = np.max(is_weights)
        is_weights = is_weights / max_weight
        
        # Convert to PyTorch tensors
        is_cuda = device.type == 'cuda'
        
        observations_tensor = torch.from_numpy(observations).to(device, non_blocking=is_cuda)
        actions_tensor = torch.from_numpy(actions).to(device, non_blocking=is_cuda)
        rewards_tensor = torch.from_numpy(rewards).to(device, non_blocking=is_cuda)
        next_observations_tensor = torch.from_numpy(next_observations).to(device, non_blocking=is_cuda)
        terminated_tensor = torch.from_numpy(terminated).to(device, non_blocking=is_cuda)
        truncated_tensor = torch.from_numpy(truncated).to(device, non_blocking=is_cuda)
        is_weights_tensor = torch.from_numpy(is_weights.astype(np.float32)).to(device, non_blocking=is_cuda)
        
        return (
            observations_tensor, actions_tensor, rewards_tensor, 
            next_observations_tensor, terminated_tensor, truncated_tensor,
            tree_indices, is_weights_tensor
        )
    
    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities in the SumTree based on TD errors.
        
        This method is called after a learning step to update the priorities
        of the sampled transitions based on their computed TD errors.
        
        Args:
            tree_indices: Array of tree indices from the sampled batch
            td_errors: Array of TD errors for the sampled transitions
        """
        # Calculate new priorities from TD errors
        # Priority = |TD_error| + epsilon
        priorities = np.abs(td_errors) + self.epsilon
        
        # Apply alpha exponent
        priorities = priorities ** self.alpha
        
        # Update priorities in SumTree
        for tree_idx, priority in zip(tree_indices, priorities):
            self.tree.update(tree_idx, priority)
        
        # Update maximum priority for new experiences
        max_new_priority = np.max(priorities)
        if max_new_priority > self.max_priority:
            self.max_priority = max_new_priority
    
    def __len__(self) -> int:
        """
        Return the current number of elements in the buffer.
        
        Returns:
            Number of stored elements
        """
        return self.size
    
    def can_sample(self, batch_size: int) -> bool:
        """
        Check if the buffer can provide a batch of the requested size.
        
        Args:
            batch_size: Desired batch size
            
        Returns:
            True if there are enough samples, False otherwise
        """
        return self.size >= batch_size
    
    def clear(self) -> None:
        """Clear the buffer completely."""
        self.size = 0
        self.ptr = 0
        self.max_priority = 1.0
        # Note: SumTree doesn't need explicit clearing as priorities will be overwritten
        logger.info("PrioritizedReplayBuffer limpiado")
    
    def get_fill_percentage(self) -> float:
        """
        Get the fill percentage of the buffer.
        
        Returns:
            Fill percentage (0.0 to 1.0)
        """
        return self.size / self.capacity
    
    def set_beta(self, beta: float) -> None:
        """
        Update the beta parameter for importance sampling.
        
        Beta is typically annealed from its initial value to 1.0 during training
        to gradually increase the correction for the biased sampling.
        
        Args:
            beta: New beta value for importance sampling
        """
        self.beta = beta
    
    def get_stats(self) -> dict:
        """
        Get buffer statistics including priority information.
        
        Returns:
            Dictionary with buffer statistics
        """
        if self.size == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'fill_percentage': 0.0,
                'ptr': self.ptr,
                'max_priority': self.max_priority,
                'total_priority': 0.0,
                'alpha': self.alpha,
                'beta': self.beta
            }
        
        # Calcular estadísticas de las recompensas
        rewards_slice = self.rewards[:self.size]
        
        return {
            'size': self.size,
            'capacity': self.capacity,
            'fill_percentage': self.get_fill_percentage(),
            'ptr': self.ptr,
            'max_priority': self.max_priority,
            'total_priority': float(self.tree.total_priority),
            'alpha': self.alpha,
            'beta': self.beta,
            'reward_mean': float(np.mean(rewards_slice)),
            'reward_std': float(np.std(rewards_slice)),
            'reward_min': float(np.min(rewards_slice)),
            'reward_max': float(np.max(rewards_slice)),
            'terminated_ratio': float(np.mean(self.terminated_flags[:self.size])),
            'truncated_ratio': float(np.mean(self.truncated_flags[:self.size]))
        }
