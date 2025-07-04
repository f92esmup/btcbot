"""
Abstracciones para las redes neuronales del agente SAC.
Aplica el Principio de Inversión de Dependencias (DIP) de SOLID.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class AbstractActor(nn.Module, ABC):
    """
    Interfaz abstracta para la red Actor.
    Define el contrato que debe cumplir cualquier implementación de Actor.
    """
    
    @abstractmethod
    def forward(self, market_data: torch.Tensor, portfolio_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass del Actor.
        
        Args:
            market_data: Tensor de forma (batch, seq_len, market_features)
            portfolio_data: Tensor de forma (batch, portfolio_features)
            
        Returns:
            Tupla (mean, log_std) para la distribución de la acción
        """
        pass


class AbstractCritic(nn.Module, ABC):
    """
    Interfaz abstracta para la red Critic.
    Define el contrato que debe cumplir cualquier implementación de Critic.
    """
    
    @abstractmethod
    def forward(self, market_data: torch.Tensor, portfolio_data: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del Critic.
        
        Args:
            market_data: Tensor de forma (batch, seq_len, market_features)
            portfolio_data: Tensor de forma (batch, portfolio_features)
            action: Tensor de forma (batch, action_dim)
            
        Returns:
            Q-value estimado de forma (batch, 1)
        """
        pass
