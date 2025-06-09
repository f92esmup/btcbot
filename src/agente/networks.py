"""
Arquitecturas de redes neuronales para el agente SAC con Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Codificación posicional para el Transformer.
    Puede ser sinusoidal fija o aprendible.
    """
    
    def __init__(self, d_model: int, max_len: int = 100, learnable: bool = False):
        """
        Args:
            d_model: Dimensión del modelo
            max_len: Longitud máxima de secuencia esperada
            learnable: Si True, usa embeddings aprendibles. Si False, usa sinusoidales fijas.
        """
        super().__init__()
        self.d_model = d_model
        self.learnable = learnable
        
        # Always create both attributes for JIT compatibility
        if learnable:
            # Embeddings posicionales aprendibles
            self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model))
            # Create dummy fixed encoding as buffer (won't be used but needed for JIT)
            dummy_pe = torch.zeros(max_len, d_model)
            self.register_buffer('pe', dummy_pe)
        else:
            # Codificación posicional sinusoidal fija
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe)
            # Create dummy learnable parameter (won't be used but needed for JIT)
            self.pos_embedding = nn.Parameter(torch.zeros(max_len, d_model), requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de forma (batch_size, seq_len, d_model)
            
        Returns:
            Tensor con codificación posicional añadida
        """
        seq_len = x.size(1)
        
        if self.learnable:
            pos_encoding = self.pos_embedding[:seq_len].unsqueeze(0)
        else:
            pos_encoding = self.pe[:seq_len].unsqueeze(0)
        
        return x + pos_encoding


class StateTransformerEncoder(nn.Module):
    """
    Codificador Transformer para procesar secuencias de observaciones de mercado.
    """
    
    def __init__(
        self,
        input_features: int,
        d_model: int = 128,
        n_head: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout_rate: float = 0.1,
        max_seq_len: int = 100
    ):
        """
        Args:
            input_features: Número de características por paso de tiempo
            d_model: Dimensión interna del Transformer
            n_head: Número de cabezales de atención
            num_encoder_layers: Número de capas del encoder
            dim_feedforward: Dimensión de la capa feedforward
            dropout_rate: Tasa de dropout
            max_seq_len: Longitud máxima de secuencia
        """
        super().__init__()
        
        self.input_features = input_features
        self.d_model = d_model
        
        # Capa de embedding lineal
        self.input_projection = nn.Linear(input_features, d_model)
        
        # Codificación posicional
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, learnable=False)
        
        # Capas del Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True  # Importante: (batch, seq, feature)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Dropout final
        self.dropout = nn.Dropout(dropout_rate)
        
        logger.info(f"StateTransformerEncoder inicializado:")
        logger.info(f"  - Input features: {input_features}")
        logger.info(f"  - d_model: {d_model}")
        logger.info(f"  - n_head: {n_head}")
        logger.info(f"  - num_layers: {num_encoder_layers}")
        logger.info(f"  - dim_feedforward: {dim_feedforward}")
        logger.info(f"  - dropout: {dropout_rate}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor de forma (batch_size, seq_len, input_features)
            mask: Máscara opcional para atención
            
        Returns:
            Tensor de forma (batch_size, d_model) con la representación de la secuencia
        """
        # Proyección de entrada
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Añadir codificación posicional
        x = self.positional_encoding(x)
        
        # Aplicar dropout
        x = self.dropout(x)
        
        # Pasar por el Transformer
        x = self.transformer_encoder(x, mask=mask)  # (batch, seq_len, d_model)
        
        # Tomar la salida del último paso de tiempo
        output = x[:, -1, :]  # (batch, d_model)
        
        return output


class ActorNetwork(nn.Module):
    """
    Red del Actor para el algoritmo SAC.
    Utiliza StateTransformerEncoder + MLP para generar parámetros de la política.
    """
    
    def __init__(
        self,
        market_features: int,
        portfolio_features: int,
        transformer_config: dict,
        mlp_hidden_dims: list,
        action_dim: int = 1,
        max_seq_len: int = 100
    ):
        """
        Args:
            market_features: Número de características de mercado por paso
            portfolio_features: Número de características del portfolio
            transformer_config: Configuración del Transformer
            mlp_hidden_dims: Dimensiones de las capas MLP ocultas
            action_dim: Dimensión del espacio de acción
            max_seq_len: Longitud máxima de secuencia
        """
        super().__init__()
        
        self.market_features = market_features
        self.portfolio_features = portfolio_features
        self.action_dim = action_dim
        
        # Encoder Transformer para datos de mercado
        self.transformer = StateTransformerEncoder(
            input_features=market_features,
            d_model=transformer_config['d_model'],
            n_head=transformer_config['n_head'],
            num_encoder_layers=transformer_config['num_encoder_layers'],
            dim_feedforward=transformer_config['dim_feedforward'],
            dropout_rate=transformer_config['dropout_rate'],
            max_seq_len=max_seq_len
        )
        
        # MLP head
        # Entrada: transformer output + portfolio features
        mlp_input_dim = transformer_config['d_model'] + portfolio_features
        
        # Capas ocultas
        layers = []
        prev_dim = mlp_input_dim
        
        for hidden_dim in mlp_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(transformer_config['dropout_rate'])
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Capas de salida para mean y log_std
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # Límites para log_std para estabilidad numérica
        self.log_std_min = -20
        self.log_std_max = 2
        
        logger.info(f"ActorNetwork inicializado:")
        logger.info(f"  - Market features: {market_features}")
        logger.info(f"  - Portfolio features: {portfolio_features}")
        logger.info(f"  - MLP input dim: {mlp_input_dim}")
        logger.info(f"  - Hidden dims: {mlp_hidden_dims}")
        logger.info(f"  - Action dim: {action_dim}")
    
    def forward(self, market_data: torch.Tensor, portfolio_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            market_data: Tensor de forma (batch, seq_len, market_features)
            portfolio_data: Tensor de forma (batch, portfolio_features)
            
        Returns:
            Tupla (mean, log_std) para la distribución de la acción
        """
        # Procesar datos de mercado con Transformer
        market_representation = self.transformer(market_data)  # (batch, d_model)
        
        # Concatenar con características del portfolio
        combined = torch.cat([market_representation, portfolio_data], dim=1)  # (batch, d_model + portfolio_features)
        
        # Pasar por MLP
        features = self.mlp(combined)
        
        # Generar mean y log_std
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        # Clamp log_std para estabilidad
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, market_data: torch.Tensor, portfolio_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Muestrea una acción de la política y calcula su log_prob.
        
        Returns:
            Tupla (action, log_prob)
        """
        mean, log_std = self.forward(market_data, portfolio_data)
        std = torch.exp(log_std)
        
        # Distribución normal
        normal = torch.distributions.Normal(mean, std)
        
        # Muestrear con reparametrización
        x_t = normal.rsample()  # Para permitir backprop
        
        # Aplicar tanh para asegurar acción en [-1, 1]
        action = torch.tanh(x_t)
        
        # Calcular log_prob con corrección de tanh
        log_prob = normal.log_prob(x_t)
        
        # Corrección de Jacobiano para tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob
    
    def log_prob(self, market_data: torch.Tensor, portfolio_data: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calcula log_prob para una acción dada.
        """
        mean, log_std = self.forward(market_data, portfolio_data)
        std = torch.exp(log_std)
        
        # Invertir tanh para obtener x_t
        # atanh está limitado a (-1, 1), así que clampeamos la acción
        action_clamped = torch.clamp(action, -0.999, 0.999)
        x_t = torch.atanh(action_clamped)
        
        # Distribución normal
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(x_t)
        
        # Corrección de Jacobiano para tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return log_prob


class CriticNetwork(nn.Module):
    """
    Red del Crítico para el algoritmo SAC.
    Estima Q(s,a) usando StateTransformerEncoder + MLP.
    """
    
    def __init__(
        self,
        market_features: int,
        portfolio_features: int,
        action_dim: int,
        transformer_config: dict,
        mlp_hidden_dims: list,
        max_seq_len: int = 100
    ):
        """
        Args:
            market_features: Número de características de mercado por paso
            portfolio_features: Número de características del portfolio
            action_dim: Dimensión del espacio de acción
            transformer_config: Configuración del Transformer
            mlp_hidden_dims: Dimensiones de las capas MLP ocultas
            max_seq_len: Longitud máxima de secuencia
        """
        super().__init__()
        
        self.market_features = market_features
        self.portfolio_features = portfolio_features
        self.action_dim = action_dim
        
        # Encoder Transformer para datos de mercado
        self.transformer = StateTransformerEncoder(
            input_features=market_features,
            d_model=transformer_config['d_model'],
            n_head=transformer_config['n_head'],
            num_encoder_layers=transformer_config['num_encoder_layers'],
            dim_feedforward=transformer_config['dim_feedforward'],
            dropout_rate=transformer_config['dropout_rate'],
            max_seq_len=max_seq_len
        )
        
        # MLP head
        # Entrada: transformer output + portfolio features + action
        mlp_input_dim = transformer_config['d_model'] + portfolio_features + action_dim
        
        # Capas ocultas
        layers = []
        prev_dim = mlp_input_dim
        
        for hidden_dim in mlp_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(transformer_config['dropout_rate'])
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Capa de salida para Q-value
        self.q_layer = nn.Linear(prev_dim, 1)
        
        logger.info(f"CriticNetwork inicializado:")
        logger.info(f"  - Market features: {market_features}")
        logger.info(f"  - Portfolio features: {portfolio_features}")
        logger.info(f"  - Action dim: {action_dim}")
        logger.info(f"  - MLP input dim: {mlp_input_dim}")
        logger.info(f"  - Hidden dims: {mlp_hidden_dims}")
    
    def forward(self, market_data: torch.Tensor, portfolio_data: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            market_data: Tensor de forma (batch, seq_len, market_features)
            portfolio_data: Tensor de forma (batch, portfolio_features)
            action: Tensor de forma (batch, action_dim)
            
        Returns:
            Q-value estimado de forma (batch, 1)
        """
        # Procesar datos de mercado con Transformer
        market_representation = self.transformer(market_data)  # (batch, d_model)
        
        # Concatenar representación de mercado, portfolio y acción
        combined = torch.cat([market_representation, portfolio_data, action], dim=1)
        
        # Pasar por MLP
        features = self.mlp(combined)
        
        # Obtener Q-value
        q_value = self.q_layer(features)
        
        return q_value
