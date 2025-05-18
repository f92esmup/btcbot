"""
Custom Transformer feature extractor module for the RL agent.

This module contains the CustomTransformerFeatureExtractor class, which implements
a feature extractor using a Transformer architecture for processing sequential
market data and portfolio information.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any

from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer for transformer models.
    
    This implementation follows the original positional encoding described
    in "Attention is All You Need" using sine and cosine functions of different frequencies.
    
    Attributes:
        d_model (int): The embedding dimension.
        dropout (nn.Dropout): Dropout layer.
        pe (Tensor): The positional encoding tensor.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize the positional encoding layer.
        
        Args:
            d_model (int): The embedding dimension.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            max_len (int, optional): Maximum sequence length. Defaults to 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register pe buffer
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the positional encoding layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model).
            
        Returns:
            torch.Tensor: Output tensor with positional encoding added.
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CustomTransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor using a Transformer architecture for RL agent.
    
    This class processes sequential market data and portfolio information using
    a Transformer Encoder architecture. It handles Dictionary observation spaces
    with 'market_features' and 'portfolio_features' components.
    
    Attributes:
        features_dim (int): Output dimension of the feature extractor.
        market_embedding (nn.Linear): Linear embedding layer for market features.
        positional_encoding (PositionalEncoding): Positional encoding layer.
        transformer_encoder (nn.TransformerEncoder): Transformer encoder layers.
        portfolio_embedding (nn.Linear): Linear embedding layer for portfolio features.
        final_embedding (nn.Linear): Final linear layer to combine transformer and portfolio embeddings.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        dim_feedforward: int = 512,
        activation: str = "relu",
        dropout: float = 0.1
    ):
        """
        Initialize the CustomTransformerFeatureExtractor.
        
        Args:
            observation_space (spaces.Dict): Observation space with 'market_features' and 'portfolio_features'.
            features_dim (int, optional): Output dimension of the feature extractor. Defaults to 256.
            d_model (int, optional): Dimension of the transformer model. Defaults to 128.
            n_heads (int, optional): Number of attention heads. Defaults to 4.
            n_encoder_layers (int, optional): Number of transformer encoder layers. Defaults to 2.
            dim_feedforward (int, optional): Dimension of feedforward network. Defaults to 512.
            activation (str, optional): Activation function. Defaults to "relu".
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            
        Raises:
            ValueError: If observation_space is not a Dict with required keys.
        """
        super().__init__(observation_space, features_dim)
        
        # Check if observation space has the expected structure
        if not isinstance(observation_space, spaces.Dict) or \
           'market_features' not in observation_space.spaces or \
           'portfolio_features' not in observation_space.spaces:
            raise ValueError("Observation space must be a Dict with 'market_features' and 'portfolio_features' keys")
        
        market_space = observation_space.spaces['market_features']
        portfolio_space = observation_space.spaces['portfolio_features']
        
        # Get shapes from spaces
        if isinstance(market_space, spaces.Box):
            seq_len, n_market_features = market_space.shape
        else:
            raise ValueError("Market features space must be a Box space")
        
        if isinstance(portfolio_space, spaces.Box):
            n_portfolio_features = portfolio_space.shape[0]
        else:
            raise ValueError("Portfolio features space must be a Box space")
        
        # Linear embedding to convert market features to d_model dimensions
        self.market_embedding = nn.Linear(n_market_features, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False  # seq_len first (seq_len, batch, feature)
        )
        
        # Stack encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers
        )
        
        # Linear layer for portfolio features
        self.portfolio_embedding = nn.Linear(n_portfolio_features, d_model)
        
        # Final embedding layer
        self.final_embedding = nn.Linear(d_model * 2, features_dim)
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the feature extractor.
        
        Process market features through the transformer encoder and combine with
        portfolio features to create a comprehensive state representation.
        
        Args:
            observations (Dict[str, torch.Tensor]): Dictionary with 'market_features'
                and 'portfolio_features' tensors.
                
        Returns:
            torch.Tensor: Extracted features tensor.
        """
        # Extract components from the dict observation
        market_features = observations['market_features']  # (batch_size, seq_len, n_features)
        portfolio_features = observations['portfolio_features']  # (batch_size, n_portfolio_features)
        
        # Process market features
        batch_size, seq_len, _ = market_features.shape
        
        # Transpose to (seq_len, batch_size, n_features) for transformer
        market_features = market_features.transpose(0, 1)
        
        # Linear embedding and positional encoding
        market_embedded = self.market_embedding(market_features)  # (seq_len, batch_size, d_model)
        market_embedded = self.positional_encoding(market_embedded)
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(market_embedded)  # (seq_len, batch_size, d_model)
        
        # Use the last sequence element as the market representation
        market_representation = transformer_output[-1]  # (batch_size, d_model)
        
        # Process portfolio features
        portfolio_embedded = self.portfolio_embedding(portfolio_features)  # (batch_size, d_model)
        
        # Concatenate market and portfolio representations
        combined_features = torch.cat([market_representation, portfolio_embedded], dim=1)  # (batch_size, d_model*2)
        
        # Final embedding
        output = self.final_embedding(combined_features)  # (batch_size, features_dim)
        
        return output
