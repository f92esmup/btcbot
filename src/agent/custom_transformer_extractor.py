"""
Implementación personalizada del extractor de características basado en Transformer 
para el agente de trading SAC dentro de Stable Baselines3.
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PositionalEncoding(nn.Module):
    """
    Implementación del Positional Encoding Sinusoidal para el Transformer.
    """
    def __init__(self, d_model: int, dropout_rate: float = 0.1, max_len: int = 100):
        """
        Inicializa el encoding posicional.
        
        Args:
            d_model: Dimensión del modelo del Transformer
            dropout_rate: Tasa de dropout para regularización
            max_len: Longitud máxima de secuencia esperada
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Crear matriz de codificación posicional
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Registrar pe como un buffer (parte del estado del módulo pero no un parámetro)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor de forma [seq_len, batch_size, d_model] o [batch_size, seq_len, d_model]
        """
        # Asumimos que x tiene forma [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)


class CustomTransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Extractor de características personalizado que:
    1. Toma observaciones en formato Dict con 'market_features' y 'portfolio_features'
    2. Replica y concatena las características de la cartera con cada paso temporal de las características del mercado
    3. Procesa la secuencia resultante a través de una arquitectura Transformer
    4. Devuelve un vector de características para las redes del actor y del crítico
    """
    
    def __init__(self, 
                 observation_space: gym.spaces.Dict, 
                 market_features_key: str = "market_features", 
                 portfolio_features_key: str = "portfolio_features",
                 features_in_transformer: int = 28, 
                 d_model: int = 128, 
                 n_heads: int = 4, 
                 n_encoder_layers: int = 3, 
                 dim_feedforward: int = 512, 
                 dropout_rate: float = 0.1):
        """
        Inicializa el extractor de características basado en Transformer.
        
        Args:
            observation_space: Espacio de observación de gymnasium (Dict)
            market_features_key: Clave para acceder a las características del mercado en el Dict
            portfolio_features_key: Clave para acceder a las características del portafolio en el Dict
            features_in_transformer: Número total de características después de concatenar (mercado+portafolio)
            d_model: Dimensión del modelo del Transformer
            n_heads: Número de cabezas de atención
            n_encoder_layers: Número de capas del encoder Transformer
            dim_feedforward: Dimensión de la capa feed-forward interna de cada capa del Transformer
            dropout_rate: Tasa de dropout para regularización
        """
        # La dimensión de salida del extractor será d_model
        features_dim = d_model
        super().__init__(observation_space, features_dim)
        
        self.market_features_key = market_features_key
        self.portfolio_features_key = portfolio_features_key
        
        # Obtener las dimensiones del espacio de observación
        self.seq_length = observation_space[market_features_key].shape[0]
        self.n_market_features = observation_space[market_features_key].shape[1]
        self.n_portfolio_features = observation_space[portfolio_features_key].shape[0]
        
        # Verificar que features_in_transformer sea consistente
        assert features_in_transformer == (self.n_market_features + self.n_portfolio_features), \
            f"features_in_transformer ({features_in_transformer}) debe ser igual a la suma de las características " \
            f"de mercado ({self.n_market_features}) y del portafolio ({self.n_portfolio_features})"
        
        # Capa de embedding lineal para proyectar las características concatenadas a la dimensión del modelo
        self.input_embedding = nn.Linear(features_in_transformer, d_model)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout_rate=dropout_rate,
            max_len=self.seq_length
        )
        
        # Crear una máscara para padding (si es necesaria)
        # self.register_buffer('src_mask', self._generate_square_subsequent_mask(self.seq_length))
        
        # Capas del Encoder Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True  # Para usar secuencias en formato [batch, seq, features]
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Procesa las observaciones a través del extractor.
        
        Args:
            observations: Diccionario con las claves 'market_features' y 'portfolio_features'
                - market_features: Tensor de forma [batch_size, seq_length, n_market_features]
                - portfolio_features: Tensor de forma [batch_size, n_portfolio_features]
        
        Returns:
            Tensor de características procesadas de forma [batch_size, features_dim]
        """
        # Extraer los componentes de la observación
        market_features = observations[self.market_features_key]
        portfolio_features = observations[self.portfolio_features_key]
        
        batch_size = market_features.shape[0]
        
        # Replicar las características del portafolio para cada paso temporal
        # [batch_size, n_portfolio_features] -> [batch_size, seq_length, n_portfolio_features]
        portfolio_features_expanded = portfolio_features.unsqueeze(1).expand(-1, self.seq_length, -1)
        
        # Concatenar las características del mercado y del portafolio
        # [batch_size, seq_length, n_market_features + n_portfolio_features]
        combined_features = torch.cat([market_features, portfolio_features_expanded], dim=2)
        
        # Pasar a través de la capa de embedding
        # [batch_size, seq_length, d_model]
        embedded_features = self.input_embedding(combined_features)
        
        # Añadir Positional Encoding
        encoded_features = self.positional_encoding(embedded_features)
        
        # Pasar a través del Transformer Encoder
        # [batch_size, seq_length, d_model]
        transformer_output = self.transformer_encoder(encoded_features)
        
        # Extraer la representación final: usar el último token de la secuencia
        # [batch_size, d_model]
        features = transformer_output[:, -1, :]
        
        # Alternativa: se puede usar Global Average Pooling en la dimensión temporal
        # features = transformer_output.mean(dim=1)
        
        return features


# Funciones de utilidad para registro de la política personalizada con Stable Baselines3
def register_policy_with_custom_extractor(observation_space, action_space, policy_kwargs=None):
    """
    Registra la política MLP de SB3 con nuestro extractor personalizado.
    
    Args:
        observation_space: Espacio de observación
        action_space: Espacio de acción
        policy_kwargs: Argumentos adicionales para la política
        
    Returns:
        Una instancia de la política SB3 configurada con nuestro extractor personalizado
    """
    from stable_baselines3.sac.policies import SACPolicy
    
    # Asegurarse de que policy_kwargs está inicializado
    if policy_kwargs is None:
        policy_kwargs = {}
    
    # Si no se especifica la clase del extractor, usar nuestro CustomTransformerFeatureExtractor
    if "features_extractor_class" not in policy_kwargs:
        policy_kwargs["features_extractor_class"] = CustomTransformerFeatureExtractor
    
    # Crear la política SAC con el extractor personalizado
    policy = SACPolicy(
        observation_space=observation_space,
        action_space=action_space,
        **policy_kwargs
    )
    
    return policy
