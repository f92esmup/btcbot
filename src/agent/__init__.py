"""
Inicializa el módulo del agente de Reinforcement Learning.
Expone las clases principales para que sean accesibles desde fuera.
"""

from src.agent.custom_transformer_extractor import CustomTransformerFeatureExtractor
from src.agent.rl_agent_manager import RLAgentManager

__all__ = ['CustomTransformerFeatureExtractor', 'RLAgentManager']
