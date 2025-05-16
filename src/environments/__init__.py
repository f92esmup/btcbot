import gymnasium as gym
from gymnasium.envs.registration import register
from .trading_env import TradingEnvironment

# Registra el entorno de trading en Gymnasium
register(
    id='FuturesTradingEnv-v0',
    entry_point='src.environments.trading_env:TradingEnvironment',
    max_episode_steps=None,  # Será establecido por el entorno basado en los datos
)

__all__ = ['TradingEnvironment']
