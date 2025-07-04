"""Módulo del agente SAC con arquitectura Transformer para trading de futuros."""

from .agent import TransformerSACAgent
from .networks import ActorNetwork, CriticNetwork, StateTransformerEncoder
from .abstractions import AbstractActor, AbstractCritic
from .replay_buffer import ReplayBuffer
from .factory import create_sac_agent

__all__ = [
    'TransformerSACAgent',
    'ActorNetwork', 
    'CriticNetwork',
    'StateTransformerEncoder',
    'AbstractActor',
    'AbstractCritic',
    'ReplayBuffer',
    'create_sac_agent'
]
