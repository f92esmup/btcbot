"""Módulo del agente SAC con arquitectura Transformer para trading de futuros."""

from .agent import TransformerSACAgent
from .networks import ActorNetwork, CriticNetwork, StateTransformerEncoder
from .abstractions import AbstractActor, AbstractCritic
from .replay_buffer import ReplayBuffer

__all__ = [
    'TransformerSACAgent',
    'ActorNetwork', 
    'CriticNetwork',
    'StateTransformerEncoder',
    'AbstractActor',
    'AbstractCritic',
    'ReplayBuffer'
]
