"""Módulo del agente SAC con arquitectura Transformer para trading de futuros."""

from .agent import TransformerSACAgent
from .networks import ActorNetwork, CriticNetwork, StateTransformerEncoder
from .abstractions import AbstractActor, AbstractCritic
from .replay_buffer import ReplayBuffer
from .factory import create_sac_agent
from .observation_parser import parse_observation, parse_observation_batch

__all__ = [
    'TransformerSACAgent',
    'ActorNetwork', 
    'CriticNetwork',
    'StateTransformerEncoder',
    'AbstractActor',
    'AbstractCritic',
    'ReplayBuffer',
    'create_sac_agent',
    'parse_observation',
    'parse_observation_batch'
]
