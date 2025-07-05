"""
Training module for btcbot.

This module contains utilities and classes for training the SAC agent,
including evaluation functionality and specialized managers.
"""

from .evaluator import AgentEvaluator
from .trainer import Trainer
from .checkpoint_manager import CheckpointManager

__all__ = ['AgentEvaluator', 'Trainer', 'CheckpointManager']
