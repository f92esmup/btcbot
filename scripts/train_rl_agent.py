#!/usr/bin/env python3
"""
Script para entrenar un agente de RL utilizando el algoritmo SAC
con la arquitectura personalizada basada en Transformer.
"""

import os
import argparse
import logging
from pathlib import Path

# Importaciones locales
from src.agent.rl_agent_manager import RLAgentManager
from src.utils.config import load_config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TrainRLAgent")


def parse_arguments():
    """
    Parsea los argumentos de la línea de comandos.
    
    Returns:
        Argumentos parseados
    """
    parser = argparse.ArgumentParser(description="Entrena un agente de RL para trading")
    
    parser.add_argument(
        "--agent-config",
        type=str,
        default="src/agent/agent_config.yaml",
        help="Ruta al archivo de configuración del agente"
    )
    
    parser.add_argument(
        "--env-config",
        type=str,
        default="src/environments/environment_config.yaml",
        help="Ruta al archivo de configuración del entorno"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Número total de pasos de entrenamiento (si se omite, se usa el valor del config)"
    )
    
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Ruta a un modelo guardado para continuar el entrenamiento"
    )
    
    return parser.parse_args()


def main():
    """
    Función principal para entrenar un agente de RL.
    """
    # Parsear argumentos
    args = parse_arguments()
    
    # Crear el administrador del agente
    agent_manager = RLAgentManager(config_path=args.agent_config)
    
    # Configurar el agente
    should_load_model = args.load_model is not None
    agent_manager.setup_agent(
        env_config_path=args.env_config,
        load_model=should_load_model,
        model_path=args.load_model
    )
    
    # Entrenar el agente
    agent_manager.train_agent(total_timesteps=args.timesteps)
    
    logger.info("Entrenamiento completado.")


if __name__ == "__main__":
    main()
