#!/usr/bin/env python3
"""
Script para entrenar un agente de RL utilizando el algoritmo SAC
con la arquitectura personalizada basada en Transformer.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Añadir el directorio raíz del proyecto al path de Python
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Importaciones locales
from src.agent.rl_agent_manager import RLAgentManager
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logger = setup_logger("TrainRLAgent")


def parse_arguments():
    """
    Parsea los argumentos de la línea de comandos.
    
    Returns:
        Argumentos parseados
    """
    parser = argparse.ArgumentParser(description="Entrena un agente de RL para trading")
    
    parser.add_argument(
        "--config",
        type=str,
        default="src/config.yaml",
        help="Ruta al archivo de configuración centralizada"
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
        help="Ruta en GCS al modelo guardado para continuar el entrenamiento (formato: path/to/model.zip)"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Desactivar el uso de GPU incluso si está disponible"
    )
    
    return parser.parse_args()


def main():
    """
    Función principal para entrenar un agente de RL.
    """
    # Parsear argumentos
    args = parse_arguments()
    
    # Cargar la configuración centralizada
    config_manager = ConfigManager(config_path=args.config)
    agent_config = config_manager.get_agent_config()
    
    # Actualizar la configuración si se solicita no usar GPU
    if args.no_gpu:
        agent_config["use_gpu"] = False
        logger.info("Uso de GPU desactivado por argumento de línea de comandos")
    
    # Crear el administrador del agente con la configuración centralizada
    agent_manager = RLAgentManager(config_path=args.config)
    
    # Si se desactivó la GPU por argumento, aplicar la configuración al administrador
    if args.no_gpu:
        agent_manager.config["use_gpu"] = False
        agent_manager.device = "cpu"
    
    # Configurar el agente
    should_load_model = args.load_model is not None
    agent_manager.setup_agent(
        load_model=should_load_model,
        model_path=args.load_model
    )
    
    # Entrenar el agente
    agent_manager.train_agent(total_timesteps=args.timesteps)
    
    logger.info("Entrenamiento completado.")


if __name__ == "__main__":
    main()
