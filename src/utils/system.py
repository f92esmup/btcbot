"""
Funciones de utilidad relacionadas con el sistema, logging y configuración de dispositivos.
"""

import logging
import sys
import random
import numpy as np
import torch


def setup_logging():
    """Configura el sistema de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            #logging.FileHandler('trading_bot.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def set_seed(seed_value: int, logger_instance: logging.Logger) -> None:
    """
    Establece la semilla aleatoria para asegurar reproducibilidad del entrenamiento.
    
    Args:
        seed_value (int): Valor de la semilla aleatoria
        logger_instance (logging.Logger): Instancia del logger para mensajes
    """
    # Establecer semilla para Python random
    random.seed(seed_value)
    
    # Establecer semilla para NumPy
    np.random.seed(seed_value)
    
    # Establecer semilla para PyTorch (CPU)
    torch.manual_seed(seed_value)
    
    # Establecer semilla para PyTorch (GPU) si está disponible
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # Configuraciones adicionales para reproducibilidad en GPU
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        #ADVERTENCIA: ESTAS DOS CONFIGURACIONES ASEGURAN MUCHA REPRODUCIBILIDAD PERO REDUCEN MUCHO EL RENDIMIENTO. NO ES RECOMENDABLE
        logger_instance.info(f"Semilla {seed_value} establecida para Python, NumPy, PyTorch (CPU y GPU)")
    else:
        logger_instance.info(f"Semilla {seed_value} establecida para Python, NumPy, PyTorch (CPU)")


def setup_device(no_cuda: bool = False) -> torch.device:
    """
    Configura el device para el entrenamiento.
    
    Args:
        no_cuda (bool): Si True, fuerza el uso de CPU
        
    Returns:
        torch.device: Device configurado
    """
    if no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        # Configurar para mejor rendimiento
        torch.backends.cudnn.benchmark = True
    
    return device
