#!/usr/bin/env python3
"""
Script para verificar la disponibilidad y funcionamiento de GPU con el entorno de trading y el agente.
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import time
from pathlib import Path

# Asegurar que se pueda importar desde el directorio raíz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importaciones locales
from src.agent.rl_agent_manager import RLAgentManager
from src.environments.trading_env import TradingEnvironment
from src.utils.config import ConfigManager

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GPUVerification")


def check_gpu_availability():
    """
    Verifica la disponibilidad de GPU en el sistema.
    """
    logger.info("=== Verificando disponibilidad de GPU ===")
    
    # Verificar CUDA (NVIDIA)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"✓ GPU CUDA disponible: {gpu_count} dispositivo(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
            logger.info(f"  - GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
            
        # Verificar memoria disponible en la primera GPU
        if gpu_count > 0:
            try:
                free_memory = torch.cuda.memory_reserved(0) / (1024 ** 3)
                allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
                logger.info(f"  - Memoria reservada: {free_memory:.2f} GB")
                logger.info(f"  - Memoria asignada: {allocated_memory:.2f} GB")
            except:
                pass
    else:
        logger.info("✗ CUDA no disponible")
    
    # Verificar MPS (Apple Silicon)
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("✓ MPS disponible (GPU Apple Silicon)")
            if torch.backends.mps.is_built():
                logger.info("  - PyTorch compilado con soporte MPS")
            else:
                logger.info("  - PyTorch no compilado con soporte MPS")
        else:
            logger.info("✗ MPS no disponible")
    except:
        logger.info("✗ Error verificando MPS")
    
    # Verificar versión de PyTorch
    logger.info(f"Versión de PyTorch: {torch.__version__}")
    
    logger.info("=========================================")


def test_tensor_operations():
    """
    Prueba operaciones básicas con tensores en diferentes dispositivos.
    """
    logger.info("=== Probando operaciones básicas de tensor ===")
    
    # Crear tensores de prueba en CPU
    logger.info("Creando tensores en CPU...")
    cpu_tensor = torch.rand(1000, 1000)
    
    # Medir tiempo de operación en CPU
    start_time = time.time()
    result_cpu = torch.matmul(cpu_tensor, cpu_tensor)
    cpu_time = time.time() - start_time
    logger.info(f"Tiempo operación en CPU: {cpu_time:.4f} segundos")
    
    # Probar en CUDA si está disponible
    if torch.cuda.is_available():
        logger.info("Creando tensores en CUDA...")
        cuda_tensor = cpu_tensor.to("cuda")
        
        # Medir tiempo de operación en CUDA (incluyendo transferencia)
        start_time = time.time()
        result_cuda = torch.matmul(cuda_tensor, cuda_tensor)
        # Forzar sincronización para medición precisa
        torch.cuda.synchronize() 
        cuda_time = time.time() - start_time
        logger.info(f"Tiempo operación en CUDA: {cuda_time:.4f} segundos")
        logger.info(f"Aceleración vs CPU: {cpu_time/cuda_time:.2f}x")
        
        # Verificar que los resultados sean similares
        diff = torch.abs(result_cpu - result_cuda.cpu()).max().item()
        logger.info(f"Diferencia máxima entre resultados CPU/CUDA: {diff}")
    
    # Probar en MPS si está disponible
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Creando tensores en MPS...")
        mps_tensor = cpu_tensor.to("mps")
        
        # Medir tiempo de operación en MPS
        start_time = time.time()
        result_mps = torch.matmul(mps_tensor, mps_tensor)
        mps_time = time.time() - start_time
        logger.info(f"Tiempo operación en MPS: {mps_time:.4f} segundos")
        logger.info(f"Aceleración vs CPU: {cpu_time/mps_time:.2f}x")
        
        # Verificar que los resultados sean similares
        diff = torch.abs(result_cpu - result_mps.cpu()).max().item()
        logger.info(f"Diferencia máxima entre resultados CPU/MPS: {diff}")
    
    logger.info("=========================================")


def test_environment_gpu_compatibility():
    """
    Prueba la compatibilidad del entorno de trading con GPU.
    """
    logger.info("=== Probando compatibilidad del entorno con GPU ===")
    
    # Crear el entorno
    try:
        env_config_path = "src/environments/environment_config.yaml"
        env = TradingEnvironment(config_path=env_config_path)
        logger.info("✓ Entorno creado correctamente")
        
        # Reiniciar el entorno
        obs, info = env.reset()
        logger.info("✓ Entorno reiniciado correctamente")
        logger.info(f"  - Forma de observación market_features: {obs['market_features'].shape}")
        logger.info(f"  - Forma de observación portfolio_features: {obs['portfolio_features'].shape}")
        
        # Si torch está disponible, probar la conversión a tensores
        if hasattr(env, 'get_torch_observation'):
            # Determinar el dispositivo a usar
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
                
            # Convertir a tensor
            logger.info(f"Convirtiendo observación a tensor en {device}...")
            torch_obs = env.get_torch_observation(obs, device=device)
            logger.info(f"✓ Conversión a tensor exitosa")
            logger.info(f"  - Dispositivo de tensores: {torch_obs['market_features'].device}")
            
            # Verificar forma de los tensores
            logger.info(f"  - Forma de tensor market_features: {torch_obs['market_features'].shape}")
            logger.info(f"  - Forma de tensor portfolio_features: {torch_obs['portfolio_features'].shape}")
    except Exception as e:
        logger.error(f"Error al probar el entorno: {str(e)}")
    
    logger.info("=========================================")


def test_agent_gpu():
    """
    Prueba el agente RL con GPU si está disponible.
    """
    logger.info("=== Probando agente RL con GPU ===")
    
    try:
        # Cargar configuración del agente
        agent_config_path = "src/agent/agent_config.yaml"
        
        # Crear el administrador del agente
        agent_manager = RLAgentManager(config_path=agent_config_path)
        logger.info(f"✓ RLAgentManager creado correctamente")
        logger.info(f"  - Dispositivo configurado: {agent_manager.device}")
        
        # Configurar el agente
        agent_manager.setup_agent(env_config_path="src/environments/environment_config.yaml")
        logger.info(f"✓ Agente configurado correctamente")
        
        # Verificar si el modelo está en el dispositivo correcto
        if hasattr(agent_manager.model, 'policy') and hasattr(agent_manager.model.policy, 'actor'):
            device = next(agent_manager.model.policy.actor.parameters()).device
            logger.info(f"  - Modelo cargado en dispositivo: {device}")
            
        # Realizar una predicción de prueba
        observation, _ = agent_manager.env.reset()
        action = agent_manager.predict_action(observation)
        logger.info(f"✓ Predicción exitosa: acción = {action}")
        
    except Exception as e:
        logger.error(f"Error al probar el agente: {str(e)}")
    
    logger.info("=========================================")


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Verificación de GPU para el bot de trading")
    parser.add_argument(
        "--no-tensor-test", 
        action="store_true",
        help="Omitir pruebas de operaciones con tensores"
    )
    parser.add_argument(
        "--no-env-test", 
        action="store_true",
        help="Omitir pruebas del entorno"
    )
    parser.add_argument(
        "--no-agent-test", 
        action="store_true",
        help="Omitir pruebas del agente"
    )
    return parser.parse_args()


def main():
    """Función principal."""
    args = parse_args()
    
    logger.info("Iniciando verificación de GPU para el bot de trading")
    
    # Verificar disponibilidad de GPU
    check_gpu_availability()
    
    # Prueba de operaciones con tensores
    if not args.no_tensor_test:
        test_tensor_operations()
    
    # Prueba de compatibilidad del entorno
    if not args.no_env_test:
        test_environment_gpu_compatibility()
    
    # Prueba del agente
    if not args.no_agent_test:
        test_agent_gpu()
    
    logger.info("Verificación de GPU completada")


if __name__ == "__main__":
    main()
