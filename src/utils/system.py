"""
Funciones de utilidad relacionadas con el sistema, logging y configuración de dispositivos.
"""

import logging
import sys
import os
import json
import random
import numpy as np
import torch
import torch.distributed as dist


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


def setup_environment_and_distribution() -> tuple[bool, int, int, int]:
    """
    Detecta si el script se está ejecutando en un entorno de entrenamiento distribuido,
    configura PyTorch distributed si es necesario, y devuelve la configuración del entorno.
    
    Esta función adhiere al Principio de Responsabilidad Única: su única responsabilidad
    es la detección y configuración del entorno distribuido.
    
    La función inspecciona las variables de entorno estándar de PyTorch (RANK, WORLD_SIZE, LOCAL_RANK)
    para determinar si debe inicializar el entrenamiento distribuido.
    
    Returns:
        tuple[bool, int, int, int]: Una tupla con:
            - is_distributed (bool): True si el entorno es distribuido, False en caso contrario
            - world_size (int): El número total de procesos (1 en modo no distribuido)
            - rank (int): El ID global del proceso actual (0 en modo no distribuido)
            - local_rank (int): El ID local del proceso en su máquina (0 en modo no distribuido)
    """
    # Obtener logger para esta función
    logger = logging.getLogger(__name__)
    
    # Detección de entorno Vertex AI
    if 'CLUSTER_SPEC' in os.environ:
        logger.info("Entorno de Vertex AI detectado - analizando CLUSTER_SPEC")
        
        try:
            # Parsear el contenido JSON de CLUSTER_SPEC
            cluster_spec = json.loads(os.environ['CLUSTER_SPEC'])
            logger.info(f"CLUSTER_SPEC parseado: {cluster_spec}")
            
            # Calcular el número total de workers en el clúster
            total_workers = 0
            for pool_name, workers in cluster_spec['cluster'].items():
                total_workers += len(workers)
            
            logger.info(f"Total de workers detectados en el clúster: {total_workers}")
            
            # Solo aplicar configuración multi-nodo si hay más de una máquina
            if total_workers > 1:
                logger.info("Escenario multi-nodo detectado - configurando comunicación entre máquinas")
                
                # Extraer MASTER_ADDR correctamente (eliminar puerto si existe)
                master_addr_raw = cluster_spec['cluster']['workerpool0'][0]
                # Eliminar el puerto de la dirección (ej: "hostname:2222" -> "hostname")
                master_addr = master_addr_raw.split(':')[0] if ':' in master_addr_raw else master_addr_raw
                master_port = '12355'  # Puerto estático
                
                # Obtener información del task actual directamente del cluster_spec
                task_type = cluster_spec['task']['type']
                task_index = int(cluster_spec['task']['index'])
                
                # Calcular RANK y WORLD_SIZE de forma robusta usando pools ordenados
                sorted_pools = sorted(cluster_spec['cluster'].keys())
                world_size = 0
                current_rank = 0
                
                # Calcular world_size total
                for pool_name in sorted_pools:
                    workers = cluster_spec['cluster'][pool_name]
                    world_size += len(workers)
                
                # Calcular el rank global de forma determinista
                for pool_name in sorted_pools:
                    workers = cluster_spec['cluster'][pool_name]
                    if pool_name == task_type:
                        current_rank += task_index
                        break
                    else:
                        current_rank += len(workers)
                
                # Establecer variables de entorno para torch.distributed
                os.environ['MASTER_ADDR'] = master_addr
                os.environ['MASTER_PORT'] = master_port
                os.environ['WORLD_SIZE'] = str(world_size)
                os.environ['RANK'] = str(current_rank)
                os.environ['LOCAL_RANK'] = '0'  # Asumimos 1 GPU por máquina en multi-nodo
                
                # Confirmar los valores establecidos
                logger.info(f"Variables de entorno multi-nodo establecidas:")
                logger.info(f"  MASTER_ADDR: {master_addr} (extraído de: {master_addr_raw})")
                logger.info(f"  MASTER_PORT: {master_port}")
                logger.info(f"  WORLD_SIZE: {world_size}")
                logger.info(f"  RANK: {current_rank}")
                logger.info(f"  LOCAL_RANK: 0")
                logger.info(f"  Task Type: {task_type}, Task Index: {task_index}")
                logger.info(f"  Pools procesados en orden: {sorted_pools}")
                
            else:
                logger.info("Escenario de un solo nodo detectado - delegando a torchrun para manejo multi-GPU local")
                logger.info("Las variables de entorno serán establecidas por torchrun automáticamente")
            
        except Exception as e:
            logger.error(f"Error al procesar CLUSTER_SPEC de Vertex AI: {e}")
            logger.info("Continuando con detección estándar de variables de entorno...")
    
    # Verificar si las variables de entorno de distribución están presentes
    rank_env = os.getenv('RANK')
    world_size_env = os.getenv('WORLD_SIZE')
    local_rank_env = os.getenv('LOCAL_RANK')
    
    # Verificar si estamos en un entorno distribuido
    if rank_env is not None and world_size_env is not None and local_rank_env is not None:
        try:
            # Convertir variables de entorno a enteros
            rank = int(rank_env)
            world_size = int(world_size_env)
            local_rank = int(local_rank_env)
            
            # Validar que los valores sean coherentes
            if rank < 0 or world_size <= 0 or local_rank < 0:
                raise ValueError(f"Valores inválidos en variables de entorno: RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
            
            if rank >= world_size:
                raise ValueError(f"RANK ({rank}) debe ser menor que WORLD_SIZE ({world_size})")
            
            logger.info(f"Entorno distribuido detectado: RANK={rank}, WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
            
            # Verificar si CUDA está disponible antes de configurar dispositivos
            if not torch.cuda.is_available():
                logger.warning("CUDA no está disponible, pero se detectó entorno distribuido. Esto puede causar problemas.")
            elif local_rank >= torch.cuda.device_count():
                logger.warning(f"LOCAL_RANK ({local_rank}) es mayor que el número de GPUs disponibles ({torch.cuda.device_count()})")
            
            # Inicializar el grupo de procesos distribuido usando backend NCCL (optimizado para GPUs NVIDIA)
            try:
                if not dist.is_initialized():
                    logger.info("Inicializando grupo de procesos distribuido con backend NCCL...")
                    dist.init_process_group(backend='nccl')
                    
                    # Asignar el dispositivo CUDA correcto al proceso actual
                    if torch.cuda.is_available():
                        torch.cuda.set_device(local_rank)
                        logger.info(f"Dispositivo CUDA {local_rank} asignado al proceso con RANK {rank}")
                    
                    logger.info(f"Entrenamiento distribuido inicializado exitosamente para proceso {rank}/{world_size}")
                else:
                    logger.info("El grupo de procesos distribuido ya está inicializado")
                    
            except Exception as e:
                logger.error(f"Error al inicializar el entrenamiento distribuido: {e}")
                logger.info("Continuando en modo no distribuido...")
                return False, 1, 0, 0
            
            return True, world_size, rank, local_rank
            
        except ValueError as e:
            logger.error(f"Error al procesar variables de entorno de distribución: {e}")
            logger.info("Continuando en modo no distribuido...")
            return False, 1, 0, 0
            
        except Exception as e:
            logger.error(f"Error inesperado al configurar entorno distribuido: {e}")
            logger.info("Continuando en modo no distribuido...")
            return False, 1, 0, 0
    
    else:
        # Entorno no distribuido (ejecución local en un solo proceso)
        missing_vars = []
        if rank_env is None:
            missing_vars.append('RANK')
        if world_size_env is None:
            missing_vars.append('WORLD_SIZE')
        if local_rank_env is None:
            missing_vars.append('LOCAL_RANK')
        
        if missing_vars:
            logger.info(f"Variables de entorno distribuido no encontradas: {', '.join(missing_vars)}")
        
        logger.info("Ejecutando en modo no distribuido (un solo proceso)")
        return False, 1, 0, 0
