"""
Script principal de entrenamiento del bot de trading de Bitcoin.
Orquesta la adquisición de datos, cálculo de indicadores y entrenamiento del modelo.
"""

import os
import sys
import logging
import tempfile
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import re

import yaml

from src.data.pipeline import DataPipeline
from src.entorno.factory import create_trading_environment
from src.agente.factory import create_sac_agent
from src.utils.observation_builder import ObservationBuilder
from src.utils.system import setup_logging, set_seed, setup_device, setup_environment_and_distribution
from src.utils.validation import validate_date_format
from src.utils.cli import parse_arguments
from src.analysis.logger import TensorboardLogger
from src.training import AgentEvaluator, Trainer
from src.training.checkpoint_manager import CheckpointManager
from src.data.artifact_manager import ArtifactManager
from src.configuration.config_manager import ConfigManager
from src.configuration.secret_utils import SecretManagerUtils
from src.configuration import AppConfig, EnvironmentConfig, AgentConfig
from src.configuration.constants import (
    CONFIG_PATH_DEFAULT, KEY_ENVIRONMENT, KEY_AGENT, KEY_GCP, 
    KEY_STORAGE_MODE, KEY_NORMALIZATION, KEY_BATCH_SIZE,
    KEY_MIN_BUFFER_FOR_LEARNING, KEY_REPLAY_BUFFER_SIZE,
    STORAGE_MODE_GCP, DEFAULT_NETWORK_INTERFACE,
    FILE_PRICE_SCALER, DIR_TRAINING_RUNS,
    KEY_EXPERIMENT_PARAMETERS, KEY_SYMBOL, KEY_INTERVAL,
    KEY_START_DATE, KEY_END_DATE,
    OPERATION_MODE_NEW_TRAINING, OPERATION_MODE_FINE_TUNING, OPERATION_MODE_RESUME_TRAINING
)


def _setup_distributed_environment() -> Tuple[bool, int, int, int, bool]:
    """
    Configure distributed environment and define process roles.
    
    Returns:
        Tuple containing:
        - is_distributed: Whether running in distributed mode
        - world_size: Total number of processes
        - rank: Global rank of current process
        - local_rank: Local rank within node
        - is_chief: Whether current process is the chief (rank 0)
    """
    is_distributed, world_size, rank, local_rank = setup_environment_and_distribution()
    is_chief = (rank == 0)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Log environment information
    logger.info("=== CONFIGURACIÓN DEL ENTORNO DE EJECUCIÓN ===")
    if is_distributed:
        logger.info(f"✅ Entorno DISTRIBUIDO detectado:")
        logger.info(f"  - World Size: {world_size} procesos")
        logger.info(f"  - Rank Global: {rank}")
        logger.info(f"  - Local Rank: {local_rank}")
        logger.info(f"  - Es Proceso Jefe (Chief): {'Sí' if is_chief else 'No'}")
    else:
        logger.info(f"📋 Entorno NO DISTRIBUIDO (un solo proceso):")
        logger.info(f"  - World Size: {world_size}")
        logger.info(f"  - Rank Global: {rank}")
        logger.info(f"  - Local Rank: {local_rank}")
        logger.info(f"  - Es Proceso Jefe (Chief): {'Sí' if is_chief else 'No'}")
    
    logger.info("=== Iniciando Bot de Trading de Bitcoin ===")
    
    return is_distributed, world_size, rank, local_rank, is_chief


def _determine_operation_mode(args) -> Tuple[str, str]:
    """
    Determine the operation mode based on command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple containing:
        - operation_mode: The determined operation mode
        - source_id: The source identifier (data_run_id or checkpoint)
    """
    logger = logging.getLogger(__name__)
    logger.info("=== DETERMINANDO MODO DE OPERACIÓN ===")
    
    if args.data_run_id:
        logger.info(f"🆕 MODO: Nuevo Entrenamiento desde data_run: {args.data_run_id}")
        operation_mode = OPERATION_MODE_NEW_TRAINING
        source_id = args.data_run_id
    elif args.checkpoint:
        if args.fine_tune_mode:
            logger.info(f"🔧 MODO: Fine-Tuning desde training_run: {args.checkpoint}")
            operation_mode = OPERATION_MODE_FINE_TUNING
        else:
            logger.info(f"▶️ MODO: Continuación desde training_run: {args.checkpoint}")
            operation_mode = OPERATION_MODE_RESUME_TRAINING
        source_id = args.checkpoint
    else:
        logger.error("❌ ERROR: Debe especificar --data-run-id o --checkpoint")
        sys.exit(1)
    
    return operation_mode, source_id


def _load_configuration(operation_mode: str, source_id: str, args) -> Tuple[Any, Dict[str, Any], str, bool]:
    """
    Load configuration based on operation mode.
    
    Args:
        operation_mode: The operation mode
        source_id: The source identifier
        args: Command line arguments
        
    Returns:
        Tuple containing:
        - local_config: The loaded configuration
        - training_run_lineage: Lineage information
        - data_run_id: The data run identifier
        - is_new_training: Whether this is a new training
    """
    logger = logging.getLogger(__name__)
    
    # Prepare gcp_config for loading operations
    gcp_config_for_load = None
    try:
        config = AppConfig.from_yaml_file(CONFIG_PATH_DEFAULT)
        if config.normalization.storage_mode == STORAGE_MODE_GCP:
            gcp_config_for_load = config.gcp.model_dump()
    except FileNotFoundError:
        logger.warning("No se encontró config.yaml local. Se asumirá que no se necesita gcp_config para cargar.")

    if operation_mode == OPERATION_MODE_NEW_TRAINING:
        # === NEW TRAINING FROM DATA_RUN ===
        logger.info(f"📊 Cargando configuración local para nuevo entrenamiento...")
        
        # Load local configuration as base
        try:
            local_config = AppConfig.from_yaml_file('src/configuration/config.yaml')
            logger.info("✅ Configuración local 'config.yaml' cargada exitosamente.")
        except FileNotFoundError:
            logger.error("❌ No se encontró el archivo 'src/configuration/config.yaml'. Abortando.")
            sys.exit(1)
        
        # Load data_run metadata to establish lineage
        logger.info(f"📋 Cargando metadatos del data_run: {args.data_run_id}")
        try:
            storage_mode = local_config.normalization.storage_mode
            artifact_manager = ArtifactManager(
                storage_mode=storage_mode,
                gcp_config=gcp_config_for_load
            )
            
            data_run_metadata = artifact_manager.load_data_run_metadata(args.data_run_id)
            
            if KEY_EXPERIMENT_PARAMETERS not in data_run_metadata:
                raise ValueError("Metadatos del data_run inválidos: falta 'experiment_parameters'")
                
            logger.info(f"📊 Dataset Info - Símbolo: {data_run_metadata[KEY_EXPERIMENT_PARAMETERS][KEY_SYMBOL]}, "
                       f"Intervalo: {data_run_metadata[KEY_EXPERIMENT_PARAMETERS][KEY_INTERVAL]}, "
                       f"Período: {data_run_metadata[KEY_EXPERIMENT_PARAMETERS][KEY_START_DATE]} → "
                       f"{data_run_metadata[KEY_EXPERIMENT_PARAMETERS].get(KEY_END_DATE, 'ahora')}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando metadatos del data_run '{args.data_run_id}': {e}")
            sys.exit(1)
            
        # Variables for new training
        is_new_training = True
        data_run_id = args.data_run_id
        training_run_lineage = {
            'data_run_id': args.data_run_id,
            'data_run_creation_timestamp': data_run_metadata['data_run_info']['creation_timestamp'],
            'data_run_description': data_run_metadata['data_run_info']['description']
        }
        
    else:
        # === RESUME OR FINE-TUNE FROM TRAINING_RUN ===
        logger.info(f"🔄 Cargando configuración desde training_run: {args.checkpoint}")
        
        # Load configuration from previous training_run
        config_dict = ConfigManager.load_training_run_config(args.checkpoint, gcp_config=gcp_config_for_load)
        if not config_dict:
            logger.error(f"❌ No se pudo cargar la configuración para el training_run: {args.checkpoint}. Abortando.")
            sys.exit(1)
        
        if operation_mode == "fine_tuning":
            # Fine-tuning: use local config but maintain lineage
            logger.info("🔧 Modo Fine-Tuning: usando configuración local con linaje del training_run original")
            try:
                local_config = AppConfig.from_yaml_file('src/configuration/config.yaml')
                logger.info("✅ Configuración local cargada para fine-tuning.")
            except FileNotFoundError:
                logger.error("❌ No se encontró el archivo 'src/configuration/config.yaml'. Abortando.")
                sys.exit(1)
        else:
            # Resume: use checkpoint configuration
            local_config = AppConfig(**config_dict.get('config', {}))
            logger.info(f"✅ Configuración del training_run '{args.checkpoint}' cargada como fuente de verdad.")
        
        # Extract data_run_id from lineage
        training_run_lineage = config_dict.get('lineage', {})
        if 'data_run_id' not in training_run_lineage:
            logger.error("❌ El training_run especificado no contiene información de linaje (data_run_id)")
            sys.exit(1)
            
        data_run_id = training_run_lineage['data_run_id']
        logger.info(f"📊 Data_run original identificado: {data_run_id}")
        
        # Variables for existing training
        is_new_training = False

    def _override_config_with_args(config, args):
        """
        Override configuration values with command line arguments if provided.
        
        Args:
            config: The configuration object to modify
            args: Command line arguments
        """
        logger = logging.getLogger(__name__)
        
        # Check and override agent hyperparameters
        if args.actor_learning_rate is not None:
            config.agent.hiperparametros_sac.actor_learning_rate = args.actor_learning_rate
            logger.info(f"🔧 Sobrescribiendo actor_learning_rate: {args.actor_learning_rate}")
            
        if args.critic_learning_rate is not None:
            config.agent.hiperparametros_sac.critic_learning_rate = args.critic_learning_rate
            logger.info(f"🔧 Sobrescribiendo critic_learning_rate: {args.critic_learning_rate}")
            
        if args.alpha_learning_rate is not None:
            config.agent.hiperparametros_sac.alpha_learning_rate = args.alpha_learning_rate
            logger.info(f"🔧 Sobrescribiendo alpha_learning_rate: {args.alpha_learning_rate}")
            
        if args.batch_size is not None:
            config.agent.batch_size = args.batch_size
            logger.info(f"🔧 Sobrescribiendo batch_size: {args.batch_size}")
            
        if args.tau is not None:
            config.agent.hiperparametros_sac.tau = args.tau
            logger.info(f"🔧 Sobrescribiendo tau: {args.tau}")
            
        if args.per_alpha is not None:
            config.agent.hiperparametros_sac.per_alpha = args.per_alpha
            logger.info(f"🔧 Sobrescribiendo per_alpha: {args.per_alpha}")
            
        if args.per_beta is not None:
            config.agent.hiperparametros_sac.per_beta = args.per_beta
            logger.info(f"🔧 Sobrescribiendo per_beta: {args.per_beta}")
    
    # Apply command line argument overrides
    _override_config_with_args(local_config, args)

    return local_config, training_run_lineage, data_run_id, is_new_training


def _extract_experiment_params(artifact_manager: ArtifactManager, data_run_id: str, local_config) -> Tuple[str, str, str, Optional[str], int]:
    """
    Extract experiment parameters from data_run metadata.
    
    Args:
        artifact_manager: The artifact manager instance
        data_run_id: The data run identifier
        local_config: The local configuration
        
    Returns:
        Tuple containing:
        - symbol: Trading symbol
        - interval: Trading interval
        - start_date: Start date
        - end_date: End date (optional)
        - seed: Random seed
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load data_run metadata to get experiment definition
        data_run_metadata = artifact_manager.load_data_run_metadata(data_run_id)
        exp_params = data_run_metadata['experiment_parameters']
        symbol = exp_params['symbol']
        interval = exp_params['interval']
        start_date = exp_params['start_date']
        end_date = exp_params.get('end_date')

        # Load seed from configuration
        seed = local_config.training_setup.seed
        
        logger.info(f"Configuración del experimento cargada desde '{data_run_id}': {symbol}/{interval}")
        logger.info(f"Rango de fechas: {start_date} - {end_date if end_date else 'presente'}")
        logger.info(f"Semilla de entrenamiento cargada desde config.yaml: {seed}")

        return symbol, interval, start_date, end_date, seed

    except KeyError as e:
        logger.error(f"Falta la clave de configuración requerida: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error cargando la configuración del experimento desde los metadatos: {e}")
        sys.exit(1)


def _synchronize_run_id(is_chief: bool, is_distributed: bool, is_new_training: bool, 
                       operation_mode: str, symbol: str, interval: str, seed: int, 
                       args, rank: int, local_rank: int) -> str:
    """
    Generate and synchronize run_id across all processes.
    
    Args:
        is_chief: Whether current process is chief
        is_distributed: Whether running in distributed mode
        is_new_training: Whether this is a new training
        operation_mode: The operation mode
        symbol: Trading symbol
        interval: Trading interval
        seed: Random seed
        args: Command line arguments
        rank: Process rank
        local_rank: Local rank
        
    Returns:
        The synchronized run_id
    """
    logger = logging.getLogger(__name__)
    
    # Only chief process generates training_run_id, then synchronizes with all processes
    if is_chief:
        # Check if run_id was provided via command line argument
        if args.run_id is not None:
            run_id = args.run_id
            logger.info(f"[Proceso Jefe] Usando run_id proporcionado por pipeline: {run_id}")
        elif is_new_training:
            # Generate new training_run_id for training from data_run
            current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
            run_id = f"training_{symbol}_{interval}_{seed}_{current_time}"
            logger.info(f"[Proceso Jefe] Nuevo Training Run ID generado: {run_id}")
        else:
            # For resume and fine-tuning, use same run_id or generate new one for fine-tuning
            if operation_mode == "fine_tuning":
                # Fine-tuning: generate new run_id but maintain lineage
                current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
                run_id = f"finetune_{symbol}_{interval}_{seed}_{current_time}"
                logger.info(f"[Proceso Jefe] Fine-Tuning Run ID generado: {run_id}")
            else:
                # Resume: use same run_id from checkpoint
                run_id = args.checkpoint
                logger.info(f"[Proceso Jefe] Reanudando Training Run ID: {run_id}")
    else:
        # Non-chief processes initialize run_id as None, will be synchronized later
        run_id = None
    
    # Synchronize run_id in distributed environments
    if is_distributed:
        # Configure device for synchronization
        sync_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        if is_chief:
            # Encode string run_id to byte tensor
            run_id_bytes = torch.tensor(bytearray(run_id, "utf-8"), dtype=torch.uint8, device=sync_device)
            # Create tensor for size and broadcast it
            size_tensor = torch.tensor([len(run_id_bytes)], dtype=torch.long, device=sync_device)
            dist.broadcast(size_tensor, src=0)
            # Broadcast byte tensor
            dist.broadcast(run_id_bytes, src=0)
            logger.info(f"[Proceso Jefe] run_id transmitido a todos los procesos")
        else:
            # Receive run_id size
            size_tensor = torch.empty(1, dtype=torch.long, device=sync_device)
            dist.broadcast(size_tensor, src=0)
            # Prepare empty tensor of correct size to receive bytes
            run_id_bytes = torch.empty(size_tensor[0].item(), dtype=torch.uint8, device=sync_device)
            dist.broadcast(run_id_bytes, src=0)
            # Decode bytes back to string
            run_id = run_id_bytes.cpu().numpy().tobytes().decode("utf-8")
        
        # All processes now have the same run_id
        logger.info(f"[Proceso {rank}] run_id sincronizado: {run_id}")
    
    return run_id


def _initialize_managers(local_config, rank: int) -> Tuple[CheckpointManager, ArtifactManager, ConfigManager]:
    """
    Initialize management components.
    
    Args:
        local_config: The local configuration
        rank: Process rank
        
    Returns:
        Tuple containing the initialized managers
    """
    logger = logging.getLogger(__name__)
    
    storage_mode = local_config.normalization.storage_mode
    gcp_config = None
    if storage_mode == "gcp":
        gcp_config = local_config.gcp.model_dump()
        logger.info(f"[Proceso {rank}] Configuración GCP cargada para los managers")

    # Create specialized instances for ALL processes
    checkpoint_manager = CheckpointManager(
        storage_mode=storage_mode,
        gcp_config=gcp_config
    )
    artifact_manager = ArtifactManager(
        storage_mode=storage_mode,
        gcp_config=gcp_config
    )
    config_manager = ConfigManager(
        storage_mode=storage_mode,
        gcp_config=gcp_config
    )
    logger.info(f"[Proceso {rank}] Managers especializados creados para modo: {storage_mode}")
    
    return checkpoint_manager, artifact_manager, config_manager


def _initialize_logging_components(is_chief: bool, run_id: str, rank: int) -> Tuple[Optional[str], Optional[Any]]:
    """
    Initialize logging components for chief process.
    
    Args:
        is_chief: Whether current process is chief
        run_id: The run identifier
        rank: Process rank
        
    Returns:
        Tuple containing tensorboard_base_dir and tb_logger
    """
    logger = logging.getLogger(__name__)
    
    if is_chief:
        logger.info("=== INICIALIZACIÓN DE COMPONENTES ADICIONALES (PROCESO JEFE) ===")

        # Simplified TensorBoard logic for local operation
        tensorboard_base_dir = "tensorboard_logs"
        logger.info(f"Configurando TensorBoard para logging local en directorio base: {tensorboard_base_dir}")

        # Initialize TensorBoard Logger with new simplified interface
        tb_logger = TensorboardLogger(
            log_dir=tensorboard_base_dir,
            run_id=run_id
        )
        
        logger.info(f"TensorBoard logs se guardarán en: {tensorboard_base_dir}/{run_id}")
        
        return tensorboard_base_dir, tb_logger
    else:
        # Non-chief processes initialize management variables they don't use as None
        logger.info(f"[Proceso {rank}] Variables de logging inicializadas como None")
        return None, None


def main():
    """Función principal del script."""
    # === PASO 1: CONFIGURACIÓN DEL ENTORNO DISTRIBUIDO ===
    is_distributed, world_size, rank, local_rank, is_chief = _setup_distributed_environment()
    logger = logging.getLogger(__name__)
    
    # === PASO 2: DETERMINAR MODO DE OPERACIÓN ===
    args = parse_arguments()
    operation_mode, source_id = _determine_operation_mode(args)

    # === PASO 3: CARGAR CONFIGURACIÓN Y LINAJE ===
    local_config, training_run_lineage, data_run_id, is_new_training = _load_configuration(
        operation_mode, source_id, args
    )

    # === PASO 4: EXTRACCIÓN DE PARÁMETROS DEL EXPERIMENTO ===
    # Create ArtifactManager instance for ALL processes
    storage_mode = local_config.normalization.storage_mode
    gcp_config = local_config.gcp.model_dump() if storage_mode == STORAGE_MODE_GCP else None
    artifact_manager = ArtifactManager(
        storage_mode=storage_mode,
        gcp_config=gcp_config
    )
    logger.info(f"[Proceso {rank}] ArtifactManager creado para modo: {storage_mode}")

    symbol, interval, start_date, end_date, seed = _extract_experiment_params(
        artifact_manager, data_run_id, local_config
    )

    # Configure random seed for reproducibility
    set_seed(seed, logger)

    # === PASO 5: GENERACIÓN Y SINCRONIZACIÓN DEL RUN_ID ===
    run_id = _synchronize_run_id(
        is_chief, is_distributed, is_new_training, operation_mode, 
        symbol, interval, seed, args, rank, local_rank
    )

    # === PASO 6: INICIALIZACIÓN DE COMPONENTES DE GESTIÓN ===
    checkpoint_manager, artifact_manager, config_manager = _initialize_managers(local_config, rank)
    
    # Assemble complete run configuration (available for all processes)
    full_run_config = {
        'run_info': {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'storage_mode': storage_mode,
            'base_path': config_manager._get_training_run_prefix(run_id),
            'operation_mode': operation_mode
        },
        'command_line_args': vars(args),
        'config': local_config.model_dump(),  # Use model_dump for Pydantic v2
        'lineage': training_run_lineage,  # Data_run origin information
        'metadata': {
            'experiment_parameters': {
                'symbol': symbol,
                'interval': interval,
                'start_date': start_date,
                'end_date': end_date
            }
        }
    }

    # === PASO 7: INICIALIZACIÓN DE COMPONENTES DE LOGGING ===
    tensorboard_base_dir, tb_logger = _initialize_logging_components(is_chief, run_id, rank)

    # Only chief process saves configuration for new trainings and fine-tuning
    if is_chief and (is_new_training or operation_mode == "fine_tuning"):
        try:
            config_manager.save_run_config(run_id, full_run_config)
            logger.info(f"✅ Configuración del training_run guardada con linaje a data_run: {data_run_id}")
        except Exception as e:
            logger.error(f"❌ Error al guardar config_training_run.yaml: {e}")
            # Continue execution as this error is not critical
def _load_training_data(artifact_manager: ArtifactManager, data_run_id: str, rank: int) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """
    Load training data from data_run.
    
    Args:
        artifact_manager: The artifact manager instance
        data_run_id: The data run identifier
        rank: Process rank
        
    Returns:
        Tuple containing dataframe, scaler, price_scaler, and data_run_metadata
    """
    logger = logging.getLogger(__name__)
    logger.info("=== CARGANDO DATOS DESDE DATA_RUN ===")
    
    # All processes load data from specified data_run using ArtifactManager
    logger.info(f"📊 Cargando artefactos desde data_run: {data_run_id}")
    
    # Load data artifacts using the new centralized method
    normalized_dataframe, scaler, price_scaler = artifact_manager.load_data_artifacts(data_run_id)
    
    # Load data run metadata using the new method
    data_run_metadata = artifact_manager.load_data_run_metadata(data_run_id)
    
    # Assign DataFrame for the rest of the code
    dataframe = normalized_dataframe
    
    logger.info(f"📊 Datos cargados exitosamente:")
    logger.info(f"  • DataFrame shape: {dataframe.shape}")
    logger.info(f"  • Rango temporal: {dataframe.index.min()} → {dataframe.index.max()}")
    logger.info(f"  • Columnas: {len(dataframe.columns)}")
    logger.info(f"  • Data_run origen: {data_run_id}")
    logger.info(f"  • Símbolo: {data_run_metadata['experiment_parameters']['symbol']}")
    logger.info(f"  • Intervalo: {data_run_metadata['experiment_parameters']['interval']}")
    logger.info(f"  • Período: {data_run_metadata['experiment_parameters']['start_date']} → "
               f"{data_run_metadata['experiment_parameters'].get('end_date', 'ahora')}")
    
    return dataframe, scaler, price_scaler, data_run_metadata


def _create_environment_and_agent(dataframe, logger, price_scaler, scaler, local_config, 
                                 full_run_config, args, rank: int, is_distributed: bool) -> Tuple[Any, Any]:
    """
    Create trading environment and agent.
    
    Args:
        dataframe: Trading data
        logger: Console logger
        price_scaler: Price scaler
        scaler: Feature scaler
        local_config: Local configuration
        full_run_config: Full run configuration
        args: Command line arguments
        rank: Process rank
        is_distributed: Whether running in distributed mode
        
    Returns:
        Tuple containing environment and agent
    """
    console_logger = logging.getLogger(__name__)
    console_logger.info(f"=== FASE 4: Creación del Entorno y Agente [Proceso {rank}] ===")
    
    # Configure device
    device = setup_device(args.no_cuda)
    console_logger.info(f"[Proceso {rank}] Usando device: {device}")
    
    # Create trading environment (all processes)
    console_logger.info(f"[Proceso {rank}] Creando entorno de trading...")
    env = create_trading_environment(
        dataframe,  # Available in all processes
        logger,
        price_scaler,  # Scaler already loaded via RunManager
        scaler,  # Main scaler for features
        env_config=local_config.environment, # Injection with Pydantic object
        run_config=full_run_config  # Complete run configuration
    )
    
    # Create agent (all processes) - CRUCIAL: Pass is_distributed
    console_logger.info(f"[Proceso {rank}] Creando agente SAC...")
    agent = create_sac_agent(env, device, logger, agent_config=local_config.agent, is_distributed=is_distributed)
    
    return env, agent


def _handle_checkpoint_loading(is_chief: bool, args, checkpoint_manager: CheckpointManager, 
                              agent, storage_mode: str, rank: int, run_id: str) -> int:
    """
    Handle checkpoint loading logic.
    
    Args:
        is_chief: Whether current process is chief
        args: Command line arguments
        checkpoint_manager: Checkpoint manager instance
        agent: The agent instance
        storage_mode: Storage mode
        rank: Process rank
        run_id: Run identifier for logging purposes
        
    Returns:
        The starting episode number
    """
    logger = logging.getLogger(__name__)
    logger.info(f"=== FASE 5: Gestión de Checkpoints [Proceso {rank}] ===")
    
    start_episode = 0
    
    # Only chief process handles checkpoint logic
    if is_chief:
        logger.info("[Proceso Jefe] Determinando configuración de checkpoint...")
        
        if args.checkpoint is None:
            # No checkpoint specified, start from scratch
            logger.info("[Proceso Jefe] Iniciando nueva ejecución (sin checkpoint)")
            start_episode = 0
        else:
            # A run_id was specified to load checkpoint
            logger.info(f"[Proceso Jefe] Intentando reanudar desde checkpoint del run_id: {args.checkpoint}")
            
            # Search for checkpoint in specific run_id using CheckpointManager (READ ONLY)
            checkpoint_info = checkpoint_manager.find_latest_checkpoint(args.checkpoint)
            
            if checkpoint_info:
                checkpoint_prefix, latest_episode = checkpoint_info
                
                logger.info(f"✅ Checkpoint encontrado del episodio {latest_episode}")
                logger.info(f"Ubicación: {checkpoint_prefix}")
                
                try:
                    logger.info(f"[Proceso Jefe] Cargando checkpoint desde: {checkpoint_prefix}")
                    # CHECKPOINT LOADING - Only chief loads agent state
                    checkpoint_manager.load_agent_from_checkpoint(agent, checkpoint_prefix, reset_optimizers=args.fine_tune_mode)
                    
                    start_episode = latest_episode
                    
                    logger.info(f"✅ Checkpoint cargado exitosamente en el proceso jefe")
                    logger.info(f"  - Run ID de origen: {args.checkpoint}")
                    logger.info(f"  - Episodio inicial: {start_episode}")
                    logger.info(f"  - Total steps: {agent.total_steps}")
                    logger.info(f"  - Learning steps: {agent.learning_steps}")
                    logger.info(f"  - Nuevos artefactos se guardarán en run_id: {run_id}")
                    
                    # Update price_scaler configuration to load from checkpoint
                    if storage_mode == "gcp":
                        blob_name_price_scaler_a_cargar = f"{args.checkpoint}/{FILE_PRICE_SCALER}"
                        logger.info(f"[Proceso Jefe] Actualizando price_scaler desde GCS (checkpoint): {blob_name_price_scaler_a_cargar}")
                    else:
                        path_price_scaler_a_cargar = f"Entrenamientos/{args.checkpoint}/{FILE_PRICE_SCALER}"
                        logger.info(f"[Proceso Jefe] Actualizando price_scaler desde local (checkpoint): {path_price_scaler_a_cargar}")
                    
                except Exception as e:
                    logger.error(f"❌ Error al cargar checkpoint desde run_id {args.checkpoint}: {e}")
                    logger.error("Terminando ejecución. Verifique que el run_id sea válido y contenga checkpoints.")
                    raise
            else:
                logger.error(f"❌ No se encontraron checkpoints en el run_id: {args.checkpoint}")
                logger.error("Terminando ejecución. Verifique que el run_id sea válido y contenga checkpoints.")
                raise RuntimeError(f"No checkpoints found in run_id: {args.checkpoint}")
    
    return start_episode


def _synchronize_initial_state(is_distributed: bool, is_chief: bool, start_episode: int, 
                              agent, rank: int, local_rank: int) -> int:
    """
    Synchronize initial state across distributed processes.
    
    Args:
        is_distributed: Whether running in distributed mode
        is_chief: Whether current process is chief
        start_episode: Starting episode number
        agent: The agent instance
        rank: Process rank
        local_rank: Local rank
        
    Returns:
        The synchronized starting episode number
    """
    logger = logging.getLogger(__name__)
    
    if is_distributed:
        logger.info(f"=== FASE 6: Sincronización del Estado Inicial [Proceso {rank}] ===")
        
        # Synchronize start_episode from chief to all processes
        sync_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        start_episode_tensor = torch.tensor([start_episode], dtype=torch.long, device=sync_device)
        
        logger.info(f"[Proceso {rank}] Sincronizando start_episode...")
        dist.broadcast(start_episode_tensor, src=0)
        
        # Update variable in worker processes
        if not is_chief:
            start_episode = start_episode_tensor.item()
            logger.info(f"[Proceso {rank}] start_episode sincronizado: {start_episode}")
        
        # Synchronize model weights from chief to all processes
        logger.info(f"[Proceso {rank}] Sincronizando pesos del modelo...")
        
        # Synchronize Actor weights
        for param in agent.actor.parameters():
            dist.broadcast(param.data, src=0)
        
        # Synchronize Critics weights
        for param in agent.critic_1.parameters():
            dist.broadcast(param.data, src=0)
        for param in agent.critic_2.parameters():
            dist.broadcast(param.data, src=0)
        
        # Synchronize Target Critics weights
        for param in agent.critic_target_1.parameters():
            dist.broadcast(param.data, src=0)
        for param in agent.critic_target_2.parameters():
            dist.broadcast(param.data, src=0)
        
        # Synchronize temperature parameter (alpha)
        if hasattr(agent, 'log_alpha'):
            dist.broadcast(agent.log_alpha.data, src=0)
        
        logger.info(f"[Proceso {rank}] ✅ Sincronización de pesos completada")
        
        # Final barrier to ensure all processes are synchronized
        dist.barrier()
        logger.info(f"[Proceso {rank}] ✅ Todos los procesos sincronizados y listos para entrenar")
    
    return start_episode


def _setup_trainer(is_chief: bool, agent, env, local_config, args, seed: str, storage_mode: str, 
                  run_id: str, tensorboard_base_dir: Optional[str], tb_logger, 
                  checkpoint_manager, config_manager, gcp_config, rank: int) -> Any:
    """
    Setup trainer instance.
    
    Args:
        is_chief: Whether current process is chief
        agent: The agent instance
        env: The environment instance
        local_config: Local configuration
        args: Command line arguments
        seed: Random seed
        storage_mode: Storage mode
        run_id: Run identifier
        tensorboard_base_dir: TensorBoard base directory
        tb_logger: TensorBoard logger
        checkpoint_manager: Checkpoint manager
        config_manager: Configuration manager
        gcp_config: GCP configuration
        rank: Process rank
        
    Returns:
        The trainer instance
    """
    logger = logging.getLogger(__name__)
    logger.info(f"=== FASE 7: Configuración Final de Entrenamiento [Proceso {rank}] ===")
    
    # Create instances for training (conditionally based on process)
    evaluator = AgentEvaluator() if is_chief else None
    
    # Configuration for trainer (all processes)
    trainer_config = {
        'seed': seed,
        KEY_BATCH_SIZE: local_config.agent.batch_size,
        KEY_MIN_BUFFER_FOR_LEARNING: local_config.agent.min_buffer_for_learning,
        KEY_REPLAY_BUFFER_SIZE: local_config.agent.replay_buffer_size,
        'eval_frequency': args.eval_frequency,
        'eval_episodes': args.eval_episodes,
        'save_frequency': args.save_frequency,
        'storage_mode': storage_mode,
        'run_id': run_id,
        'tensorboard_dir': tensorboard_base_dir if is_chief else None,
        'gcs_bucket_name': gcp_config.get('storage', {}).get('bucket_name') if storage_mode == STORAGE_MODE_GCP else None,
        # Prioritized Experience Replay parameters
        'per_alpha': local_config.agent.hiperparametros_sac.per_alpha,
        'per_beta': local_config.agent.hiperparametros_sac.per_beta
    }
    
    # Create trainer (all processes) with conditional instantiation
    trainer = Trainer(
        agent=agent,
        env=env,
        evaluator=evaluator,  # Only chief has evaluator
        logger=tb_logger if is_chief else None,  # Only chief has tb_logger
        checkpoint_manager=checkpoint_manager if is_chief else None,  # Only chief uses checkpoint_manager for writing
        config_manager=config_manager if is_chief else None,  # Only chief uses config_manager for writing
        training_run_id=run_id if is_chief else None,  # Only chief needs training_run_id
        trainer_config=trainer_config,
        logger_console=logger
    )
    
    return trainer


def _execute_training_and_cleanup(trainer, start_episode: int, args, is_chief: bool, 
                                 is_distributed: bool, tb_logger, rank: int):
    """
    Execute training and handle cleanup.
    
    Args:
        trainer: The trainer instance
        start_episode: Starting episode number
        args: Command line arguments
        is_chief: Whether current process is chief
        is_distributed: Whether running in distributed mode
        tb_logger: TensorBoard logger
        rank: Process rank
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"  - Episodio inicial: {start_episode}")
        logger.info(f"  - Episodios totales: {args.episodes}")
        logger.info(f"  - Episodios por entrenar: {args.episodes - start_episode}")
        
        # Verify that episodes remain to be trained
        if start_episode >= args.episodes:
            logger.warning(f"El checkpoint ya alcanzó o superó el número de episodios objetivo ({args.episodes})")
            logger.info("No hay episodios adicionales para entrenar. Terminando...")
            return
        
        # === PHASE 8: DISTRIBUTED TRAINING EXECUTION ===
        logger.info(f"=== FASE 8: Iniciando Entrenamiento Distribuido [Proceso {rank}] ===")
        
        # ALL processes participate in training - DDP handles synchronization
        trainer.train(start_episode=start_episode, total_episodes=args.episodes)
        
        logger.info(f"=== Proceso {rank} Completado Exitosamente ===")
        
    except KeyboardInterrupt:
        logger.info(f"[Proceso {rank}] Proceso interrumpido por el usuario")
        
    except Exception as e:
        logger.error(f"[Proceso {rank}] Error during execution: {e}")
        logger.exception("Detalles del error:")
        
    finally:
        # === CLEANUP AND ORDERLY FINALIZATION ===
        logger.info(f"[Proceso {rank}] Iniciando limpieza final...")
        
        # Close TensorBoard logger (only chief)
        if is_chief and tb_logger is not None:
            try:
                tb_logger.close()
                logger.info("[Proceso Jefe] TensorBoard logger cerrado exitosamente")
            except Exception as e:
                logger.warning(f"[Proceso Jefe] Error al cerrar TensorBoard logger: {e}")
        
        # Clean up distributed environment if necessary
        if is_distributed:
            try:
                logger.info(f"[Proceso {rank}] Cerrando grupo de procesos distribuidos...")
                dist.destroy_process_group()
                logger.info(f"[Proceso {rank}] ✅ Grupo de procesos cerrado exitosamente")
            except Exception as e:
                logger.warning(f"[Proceso {rank}] Error al cerrar proceso distribuido: {e}")
        
        logger.info(f"[Proceso {rank}] ✅ Limpieza completada - Finalizando proceso")


def main():
    """Función principal del script."""
    # === PASO 1: CONFIGURACIÓN DEL ENTORNO DISTRIBUIDO ===
    is_distributed, world_size, rank, local_rank, is_chief = _setup_distributed_environment()
    logger = logging.getLogger(__name__)
    
    # === PASO 2: DETERMINAR MODO DE OPERACIÓN ===
    args = parse_arguments()
    operation_mode, source_id = _determine_operation_mode(args)

    # === PASO 3: CARGAR CONFIGURACIÓN Y LINAJE ===
    local_config, training_run_lineage, data_run_id, is_new_training = _load_configuration(
        operation_mode, source_id, args
    )

    # === PASO 4: EXTRACCIÓN DE PARÁMETROS DEL EXPERIMENTO ===
    # Create ArtifactManager instance for ALL processes
    storage_mode = local_config.normalization.storage_mode
    gcp_config = local_config.gcp.model_dump() if storage_mode == STORAGE_MODE_GCP else None
    artifact_manager = ArtifactManager(
        storage_mode=storage_mode,
        gcp_config=gcp_config
    )
    logger.info(f"[Proceso {rank}] ArtifactManager creado para modo: {storage_mode}")

    symbol, interval, start_date, end_date, seed = _extract_experiment_params(
        artifact_manager, data_run_id, local_config
    )

    # Configure random seed for reproducibility
    set_seed(seed, logger)

    # === PASO 5: GENERACIÓN Y SINCRONIZACIÓN DEL RUN_ID ===
    run_id = _synchronize_run_id(
        is_chief, is_distributed, is_new_training, operation_mode, 
        symbol, interval, seed, args, rank, local_rank
    )

    # === PASO 6: INICIALIZACIÓN DE COMPONENTES DE GESTIÓN ===
    checkpoint_manager, artifact_manager, config_manager = _initialize_managers(local_config, rank)
    
    # Assemble complete run configuration (available for all processes)
    full_run_config = {
        'run_info': {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'storage_mode': storage_mode,
            'base_path': config_manager._get_training_run_prefix(run_id),
            'operation_mode': operation_mode
        },
        'command_line_args': vars(args),
        'config': local_config.model_dump(),  # Use model_dump for Pydantic v2
        'lineage': training_run_lineage,  # Data_run origin information
        'metadata': {
            'experiment_parameters': {
                'symbol': symbol,
                'interval': interval,
                'start_date': start_date,
                'end_date': end_date
            }
        }
    }

    # === PASO 7: INICIALIZACIÓN DE COMPONENTES DE LOGGING ===
    tensorboard_base_dir, tb_logger = _initialize_logging_components(is_chief, run_id, rank)

    # Only chief process saves configuration for new trainings and fine-tuning
    if is_chief and (is_new_training or operation_mode == "fine_tuning"):
        try:
            config_manager.save_run_config(run_id, full_run_config)
            logger.info(f"✅ Configuración del training_run guardada con linaje a data_run: {data_run_id}")
        except Exception as e:
            logger.error(f"❌ Error al guardar config_training_run.yaml: {e}")
            # Continue execution as this error is not critical
    elif is_chief:
        logger.info(f"ℹ️ Reanudando entrenamiento existente - no se actualiza la configuración")

    # === PASO 8: CARGAR DATOS DE ENTRENAMIENTO ===
    dataframe, scaler, price_scaler, data_run_metadata = _load_training_data(artifact_manager, data_run_id, rank)

    # === PASO 9: CREAR ENTORNO Y AGENTE ===
    env, agent = _create_environment_and_agent(
        dataframe, logger, price_scaler, scaler, local_config, 
        full_run_config, args, rank, is_distributed
    )

    # === PASO 10: GESTIÓN DE CHECKPOINTS ===
    start_episode = _handle_checkpoint_loading(is_chief, args, checkpoint_manager, agent, storage_mode, rank, run_id)

    # === PASO 11: SINCRONIZACIÓN DEL ESTADO INICIAL ===
    start_episode = _synchronize_initial_state(is_distributed, is_chief, start_episode, agent, rank, local_rank)

    # === PASO 12: CONFIGURAR TRAINER ===
    trainer = _setup_trainer(
        is_chief, agent, env, local_config, args, seed, storage_mode, 
        run_id, tensorboard_base_dir, tb_logger, checkpoint_manager, 
        config_manager, gcp_config, rank
    )

    # === PASO 13: EJECUTAR ENTRENAMIENTO Y LIMPIEZA ===
    _execute_training_and_cleanup(trainer, start_episode, args, is_chief, is_distributed, tb_logger, rank)


if __name__ == "__main__":
    # Forzar a NCCL a usar una interfaz de red común en entornos cloud para evitar timeouts.
    # Leer desde la configuración para mayor flexibilidad.
    try:
        config = AppConfig.from_yaml_file('src/configuration/config.yaml')
        nccl_socket_ifname = config.system.nccl_socket_ifname
        os.environ['NCCL_SOCKET_IFNAME'] = nccl_socket_ifname
        print(f"Establecida variable de entorno NCCL_SOCKET_IFNAME='{nccl_socket_ifname}'")
    except Exception as e:
        print(f"Advertencia: No se pudo leer nccl_socket_ifname de config.yaml. Usando valor por defecto 'eth0'. Error: {e}")
        os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'

    # Forzar el método 'spawn' para multiprocessing para evitar problemas de CUDA
    # en los procesos hijos que guardan los modelos. Es la solución estándar.
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()