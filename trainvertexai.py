# trainvertexai.py (NUEVO ARCHIVO)

import os
import sys
import logging
import json
from datetime import datetime
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
import joblib

from src.data.pipeline import DataPipeline
from src.entorno.environment import FuturesTradingEnv
from src.agente.agent import TransformerSACAgent
from src.configuration.config import config
from src.utils.system import setup_logging, set_seed, setup_device
from src.utils.validation import validate_date_format
from src.utils.cli import parse_arguments
from src.analysis.logger import TensorboardLogger
from src.training import RunManager, AgentEvaluator, Trainer


def setup_distributed_training(logger_instance: logging.Logger):
    """
    Configura el entrenamiento distribuido usando las variables de entorno de Vertex AI.
    
    Args:
        logger_instance: Logger para escribir mensajes
        
    Returns:
        Tuple de (is_distributed, world_size, rank)
    """
    if 'RANK' not in os.environ:
        logger_instance.info("Modo de un solo proceso.")
        return False, 1, 0
    
    try:
        dist.init_process_group(backend='nccl')
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        torch.cuda.set_device(rank % torch.cuda.device_count())
        logger_instance.info(f"Entorno distribuido inicializado. Rank: {rank}/{world_size}")
        return True, world_size, rank
    except Exception as e:
        logger_instance.error(f"Error inicializando grupo distribuido: {e}")
        sys.exit(1)


def main():
    """Función principal para el entrenamiento distribuido en Vertex AI."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configurar entrenamiento distribuido
    is_distributed, world_size, rank = setup_distributed_training(logger)
    is_chief = (rank == 0)
    
    logger.info(f"[Rank {rank}] Iniciando proceso.")
    
    # Parsear argumentos y configurar semilla
    args = parse_arguments()
    final_seed = args.seed + rank
    set_seed(final_seed, logger)
    
    # Generar run_id solo en el proceso chief
    run_id = ""
    if is_chief:
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_id = f"{args.symbol}_{args.interval}_{args.seed}_{current_time}"
        logger.info(f"Run ID (controlado por Chief): {run_id}")
    
    # Compartir run_id del chief a todos los workers
    if is_distributed:
        # Compartir run_id del chief a los workers
        run_id_tensor = torch.tensor(list(map(ord, run_id)), dtype=torch.int, device=f'cuda:{rank}') if is_chief else torch.empty(100, dtype=torch.int, device=f'cuda:{rank}')
        dist.broadcast(run_id_tensor, src=0)
        if not is_chief:
            run_id = "".join([chr(c) for c in run_id_tensor if c != 0])
    
    # Configurar paths y loggers solo en el proceso chief
    base_path = ""
    tb_logger, run_manager = None, None
    if is_chief:
        if config.storage_mode == "gcp":
            base_path = f"gs://{config.gcs_bucket_name}/{run_id}"
        else:
            base_path = Path("Entrenamientos") / run_id
            base_path.mkdir(parents=True, exist_ok=True)
        
        tb_logger = TensorboardLogger(log_dir=None, run_id=run_id)
        run_manager = RunManager(base_path=str(base_path), run_id=run_id)
        hparams = {'run_id': run_id, 'world_size': world_size, **vars(args)}
        tb_logger.log_hyperparameters(hparams)
        run_manager.save_run_config(hparams=hparams, args=args)
    
    # Pipeline de datos
    logger.info(f"[Rank {rank}] Ejecutando Pipeline de Datos...")
    pipeline = DataPipeline(
        args.symbol, 
        args.interval, 
        args.start_date, 
        args.end_date, 
        run_id, 
        base_path if is_chief else "temp_data"
    )
    df, scaler_path = pipeline.run()
    
    # Configurar entorno y agente
    price_scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    env = FuturesTradingEnv(data_df=df, price_scaler=price_scaler)
    device = setup_device(args.no_cuda)
    
    agent = TransformerSACAgent(
        observation_space_shape=env.observation_space.shape,
        action_space_shape=env.action_space.shape,
        market_features=len(env.column_names),
        portfolio_features=4,
        sequence_length=config.ventana_observacion_size,
        device=device,
        is_distributed=is_distributed
    )
    
    # Sincronizar todos los procesos antes de comenzar el entrenamiento
    if is_distributed:
        dist.barrier()
    
    # Entrenar el agente
    trainer = Trainer(
        agent, 
        env, 
        AgentEvaluator() if is_chief else None, 
        tb_logger, 
        run_manager, 
        config.agent_config, 
        logger
    )
    trainer.train(start_episode=0, total_episodes=args.episodes)
    
    # Limpiar el grupo de procesos distribuidos
    if is_distributed:
        dist.destroy_process_group()
    
    if is_chief:
        logger.info("=== Proceso (Chief) Completado Exitosamente ===")


if __name__ == "__main__":
    main()
