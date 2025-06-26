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
        Tuple de (is_distributed, world_size, rank, local_rank)
    """
    if 'RANK' not in os.environ:
        logger_instance.info("Modo de un solo proceso.")
        return False, 1, 0, 0
    
    try:
        dist.init_process_group(backend='nccl')
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        # CORRECCIÓN: Usar LOCAL_RANK para la asignación de dispositivo.
        # En un nodo con una sola GPU, LOCAL_RANK será 0.
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        logger_instance.info(f"Entorno distribuido: Rank {rank}/{world_size}, GPU Local: {local_rank}")
        return True, world_size, rank, local_rank
    except Exception as e:
        logger_instance.error(f"Error inicializando grupo distribuido: {e}")
        sys.exit(1)


def main():
    """Función principal para el entrenamiento distribuido en Vertex AI."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configurar entrenamiento distribuido
    is_distributed, world_size, rank, local_rank = setup_distributed_training(logger)
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
        # CORRECCIÓN: Usar local_rank para el dispositivo del tensor de comunicación.
        device_for_comm = f'cuda:{local_rank}'
        if is_chief:
            # Asegurarse de que la lista tenga un tamaño fijo para que no haya errores de tamaño
            run_id_list = list(map(ord, run_id))
            run_id_list.extend([0] * (100 - len(run_id_list)))  # Rellenar con ceros
            run_id_tensor = torch.tensor(run_id_list, dtype=torch.int, device=device_for_comm)
        else:
            run_id_tensor = torch.empty(100, dtype=torch.int, device=device_for_comm)
        
        dist.broadcast(run_id_tensor, src=0)
        
        if not is_chief:
            # Decodificar el run_id recibido
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
    
    # --- INICIO DE LA SECCIÓN CORREGIDA ---
    
    # 1. SOLO el Chief ejecuta el pipeline para escribir los artefactos
    if is_chief:
        logger.info(f"[Rank 0 - Chief] Ejecutando Pipeline de Datos para preparar artefactos...")
        pipeline = DataPipeline(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date,
            run_id=run_id,
            base_path=base_path if is_chief else "temp_data",
            save_artifacts=is_chief  # <-- Esta es la línea clave.
        )
        pipeline.run()
        logger.info(f"[Rank 0 - Chief] Pipeline completado. Artefactos guardados en: {base_path}")
    
    # 2. TODOS los procesos esperan aquí. Esto garantiza que los workers no continúen
    #    hasta que el chief haya terminado de guardar los archivos.
    if is_distributed:
        logger.info(f"[Rank {rank}] Esperando a que el Chief termine la preparación de datos...")
        dist.barrier()
    
    # 3. TODOS los procesos cargan los artefactos desde la ubicación compartida.
    logger.info(f"[Rank {rank}] Cargando los artefactos preparados por el Chief.")
    
    # Para cargar, cada proceso necesita una instancia de RunManager para acceder a los métodos de carga
    loader_run_manager = RunManager(run_id=run_id, base_path=base_path)

    # Cada proceso también necesita generar el dataframe en memoria para trabajar con él.
    # La parte crítica es que TODOS usarán el MISMO price_scaler cargado desde GCS.
    temp_pipeline = DataPipeline(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        run_id=run_id,
        base_path="temp_data",  # La ruta base aquí es irrelevante
        save_artifacts=False  # Los workers no deben guardar artefactos
    )
    df, _ = temp_pipeline.run()
    
    try:
        price_scaler_blob_name = f"{run_id}/price_scaler.pkl"
        price_scaler = loader_run_manager.load_price_scaler(blob_name=price_scaler_blob_name)
        logger.info(f"[Rank {rank}] Price scaler cargado exitosamente desde {price_scaler_blob_name}.")
    except Exception as e:
        logger.error(f"[Rank {rank}] CRÍTICO: No se pudo cargar el price_scaler desde GCS. Error: {e}")
        sys.exit(1)
    
    # --- FIN DE LA SECCIÓN CORREGIDA ---
    
    # El resto del script continúa como estaba, ahora con el 'price_scaler' cargado correctamente.
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
    # Construir trainer_config completo con todos los parámetros necesarios
    trainer_config = dict(config.agent_config)
    trainer_config['seed'] = args.seed
    trainer_config['eval_frequency'] = getattr(args, 'eval_frequency', config.agent_config.get('eval_frequency', 100))
    trainer_config['save_frequency'] = getattr(args, 'save_frequency', config.agent_config.get('save_frequency', 100))
    trainer_config['eval_episodes'] = getattr(args, 'eval_episodes', config.agent_config.get('eval_episodes', 5))
    trainer_config['batch_size'] = config.agent_config.get('batch_size', 256)
    trainer_config['min_buffer_for_learning'] = config.agent_config.get('min_buffer_for_learning', 10000)
    trainer_config['replay_buffer_capacity'] = config.agent_config.get('replay_buffer_capacity', 100000)
    
    trainer = Trainer(
        agent, 
        env, 
        AgentEvaluator() if is_chief else None, 
        tb_logger, 
        run_manager, 
        trainer_config,  # Usar la configuración completa
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
