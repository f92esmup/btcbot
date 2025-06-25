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
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import re

from src.data.pipeline import DataPipeline
from src.entorno.environment import FuturesTradingEnv
from src.agente.agent import TransformerSACAgent
from src.configuration.config import config
from src.configuration.gcs_utils import GCSUtils
from src.utils.system import setup_logging, set_seed, setup_device
from src.utils.validation import validate_date_format
from src.utils.cli import parse_arguments
from src.analysis.logger import TensorboardLogger
from src.training import RunManager, AgentEvaluator, Trainer


def create_trading_environment(dataframe: Any, logger, run_manager: RunManager, price_scaler_path: Optional[str] = None, price_scaler_blob_name: Optional[str] = None) -> FuturesTradingEnv:
    """
    Crea el entorno de trading con los datos procesados.
    
    Args:
        dataframe: DataFrame con datos normalizados
        logger: Logger para mensajes
        run_manager: Instancia centralizada de RunManager
        price_scaler_path: Ruta específica del price_scaler (opcional, para checkpoint loading)
        price_scaler_blob_name: Blob name específico en GCS (opcional, para checkpoint loading)
        
    Returns:
        FuturesTradingEnv: Entorno configurado
    """
    logger.info("Creando entorno de trading...")
    
    # Cargar el price_scaler usando RunManager
    logger.info("Cargando price_scaler desde almacenamiento...")
    try:
        price_scaler = run_manager.load_price_scaler(price_scaler_path, price_scaler_blob_name)
        
        # Obtener información del rango para logging
        if hasattr(price_scaler, 'data_min_') and hasattr(price_scaler, 'data_max_'):
            close_min = price_scaler.data_min_[0]
            close_max = price_scaler.data_max_[0]
            logger.info(f"Price scaler cargado exitosamente - Rango Close: {close_min:.2f} - {close_max:.2f}")
        else:
            logger.info("Price scaler cargado exitosamente")
        
    except Exception as e:
        logger.error(f"Error crítico al cargar price_scaler: {e}")
        logger.error("No se puede continuar sin el price_scaler. Deteniendo ejecución.")
        raise RuntimeError(f"Fallo al cargar price_scaler: {e}")
    
    env = FuturesTradingEnv(
        data_df=dataframe,
        price_scaler=price_scaler
    )
    
    logger.info(f"Entorno creado:")
    logger.info(f"  - Balance inicial: ${config.capital_inicial:,.2f}")
    logger.info(f"  - Apalancamiento: {config.apalancamiento}x")
    logger.info(f"  - Ventana observación: {config.ventana_observacion_size}")
    logger.info(f"  - Espacio de observación: {env.observation_space}")
    logger.info(f"  - Espacio de acción: {env.action_space}")
    
    return env


def create_sac_agent(env: FuturesTradingEnv, device: torch.device, logger) -> TransformerSACAgent:
    """
    Crea el agente SAC con arquitectura Transformer.
    
    Args:
        env: Entorno de trading
        device: Device para el entrenamiento
        logger: Logger para mensajes
        
    Returns:
        TransformerSACAgent: Agente configurado
    """
    logger.info("Creando agente SAC con Transformer...")
    
    # Obtener parámetros del entorno
    observation_space_shape = env.observation_space.shape
    action_space_shape = env.action_space.shape
    
    # Calcular características de mercado y portfolio
    ventana_size = config.ventana_observacion_size
    num_features_mercado = len(env.column_names)
    market_features = num_features_mercado
    portfolio_features = 4  # tipo_posicion, pnl_roe, pasos_posicion, precio_entrada
    sequence_length = ventana_size
    
    agent = TransformerSACAgent(
        observation_space_shape=observation_space_shape,
        action_space_shape=action_space_shape,
        market_features=market_features,
        portfolio_features=portfolio_features,
        sequence_length=sequence_length,
        device=device
    )
    
    # Contar parámetros del modelo
    total_params = sum(p.numel() for p in agent.actor.parameters())
    trainable_params = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    
    logger.info(f"Agente SAC creado:")
    logger.info(f"  - Parámetros totales: {total_params:,}")
    logger.info(f"  - Parámetros entrenables: {trainable_params:,}")
    logger.info(f"  - Market features: {market_features}")
    logger.info(f"  - Portfolio features: {portfolio_features}")
    logger.info(f"  - Sequence length: {sequence_length}")
    logger.info(f"  - Gamma: {config.gamma}")
    logger.info(f"  - Tau: {config.tau}")
    logger.info(f"  - Alpha inicial: {config.initial_log_alpha}")
    logger.info(f"  - Learning rates: Actor={config.actor_learning_rate}, Critic={config.critic_learning_rate}")
    
    return agent






def main():
    """Función principal del script."""
    # Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== Iniciando Bot de Trading de Bitcoin ===")
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Configurar semilla aleatoria para reproducibilidad
    set_seed(args.seed, logger)
    
    # Validar fecha de inicio
    if not validate_date_format(args.start_date):
        logger.error(f"Formato de fecha inválido: {args.start_date}. Use YYYY-MM-DD")
        sys.exit(1)
    
    logger.info(f"Parámetros: Symbol={args.symbol}, Interval={args.interval}, Start Date={args.start_date}, Seed={args.seed}")

    # Generar run_id único incluyendo la semilla
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_id = f"{args.symbol}_{args.interval}_{args.seed}_{current_time}"
    logger.info(f"Run ID generado: {run_id}")
    
    # Crear instancia única de GCSUtils para todo el proceso
    gcs_utils = None
    if config.storage_mode == "gcp":
        from src.configuration.gcs_utils import gcs_utils
        logger.info("Usando instancia global de GCSUtils para modo GCP")

    # Determinar base_path según storage_mode
    if config.storage_mode == "gcp":
        base_path = f"gs://{config.gcs_bucket_name}/{run_id}"
        logger.info(f"Modo GCP: Los artefactos se guardarán en {base_path}")
    else:
        base_path = Path("Entrenamientos") / run_id
        base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Modo Local: Los artefactos se guardarán en {base_path}")

    # Lógica de TensorBoard modificada
    # Para el modo local, seguimos creando un directorio.
    # Para el modo GCP, log_dir no es estrictamente necesario, pero lo mantenemos por consistencia.
    # El logger interno decidirá qué hacer.
    if config.storage_mode == "local":
        tensorboard_dir = Path(base_path) / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
    else:
        # En modo GCP, los logs se envían directamente a la API de Vertex,
        # no se necesita un directorio local persistente.
        tensorboard_dir = None

    # Inicializar TensorBoard Logger
    # Pasamos el run_id para que lo use como nombre del "run" en el experimento
    tb_logger = TensorboardLogger(log_dir=str(tensorboard_dir) if tensorboard_dir else None, run_id=run_id)
    
    if config.storage_mode == "local":
        logger.info(f"TensorBoard logs se guardarán localmente en: {tensorboard_dir}")
    else:
        logger.info(f"TensorBoard logs se enviarán directamente a Vertex AI TensorBoard")

    # Registrar Hiperparámetros
    hparams = {
        'run_id': run_id,
        'symbol': args.symbol,
        'interval': args.interval,
        'start_date': args.start_date,
        'seed': args.seed,
        'episodes': args.episodes,
        'eval_frequency': args.eval_frequency,
        'save_frequency': args.save_frequency,
        'actor_lr': config.actor_learning_rate,
        'critic_lr': config.critic_learning_rate,
        'alpha_lr': config.alpha_learning_rate,
        'gamma': config.gamma,
        'tau': config.tau,
        'batch_size': config.batch_size,
        'buffer_size': config.replay_buffer_size,
        'd_model': config.d_model,
        'n_head': config.n_head,
        'num_encoder_layers': config.num_encoder_layers,
        'ventana_observacion': config.ventana_observacion_size,
        'capital_inicial': config.capital_inicial,
        'apalancamiento': config.apalancamiento,
        'storage_mode': config.storage_mode,
        'base_path': str(base_path)
    }
    # Log hyperparameters
    tb_logger.log_hyperparameters(hparams)
    
    # Crear instancia única de RunManager para todo el proceso
    run_manager = RunManager(base_path=str(base_path), run_id=run_id, gcs_utils=gcs_utils)
    logger.info(f"RunManager centralizado creado - Base path: {base_path}")
    
    # Guardar configuración del run usando RunManager
    try:
        run_manager.save_run_config(hparams=hparams, args=args)
    except Exception as e:
        logger.error(f"Error al guardar config_run.yaml: {e}")
        # Continuar ejecución ya que este error no es crítico
    
    try:
        logger.info("=== Ejecutando Pipeline de Datos ===")
        data_pipeline = DataPipeline(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            run_id=run_id,
            base_path=str(base_path)
        )
        normalized_dataframe, price_scaler_path = data_pipeline.run()

        # Actualizar referencia al dataframe
        dataframe = normalized_dataframe
        
        # 4. Entrenamiento del Modelo SAC
        logger.info("=== FASE 4: Entrenamiento del Modelo SAC ===")
        
        # Configurar device
        device = setup_device(args.no_cuda)
        logger.info(f"Usando device: {device}")
        
        # Variables para resumir entrenamiento
        start_episode = 0
        
        # Nueva lógica de carga de checkpoint basada en argumento --checkpoint
        logger.info("\n=== CONFIGURACIÓN DE CHECKPOINT O EJECUCIÓN NUEVA ===")
        
        # Variables para pasar la ruta/blob del price_scaler a create_trading_environment
        path_price_scaler_a_cargar = None
        blob_name_price_scaler_a_cargar = None
        
        if args.checkpoint is None:
            # No se especificó checkpoint, comenzar desde cero
            logger.info("Iniciando nueva ejecución (sin checkpoint).")
            start_episode = 0
            
            # Usar la ruta del price_scaler que devolvió el pipeline
            if config.storage_mode == "gcp":
                blob_name_price_scaler_a_cargar = f"{run_id}/price_scaler.pkl"
                logger.info(f"Se cargará price_scaler desde GCS (nueva ejecución): {blob_name_price_scaler_a_cargar}")
            else:
                path_price_scaler_a_cargar = price_scaler_path
                logger.info(f"Se cargará price_scaler desde local (nueva ejecución): {path_price_scaler_a_cargar}")
        else:
            # Se especificó un run_id para cargar checkpoint
            logger.info(f"Intentando reanudar desde checkpoint del run_id: {args.checkpoint}")
            
            # Buscar checkpoint en el run_id específico usando RunManager
            checkpoint_info = run_manager.find_latest_checkpoint(args.checkpoint)
            
            if checkpoint_info:
                checkpoint_prefix, latest_episode = checkpoint_info
                
                logger.info(f"✅ Checkpoint encontrado del episodio {latest_episode}")
                logger.info(f"Ubicación: {checkpoint_prefix}")
                
                # Configurar rutas de scalers para cargar desde el run_id específico del checkpoint
                if config.storage_mode == "gcp":
                    # Para GCS, construir blob names específicos del run_id del checkpoint
                    blob_name_price_scaler_a_cargar = f"{args.checkpoint}/price_scaler.pkl"
                    logger.info(f"Se cargará price_scaler desde GCS (checkpoint): {blob_name_price_scaler_a_cargar}")
                else:
                    # Para local, construir paths específicos del run_id del checkpoint
                    path_price_scaler_a_cargar = f"Entrenamientos/{args.checkpoint}/price_scaler.pkl"
                    logger.info(f"Se cargará price_scaler desde local (checkpoint): {path_price_scaler_a_cargar}")
            else:
                logger.error(f"❌ No se encontraron checkpoints en el run_id: {args.checkpoint}")
                logger.error("Terminando ejecución. Verifique que el run_id sea válido y contenga checkpoints.")
                return
        
        # Crear entorno de trading
        env = create_trading_environment(
            dataframe,  # Este debe ser el dataframe normalizado
            logger,
            run_manager,
            price_scaler_path=path_price_scaler_a_cargar,
            price_scaler_blob_name=blob_name_price_scaler_a_cargar
        )
        
        # Crear agente
        agent = create_sac_agent(env, device, logger)
        
        # Cargar checkpoint si corresponde
        if args.checkpoint is not None:
            checkpoint_info = run_manager.find_latest_checkpoint(args.checkpoint)
            if checkpoint_info:
                checkpoint_prefix, latest_episode = checkpoint_info
                
                try:
                    logger.info(f"Cargando checkpoint desde: {checkpoint_prefix}")
                    run_manager.load_agent_from_checkpoint(agent, checkpoint_prefix)
                    
                    start_episode = latest_episode
                    
                    logger.info(f"✅ Checkpoint cargado exitosamente")
                    logger.info(f"  - Run ID de origen: {args.checkpoint}")
                    logger.info(f"  - Episodio inicial: {start_episode}")
                    logger.info(f"  - Total steps: {agent.total_steps}")
                    logger.info(f"  - Learning steps: {agent.learning_steps}")
                    logger.info(f"  - Nuevos artefactos se guardarán en run_id: {run_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Error al cargar checkpoint desde run_id {args.checkpoint}: {e}")
                    logger.error("Terminando ejecución. Verifique que el run_id sea válido y contenga checkpoints.")
                    return
        
        logger.info(f"\n=== CONFIGURACIÓN FINAL DE ENTRENAMIENTO ===")
        logger.info(f"  - Episodio inicial: {start_episode}")
        logger.info(f"  - Episodios totales: {args.episodes}")
        logger.info(f"  - Episodios por entrenar: {args.episodes - start_episode}")
        logger.info(f"  - Run ID actual: {run_id}")
        
        # Verificar que queden episodios por entrenar
        if start_episode >= args.episodes:
            logger.warning(f"El checkpoint ya alcanzó o superó el número de episodios objetivo ({args.episodes})")
            logger.info("No hay episodios adicionales para entrenar. Terminando...")
            return
        
        # Crear instancias para el entrenamiento
        evaluator = AgentEvaluator()
        
        # Configuración para el trainer
        trainer_config = {
            'seed': args.seed,
            'batch_size': config.batch_size,
            'min_buffer_for_learning': config.min_buffer_for_learning,
            'replay_buffer_size': config.replay_buffer_size,
            'eval_frequency': args.eval_frequency,
            'eval_episodes': args.eval_episodes,
            'save_frequency': args.save_frequency,
            'storage_mode': config.storage_mode,
            'run_id': run_id,
            'tensorboard_dir': tensorboard_dir,
            'gcs_bucket_name': getattr(config, 'gcs_bucket_name', None),
            'gcs_utils': gcs_utils
        }
        
        # Crear trainer e iniciar entrenamiento
        trainer = Trainer(
            agent=agent,
            env=env,
            evaluator=evaluator,
            logger=tb_logger,
            run_manager=run_manager,
            trainer_config=trainer_config,
            logger_console=logger
        )
        
        # Ejecutar entrenamiento
        trainer.train(start_episode=start_episode, total_episodes=args.episodes)
        
        logger.info("=== Proceso Completado Exitosamente ===")
        
    except KeyboardInterrupt:
        logger.info("Proceso interrumpido por el usuario")
        tb_logger.close()
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")
        logger.exception("Detalles del error:")
        tb_logger.close()
        sys.exit(1)
    
    finally:
        # Asegurar que el writer se cierre
        if 'tb_logger' in locals():
            tb_logger.close()


if __name__ == "__main__":
    # AÑADE ESTA LÍNEA AQUÍ DENTRO:
    # Forzar el método 'spawn' para multiprocessing para evitar problemas de CUDA
    # en los procesos hijos que guardan los modelos. Es la solución estándar.
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()