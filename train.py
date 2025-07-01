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
from src.entorno.environment import FuturesTradingEnv
from src.entorno.portfolio import Portfolio
from src.agente.agent import TransformerSACAgent
from src.utils.system import setup_logging, set_seed, setup_device, setup_environment_and_distribution
from src.utils.validation import validate_date_format
from src.utils.cli import parse_arguments
from src.analysis.logger import TensorboardLogger
from src.training import RunManager, AgentEvaluator, Trainer


def create_trading_environment(dataframe: Any, logger, run_manager: RunManager, env_config: dict, price_scaler_path: Optional[str] = None, price_scaler_blob_name: Optional[str] = None) -> FuturesTradingEnv:
    """
    Crea el entorno de trading con los datos procesados.
    
    Args:
        dataframe: DataFrame con datos normalizados
        logger: Logger para mensajes
        run_manager: Instancia centralizada de RunManager
        env_config: Diccionario con la configuración del entorno
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
    
    sim_portfolio = Portfolio(env_config)

    env = FuturesTradingEnv(
        data_df=dataframe,
        price_scaler=price_scaler,
        env_config=env_config, # Inyección de configuración
        portfolio=sim_portfolio
    )
    
    logger.info(f"Entorno creado:")
    logger.info(f"  - Balance inicial: ${env_config['capital_inicial']:,.2f}")
    logger.info(f"  - Apalancamiento: {env_config['apalancamiento']}x")
    logger.info(f"  - Ventana observación: {env_config['ventana_observacion_size']}")
    logger.info(f"  - Espacio de observación: {env.observation_space}")
    logger.info(f"  - Espacio de acción: {env.action_space}")
    
    return env


def create_sac_agent(env: FuturesTradingEnv, device: torch.device, logger, agent_config: dict, is_distributed: bool = False) -> TransformerSACAgent:
    """
    Crea el agente SAC con arquitectura Transformer.
    
    Args:
        env: Entorno de trading
        device: Device para el entrenamiento
        logger: Logger para mensajes
        agent_config: Diccionario con la configuración del agente
        is_distributed: Si el entrenamiento es distribuido
        
    Returns:
        TransformerSACAgent: Agente configurado
    """
    logger.info("Creando agente SAC con Transformer...")
    
    # Obtener parámetros del entorno
    observation_space_shape = env.observation_space.shape
    action_space_shape = env.action_space.shape
    
    # Calcular características de mercado y portfolio
    ventana_size = env.config_entorno['ventana_observacion_size']
    num_features_mercado = len(env.column_names)
    market_features = num_features_mercado
    # Leer portfolio_features desde la configuración del agente
    portfolio_features = agent_config.get('architecture', {}).get('portfolio_features', 4)
    sequence_length = ventana_size
    
    agent = TransformerSACAgent(
        observation_space_shape=observation_space_shape,
        action_space_shape=action_space_shape,
        market_features=market_features,
        portfolio_features=portfolio_features,
        sequence_length=sequence_length,
        config_override=agent_config, # Inyección de configuración
        device=device,
        is_distributed=is_distributed
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
    logger.info(f"  - Gamma: {agent_config['hiperparametros_sac']['gamma']}")
    logger.info(f"  - Tau: {agent_config['hiperparametros_sac']['tau']}")
    logger.info(f"  - Alpha inicial: {agent_config['hiperparametros_sac']['initial_log_alpha']}")
    logger.info(f"  - Learning rates: Actor={agent_config['hiperparametros_sac']['actor_learning_rate']}, Critic={agent_config['hiperparametros_sac']['critic_learning_rate']}")
    logger.info(f"  - Entrenamiento distribuido: {'Sí' if is_distributed else 'No'}")
    
    return agent






def main():
    """Función principal del script."""
    # === DETECCIÓN Y CONFIGURACIÓN DEL ENTORNO DISTRIBUIDO ===
    # Esta debe ser la PRIMERA acción en main() para configurar correctamente
    # el entorno de ejecución antes de cualquier otra operación
    is_distributed, world_size, rank, local_rank = setup_environment_and_distribution()
    
    # Definir el rol "jefe" (chief) - solo el proceso con rank 0
    is_chief = (rank == 0)
    
    # Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Logging informativo del estado del entorno detectado
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
    
    # Parsear argumentos
    args = parse_arguments()

    # --- LÓGICA DE CARGA DE CONFIGURACIÓN ---
    # Determinar qué configuración usar: la del checkpoint (continuación) o la local (nuevo/fine-tune)
    gcp_config_for_load = None
    try:
        with open('src/configuration/config.yaml', 'r') as f:
            # Cargar gcp_config de la configuración local para poder usar RunManager
            temp_config = yaml.safe_load(f)
            if temp_config.get('normalization', {}).get('storage_mode') == 'gcp':
                gcp_config_for_load = temp_config.get('gcp', {})
    except FileNotFoundError:
        logger.warning("No se encontró config.yaml local. Se asumirá que no se necesita gcp_config para cargar.")

    if args.checkpoint and not args.fine_tune_mode:
        logger.info(f"Modo 'Continuación Pura' detectado. Cargando configuración desde el run_id: {args.checkpoint}")
        # Cargar la configuración del run anterior para una continuación exacta
        config_dict = RunManager.load_run_config(args.checkpoint, gcp_config=gcp_config_for_load)
        if not config_dict:
            logger.error(f"No se pudo cargar la configuración para el run_id: {args.checkpoint}. Abortando.")
            sys.exit(1)
        # La configuración cargada del checkpoint ya contiene la sección 'config'
        local_config_dict = config_dict.get('config', {})
        logger.info(f"Configuración del run '{args.checkpoint}' cargada exitosamente como fuente de verdad.")
    else:
        # Modo 'Nuevo Entrenamiento' o 'Fine-Tuning': usar la configuración local
        if args.checkpoint and args.fine_tune_mode:
            logger.info(f"Modo 'Fine-Tuning' detectado. Usando configuración local 'config.yaml' para el nuevo run.")
        else:
            logger.info("Modo 'Nuevo Entrenamiento' detectado. Usando 'config.yaml' local.")
        
        try:
            with open('src/configuration/config.yaml', 'r') as f:
                local_config_dict = yaml.safe_load(f)
            logger.info("Configuración local 'config.yaml' cargada exitosamente.")
        except FileNotFoundError:
            logger.error("No se encontró el archivo 'src/configuration/config.yaml'. Abortando.")
            sys.exit(1)

    # --- EXTRACCIÓN DE PARÁMETROS DEL EXPERIMENTO ---
    try:
        # Ahora extraemos la configuración de 'local_config_dict', que es la fuente de verdad correcta
        exp_config = local_config_dict['experiment_definition']
        symbol = exp_config['symbol']
        interval = exp_config['interval']
        start_date = exp_config['training_start_date']
        end_date = exp_config.get('training_end_date')
        seed = exp_config['seed']
        logger.info(f"Configuración del experimento: {symbol}/{interval} desde {start_date}, Seed: {seed}")
    except KeyError as e:
        logger.error(f"Falta la clave de configuración requerida en 'experiment_definition': {e}")
        sys.exit(1)

    # Configurar semilla aleatoria para reproducibilidad
    set_seed(seed, logger)

    # === GENERACIÓN Y SINCRONIZACIÓN DEL RUN_ID ===
    # Solo el proceso jefe genera el run_id, luego lo sincroniza con todos los procesos
    if is_chief:
        # Generar run_id único incluyendo la semilla
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        run_id = f"{symbol}_{interval}_{seed}_{current_time}"
        logger.info(f"[Proceso Jefe] Run ID generado: {run_id}")
    else:
        # Los procesos no-jefe inicializan run_id como None, se sincronizará después
        run_id = None
    
    # Sincronización del run_id en entornos distribuidos
    if is_distributed:
        # Configurar dispositivo para la sincronización
        sync_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        if is_chief:
            # Codificar el string run_id a un tensor de bytes
            run_id_bytes = torch.tensor(bytearray(run_id, "utf-8"), dtype=torch.uint8, device=sync_device)
            # Crear un tensor para el tamaño y transmitirlo
            size_tensor = torch.tensor([len(run_id_bytes)], dtype=torch.long, device=sync_device)
            dist.broadcast(size_tensor, src=0)
            # Transmitir el tensor de bytes
            dist.broadcast(run_id_bytes, src=0)
            logger.info(f"[Proceso Jefe] run_id transmitido a todos los procesos")
        else:
            # Recibir el tamaño del run_id
            size_tensor = torch.empty(1, dtype=torch.long, device=sync_device)
            dist.broadcast(size_tensor, src=0)
            # Preparar un tensor vacío del tamaño correcto para recibir los bytes
            run_id_bytes = torch.empty(size_tensor[0].item(), dtype=torch.uint8, device=sync_device)
            dist.broadcast(run_id_bytes, src=0)
            # Decodificar los bytes de vuelta a un string
            run_id = run_id_bytes.cpu().numpy().tobytes().decode("utf-8")
        
        # Todos los procesos ahora tienen el mismo run_id
        logger.info(f"[Proceso {rank}] run_id sincronizado: {run_id}")
    
    # === INICIALIZACIÓN DE COMPONENTES DE GESTIÓN ===
    # Crear instancia única de RunManager para TODOS los procesos (lectura)
    # pero solo el jefe realizará operaciones de escritura
    storage_mode = local_config_dict.get('normalization', {}).get('storage_mode', 'local')
    gcp_config = None
    if storage_mode == "gcp":
        gcp_config = local_config_dict.get('gcp', {})
        logger.info(f"[Proceso {rank}] Configuración GCP cargada para RunManager")

    # Determinar base_path según storage_mode (todos los procesos)
    if storage_mode == "gcp":
        gcs_bucket_name = gcp_config.get('storage', {}).get('bucket_name')
        base_path = f"gs://{gcs_bucket_name}/{run_id}"
        logger.info(f"[Proceso {rank}] Modo GCP: Los artefactos se accederán desde {base_path}")
    else:
        base_path = Path("Entrenamientos") / run_id
        if is_chief:
            base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Proceso {rank}] Modo Local: Los artefactos se accederán desde {base_path}")

    # Crear instancia de RunManager para TODOS los procesos
    run_manager = RunManager(
        base_path=str(base_path), 
        run_id=run_id, 
        storage_mode=storage_mode,
        gcp_config=gcp_config
    )
    logger.info(f"[Proceso {rank}] RunManager creado - Base path: {base_path}")
    
    # Solo el proceso jefe inicializa los componentes de logging y configuración
    if is_chief:
        logger.info("=== INICIALIZACIÓN DE COMPONENTES ADICIONALES (PROCESO JEFE) ===")

        # Lógica de TensorBoard modificada
        if storage_mode == "local":
            tensorboard_dir = Path(base_path) / "tensorboard"
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
        else:
            tensorboard_dir = None

        # Inicializar TensorBoard Logger
        vertex_ai_config = None
        if storage_mode == "gcp":
            # Pasar la configuración completa que incluye tanto tensorboard_vertex_ai como gcp
            vertex_ai_config = local_config_dict
            vertex_ai_config['storage_mode'] = storage_mode
        
        tb_logger = TensorboardLogger(
            log_dir=str(tensorboard_dir) if tensorboard_dir else None, 
            run_id=run_id,
            vertex_ai_config=vertex_ai_config
        )
        
        if storage_mode == "local":
            logger.info(f"TensorBoard logs se guardarán localmente en: {tensorboard_dir}")
        else:
            logger.info(f"TensorBoard logs se enviarán directamente a Vertex AI TensorBoard")

        # Ensamblar la configuración completa del run
        full_run_config = {
            'run_info': {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'storage_mode': storage_mode,
                'base_path': str(base_path)
            },
            'command_line_args': vars(args),
            'config': local_config_dict
        }

        # Guardar configuración del run usando RunManager (SOLO EL JEFE ESCRIBE)
        try:
            run_manager.save_run_config(full_run_config)
        except Exception as e:
            logger.error(f"Error al guardar config_run.yaml: {e}")
            # Continuar ejecución ya que este error no es crítico
    else:
        # Los procesos no-jefe inicializan variables de gestión que no usan a None
        tensorboard_dir = None
        tb_logger = None
        hparams = None
        logger.info(f"[Proceso {rank}] Variables de logging inicializadas como None")
    
    try:
        # === FASE 1: EL PROCESO JEFE GENERA Y GUARDA LOS ARTEFACTOS ===
        if is_chief:
            logger.info("=== FASE 1: Generando y Guardando Artefactos (PROCESO JEFE) ===")
            
            # Obtener credenciales de API desde variables de entorno
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                logger.warning("Credenciales de Binance no encontradas en variables de entorno. Continuando sin ellas.")
                logger.warning("Para usar la API de Binance, define BINANCE_API_KEY y BINANCE_API_SECRET")
            else:
                logger.info("Credenciales de Binance cargadas desde variables de entorno")
            
            # Crear instancia local de GCSUtils si es necesario para compatibilidad
            gcs_utils_for_pipeline = None
            if storage_mode == "gcp":
                from src.configuration.gcs_utils import GCSUtils
                gcs_utils_for_pipeline = GCSUtils(gcp_config)
            
            data_pipeline_chief = DataPipeline(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                run_id=run_id,
                base_path=str(base_path),
                full_config=local_config_dict,
                save_artifacts=True,
                api_key=api_key,
                api_secret=api_secret,
                gcs_utils=gcs_utils_for_pipeline
            )
            # El jefe ejecuta con save_artifacts=True para guardar scalers y metadatos
            _, _ = data_pipeline_chief.run()
            logger.info("✅ FASE 1 completada - Artefactos generados y guardados por el proceso jefe")
        
        # === FASE 2: SINCRONIZACIÓN CON BARRERA ===
        # Todos los procesos esperan a que el jefe termine de guardar los artefactos
        if is_distributed:
            logger.info(f"[Proceso {rank}] Esperando en barrera de sincronización...")
            dist.barrier()
            logger.info(f"[Proceso {rank}] ✅ Sincronización completada - Artefactos disponibles para todos")
        
        # === FASE 3: TODOS LOS PROCESOS CARGAN LOS DATOS EN MEMORIA ===
        logger.info(f"=== FASE 3: Cargando Datos en Memoria [Proceso {rank}] ===")
        
        # Obtener credenciales de API desde variables de entorno (para todos los procesos)
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        # Crear instancia local de GCSUtils si es necesario para compatibilidad
        gcs_utils_for_pipeline = None
        if storage_mode == "gcp":
            from src.configuration.gcs_utils import GCSUtils
            gcs_utils_for_pipeline = GCSUtils(gcp_config)
        
        data_pipeline = DataPipeline(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            run_id=run_id,
            base_path=str(base_path),
            full_config=local_config_dict,
            save_artifacts=False,
            api_key=api_key,
            api_secret=api_secret,
            gcs_utils=gcs_utils_for_pipeline
        )
        # Todos los procesos (incluido el jefe) ejecutan con save_artifacts=False
        # Esto carga los datos y los procesa en memoria, usando los scalers ya guardados
        normalized_dataframe, price_scaler_path = data_pipeline.run()
        
        # Ahora todos los procesos tienen los datos cargados en su memoria
        dataframe = normalized_dataframe
        logger.info(f"[Proceso {rank}] ✅ FASE 3 completada - Datos cargados en memoria")
        
        # === FASE 4: CREACIÓN DEL ENTORNO Y AGENTE (TODOS LOS PROCESOS) ===
        logger.info(f"=== FASE 4: Creación del Entorno y Agente [Proceso {rank}] ===")
        
        # Configurar device
        device = setup_device(args.no_cuda)
        logger.info(f"[Proceso {rank}] Usando device: {device}")
        
        # Variables para configuración de checkpoint (se determinarán después)
        path_price_scaler_a_cargar = None
        blob_name_price_scaler_a_cargar = None
        start_episode = 0
        
        # Usar la ruta del price_scaler que devolvió el pipeline para nueva ejecución por defecto
        if storage_mode == "gcp":
            blob_name_price_scaler_a_cargar = f"{run_id}/price_scaler.pkl"
        else:
            path_price_scaler_a_cargar = price_scaler_path
        
        # Crear entorno de trading (todos los procesos)
        logger.info(f"[Proceso {rank}] Creando entorno de trading...")
        env = create_trading_environment(
            dataframe,  # Disponible en todos los procesos
            logger,
            run_manager,  # Ahora todos los procesos tienen run_manager
            env_config=local_config_dict['environment'], # Inyección
            price_scaler_path=path_price_scaler_a_cargar,
            price_scaler_blob_name=blob_name_price_scaler_a_cargar
        )
        
        # Crear agente (todos los procesos) - CRUCIAL: Pasar is_distributed
        logger.info(f"[Proceso {rank}] Creando agente SAC...")
        agent = create_sac_agent(env, device, logger, agent_config=local_config_dict['agent'], is_distributed=is_distributed)
        
        # === FASE 5: GESTIÓN DE CHECKPOINTS Y SINCRONIZACIÓN (CENTRALIZADA EN EL JEFE) ===
        logger.info(f"=== FASE 5: Gestión de Checkpoints [Proceso {rank}] ===")
        
        # Solo el proceso jefe maneja la lógica de checkpoints
        if is_chief:
            logger.info("[Proceso Jefe] Determinando configuración de checkpoint...")
            
            if args.checkpoint is None:
                # No se especificó checkpoint, comenzar desde cero
                logger.info("[Proceso Jefe] Iniciando nueva ejecución (sin checkpoint)")
                start_episode = 0
            else:
                # Se especificó un run_id para cargar checkpoint
                logger.info(f"[Proceso Jefe] Intentando reanudar desde checkpoint del run_id: {args.checkpoint}")
                
                # Buscar checkpoint en el run_id específico usando RunManager (SOLO LECTURA)
                checkpoint_info = run_manager.find_latest_checkpoint(args.checkpoint)
                
                if checkpoint_info:
                    checkpoint_prefix, latest_episode = checkpoint_info
                    
                    logger.info(f"✅ Checkpoint encontrado del episodio {latest_episode}")
                    logger.info(f"Ubicación: {checkpoint_prefix}")
                    
                    try:
                        logger.info(f"[Proceso Jefe] Cargando checkpoint desde: {checkpoint_prefix}")
                        # CARGA DE CHECKPOINT - Solo el jefe carga el estado del agente
                        run_manager.load_agent_from_checkpoint(agent, checkpoint_prefix, reset_optimizers=args.fine_tune_mode)
                        
                        start_episode = latest_episode
                        
                        logger.info(f"✅ Checkpoint cargado exitosamente en el proceso jefe")
                        logger.info(f"  - Run ID de origen: {args.checkpoint}")
                        logger.info(f"  - Episodio inicial: {start_episode}")
                        logger.info(f"  - Total steps: {agent.total_steps}")
                        logger.info(f"  - Learning steps: {agent.learning_steps}")
                        logger.info(f"  - Nuevos artefactos se guardarán en run_id: {run_id}")
                        
                        # Actualizar configuración de price_scaler para cargar desde checkpoint
                        if storage_mode == "gcp":
                            blob_name_price_scaler_a_cargar = f"{args.checkpoint}/price_scaler.pkl"
                            logger.info(f"[Proceso Jefe] Actualizando price_scaler desde GCS (checkpoint): {blob_name_price_scaler_a_cargar}")
                        else:
                            path_price_scaler_a_cargar = f"Entrenamientos/{args.checkpoint}/price_scaler.pkl"
                            logger.info(f"[Proceso Jefe] Actualizando price_scaler desde local (checkpoint): {path_price_scaler_a_cargar}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error al cargar checkpoint desde run_id {args.checkpoint}: {e}")
                        logger.error("Terminando ejecución. Verifique que el run_id sea válido y contenga checkpoints.")
                        return
                else:
                    logger.error(f"❌ No se encontraron checkpoints en el run_id: {args.checkpoint}")
                    logger.error("Terminando ejecución. Verifique que el run_id sea válido y contenga checkpoints.")
                    return
        
        # === FASE 6: SINCRONIZACIÓN DEL ESTADO INICIAL (DISTRIBUIDO) ===
        if is_distributed:
            logger.info(f"=== FASE 6: Sincronización del Estado Inicial [Proceso {rank}] ===")
            
            # Sincronizar start_episode desde el jefe a todos los procesos
            sync_device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
            start_episode_tensor = torch.tensor([start_episode], dtype=torch.long, device=sync_device)
            
            logger.info(f"[Proceso {rank}] Sincronizando start_episode...")
            dist.broadcast(start_episode_tensor, src=0)
            
            # Actualizar la variable en los procesos trabajadores
            if not is_chief:
                start_episode = start_episode_tensor.item()
                logger.info(f"[Proceso {rank}] start_episode sincronizado: {start_episode}")
            
            # Sincronizar los pesos del modelo desde el jefe a todos los procesos
            logger.info(f"[Proceso {rank}] Sincronizando pesos del modelo...")
            
            # Sincronizar pesos del Actor
            for param in agent.actor.parameters():
                dist.broadcast(param.data, src=0)
            
            # Sincronizar pesos de los Critics
            for param in agent.critic_1.parameters():
                dist.broadcast(param.data, src=0)
            for param in agent.critic_2.parameters():
                dist.broadcast(param.data, src=0)
            
            # Sincronizar pesos de los Critics Target
            for param in agent.critic_target_1.parameters():
                dist.broadcast(param.data, src=0)
            for param in agent.critic_target_2.parameters():
                dist.broadcast(param.data, src=0)
            
            # Sincronizar parámetro de temperatura (alpha)
            if hasattr(agent, 'log_alpha'):
                dist.broadcast(agent.log_alpha.data, src=0)
            
            logger.info(f"[Proceso {rank}] ✅ Sincronización de pesos completada")
            
            # Barrera final para asegurar que todos los procesos estén sincronizados
            dist.barrier()
            logger.info(f"[Proceso {rank}] ✅ Todos los procesos sincronizados y listos para entrenar")
        
        # === FASE 7: CONFIGURACIÓN FINAL DE ENTRENAMIENTO (TODOS LOS PROCESOS) ===
        logger.info(f"=== FASE 7: Configuración Final de Entrenamiento [Proceso {rank}] ===")
        logger.info(f"  - Episodio inicial: {start_episode}")
        logger.info(f"  - Episodios totales: {args.episodes}")
        logger.info(f"  - Episodios por entrenar: {args.episodes - start_episode}")
        logger.info(f"  - Run ID actual: {run_id}")
        
        # Verificar que queden episodios por entrenar
        if start_episode >= args.episodes:
            logger.warning(f"El checkpoint ya alcanzó o superó el número de episodios objetivo ({args.episodes})")
            logger.info("No hay episodios adicionales para entrenar. Terminando...")
            return
        
        # Crear instancias para el entrenamiento (condicionalmente según el proceso)
        evaluator = AgentEvaluator() if is_chief else None
        
        # Configuración para el trainer (todos los procesos)
        trainer_config = {
            'seed': seed,
            'batch_size': local_config_dict['agent']['batch_size'],
            'min_buffer_for_learning': local_config_dict['agent']['min_buffer_for_learning'],
            'replay_buffer_size': local_config_dict['agent']['replay_buffer_size'],
            'eval_frequency': args.eval_frequency,
            'eval_episodes': args.eval_episodes,
            'save_frequency': args.save_frequency,
            'storage_mode': storage_mode,
            'run_id': run_id,
            'tensorboard_dir': tensorboard_dir if is_chief else None,
            'gcs_bucket_name': gcp_config.get('storage', {}).get('bucket_name') if storage_mode == 'gcp' else None
        }
        
        # Crear trainer (todos los procesos) con instanciación condicional
        trainer = Trainer(
            agent=agent,
            env=env,
            evaluator=evaluator,  # Solo el jefe tiene evaluador
            logger=tb_logger if is_chief else None,  # Solo el jefe tiene tb_logger
            run_manager=run_manager if is_chief else None,  # Solo el jefe usa run_manager para escritura
            trainer_config=trainer_config,
            logger_console=logger
        )
        
        # === FASE 8: EJECUCIÓN DEL ENTRENAMIENTO DISTRIBUIDO ===
        logger.info(f"=== FASE 8: Iniciando Entrenamiento Distribuido [Proceso {rank}] ===")
        
        # TODOS los procesos participan en el entrenamiento - DDP maneja la sincronización
        trainer.train(start_episode=start_episode, total_episodes=args.episodes)
        
        logger.info(f"=== Proceso {rank} Completado Exitosamente ===")
        
    except KeyboardInterrupt:
        logger.info(f"[Proceso {rank}] Proceso interrumpido por el usuario")
        
    except Exception as e:
        logger.error(f"[Proceso {rank}] Error during execution: {e}")
        logger.exception("Detalles del error:")
        
    finally:
        # === LIMPIEZA Y FINALIZACIÓN ORDENADA ===
        logger.info(f"[Proceso {rank}] Iniciando limpieza final...")
        
        # Cerrar TensorBoard logger (solo el jefe)
        if is_chief and 'tb_logger' in locals() and tb_logger is not None:
            try:
                tb_logger.close()
                logger.info("[Proceso Jefe] TensorBoard logger cerrado exitosamente")
            except Exception as e:
                logger.warning(f"[Proceso Jefe] Error al cerrar TensorBoard logger: {e}")
        
        # Limpiar el entorno distribuido si es necesario
        if is_distributed:
            try:
                logger.info(f"[Proceso {rank}] Cerrando grupo de procesos distribuidos...")
                dist.destroy_process_group()
                logger.info(f"[Proceso {rank}] ✅ Grupo de procesos cerrado exitosamente")
            except Exception as e:
                logger.warning(f"[Proceso {rank}] Error al cerrar proceso distribuido: {e}")
        
        logger.info(f"[Proceso {rank}] ✅ Limpieza completada - Finalizando proceso")


if __name__ == "__main__":
    # Forzar a NCCL a usar una interfaz de red común en entornos cloud para evitar timeouts.
    # Leer desde la configuración para mayor flexibilidad.
    try:
        with open('src/configuration/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        nccl_socket_ifname = config.get('system', {}).get('nccl_socket_ifname', 'eth0')
        os.environ['NCCL_SOCKET_IFNAME'] = nccl_socket_ifname
        print(f"Establecida variable de entorno NCCL_SOCKET_IFNAME='{nccl_socket_ifname}'")
    except Exception as e:
        print(f"Advertencia: No se pudo leer nccl_socket_ifname de config.yaml. Usando valor por defecto 'eth0'. Error: {e}")
        os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'

    # Forzar el método 'spawn' para multiprocessing para evitar problemas de CUDA
    # en los procesos hijos que guardan los modelos. Es la solución estándar.
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()