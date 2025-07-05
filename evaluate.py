#!/usr/bin/env python3
"""
Script de evaluación para modelos entrenados de btcbot.

Este script realiza un backtesting exhaustivo de un modelo ya entrenado,
generando artefactos de evaluación incluyendo métricas y logs de TensorBoard.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Imports from project modules
from src.utils.cli import parse_evaluation_arguments
from src.configuration.config_manager import ConfigManager
from src.data.artifact_manager import ArtifactManager
from src.training.checkpoint_manager import CheckpointManager
from src.entorno.factory import create_trading_environment
from src.agente.factory import create_sac_agent
from src.training.evaluator import AgentEvaluator
from src.analysis.logger import TensorboardLogger
from src.configuration.config_model import AppConfig
from src.configuration import EnvironmentConfig, AgentConfig
from src.utils.system import setup_device, setup_logging


def generate_evaluation_id(run_id: str) -> str:
    """
    Genera un ID único para la evaluación basado en el run_id y timestamp.
    
    Args:
        run_id: ID del training run
        
    Returns:
        str: ID único de evaluación
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval_{run_id}_{timestamp}"


def create_evaluation_directories(evaluation_id: str) -> tuple[Path, Path]:
    """
    Crea los directorios necesarios para almacenar los artefactos de evaluación.
    
    Args:
        evaluation_id: ID único de la evaluación
        
    Returns:
        tuple[Path, Path]: Rutas del directorio de evaluación y logs de TensorBoard
    """
    # Crear directorio principal de evaluaciones
    eval_dir = Path("evaluations") / evaluation_id
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear directorio para logs de TensorBoard de evaluaciones
    tensorboard_dir = Path("tensorboard_logs_evaluations") / evaluation_id
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    return eval_dir, tensorboard_dir


def save_evaluation_summary(eval_dir: Path, metrics: Dict[str, float], evaluation_id: str) -> None:
    """
    Guarda el resumen de métricas de evaluación en formato JSON.
    
    Args:
        eval_dir: Directorio de evaluación
        metrics: Diccionario con las métricas finales
        evaluation_id: ID de la evaluación
    """
    summary = {
        "evaluation_id": evaluation_id,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics
    }
    
    summary_file = eval_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Resumen de evaluación guardado en: {summary_file}")


def upload_evaluation_artifacts(config_manager: ConfigManager, evaluation_id: str, 
                               eval_dir: Path, tensorboard_dir: Path, gcp_config: Dict[str, Any]) -> None:
    """
    Sube los artefactos de evaluación a GCS si el storage_mode es 'gcp'.
    
    Args:
        config_manager: Manager de configuración
        evaluation_id: ID de la evaluación
        eval_dir: Directorio local de evaluación
        tensorboard_dir: Directorio local de logs de TensorBoard
        gcp_config: Configuración GCP
    """
    if config_manager.storage_mode == "gcp":
        logging.info("Subiendo artefactos de evaluación a GCS...")
        
        # Crear instancia de GCSUtils
        from src.configuration.gcs_utils import GCSUtils
        gcs_utils = GCSUtils(gcp_config)
        
        # Subir el resumen JSON
        summary_file = eval_dir / "evaluation_summary.json"
        if summary_file.exists():
            gcs_path = f"evaluations/{evaluation_id}/evaluation_summary.json"
            gcs_utils.upload_file_to_gcs(str(summary_file), gcs_path)
        
        # Subir logs de TensorBoard
        if tensorboard_dir.exists():
            for file_path in tensorboard_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(tensorboard_dir)
                    gcs_path = f"evaluations/{evaluation_id}/tensorboard_logs/{relative_path}"
                    gcs_utils.upload_file_to_gcs(str(file_path), gcs_path)
        
        logging.info("Artefactos de evaluación subidos exitosamente a GCS")


def validate_run_id(run_id: str) -> None:
    """
    Valida que el formato del run_id sea correcto.
    
    Args:
        run_id: ID del training run a validar
        
    Raises:
        ValueError: Si el format del run_id no es válido
    """
    # Validación básica del formato (debe empezar con training_)
    if not run_id.startswith('training_'):
        raise ValueError(f"El run_id debe empezar con 'training_', recibido: {run_id}")
    
    # Verificar que no esté vacío después del prefijo
    if len(run_id) <= len('training_'):
        raise ValueError(f"El run_id parece estar incompleto: {run_id}")


def main():
    """Función principal del script de evaluación."""
    # Parsear argumentos
    args = parse_evaluation_arguments()
    run_id = args.run_id
    model_type = args.model_type
    
    # Validar formato de run_id
    validate_run_id(run_id)
    
    # Generar ID único de evaluación
    evaluation_id = generate_evaluation_id(run_id)
    
    # Configurar logging básico
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Iniciando evaluación del modelo {model_type} del run: {run_id}")
    logger.info(f"ID de evaluación: {evaluation_id}")
    
    # Validar formato del run_id
    try:
        validate_run_id(run_id)
        logger.info("✅ Formato del run_id válido")
    except ValueError as e:
        logger.error(f"❌ Error en el formato del run_id: {e}")
        sys.exit(1)
    
    try:
        # 1. Centralized configuration loading using ConfigManager
        logger.info("Cargando configuración del training run...")
        
        # First, try to load local configuration to get GCP details if needed
        gcp_config_for_load = None
        try:
            from src.configuration import AppConfig
            local_config_obj = AppConfig.from_yaml_file('src/configuration/config.yaml')
            gcp_config_for_load = local_config_obj.gcp.model_dump() if hasattr(local_config_obj, 'gcp') else None
        except FileNotFoundError:
            logger.warning("No se encontró config.yaml local. Se asumirá que no se necesita gcp_config para cargar.")
        
        # Load the complete training run configuration using ConfigManager
        config_dict = ConfigManager.load_training_run_config(run_id, gcp_config=gcp_config_for_load)
        if config_dict is None:
            raise ValueError(f"No se pudo cargar la configuración para el run: {run_id}")
        
        # Extract data_run_id from lineage section
        lineage = config_dict.get('lineage', {})
        data_run_id = lineage.get('data_run_id')
        if not data_run_id:
            raise ValueError(f"No se encontró data_run_id en el linaje del run: {run_id}")
        
        logger.info(f"Data run ID extraído del linaje: {data_run_id}")
        
        # Create configuration object from loaded data
        config = AppConfig(**config_dict.get('config', {}))
        
        # 2. Setup device and logging
        device = setup_device(no_cuda=False)
        logger.info(f"Usando dispositivo: {device}")
        
        # 3. Initialize managers with the loaded configuration
        main_config = config_dict.get('config', {})
        storage_mode = main_config.get('normalization', {}).get('storage_mode', 'local')
        gcp_config = main_config.get('gcp') if storage_mode == 'gcp' else None
        
        config_manager = ConfigManager(storage_mode=storage_mode, gcp_config=gcp_config)
        artifact_manager = ArtifactManager(storage_mode=storage_mode, gcp_config=gcp_config)
        checkpoint_manager = CheckpointManager(storage_mode=storage_mode, gcp_config=gcp_config)
        
        # 4. Load data artifacts using ArtifactManager
        logger.info(f"Cargando artefactos del data_run: {data_run_id}")
        normalized_dataframe, scaler, price_scaler = artifact_manager.load_data_artifacts(data_run_id)
        
        if normalized_dataframe is None or scaler is None or price_scaler is None:
            raise ValueError(f"No se pudieron cargar los artefactos del data_run: {data_run_id}")
        
        logger.info("✅ Artefactos de datos cargados exitosamente mediante ArtifactManager")
        
        # 5. Reconstruir el entorno
        logger.info("Reconstruyendo entorno de trading...")
        env = create_trading_environment(
            dataframe=normalized_dataframe,
            logger=logger,
            price_scaler=price_scaler,
            scaler=scaler,
            env_config=config.environment,
            run_config=config_dict
        )
        
        # 6. Reconstruir el agente
        logger.info("Reconstruyendo agente...")
        agent = create_sac_agent(
            env=env,
            device=device,
            logger=logger,
            agent_config=config.agent,
            is_distributed=False
        )
        
        # 7. Cargar pesos del modelo usando CheckpointManager
        logger.info(f"Cargando pesos del {model_type}...")
        
        # Construct the correct checkpoint prefix path
        checkpoint_prefix = f"training_runs/{run_id}/{model_type}"
        logger.info(f"Ruta del checkpoint: {checkpoint_prefix}")
        
        checkpoint_manager.load_agent_from_checkpoint(
            agent=agent,
            checkpoint_prefix=checkpoint_prefix,
            reset_optimizers=False
        )
        
        # 8. Crear directorios de evaluación
        eval_dir, tensorboard_dir = create_evaluation_directories(evaluation_id)
        
        # 9. Realizar evaluación
        logger.info("Ejecutando evaluación completa...")
        evaluator = AgentEvaluator()
        final_metrics, equity_curve, trade_pnl_list = evaluator.evaluate(agent, env)
        
        logger.info("Evaluación completada exitosamente")
        
        # Mostrar métricas principales
        logger.info("=" * 60)
        logger.info("📊 RESUMEN DE MÉTRICAS PRINCIPALES:")
        logger.info("=" * 60)
        
        key_metrics = [
            ('Total Return', final_metrics.get('total_return', 'N/A')),
            ('Sharpe Ratio', final_metrics.get('sharpe_ratio', 'N/A')),
            ('Max Drawdown', final_metrics.get('max_drawdown', 'N/A')),
            ('Win Rate', final_metrics.get('win_rate', 'N/A')),
            ('Total Trades', final_metrics.get('total_trades', 'N/A')),
        ]
        
        for metric_name, metric_value in key_metrics:
            if isinstance(metric_value, float):
                logger.info(f"  {metric_name:15s}: {metric_value:.4f}")
            else:
                logger.info(f"  {metric_name:15s}: {metric_value}")
        
        logger.info("=" * 60)
        
        # 10. Guardar resumen de métricas
        save_evaluation_summary(eval_dir, final_metrics, evaluation_id)
        
        # 11. Configurar TensorBoard logger y guardar resultados completos
        logger.info("Guardando resultados en TensorBoard...")
        tensorboard_logger = TensorboardLogger(log_dir=str(tensorboard_dir))
        
        # Preparar hiperparámetros para TensorBoard
        hparams = {
            "training_run_id": run_id,
            "data_run_id": data_run_id,
            "model_type": model_type,
            "evaluation_id": evaluation_id,
            # Añadir algunos hiperparámetros clave del entrenamiento
            "learning_rate": config.agent.hiperparametros_sac.actor_learning_rate,
            "batch_size": config.agent.batch_size,
            "ventana_observacion_size": config.environment.ventana_observacion_size,
            "initial_balance": config.environment.capital_inicial,
        }
        
        # Guardar en TensorBoard
        tensorboard_logger.log_evaluation_summary(
            hparams=hparams,
            final_metrics=final_metrics,
            equity_curve=equity_curve,
            trade_pnl_list=trade_pnl_list
        )
        
        tensorboard_logger.close()
        
        # 12. Subir artefactos a GCS si corresponde
        upload_evaluation_artifacts(config_manager, evaluation_id, eval_dir, tensorboard_dir, gcp_config)
        
        logger.info(f"🎉 Evaluación completada exitosamente!")
        logger.info(f"📊 Resultados guardados en: {eval_dir}")
        logger.info(f"📈 Logs de TensorBoard en: {tensorboard_dir}")
        logger.info(f"🆔 ID de evaluación: {evaluation_id}")
        
    except Exception as e:
        logger.error(f"Error durante la evaluación: {str(e)}")
        raise


if __name__ == "__main__":
    main()
