#!/usr/bin/env python3
"""
Script de evaluación de modelos pre-entrenados para btcbot.

Este script permite evaluar un modelo pre-entrenado cargándolo desde Google Cloud Storage
o almacenamiento local y ejecutándolo en un rango de fechas específico para obtener
métricas de rendimiento detalladas.

Uso:
    python evaluate.py --run-id XXXXXXXX --start-date 2024-01-01 --end-date 2024-01-31 
                       --symbol BTCUSDT --interval 1h
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Importaciones del proyecto
from src.data.pipeline import DataPipeline
from src.entorno.environment import FuturesTradingEnv
from src.entorno.portfolio import Portfolio
from src.agente.agent import TransformerSACAgent
from src.training.run_manager import RunManager
from src.training.evaluator import AgentEvaluator
from src.utils.system import setup_logging, setup_device

def parse_arguments() -> argparse.Namespace:
    """
    Parsea los argumentos de línea de comandos.
    
    Returns:
        argparse.Namespace: Argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description="Evalúa un modelo pre-entrenado de trading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argumentos requeridos
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="ID del entrenamiento a evaluar"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Fecha de inicio para la evaluación (formato: YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Fecha de fin para la evaluación (formato: YYYY-MM-DD)"
    )
    
    # Leer el valor por defecto desde el archivo de configuración principal
    try:
        with open('src/configuration/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        default_model_type = config.get('evaluation', {}).get('default_model_to_load', 'best')
    except (FileNotFoundError, yaml.YAMLError):
        default_model_type = 'best'

    # Argumentos opcionales
    parser.add_argument(
        "--model-type",
        type=str,
        default=default_model_type,
        choices=["best", "final"],
        help=f"Tipo de modelo a cargar (por defecto: {default_model_type})"
    )
    
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Fuerza el uso de CPU en lugar de GPU"
    )
    
    return parser.parse_args()


def validate_dates(start_date: str, end_date: str) -> bool:
    """
    Valida el formato y coherencia de las fechas.
    
    Args:
        start_date: Fecha de inicio en formato YYYY-MM-DD
        end_date: Fecha de fin en formato YYYY-MM-DD
        
    Returns:
        bool: True si las fechas son válidas
        
    Raises:
        ValueError: Si las fechas no son válidas
    """
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        if start_dt >= end_dt:
            raise ValueError("La fecha de inicio debe ser anterior a la fecha de fin")
            
        return True
        
    except ValueError as e:
        if "time data" in str(e):
            raise ValueError("Formato de fecha inválido. Use YYYY-MM-DD")
        raise


def create_agent_from_config(
    observation_space_shape: tuple,
    action_space_shape: tuple,
    market_features: int,
    portfolio_features: int,
    sequence_length: int,
    device,
    agent_config: Dict[str, Any]
) -> TransformerSACAgent:
    """
    Crea una instancia del agente usando la configuración del entrenamiento.
    
    Args:
        observation_space_shape: Forma del espacio de observación
        action_space_shape: Forma del espacio de acción
        market_features: Número de características de mercado
        portfolio_features: Número de características de portfolio
        sequence_length: Longitud de la secuencia
        device: Dispositivo (CPU/GPU)
        agent_config: Configuración del agente desde el run de entrenamiento.
        
    Returns:
        TransformerSACAgent: Instancia del agente
    """
    logger = logging.getLogger(__name__)
    logger.info("Usando configuración del agente desde el entrenamiento original.")
    
    return TransformerSACAgent(
        observation_space_shape=observation_space_shape,
        action_space_shape=action_space_shape,
        market_features=market_features,
        portfolio_features=portfolio_features,
        sequence_length=sequence_length,
        config_override=agent_config, # Inyección directa
        device=device
    )


def print_evaluation_report(metrics: Dict[str, Any], args: argparse.Namespace, exp_config: Dict[str, Any]) -> None:
    """
    Imprime un informe detallado de la evaluación.
    
    Args:
        metrics: Diccionario con las métricas de evaluación
        args: Argumentos de línea de comandos
        exp_config: Configuración del experimento extraída del run
    """
    print("\n" + "="*80)
    print("📊 INFORME DE EVALUACIÓN DEL MODELO")
    print("="*80)
    
    # Información del modelo
    print(f"🎯 Modelo evaluado:")
    print(f"   • Run ID: {args.run_id}")
    print(f"   • Tipo de modelo: {args.model_type}")
    print(f"   • Símbolo: {exp_config['symbol']}")
    print(f"   • Intervalo: {exp_config['interval']}")

    print(f"   • Período: {args.start_date} a {args.end_date}")
    
    print(f"\n💰 Métricas de Rendimiento:")
    print(f"   • Return total: {metrics.get('mean_return', 0):.4f}")
    print(f"   • Profit promedio: {metrics.get('mean_profit', 0):.2f}%")
    print(f"   • Profit total: {metrics.get('total_profit', 0):.2f}%")
    
    print(f"\n📈 Métricas de Riesgo:")
    print(f"   • Máximo Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
    print(f"   • Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"   • Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}")
    
    print(f"\n🎯 Métricas de Trading:")
    print(f"   • Trades totales: {metrics.get('total_trades', 0)}")
    print(f"   • Trades exitosos: {metrics.get('successful_trades', 0)}")
    print(f"   • Tasa de éxito: {metrics.get('win_rate', 0):.2f}%")
    print(f"   • Duración promedio episodio: {metrics.get('mean_episode_length', 0):.0f} pasos")
    
    # Información adicional si está disponible
    if 'episode_summary' in metrics:
        summary = metrics['episode_summary']
        print(f"\n📋 Resumen del Episodio:")
        print(f"   • Balance inicial: ${summary.get('initial_balance', 0):,.2f}")
        print(f"   • Balance final: ${summary.get('final_balance', 0):,.2f}")
        print(f"   • Equity final: ${summary.get('final_equity', 0):,.2f}")
        print(f"   • Rendimiento: {summary.get('total_return_pct', 0):.2f}%")
    
    print("\n" + "="*80)
    print("✅ Evaluación completada exitosamente")
    print("="*80 + "\n")


def main():
    """Función principal del script de evaluación."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        args = parse_arguments()
        validate_dates(args.start_date, args.end_date)

        logger.info(f"🚀 Iniciando evaluación del modelo para Run ID: {args.run_id}")

        # --- 1. Carga de Configuración como Única Fuente de Verdad ---
        logger.info("📁 Cargando configuración del run...")
        
        # Cargar la configuración local solo para obtener los detalles de GCP
        # Esto es necesario para que RunManager sepa a qué bucket conectarse.
        try:
            with open('src/configuration/config.yaml', 'r') as f:
                local_config = yaml.safe_load(f)
            gcp_config_local = local_config.get('gcp')
        except FileNotFoundError:
            logger.warning("No se encontró config.yaml local. Se asumirá que no se necesita gcp_config.")
            gcp_config_local = None

        run_config = RunManager.load_training_run_config(args.run_id, gcp_config=gcp_config_local)
        if not run_config or 'config' not in run_config:
            logger.error(f"No se pudo cargar o es inválida la configuración para el run_id: {args.run_id}. Abortando.")
            sys.exit(1)
        
        main_config = run_config['config']
        env_config = main_config['environment']
        agent_config = main_config['agent']
        storage_mode = main_config.get('normalization', {}).get('storage_mode', 'local')
        gcp_config = main_config.get('gcp') if storage_mode == 'gcp' else None
        logger.info("✅ Configuración del run cargada y validada.")

        # --- EXTRAER SYMBOL E INTERVALO DEL RUN CONFIG ---
        try:
            # Intentar cargar desde experiment_definition (legacy) primero
            if 'experiment_definition' in main_config:
                exp_config = main_config['experiment_definition']
                symbol = exp_config['symbol']
                interval = exp_config['interval']
                logger.info(f"Símbolo ({symbol}) e Intervalo ({interval}) extraídos de 'experiment_definition' (legacy).")
            # Si no existe, cargar desde metadata.experiment_parameters (nueva estructura)
            elif 'metadata' in main_config and 'experiment_parameters' in main_config['metadata']:
                exp_params = main_config['metadata']['experiment_parameters']
                symbol = exp_params['symbol']
                interval = exp_params['interval']
                logger.info(f"Símbolo ({symbol}) e Intervalo ({interval}) extraídos de 'metadata.experiment_parameters'.")
            else:
                raise KeyError("No se encontró ni 'experiment_definition' ni 'metadata.experiment_parameters'")
                
        except KeyError as e:
            logger.error(f"La configuración para '{args.run_id}' no contiene información del experimento: {e}")
            sys.exit(1)

        # --- 2. Inicialización de Componentes Esenciales ---
        device = setup_device(args.no_cuda)
        logger.info(f"Dispositivo configurado: {device}")

        run_manager = RunManager(
            run_id=args.run_id,
            storage_mode=storage_mode,
            gcp_config=gcp_config
        )
        logger.info(f"RunManager inicializado en modo '{storage_mode}'.")

        # --- 3. Pipeline de Datos para Evaluación ---
        logger.info("📊 Ejecutando pipeline de datos para el período de evaluación...")
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        gcs_utils_pipeline = None
        if storage_mode == "gcp":
            from src.configuration.gcs_utils import GCSUtils
            gcs_utils_pipeline = GCSUtils(gcp_config)

        # Usar el nombre del directorio temporal desde la configuración
        temp_dir_name = main_config.get('evaluation', {}).get('temp_directory', 'temp_evaluation')

        data_pipeline = DataPipeline(
            symbol=symbol, interval=interval, start_date=args.start_date, end_date=args.end_date,
            run_id=f"evaluation_{args.run_id}", base_path=temp_dir_name,
            full_config=main_config,  # Inyectar la configuración completa del run
            save_artifacts=False, # No guardar artefactos durante la evaluación
            api_key=api_key, api_secret=api_secret,
            gcs_utils=gcs_utils_pipeline
        )
        normalized_dataframe, price_scaler = data_pipeline.run()
        logger.info(f"Pipeline completado. Datos para evaluación: {normalized_dataframe.shape}")

        # --- 4. Creación del Entorno y Agente ---
        logger.info("🏗️ Creando entorno de trading...")
        sim_portfolio = Portfolio(env_config)
        env = FuturesTradingEnv(
            data_df=normalized_dataframe,
            price_scaler=price_scaler, # Inyectar el scaler devuelto por el pipeline
            env_config=env_config,  # Inyección de configuración del run
            portfolio=sim_portfolio
        )

        obs, _ = env.reset()
        market_features = len(env.column_names)
        # Leer portfolio_features desde la configuración
        portfolio_features = agent_config.get('architecture', {}).get('portfolio_features', 4)
        sequence_length = env_config.get('ventana_observacion_size', 24)

        logger.info("🤖 Creando agente desde la configuración del run...")
        agent = create_agent_from_config(
            observation_space_shape=obs.shape,
            action_space_shape=env.action_space.shape,
            market_features=market_features, portfolio_features=portfolio_features, sequence_length=sequence_length,
            device=device,
            agent_config=agent_config  # Inyección de configuración del run
        )

        # --- 5. Carga del Modelo y Ejecución de la Evaluación ---
        logger.info(f"📥 Cargando pesos del modelo '{args.model_type}'...")
        model_prefix = f"{args.run_id}/{args.model_type}_model"
        run_manager.load_agent_from_checkpoint(agent, model_prefix, reset_optimizers=True)
        logger.info("Modelo cargado exitosamente.")

        logger.info("🎯 Ejecutando evaluación...")
        evaluator = AgentEvaluator()
        metrics = evaluator.evaluate(agent=agent, env=env, num_episodes=1)
        logger.info("Evaluación completada.")

        # --- 6. Informe y Limpieza ---
        print_evaluation_report(metrics, args, exp_config)

        try:
            import shutil
            if Path(temp_dir_name).exists():
                shutil.rmtree(temp_dir_name)
                logger.info("Archivos temporales de evaluación eliminados.")
        except Exception as e:
            logger.warning(f"No se pudieron eliminar archivos temporales: {e}")

        logger.info("✅ Proceso de evaluación finalizado exitosamente.")

    except KeyboardInterrupt:
        logger.info("❌ Evaluación interrumpida por el usuario.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error catastrófico durante la evaluación: {str(e)}")
        import traceback
        logger.error(f"Traceback completo:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
