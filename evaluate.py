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
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Importaciones del proyecto
from src.data.pipeline import DataPipeline
from src.entorno.environment import FuturesTradingEnv
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
    
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Símbolo del par de trading (ej: BTCUSDT)"
    )
    
    parser.add_argument(
        "--interval",
        type=str,
        required=True,
        help="Intervalo de tiempo para las velas (ej: 1h, 4h, 1d)"
    )
    
    # Argumentos opcionales
    parser.add_argument(
        "--model-type",
        type=str,
        default="best",
        choices=["best", "final"],
        help="Tipo de modelo a cargar"
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
        device=device,
        config_override=agent_config # Inyección directa
    )


def print_evaluation_report(metrics: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Imprime un informe detallado de la evaluación.
    
    Args:
        metrics: Diccionario con las métricas de evaluación
        args: Argumentos de línea de comandos
    """
    print("\n" + "="*80)
    print("📊 INFORME DE EVALUACIÓN DEL MODELO")
    print("="*80)
    
    # Información del modelo
    print(f"🎯 Modelo evaluado:")
    print(f"   • Run ID: {args.run_id}")
    print(f"   • Tipo de modelo: {args.model_type}")
    print(f"   • Símbolo: {args.symbol}")
    print(f"   • Intervalo: {args.interval}")
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
    
    # Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Parsear argumentos
        args = parse_arguments()
        
        # Validar fechas
        validate_dates(args.start_date, args.end_date)
        
        logger.info("🚀 Iniciando evaluación del modelo")
        logger.info(f"Run ID: {args.run_id}")
        logger.info(f"Modelo: {args.model_type}")
        logger.info(f"Período: {args.start_date} a {args.end_date}")
        logger.info(f"Símbolo: {args.symbol} ({args.interval})")
        
        # Configurar dispositivo
        device = setup_device(args.no_cuda)
        logger.info(f"Dispositivo configurado: {device}")
        
        # Instanciar RunManager y cargar configuración del run
        run_manager = RunManager()
        run_manager.set_run_context(args.run_id)
        run_config = run_manager.download_and_load_yaml_config(args.run_id)
        if not run_config:
            logger.error(f"No se pudo cargar la configuración para el run_id: {args.run_id}. Abortando.")
            sys.exit(1)
        logger.info("Configuración del run cargada exitosamente.")

        # Extraer configuraciones específicas
        env_config = run_config['config_snapshot']['environment']
        agent_config = run_config['config_snapshot']['agent']
        
        # Pipeline de datos
        logger.info("📊 Ejecutando pipeline de datos...")
        data_pipeline = DataPipeline(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date,
            run_id=f"evaluation_{args.run_id}",
            base_path="temp_evaluation",
            save_artifacts=False # No guardar artefactos durante la evaluación
        )
        
        normalized_dataframe, _ = data_pipeline.run()
        logger.info(f"Pipeline completado. Datos: {normalized_dataframe.shape}")
        
        # Cargar price_scaler del entrenamiento original
        logger.info("📦 Cargando price_scaler del entrenamiento original...")
        price_scaler = run_manager.load_price_scaler(
            blob_name=f"{args.run_id}/price_scaler.pkl"
        )
        logger.info("Price scaler cargado exitosamente")
        
        # Crear entorno con la configuración inyectada
        logger.info("🏗️ Creando entorno de trading...")
        env = FuturesTradingEnv(
            data_df=normalized_dataframe,
            price_scaler=price_scaler,
            env_config=env_config # Inyección de configuración
        )
        
        # Obtener información del entorno para crear el agente
        obs, _ = env.reset()
        observation_space_shape = obs.shape
        action_space_shape = env.action_space.shape
        
        # Obtener características del agente desde la configuración del run
        hyperparams = run_config.get('hyperparameters', {})
        market_features = hyperparams.get('market_features', len(env.column_names))
        portfolio_features = hyperparams.get('portfolio_features', 4)
        sequence_length = env_config.get('ventana_observacion_size', 24)
        
        logger.info(f"Espacio de observación: {observation_space_shape}")
        logger.info(f"Espacio de acción: {action_space_shape}")
        logger.info(f"Características de mercado: {market_features}")
        logger.info(f"Características de portfolio: {portfolio_features}")
        logger.info(f"Longitud de secuencia: {sequence_length}")
        
        # Crear agente
        logger.info("🤖 Creando agente...")
        agent = create_agent_from_config(
            observation_space_shape=observation_space_shape,
            action_space_shape=action_space_shape,
            market_features=market_features,
            portfolio_features=portfolio_features,
            sequence_length=sequence_length,
            device=device,
            agent_config=agent_config # Inyección de configuración
        )
        
        # Cargar pesos del modelo
        logger.info("📥 Cargando pesos del modelo...")
        # Corregir la ruta para que apunte al directorio de entrenamientos
        model_prefix = f"Entrenamientos/{args.run_id}/{args.model_type}_model/{args.model_type}_model"
        logger.info(f"Cargando modelo con prefijo: {model_prefix}")
        run_manager.load_agent_from_checkpoint(agent, model_prefix, reset_optimizers=True)
        logger.info("Modelo cargado exitosamente")
        
        # Ejecutar evaluación
        logger.info("🎯 Ejecutando evaluación...")
        evaluator = AgentEvaluator()
        
        metrics = evaluator.evaluate(
            agent=agent,
            env=env,
            num_episodes=1  # Una sola pasada por los datos
        )
        
        logger.info("Evaluación completada")
        
        # Generar y mostrar informe
        print_evaluation_report(metrics, args)
        
        # Limpiar archivos temporales del pipeline
        try:
            import shutil
            if Path("temp_evaluation").exists():
                shutil.rmtree("temp_evaluation")
                logger.info("Archivos temporales eliminados")
        except Exception as e:
            logger.warning(f"No se pudieron eliminar archivos temporales: {e}")
        
        logger.info("✅ Evaluación completada exitosamente")
        
    except KeyboardInterrupt:
        logger.info("❌ Evaluación interrumpida por el usuario")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Error durante la evaluación: {str(e)}")
        import traceback
        logger.error(f"Traceback completo:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
