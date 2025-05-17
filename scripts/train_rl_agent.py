#!/usr/bin/env python
"""
Script para entrenar un agente de RL para trading de criptomonedas.
Optimizado para ejecutarse como un componente de Vertex AI Pipelines.
"""
import argparse
import logging
import os
import json
import time

from src.agent.rl_agent_manager import RLAgentManagerCloud

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Función principal para entrenar un agente RL."""
    parser = argparse.ArgumentParser(description="Entrena un agente RL para trading")
    
    # Parámetros para datos de entrenamiento
    parser.add_argument("--input-data-gcs", type=str, required=True,
                        help="Ruta completa al archivo de datos procesados en GCS")
    parser.add_argument("--output-model-gcs", type=str, required=True,
                        help="Ruta GCS completa donde guardar el modelo entrenado")
    parser.add_argument("--export-model-gcs", type=str, required=False,
                        default=None,
                        help="Ruta GCS donde guardar el modelo exportado para servir (opcional)")
    
    # Parámetros de GCP
    parser.add_argument("--project-id", type=str, required=False,
                        default=os.getenv("GCP_PROJECT_ID"),
                        help="ID del proyecto GCP (default: desde variable GCP_PROJECT_ID)")
    parser.add_argument("--models-bucket", type=str, required=False,
                        default=os.getenv("MODELS_STAGING_BUCKET"),
                        help="Bucket para modelos (default: desde variable MODELS_STAGING_BUCKET)")
    
    # Parámetros del entorno
    parser.add_argument("--sequence-length", type=int, required=False,
                        default=int(os.getenv("SEQUENCE_LENGTH_L", "96")),
                        help="Longitud de la secuencia (default: desde SEQUENCE_LENGTH_L o 96)")
    parser.add_argument("--initial-equity", type=float, required=False,
                        default=float(os.getenv("INITIAL_EQUITY", "10000.0")),
                        help="Equity inicial (default: desde INITIAL_EQUITY o 10000.0)")
    parser.add_argument("--leverage", type=int, required=False,
                        default=int(os.getenv("LEVERAGE", "1")),
                        help="Apalancamiento (default: desde LEVERAGE o 1)")
    parser.add_argument("--position-size", type=float, required=False,
                        default=float(os.getenv("POSITION_SIZE_PERCENTAGE", "0.2")),
                        help="Tamaño de posición como % de equity (default: desde POSITION_SIZE_PERCENTAGE o 0.2)")
    parser.add_argument("--stop-loss", type=float, required=False,
                        default=None if os.getenv("STOP_LOSS_PERCENTAGE") is None else float(os.getenv("STOP_LOSS_PERCENTAGE")),
                        help="Porcentaje de stop loss (default: None)")
    parser.add_argument("--take-profit", type=float, required=False,
                        default=None if os.getenv("TAKE_PROFIT_PERCENTAGE") is None else float(os.getenv("TAKE_PROFIT_PERCENTAGE")),
                        help="Porcentaje de take profit (default: None)")
    parser.add_argument("--trading-fees", type=float, required=False,
                        default=float(os.getenv("TRADING_FEES", "0.0004")),
                        help="Comisiones de trading (default: desde TRADING_FEES o 0.0004)")
    parser.add_argument("--slippage", type=float, required=False,
                        default=float(os.getenv("SLIPPAGE", "0.0001")),
                        help="Deslizamiento (default: desde SLIPPAGE o 0.0001)")
    parser.add_argument("--random-start", type=bool, required=False,
                        default=True if os.getenv("RANDOM_START", "true").lower() in ['true', '1', 'yes'] else False,
                        help="Usar inicio aleatorio en entrenamiento (default: true)")
    
    # Parámetros del agente RL
    parser.add_argument("--algorithm", type=str, required=False,
                        default=os.getenv("AGENT_ALGORITHM", "SAC"),
                        help="Algoritmo RL (default: desde AGENT_ALGORITHM o 'SAC')")
    parser.add_argument("--learning-rate", type=float, required=False,
                        default=float(os.getenv("AGENT_LEARNING_RATE", "0.0003")),
                        help="Tasa de aprendizaje (default: desde AGENT_LEARNING_RATE o 0.0003)")
    parser.add_argument("--buffer-size", type=int, required=False,
                        default=int(os.getenv("AGENT_BUFFER_SIZE", "100000")),
                        help="Tamaño del buffer (default: desde AGENT_BUFFER_SIZE o 100000)")
    parser.add_argument("--batch-size", type=int, required=False,
                        default=int(os.getenv("AGENT_BATCH_SIZE", "256")),
                        help="Tamaño del batch (default: desde AGENT_BATCH_SIZE o 256)")
    parser.add_argument("--gamma", type=float, required=False,
                        default=float(os.getenv("AGENT_GAMMA", "0.99")),
                        help="Factor de descuento (default: desde AGENT_GAMMA o 0.99)")
    parser.add_argument("--learning-starts", type=int, required=False,
                        default=int(os.getenv("AGENT_LEARNING_STARTS", "10000")),
                        help="Pasos antes de empezar a entrenar (default: desde AGENT_LEARNING_STARTS o 10000)")
    parser.add_argument("--total-timesteps", type=int, required=False,
                        default=int(os.getenv("PIPELINE_TOTAL_TIMESTEPS", "500000")),
                        help="Pasos totales de entrenamiento (default: desde PIPELINE_TOTAL_TIMESTEPS o 500000)")
    parser.add_argument("--eval-freq", type=int, required=False,
                        default=int(os.getenv("AGENT_EVAL_FREQ", "10000")),
                        help="Frecuencia de evaluación (default: desde AGENT_EVAL_FREQ o 10000)")
    parser.add_argument("--save-freq", type=int, required=False,
                        default=int(os.getenv("AGENT_SAVE_FREQ", "50000")),
                        help="Frecuencia de guardado (default: desde AGENT_SAVE_FREQ o 50000)")
    parser.add_argument("--n-eval-episodes", type=int, required=False,
                        default=int(os.getenv("AGENT_N_EVAL_EPISODES", "5")),
                        help="Número de episodios para evaluación (default: desde AGENT_N_EVAL_EPISODES o 5)")
    parser.add_argument("--device", type=str, required=False,
                        default=os.getenv("AGENT_DEVICE", "auto"),
                        help="Dispositivo para entrenamiento (default: desde AGENT_DEVICE o 'auto')")
    parser.add_argument("--transformer-config", type=str, required=False,
                        default=os.getenv("AGENT_TRANSFORMER_CONFIG", None),
                        help="Configuración JSON del Transformer (default: config interna)")
    parser.add_argument("--save-replay-buffer", type=bool, required=False,
                        default=True if os.getenv("SAVE_REPLAY_BUFFER", "false").lower() in ['true', '1', 'yes'] else False,
                        help="Guardar también el buffer de experiencia (default: false)")
    
    args = parser.parse_args()
    
    # Verificar parámetros obligatorios
    if not args.project_id:
        raise ValueError("Se requiere --project-id o la variable de entorno GCP_PROJECT_ID")
    
    # Parsear configuración del Transformer si se proporciona
    policy_kwargs_dict = None
    if args.transformer_config:
        try:
            policy_kwargs_dict = json.loads(args.transformer_config)
        except json.JSONDecodeError:
            logger.warning("Error decodificando transformer-config JSON. Usando configuración por defecto.")
    
    try:
        start_time = time.time()
        logger.info(f"Iniciando entrenamiento del agente RL con datos de: {args.input_data_gcs}")
        
        # Inicializar el administrador del agente
        agent_manager = RLAgentManagerCloud(
            project_id=args.project_id,
            algorithm=args.algorithm,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            policy_kwargs_dict=policy_kwargs_dict,
            device=args.device,
            models_bucket=args.models_bucket
        )
        
        # Configurar el entorno y el agente
        agent_manager.setup_agent(
            sequence_length_L=args.sequence_length,
            initial_equity=args.initial_equity,
            leverage=args.leverage,
            position_size_percentage=args.position_size,
            stop_loss_percentage=args.stop_loss,
            take_profit_percentage=args.take_profit,
            trading_fees=args.trading_fees,
            slippage=args.slippage,
            data_gcs_path=args.input_data_gcs,
            random_start=args.random_start
        )
        
        # Entrenar el agente
        training_stats = agent_manager.train_agent(
            total_timesteps=args.total_timesteps,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            n_eval_episodes=args.n_eval_episodes,
            save_path_gcs=args.output_model_gcs
        )
        
        # Guardar el modelo final
        model_gcs_path = agent_manager.save_model(
            args.output_model_gcs,
            save_replay_buffer=args.save_replay_buffer
        )
        
        # Evaluación final
        eval_metrics = agent_manager.evaluate_agent(n_eval_episodes=10)
        
        # Exportar modelo para servir si se solicita
        serving_model_path = None
        if args.export_model_gcs:
            serving_model_path = agent_manager.export_model_for_serving(
                args.export_model_gcs
            )
        
        # Combinar estadísticas
        result = {
            "training_duration_seconds": int(time.time() - start_time),
            "training_stats": training_stats,
            "eval_metrics": eval_metrics,
            "model_gcs_path": model_gcs_path,
            "serving_model_path": serving_model_path
        }
        
        # Guardar métricas como JSON para el Pipeline
        metrics_json = json.dumps(eval_metrics)
        with open('/tmp/metrics.json', 'w') as f:
            f.write(metrics_json)
            
        # Para integrarse con Vertex AI Pipelines
        if 'PIPELINE_OUTPUT_FILE' in os.environ:
            with open(os.environ['PIPELINE_OUTPUT_FILE'], 'w') as f:
                json.dump({
                    "model_path": model_gcs_path,
                    "metrics": eval_metrics,
                    "serving_model_path": serving_model_path
                }, f)
        
        logger.info(f"Entrenamiento completado en {result['training_duration_seconds']} segundos")
        logger.info(f"Modelo guardado en: {model_gcs_path}")
        logger.info(f"Métricas de evaluación: {json.dumps(eval_metrics, indent=2)}")
        
        return 0
            
    except Exception as e:
        logger.exception(f"Error no controlado: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
