#!/usr/bin/env python
"""
Script para evaluar un agente RL de trading entrenado.
Optimizado para ejecutarse como un componente de Vertex AI Pipelines.
"""
import argparse
import logging
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import storage

from src.agent.rl_agent_manager_cloud import RLAgentManagerCloud

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Función principal para evaluar un agente RL."""
    parser = argparse.ArgumentParser(description="Evalúa un agente RL para trading")
    
    # Parámetros básicos
    parser.add_argument("--model-gcs-path", type=str, required=True,
                        help="Ruta GCS completa al modelo entrenado")
    parser.add_argument("--test-data-gcs", type=str, required=True,
                        help="Ruta GCS a los datos de prueba")
    parser.add_argument("--output-metrics-path", type=str, required=False,
                        default="/tmp/metrics.json",
                        help="Ruta para guardar métricas de evaluación (local o GCS)")
    parser.add_argument("--output-plots-dir", type=str, required=False,
                        default="/tmp/eval_plots",
                        help="Directorio para guardar gráficos de evaluación")
    
    # Parámetros de GCP
    parser.add_argument("--project-id", type=str, required=False,
                        default=os.getenv("GCP_PROJECT_ID"),
                        help="ID del proyecto GCP (default: desde variable GCP_PROJECT_ID)")
    parser.add_argument("--evaluation-bucket", type=str, required=False,
                        default=os.getenv("EVALUATION_RESULTS_BUCKET"),
                        help="Bucket para resultados (default: desde variable EVALUATION_RESULTS_BUCKET)")
    
    # Parámetros de evaluación
    parser.add_argument("--num-episodes", type=int, required=False,
                        default=int(os.getenv("EVAL_NUM_EPISODES", "10")),
                        help="Número de episodios para evaluación")
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
                        help="Tamaño de posición como % de equity (default: 0.2)")
    parser.add_argument("--trading-fees", type=float, required=False,
                        default=float(os.getenv("TRADING_FEES", "0.0004")),
                        help="Comisiones por trade (default: 0.0004)")
    parser.add_argument("--success-threshold-sharpe", type=float, required=False,
                        default=float(os.getenv("SUCCESS_THRESHOLD_SHARPE", "0.5")),
                        help="Umbral de Sharpe ratio para despliegue (default: 0.5)")
    parser.add_argument("--success-threshold-drawdown", type=float, required=False,
                        default=float(os.getenv("SUCCESS_THRESHOLD_DRAWDOWN", "0.2")),
                        help="Umbral máximo de drawdown para despliegue (default: 0.2)")
    parser.add_argument("--success-threshold-winrate", type=float, required=False,
                        default=float(os.getenv("SUCCESS_THRESHOLD_WINRATE", "0.5")),
                        help="Umbral de win rate para despliegue (default: 0.5)")
    
    args = parser.parse_args()
    
    # Verificar parámetros obligatorios
    if not args.project_id:
        raise ValueError("Se requiere --project-id o la variable de entorno GCP_PROJECT_ID")
    
    # Crear directorio para gráficos
    os.makedirs(args.output_plots_dir, exist_ok=True)
    
    try:
        start_time = time.time()
        logger.info(f"Iniciando evaluación del modelo: {args.model_gcs_path}")
        logger.info(f"Datos de prueba: {args.test_data_gcs}")
        
        # Inicializar el administrador del agente
        agent_manager = RLAgentManagerCloud(
            project_id=args.project_id,
            device="cpu"  # Para evaluación es suficiente con CPU
        )
        
        # Configurar el entorno y el agente con datos de prueba
        agent_manager.setup_agent(
            sequence_length_L=args.sequence_length,
            initial_equity=args.initial_equity,
            leverage=args.leverage,
            position_size_percentage=args.position_size,
            trading_fees=args.trading_fees,
            data_gcs_path=args.test_data_gcs,
            random_start=False  # Para evaluación, comenzar desde el principio
        )
        
        # Cargar el modelo entrenado
        agent_manager.load_model(args.model_gcs_path)
        logger.info("Modelo cargado exitosamente")
        
        # Evaluar el modelo
        eval_metrics = agent_manager.evaluate_agent(
            n_eval_episodes=args.num_episodes,
            deterministic=True
        )
        
        # Crear visualizaciones
        # (Simplificado - en una implementación real harías más gráficos)
        equity_curve_data = []
        
        # Recopilar datos para equity curve
        for i in range(min(5, args.num_episodes)):  # Limitamos a 5 episodios para la curva
            obs, info = agent_manager.eval_env.reset()
            done = False
            truncated = False
            equities = [args.initial_equity]
            
            while not (done or truncated):
                action, _ = agent_manager.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = agent_manager.eval_env.step(action)
                done = terminated or truncated
                equities.append(info['equity'])
            
            equity_curve_data.append(equities)
        
        # Crear gráfico de equity curve
        plt.figure(figsize=(12, 6))
        for i, equity_curve in enumerate(equity_curve_data):
            plt.plot(equity_curve, label=f"Episodio {i+1}")
        plt.axhline(y=args.initial_equity, color='r', linestyle='--', label="Equity Inicial")
        plt.title("Curvas de Equity durante Evaluación")
        plt.xlabel("Pasos")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.grid(True)
        
        # Guardar gráfico
        equity_plot_path = os.path.join(args.output_plots_dir, "equity_curves.png")
        plt.savefig(equity_plot_path)
        
        # Añadir informe de evaluación más detallado
        plt.figure(figsize=(10, 8))
        metrics_text = [
            f"Evaluación del Modelo: {os.path.basename(args.model_gcs_path)}",
            f"Episodios: {args.num_episodes}",
            f"Equity Media Final: ${eval_metrics['avg_final_equity']:.2f}",
            f"Cambio de Equity: {eval_metrics['equity_change_pct']:.2f}%",
            f"Ratio de Sharpe: {eval_metrics['avg_sharpe_ratio']:.2f}",
            f"Drawdown Máximo: {eval_metrics['avg_max_drawdown']*100:.2f}%",
            f"Win Rate: {eval_metrics['avg_win_rate']*100:.2f}%"
        ]
        plt.text(0.5, 0.5, "\n".join(metrics_text), ha='center', va='center', fontsize=12)
        plt.axis('off')
        summary_plot_path = os.path.join(args.output_plots_dir, "evaluation_summary.png")
        plt.savefig(summary_plot_path)
        
        # Determinar si el modelo cumple con los criterios de calidad para despliegue
        success_threshold_met = (
            eval_metrics['avg_sharpe_ratio'] >= args.success_threshold_sharpe and
            eval_metrics['avg_max_drawdown'] <= args.success_threshold_drawdown and
            eval_metrics['avg_win_rate'] >= args.success_threshold_winrate
        )
        
        # Añadir recomendación de despliegue
        eval_metrics['success_threshold_met'] = success_threshold_met
        eval_metrics['deploy_recommendation'] = success_threshold_met
        eval_metrics['thresholds'] = {
            'sharpe': args.success_threshold_sharpe,
            'drawdown': args.success_threshold_drawdown,
            'win_rate': args.success_threshold_winrate
        }
        
        # Guardar métricas en JSON
        with open(args.output_metrics_path, 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        
        # Si evaluation_bucket está definido, subir los resultados a GCS
        if args.evaluation_bucket:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_path = f"evaluation_results/{os.path.basename(args.model_gcs_path)}_{timestamp}"
            
            # Subir métricas JSON
            storage_client = storage.Client(project=args.project_id)
            bucket = storage_client.bucket(args.evaluation_bucket)
            
            # Subir archivo de métricas
            metrics_blob_name = f"{base_path}/metrics.json"
            metrics_blob = bucket.blob(metrics_blob_name)
            metrics_blob.upload_from_filename(args.output_metrics_path)
            
            # Subir gráficos
            for plot_file in os.listdir(args.output_plots_dir):
                plot_path = os.path.join(args.output_plots_dir, plot_file)
                plot_blob_name = f"{base_path}/plots/{plot_file}"
                plot_blob = bucket.blob(plot_blob_name)
                plot_blob.upload_from_filename(plot_path)
            
            logger.info(f"Resultados de evaluación subidos a gs://{args.evaluation_bucket}/{base_path}")
            
            # Actualizar ruta en métricas para KFP
            eval_metrics['plots_gcs_path'] = f"gs://{args.evaluation_bucket}/{base_path}/plots"
            
            # Escribir archivo actualizado para KFP
            with open('/tmp/kfp_metrics.json', 'w') as f:
                json.dump(eval_metrics, f)
                
            # Para integrarse con Vertex AI Pipelines
            if 'PIPELINE_OUTPUT_FILE' in os.environ:
                with open(os.environ['PIPELINE_OUTPUT_FILE'], 'w') as f:
                    json.dump({
                        "model_path": args.model_gcs_path,
                        "metrics": eval_metrics,
                        "deploy_recommendation": success_threshold_met,
                        "plots_gcs_path": f"gs://{args.evaluation_bucket}/{base_path}/plots"
                    }, f)
        
        eval_duration = time.time() - start_time
        logger.info(f"Evaluación completada en {eval_duration:.2f} segundos")
        logger.info(f"Resumen: Equity final ${eval_metrics['avg_final_equity']:.2f} "
                   f"({eval_metrics['equity_change_pct']:.2f}%), "
                   f"Sharpe={eval_metrics['avg_sharpe_ratio']:.2f}")
        
        if success_threshold_met:
            logger.info("✅ El modelo cumple con los criterios de calidad para despliegue")
        else:
            logger.warning("⚠️ El modelo NO cumple con los criterios de calidad para despliegue")
        
        return 0
            
    except Exception as e:
        logger.exception(f"Error no controlado: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
