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
    parser.add_argument("--output-metrics-path", type=str, required=True,
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
            data_gcs_path=args.test_data_gcs,
            random_start=False  # Para evaluación, comenzar desde el principio
        )
        
        # Cargar el modelo entrenado
        success = agent_manager.load_model(args.model_gcs_path)
        if not success:
            logger.error(f"Error cargando modelo desde {args.model_gcs_path}")
            return 1
        
        # Evaluar el modelo
        eval_metrics = agent_manager.evaluate_agent(
            n_eval_episodes=args.num_episodes,
            deterministic=True
        )
        
        # Crear visualizaciones
        # (Simplificado - en una implementación real harías más gráficos)
        equity_curve_data = []
        
        # Recopilar datos para equity curve
        for i in range(5):  # Limitamos a 5 episodios para la curva
            obs, info = agent_manager.eval_env.reset()
            done = False
            truncated = False
            equities = [args.initial_equity]
            
            while not (done or truncated):
                action, _ = agent_manager.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = agent_manager.eval_env.step(action)
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
        
        # Guardar métricas en JSON
        with open('/tmp/evaluation_metrics.json', 'w') as f:
            json.dump(eval_metrics, f, indent=2)
        
        # Si output_metrics_path es GCS, subir los resultados
        if args.output_metrics_path.startswith("gs://"):
            # Subir métricas JSON
            storage_client = storage.Client(project=args.project_id)
            
            # Extraer bucket y blob
            gcs_path = args.output_metrics_path.replace("gs://", "")
            bucket_name = gcs_path.split("/")[0]
            metrics_blob_name = "/".join(gcs_path.split("/")[1:])
            plots_prefix = os.path.dirname(metrics_blob_name)
            
            bucket = storage_client.bucket(bucket_name)
            
            # Subir archivo de métricas
            metrics_blob = bucket.blob(metrics_blob_name)
            metrics_blob.upload_from_filename('/tmp/evaluation_metrics.json')
            
            # Subir gráficos
            for plot_file in os.listdir(args.output_plots_dir):
                plot_path = os.path.join(args.output_plots_dir, plot_file)
                plot_blob_name = f"{plots_prefix}/plots/{plot_file}"
                plot_blob = bucket.blob(plot_blob_name)
                plot_blob.upload_from_filename(plot_path)
            
            logger.info(f"Resultados de evaluación subidos a gs://{bucket_name}/{plots_prefix}")
        else:
            # Escribir en la ruta local especificada
            os.makedirs(os.path.dirname(args.output_metrics_path), exist_ok=True)
            with open(args.output_metrics_path, 'w') as f:
                json.dump(eval_metrics, f, indent=2)
            logger.info(f"Métricas de evaluación guardadas en: {args.output_metrics_path}")
        
        eval_duration = time.time() - start_time
        logger.info(f"Evaluación completada en {eval_duration:.2f} segundos")
        logger.info(f"Resumen: Equity final ${eval_metrics['avg_final_equity']:.2f} "
                   f"({eval_metrics['equity_change_pct']:.2f}%), "
                   f"Sharpe={eval_metrics['avg_sharpe_ratio']:.2f}")
        
        return 0
            
    except Exception as e:
        logger.exception(f"Error no controlado: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
