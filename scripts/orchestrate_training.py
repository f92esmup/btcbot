#!/usr/bin/env python3
"""
Orquestador centralizado para el pipeline de MLOps de BTCBot

Este script es el punto único de entrada para ejecutar el pipeline completo de MLOps:
1. Adquisición de datos históricos de Binance
2. Preprocesamiento y generación de secuencias de características
3. Entrenamiento del modelo de RL con la estrategia "Best Model Only"
4. Evaluación del modelo entrenado

Ya no se requieren scripts individuales separados para cada fase, ya que toda
la funcionalidad está integrada en este orquestador centralizado.

Uso:
  python orchestrate_training.py                    # Ejecutar pipeline completo
  python orchestrate_training.py --timesteps 10000  # Especificar pasos de entrenamiento
  python orchestrate_training.py --phase data --start-date 2023-01-01  # Solo fase de datos
  python orchestrate_training.py --phase train      # Solo entrenamiento
  python orchestrate_training.py --phase evaluate --episodes 5  # Solo evaluación
"""

import argparse
import logging
import sys
import os
import asyncio
import numpy as np
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Añadir el directorio raíz del proyecto al path de Python
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Importaciones de los módulos del proyecto
from src.data.data_pipeline import IntegratedDataPipeline
from src.agent.rl_agent_manager import RLAgentManager
from src.utils.config import ConfigManager
from src.utils.logging_utils import setup_logger, get_madrid_timestamp_str, get_madrid_timestamp
from src.utils.bigquery_utils import stream_data_to_bigquery
from src.callbacks.bigquery_evaluation_schema import EVALUATION_LOG_SCHEMA
from src.callbacks import BigQueryLoggingCallback
from google.cloud import bigquery

class TrainingOrchestrator:
    """
    Orquestador centralizado para el pipeline completo de MLOps.
    
    Esta clase implementa directamente todas las funcionalidades de:
    - Adquisición y preprocesamiento de datos (anteriormente run_data_pipeline.py)
    - Entrenamiento del modelo (anteriormente train_rl_agent.py) 
    - Evaluación del modelo (anteriormente evaluate_rl_agent.py)
    """
    
    def __init__(self, config_path="src/config.yaml"):
        """
        Inicializa el orquestador con la configuración centralizada.
        
        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        self.config_path = config_path
        self.config_manager = ConfigManager(config_path=config_path)
        
        # Verificar que GCS_BUCKET_NAME esté configurado (requerido para todas las fases)
        self.gcs_bucket_name = os.environ.get("GCS_BUCKET_NAME")
        if not self.gcs_bucket_name:
            logger.warning("⚠️ GCS_BUCKET_NAME no está configurado. El pipeline puede no funcionar correctamente.")
        
        # Verificar configuración de BigQuery para logs
        self.gcp_project_id = os.environ.get('GCP_PROJECT_ID')
        self.bigquery_dataset_id = os.environ.get('BIGQUERY_LOG_DATASET_ID')
        
        if not self.gcp_project_id or not self.bigquery_dataset_id:
            logger.warning("⚠️ Configuración de BigQuery incompleta. Los logs se guardarán solo localmente.")
        
        # Crear cliente BigQuery si está configurado
        self.bq_client = None
        if self.gcp_project_id and self.bigquery_dataset_id:
            try:
                self.bq_client = bigquery.Client(project=self.gcp_project_id)
                logger.info(f"Cliente BigQuery inicializado para proyecto {self.gcp_project_id}")
            except Exception as e:
                logger.error(f"Error al inicializar el cliente BigQuery: {e}")
    
    async def phase_1_data_pipeline(self, start_date=None, end_date=None, force_reprocess=False):
        """
        Fase 1: Adquisición y preprocesamiento de datos.
        Implementación directa de la funcionalidad de run_data_pipeline.py.
        
        Args:
            start_date: Fecha inicial para los datos (YYYY-MM-DD)
            end_date: Fecha final para los datos (YYYY-MM-DD)
            force_reprocess: Si True, reprocesa todos los chunks incluso si ya existen
            
        Returns:
            bool: True si la operación fue exitosa, False en caso contrario
        """
        logger.info("=" * 50)
        logger.info("FASE 1: ADQUISICIÓN Y PREPROCESAMIENTO DE DATOS")
        logger.info("=" * 50)
        
        try:
            # Inicializar el pipeline de datos integrado
            pipeline = IntegratedDataPipeline(config_path=self.config_path)
            
            # Usar fecha por defecto desde la configuración si no se proporciona
            if not start_date:
                start_date = pipeline.config['data_acquisition_defaults'].get(
                    'historical_start_date', "2025-01-01"
                )
                logger.info(f"Usando fecha de inicio por defecto: {start_date}")
            
            # Usar la fecha actual si no se proporciona una fecha de fin
            if not end_date:
                end_date = datetime.now().strftime("%Y-%m-%d")
                logger.info(f"Usando fecha actual como fecha de fin: {end_date}")
                
            logger.info(f"Ejecutando pipeline de datos desde {start_date} hasta {end_date}")
            logger.info(f"Force reprocess: {force_reprocess}")
            
            # Ejecutar el pipeline de datos integrado
            results = await pipeline.run_pipeline(
                start_date=start_date, 
                end_date=end_date,
                force_reprocess=force_reprocess
            )
            
            # Mostrar resultados
            logger.info(f"Pipeline completado con estado: {results['status']}")
            logger.info(f"Total chunks: {results['total_chunks']}")
            logger.info(f"Chunks existentes: {results['existing_chunks']}")
            logger.info(f"Chunks procesados: {results['newly_processed_chunks']}")
            logger.info(f"Total secuencias: {results['total_sequences']}")
            
            return results['status'] == 'success'
            
        except Exception as e:
            logger.error(f"Error en la fase de datos: {e}", exc_info=True)
            return False
    
    def phase_2_train_model(self, timesteps=None):
        """
        Fase 2: Entrenamiento del modelo de RL.
        Implementación directa de la funcionalidad de train_rl_agent.py.
        
        Args:
            timesteps: Número de pasos de entrenamiento (None para usar config)
            
        Returns:
            bool: True si la operación fue exitosa, False en caso contrario
        """
        logger.info("=" * 50)
        logger.info("FASE 2: ENTRENAMIENTO DEL MODELO")
        logger.info("=" * 50)
        
        try:
            # Cargar la configuración del agente
            agent_config = self.config_manager.get_agent_config()
            
            # Obtener el número de timesteps
            if timesteps is None:
                timesteps = agent_config.get("total_training_timesteps", 1000000)
                logger.info(f"Usando timesteps desde configuración: {timesteps}")
            else:
                logger.info(f"Usando timesteps proporcionados: {timesteps}")
                
            # Inicializar RLAgentManager
            agent_manager = RLAgentManager(config_path=self.config_path)
            
            # Configurar el agente (sin cargar un modelo previo para entrenamiento desde cero)
            agent_manager.setup_agent()
            logger.info("Agente configurado correctamente")
            
            # Configurar callback de BigQuery para logging, si está disponible
            callbacks = []
            if self.gcp_project_id and self.bigquery_dataset_id:
                try:
                    bq_callback = BigQueryLoggingCallback(
                        project_id=self.gcp_project_id,
                        dataset_id=self.bigquery_dataset_id,
                        config_manager=self.config_manager,
                        bq_client=self.bq_client
                    )
                    callbacks.append(bq_callback)
                    logger.info("Callback de BigQuery inicializado para logging de entrenamiento")
                except Exception as e:
                    logger.error(f"Error al inicializar BigQuery callback: {e}")
            
            # Entrenar el modelo
            logger.info(f"Iniciando entrenamiento por {timesteps} pasos")
            agent_manager.train_agent(
                total_timesteps=timesteps,
                user_callbacks=callbacks
            )
            
            logger.info("Entrenamiento completado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error en la fase de entrenamiento: {e}", exc_info=True)
            return False
    
    def phase_3_evaluate_model(self, episodes=1, output_dir="results"):
        """
        Fase 3: Evaluación del modelo entrenado.
        Implementación directa de la funcionalidad de evaluate_rl_agent.py.
        
        Args:
            episodes: Número de episodios para evaluar
            output_dir: Directorio para guardar resultados
            
        Returns:
            bool: True si la operación fue exitosa, False en caso contrario
        """
        logger.info("=" * 50)
        logger.info("FASE 3: EVALUACIÓN DEL MODELO")
        logger.info("=" * 50)
        
        try:
            # Crear directorio para resultados si no existe
            os.makedirs(output_dir, exist_ok=True)
            
            # Generar un ID de evaluación único
            evaluation_id = str(uuid.uuid4())
            
            # Configurar logging para BigQuery
            save_to_bigquery = self.gcp_project_id and self.bigquery_dataset_id and self.bq_client
            
            # Inicializar RLAgentManager
            agent_manager = RLAgentManager(config_path=self.config_path)
            
            # Determinar la ruta al mejor modelo usando la función centralizada
            model_path = self.config_manager.get_best_model_default_gcs_path()
            logger.info(f"Evaluando modelo en: {model_path}")
            
            # Configurar el agente con el modelo a evaluar
            agent_manager.setup_agent(
                load_model=True,
                model_path=model_path
            )
            logger.info("Modelo cargado exitosamente para evaluación")
            
            # Crear entorno para evaluación
            env = agent_manager.setup_environment(is_eval=True)
            
            # Ejecutar evaluación para cada episodio
            logger.info(f"Evaluando {episodes} episodio(s)")
            all_episode_stats = []
            
            for episode_idx in range(episodes):
                logger.info(f"Iniciando episodio de evaluación {episode_idx + 1}/{episodes}")
                
                # Reiniciar el entorno
                obs, _ = env.reset()
                done = False
                truncated = False
                step = 0
                
                # Datos a registrar
                episode_data = {
                    "observations": [],
                    "actions": [],
                    "rewards": [],
                    "dones": [],
                    "infos": [],
                    "equity_curve": [],
                    "positions": []
                }
                
                # Ejecutar episodio
                while not done and not truncated:
                    # Predecir acción
                    action, _ = agent_manager.model.predict(obs, deterministic=True)
                    
                    # Ejecutar acción en el entorno
                    next_obs, reward, done, truncated, info = env.step(action)
                    
                    # Guardar datos
                    episode_data["observations"].append(obs)
                    episode_data["actions"].append(action)
                    episode_data["rewards"].append(reward)
                    episode_data["dones"].append(done)
                    episode_data["infos"].append(info)
                    
                    # Actualizar posición actual
                    if "portfolio_state" in info and "equity" in info["portfolio_state"]:
                        episode_data["equity_curve"].append(info["portfolio_state"]["equity"])
                        if "position_direction" in info["portfolio_state"]:
                            episode_data["positions"].append(info["portfolio_state"]["position_direction"])
                        else:
                            episode_data["positions"].append(0)  # Posición neutra
                    
                    # Actualizar para siguiente paso
                    obs = next_obs
                    step += 1
                
                # Calcular estadísticas del episodio
                logger.info(f"Episodio {episode_idx + 1} completado en {step} pasos")
                
                # Calcular estadísticas básicas
                rewards = np.array(episode_data["rewards"])
                initial_equity = episode_data["equity_curve"][0] if episode_data["equity_curve"] else 10000.0
                final_equity = episode_data["equity_curve"][-1] if episode_data["equity_curve"] else 10000.0
                pct_return = ((final_equity / initial_equity) - 1.0) * 100.0
                
                logger.info(f"Rendimiento del episodio: {pct_return:.2f}%")
                
                # Guardar estadísticas básicas
                episode_stats = {
                    "episode": episode_idx + 1,
                    "steps": step,
                    "total_reward": pct_return,
                    "avg_reward": rewards.mean(),
                    "cumulative_reward": rewards.sum(),
                    "initial_value": initial_equity,
                    "final_value": final_equity
                }
                all_episode_stats.append(episode_stats)
                
                # Enviar a BigQuery si está configurado
                if save_to_bigquery:
                    try:
                        # Crear un registro para BigQuery con los resultados del episodio
                        bq_record = {
                            'evaluation_id': evaluation_id,
                            'model_path': model_path,
                            'timestamp_evaluation': datetime.now().isoformat(),
                            'config_path': self.config_path,
                            'num_episodes': episodes,
                            'episode_number': episode_idx + 1,
                            'total_return_percent': pct_return,
                            'initial_value': initial_equity,
                            'final_value': final_equity,
                            'avg_reward_per_step': episode_stats['avg_reward'],
                            'cumulative_reward': episode_stats['cumulative_reward'],
                            'episode_length_steps': step
                        }
                        
                        # Crear tabla con fecha actual
                        current_date = datetime.now().strftime('%Y%m%d')
                        table_id = f"evaluacion_{current_date}"
                        
                        # Enviar a BigQuery
                        success = stream_data_to_bigquery(
                            project_id=self.gcp_project_id,
                            dataset_id=self.bigquery_dataset_id,
                            table_id=table_id,
                            rows_to_insert=[bq_record],
                            client=self.bq_client,
                            schema=EVALUATION_LOG_SCHEMA
                        )
                        
                        if success:
                            logger.info(f"Datos del episodio {episode_idx+1} guardados en BigQuery")
                        else:
                            logger.error(f"Error guardando datos del episodio {episode_idx+1} en BigQuery")
                    except Exception as e:
                        logger.error(f"Error enviando datos a BigQuery: {e}")
            
            # Crear resumen de todos los episodios
            summary_file = os.path.join(output_dir, "resumen_evaluacion.csv")
            with open(summary_file, 'w') as f:
                # Encabezados
                f.write("episode,steps,total_reward,avg_reward,cumulative_reward,initial_value,final_value\n")
                
                # Datos por episodio
                for stats in all_episode_stats:
                    f.write(f"{stats['episode']},{stats['steps']},{stats['total_reward']:.2f},{stats['avg_reward']:.4f},{stats['cumulative_reward']:.2f},{stats['initial_value']:.2f},{stats['final_value']:.2f}\n")
            
            logger.info(f"Resumen de evaluación guardado en {summary_file}")
            
            # Mostrar estadísticas promedio
            avg_return = np.mean([s['total_reward'] for s in all_episode_stats])
            avg_reward = np.mean([s['avg_reward'] for s in all_episode_stats])
            
            logger.info(f"=== Resumen de Evaluación ({episodes} episodios) ===")
            logger.info(f"Retorno Promedio: {avg_return:.2f}%")
            logger.info(f"Recompensa Promedio por Paso: {avg_reward:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en la fase de evaluación: {e}", exc_info=True)
            return False
    
    async def run_full_pipeline(self, timesteps=None, episodes=1, start_date=None, end_date=None):
        """
        Ejecuta el pipeline completo de entrenamiento.
        
        Args:
            timesteps: Número de timesteps para entrenamiento (None para usar config)
            episodes: Número de episodios para evaluación
            start_date: Fecha de inicio para datos (None para usar la configuración)
            end_date: Fecha de fin para datos (None para usar la fecha actual)
            
        Returns:
            bool: True si todas las fases fueron exitosas, False en caso contrario
        """
        start_time = datetime.now()
        logger.info("🚀 INICIANDO PIPELINE COMPLETA DE ENTRENAMIENTO")
        logger.info(f"Hora de inicio: {start_time}")
        
        # Inicializar seguimiento de fases
        completed_phases = 0
        failed_phases = []
        
        # Fase 1: Datos
        logger.info("\n🔄 Iniciando Fase 1: Adquisición y Preprocesamiento de Datos")
        try:
            if await self.phase_1_data_pipeline(start_date, end_date):
                completed_phases += 1
                logger.info("✅ Fase 1 (Datos) completada exitosamente")
            else:
                failed_phases.append("Datos")
                logger.error("❌ Fase 1 (Datos) falló")
        except Exception as e:
            failed_phases.append("Datos")
            logger.error(f"❌ Excepción en Fase 1 (Datos): {str(e)}")
        
        # Fase 2: Entrenamiento
        logger.info("\n🔄 Iniciando Fase 2: Entrenamiento del Modelo")
        try:
            if self.phase_2_train_model(timesteps):
                completed_phases += 1
                logger.info("✅ Fase 2 (Entrenamiento) completada exitosamente")
            else:
                failed_phases.append("Entrenamiento")
                logger.error("❌ Fase 2 (Entrenamiento) falló")
        except Exception as e:
            failed_phases.append("Entrenamiento")
            logger.error(f"❌ Excepción en Fase 2 (Entrenamiento): {str(e)}")
        
        # Fase 3: Evaluación
        logger.info("\n🔄 Iniciando Fase 3: Evaluación del Modelo")
        try:
            if self.phase_3_evaluate_model(episodes):
                completed_phases += 1
                logger.info("✅ Fase 3 (Evaluación) completada exitosamente")
            else:
                failed_phases.append("Evaluación")
                logger.error("❌ Fase 3 (Evaluación) falló")
        except Exception as e:
            failed_phases.append("Evaluación")
            logger.error(f"❌ Excepción en Fase 3 (Evaluación): {str(e)}")
        
        # Resumen final
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("📊 RESUMEN DE EJECUCIÓN")
        logger.info("=" * 60)
        logger.info(f"Hora de inicio: {start_time}")
        logger.info(f"Hora de fin: {end_time}")
        logger.info(f"Duración total: {duration}")
        logger.info(f"Fases completadas: {completed_phases}/3")
        
        if failed_phases:
            logger.warning(f"Fases fallidas: {', '.join(failed_phases)}")
            return False
        else:
            logger.info("🎉 ¡PIPELINE COMPLETADA EXITOSAMENTE!")
            return True

async def main():
    parser = argparse.ArgumentParser(
        description="Orquestador centralizado de MLOps para BTCBot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python orchestrate_training.py                    # Ejecutar pipeline completo
  python orchestrate_training.py --timesteps 10000  # Especificar pasos de entrenamiento
  python orchestrate_training.py --phase data --start-date 2023-01-01  # Solo fase de datos
  python orchestrate_training.py --phase train      # Solo entrenamiento
  python orchestrate_training.py --phase evaluate --episodes 5  # Solo evaluación
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default="src/config.yaml",
        help='Ruta al archivo de configuración centralizada'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Número de timesteps para entrenamiento (si se omite, usa valor de config.yaml)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='Número de episodios para evaluación (por defecto: 1)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Fecha de inicio para datos históricos (formato: YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='Fecha de fin para datos históricos (formato: YYYY-MM-DD, por defecto: fecha actual)'
    )
    
    parser.add_argument(
        '--force-reprocess',
        action='store_true',
        help='Forzar el reprocesamiento de todos los chunks de datos'
    )
    
    parser.add_argument(
        '--phase',
        choices=['data', 'train', 'evaluate'],
        help='Ejecutar solo una fase específica (omitir para pipeline completa)'
    )
    
    args = parser.parse_args()
    
    # Validar argumentos
    if args.timesteps is not None and args.timesteps <= 0:
        logger.error("El número de timesteps debe ser positivo")
        sys.exit(1)
    
    if args.episodes <= 0:
        logger.error("El número de episodios debe ser positivo")
        sys.exit(1)
    
    # Crear orquestador
    orchestrator = TrainingOrchestrator(config_path=args.config)
    
    try:
        if args.phase:
            # Ejecutar solo una fase específica
            logger.info(f"Ejecutando solo la fase: {args.phase}")
            
            if args.phase == 'data':
                success = await orchestrator.phase_1_data_pipeline(
                    start_date=args.start_date, 
                    end_date=args.end_date,
                    force_reprocess=args.force_reprocess
                )
            elif args.phase == 'train':
                success = orchestrator.phase_2_train_model(args.timesteps)
            elif args.phase == 'evaluate':
                success = orchestrator.phase_3_evaluate_model(args.episodes)
            
            if success:
                logger.info(f"✅ Fase '{args.phase}' completada exitosamente")
                sys.exit(0)
            else:
                logger.error(f"❌ Fase '{args.phase}' falló")
                sys.exit(1)
        else:
            # Ejecutar pipeline completa
            success = await orchestrator.run_full_pipeline(
                timesteps=args.timesteps,
                episodes=args.episodes,
                start_date=args.start_date,
                end_date=args.end_date
            )
            
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Ejecución interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Error inesperado: {str(e)}")
        logger.exception(e)  # Añadir traceback para mejor diagnóstico
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())