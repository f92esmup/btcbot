
import logging
import datetime
import uuid
import csv
import os
import tempfile
from typing import List, Dict, Any
from google.cloud import bigquery
from stable_baselines3.common.callbacks import BaseCallback

from src.utils.bigquery_utils import stream_data_to_bigquery
from src.utils.config import ConfigManager  # For accessing config values
from src.utils.logging_utils import get_madrid_timestamp_str, get_madrid_timestamp

logger = logging.getLogger(__name__)

# Define the BigQuery schema for entrenamiento_{FECHA} tables
# This should match Step 3 of the plan.
TRAINING_LOG_SCHEMA = [
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),  # model.learn() call ID
    bigquery.SchemaField("episode_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("step_in_episode", "INTEGER", mode="NULLABLE"),  # Null for episode_summary, training_metric
    bigquery.SchemaField("total_steps_elapsed", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("timestamp_event", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("event_type", "STRING", mode="REQUIRED"),  # 'step_info', 'episode_summary', 'training_metric'
    # Fields for 'step_info'
    bigquery.SchemaField("reward_step", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("action_value_raw", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("current_equity_step", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("current_position_side_step", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("current_position_avg_price_step", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("current_position_size_step", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("market_price_at_step", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("obs_market_feat_0_step", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("obs_market_feat_1_step", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("obs_market_feat_2_step", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("obs_portfolio_pnl_norm_step", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("obs_portfolio_steps_in_pos_norm_step", "FLOAT", mode="NULLABLE"),
    # Fields for 'episode_summary'
    bigquery.SchemaField("total_reward_episode", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("pnl_realized_episode", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("num_trades_episode", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("total_fees_episode", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("episode_duration_steps", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("final_equity_episode", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("termination_reason", "STRING", mode="NULLABLE"),
    # Fields for 'training_metric'
    bigquery.SchemaField("actor_loss", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("critic_loss", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("entropy_coefficient", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("learning_rate", "FLOAT", mode="NULLABLE"),
]


def convert_madrid_timestamp_to_utc_datetime(timestamp_str):
    """
    Convierte un timestamp string de Madrid a datetime UTC naive para BigQuery.
    
    Args:
        timestamp_str: String de timestamp en formato ISO con timezone de Madrid
        
    Returns:
        datetime: Datetime naive en UTC para BigQuery
    """
    if not timestamp_str:
        return None
        
    try:
        from datetime import datetime
        import pytz
        from src.utils.logging_utils import MADRID_TZ
        
        # Parse el timestamp que viene en formato ISO
        if timestamp_str.endswith('+01:00') or timestamp_str.endswith('+02:00'):
            # Ya tiene timezone info
            madrid_dt = datetime.fromisoformat(timestamp_str)
        else:
            # Parse como ISO y asumir que es Madrid time
            madrid_dt = datetime.fromisoformat(timestamp_str.replace('Z', ''))
            madrid_dt = MADRID_TZ.localize(madrid_dt)
        
        # Convertir a UTC
        utc_dt = madrid_dt.astimezone(pytz.utc)
        
        # Retornar como naive datetime (sin timezone info) para BigQuery
        return utc_dt.replace(tzinfo=None)
        
    except Exception as e:
        logger.error(f"Error convirtiendo timestamp {timestamp_str} a UTC: {e}")
        # Fallback: usar timestamp actual UTC
        from datetime import datetime
        import pytz
        return datetime.now(pytz.utc).replace(tzinfo=None)


class BigQueryLoggingCallback(BaseCallback):
    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        config_manager: ConfigManager,
        bq_client: bigquery.Client = None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.bq_client = bq_client if bq_client else bigquery.Client(project=project_id)
        self.run_id = str(uuid.uuid4())  # Unique ID for this training run
        self.session_id = str(uuid.uuid4())  # Unique ID for this model.learn() call
        
        # 📁 Almacenamiento local en CSV durante entrenamiento
        self.local_csv_file = None
        self.csv_writer = None
        self.csv_fieldnames = [field.name for field in TRAINING_LOG_SCHEMA]
        
        agent_config = config_manager.get_agent_config()
        bq_log_config = agent_config.get('bigquery_logging', {})
        self.metrics_log_interval = bq_log_config.get('training_metrics_log_interval_steps', 1000)
        self.last_metrics_log_step = 0

    def _on_training_start(self):
        # Reset session_id for a new training session (if model.learn is called multiple times)
        self.session_id = str(uuid.uuid4())
        self.last_metrics_log_step = 0
        
        # 📁 Crear archivo CSV local temporal para almacenar datos durante entrenamiento
        timestamp = get_madrid_timestamp().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"training_logs_{self.run_id}_{timestamp}.csv"
        self.local_csv_file = os.path.join(tempfile.gettempdir(), csv_filename)
        
        # Abrir archivo CSV y escribir headers
        self.csv_file_handle = open(self.local_csv_file, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.csv_file_handle, fieldnames=self.csv_fieldnames)
        self.csv_writer.writeheader()
        
        logger.info(f"🚀 BigQuery Logging Started. Run ID: {self.run_id}, Session ID: {self.session_id}")
        logger.info(f"📁 Almacenando datos localmente en: {self.local_csv_file}")

    def _on_step(self) -> bool:
        # Retrieve data from the environment
        if self.training_env is None:
            logger.warning("BigQueryLoggingCallback: training_env is None, cannot retrieve logs.")
            return True

        try:
            # For VecEnv, use get_attr; for single env, directly call method
            if hasattr(self.training_env, 'envs'):  # Likely a VecEnv
                step_and_summary_data_list = self.training_env.env_method("get_current_episode_step_data")
                if step_and_summary_data_list and isinstance(step_and_summary_data_list, list):
                    step_and_summary_data = step_and_summary_data_list[0]
                else:
                    step_and_summary_data = []
            else:  # Single environment
                step_and_summary_data = self.training_env.get_current_episode_step_data()

        except AttributeError as e:
            logger.error(f"Error accessing get_current_episode_step_data from environment: {e}")
            step_and_summary_data = []

        # 📝 Escribir datos al CSV local inmediatamente (muy rápido)
        if step_and_summary_data and self.csv_writer:
            for record in step_and_summary_data:
                # Add common fields
                record['run_id'] = self.run_id
                record['session_id'] = self.session_id
                record['total_steps_elapsed'] = self.num_timesteps  # SB3's total steps
                
                # Convertir timestamp_event de Madrid a UTC string ISO para CSV
                if 'timestamp_event' in record and record['timestamp_event']:
                    utc_datetime = convert_madrid_timestamp_to_utc_datetime(record['timestamp_event'])
                    if utc_datetime:
                        record['timestamp_event'] = utc_datetime.isoformat()
                
                # Preparar registro con todos los campos del schema
                csv_record = {name: record.get(name) for name in self.csv_fieldnames}
                self.csv_writer.writerow(csv_record)
                
                # Log cada ciertos pasos para mostrar progreso sin saturar
                if self.num_timesteps % 500 == 0:
                    logger.info(f"💾 Guardado step {self.num_timesteps} en CSV local")
        elif self.num_timesteps % 500 == 0:
            logger.debug(f"⚠️ Step {self.num_timesteps}: No hay datos para guardar")

        # Log training metrics periodically
        if (self.num_timesteps - self.last_metrics_log_step) >= self.metrics_log_interval:
            sb3_logs = getattr(self.logger, 'name_to_value', {}) if self.logger else {}
            if sb3_logs and self.csv_writer:
                metric_log = {
                    'run_id': self.run_id,
                    'session_id': self.session_id,
                    'episode_id': -1,
                    'step_in_episode': -1,
                    'total_steps_elapsed': self.num_timesteps,
                    'timestamp_event': convert_madrid_timestamp_to_utc_datetime(get_madrid_timestamp_str()).isoformat(),
                    'event_type': 'training_metric',
                    'actor_loss': sb3_logs.get('train/actor_loss'),
                    'critic_loss': sb3_logs.get('train/critic_loss'),
                    'entropy_coefficient': sb3_logs.get('train/ent_coef'),
                    'learning_rate': sb3_logs.get('train/learning_rate'),
                }
                
                # Preparar registro con todos los campos del schema
                csv_record = {name: metric_log.get(name) for name in self.csv_fieldnames}
                self.csv_writer.writerow(csv_record)
                
                logger.info(f"📊 Métricas de entrenamiento guardadas en CSV - Step {self.num_timesteps}")
            
            self.last_metrics_log_step = self.num_timesteps

        # 🚀 Flush del archivo CSV cada cierto número de pasos para asegurar escritura
        if self.num_timesteps % 500 == 0 and hasattr(self, 'csv_file_handle'):
            self.csv_file_handle.flush()
        
        return True

    def _on_training_end(self) -> None:
        logger.info(f"🏁 Entrenamiento finalizado. Total de pasos: {self.num_timesteps}")
        
        # 📁 Cerrar archivo CSV
        if hasattr(self, 'csv_file_handle') and self.csv_file_handle:
            self.csv_file_handle.close()
            logger.info(f"📁 Archivo CSV local cerrado: {self.local_csv_file}")
        
        # 🚀 Subir todos los datos del CSV a BigQuery
        if self.local_csv_file and os.path.exists(self.local_csv_file):
            logger.info(f"📤 Subiendo datos del CSV local a BigQuery...")
            success = self._upload_csv_to_bigquery()
            
            if success:
                # 🗑️ Eliminar archivo local después de subida exitosa
                try:
                    os.remove(self.local_csv_file)
                    logger.info(f"✅ Archivo CSV local eliminado después de subida exitosa: {self.local_csv_file}")
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo eliminar el archivo CSV local: {e}")
            else:
                logger.error(f"❌ La subida a BigQuery falló. El archivo CSV se mantiene en: {self.local_csv_file}")
        
        logger.info(f"🏁 BigQuery Logging finalizado. Run ID: {self.run_id}, Session ID: {self.session_id}")

    def _upload_csv_to_bigquery(self) -> bool:
        """
        Sube todos los datos del archivo CSV local a BigQuery de una vez.
        
        Returns:
            bool: True si la subida fue exitosa, False en caso contrario
        """
        if not self.local_csv_file or not os.path.exists(self.local_csv_file):
            logger.error("❌ No hay archivo CSV local para subir")
            return False
        
        try:
            # Leer todos los registros del CSV
            records_to_upload = []
            with open(self.local_csv_file, 'r', encoding='utf-8') as f:
                csv_reader = csv.DictReader(f)
                for row in csv_reader:
                    # Convertir campos numéricos y preparar para BigQuery
                    processed_row = {}
                    for field_name, value in row.items():
                        if value == '' or value is None:
                            processed_row[field_name] = None
                        elif field_name == 'timestamp_event' and value:
                            # Convertir timestamp de Madrid a UTC datetime naive para BigQuery
                            processed_row[field_name] = convert_madrid_timestamp_to_utc_datetime(value)
                        elif field_name in ['episode_id', 'step_in_episode', 'total_steps_elapsed', 
                                          'current_position_side_step', 'num_trades_episode', 'episode_duration_steps']:
                            # Campos enteros
                            try:
                                processed_row[field_name] = int(float(value)) if value else None
                            except:
                                processed_row[field_name] = None
                        elif field_name in ['reward_step', 'action_value_raw', 'current_equity_step',
                                          'current_position_avg_price_step', 'current_position_size_step',
                                          'market_price_at_step', 'obs_market_feat_0_step', 'obs_market_feat_1_step',
                                          'obs_market_feat_2_step', 'obs_portfolio_pnl_norm_step',
                                          'obs_portfolio_steps_in_pos_norm_step', 'total_reward_episode',
                                          'pnl_realized_episode', 'total_fees_episode', 'final_equity_episode',
                                          'actor_loss', 'critic_loss', 'entropy_coefficient', 'learning_rate']:
                            # Campos float
                            try:
                                processed_row[field_name] = float(value) if value else None
                            except:
                                processed_row[field_name] = None
                        else:
                            # Campos string
                            processed_row[field_name] = value if value else None
                    
                    records_to_upload.append(processed_row)
            
            num_records = len(records_to_upload)
            logger.info(f"📊 Preparando subida de {num_records} registros a BigQuery")
            
            if num_records == 0:
                logger.warning("⚠️ No hay registros para subir a BigQuery")
                return True
            
            # Subir a BigQuery
            from src.utils.logging_utils import get_madrid_timestamp
            table_id_date_suffix = get_madrid_timestamp().strftime('%Y%m%d')
            table_id = f"entrenamiento_{table_id_date_suffix}"
            
            success = stream_data_to_bigquery(
                project_id=self.project_id,
                dataset_id=self.dataset_id,
                table_id=table_id,
                rows_to_insert=records_to_upload,
                client=self.bq_client,
                schema=TRAINING_LOG_SCHEMA
            )
            
            if success:
                logger.info(f"✅ {num_records} registros subidos exitosamente a BigQuery tabla {table_id}")
                return True
            else:
                logger.error(f"❌ Error al subir registros a BigQuery")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error procesando archivo CSV para subir a BigQuery: {e}")
            return False