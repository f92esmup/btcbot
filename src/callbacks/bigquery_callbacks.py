# src/callbacks/bigquery_callbacks.py
import logging
import datetime
import uuid
from typing import List, Dict, Any
from google.cloud import bigquery
from stable_baselines3.common.callbacks import BaseCallback

from src.utils.bigquery_utils import stream_data_to_bigquery
from src.utils.config import ConfigManager # For accessing config values

logger = logging.getLogger(__name__)

# Define the BigQuery schema for entrenamiento_{FECHA} tables
# This should match Step 3 of the plan.
TRAINING_LOG_SCHEMA = [
    bigquery.SchemaField("run_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"), # model.learn() call ID
    bigquery.SchemaField("episode_id", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("step_in_episode", "INTEGER", mode="NULLABLE"), # Null for episode_summary, training_metric
    bigquery.SchemaField("total_steps_elapsed", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("timestamp_event", "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("event_type", "STRING", mode="REQUIRED"), # 'step_info', 'episode_summary', 'training_metric'

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
        self.run_id = str(uuid.uuid4()) # Unique ID for this training run
        self.session_id = str(uuid.uuid4()) # Unique ID for this model.learn() call
        self.log_buffer: List[Dict[str, Any]] = []
        
        agent_config = config_manager.get_agent_config()
        bq_log_config = agent_config.get('bigquery_logging', {})
        self.batch_size = bq_log_config.get('training_log_batch_size', 100)
        self.metrics_log_interval = bq_log_config.get('training_metrics_log_interval_steps', 1000)
        self.last_metrics_log_step = 0

    def _on_training_start(self):
        # Reset session_id for a new training session (if model.learn is called multiple times)
        self.session_id = str(uuid.uuid4())
        self.last_metrics_log_step = 0
        logger.info(f"BigQuery Logging Started. Run ID: {self.run_id}, Session ID: {self.session_id}")

    def _on_step(self) -> bool:
        # Retrieve data from the environment
        # Assuming self.training_env is a VecEnv, access attributes of the first env
        if self.training_env is None:
            logger.warning("BigQueryLoggingCallback: training_env is None, cannot retrieve logs.")
            return True

        try:
            # For VecEnv, use get_attr; for single env, directly call method
            if hasattr(self.training_env, 'envs'): # Likely a VecEnv
                # This gets a list of lists, one for each env. We assume one env for detailed logging.
                step_and_summary_data_list = self.training_env.env_method("get_current_episode_step_data")
                # env_method returns a list of results, one for each sub-environment.
                # We expect data from the first (and likely only) environment.
                if step_and_summary_data_list and isinstance(step_and_summary_data_list, list):
                    step_and_summary_data = step_and_summary_data_list[0]
                else: # Fallback or unexpected structure
                    step_and_summary_data = []
            else: # Single environment
                step_and_summary_data = self.training_env.get_current_episode_step_data()

        except AttributeError as e:
            logger.error(f"Error accessing get_current_episode_step_data from environment: {e}")
            step_and_summary_data = []


        episode_ended = False
        if step_and_summary_data:
            for record in step_and_summary_data:
                # Add common fields
                record['run_id'] = self.run_id
                record['session_id'] = self.session_id
                record['total_steps_elapsed'] = self.num_timesteps # SB3's total steps
                self.log_buffer.append(record)
                if record.get('event_type') == 'episode_summary':
                    episode_ended = True
        
        # Log training metrics periodically
        if (self.num_timesteps - self.last_metrics_log_step) >= self.metrics_log_interval:
            sb3_logs = self.logger.get_latest_values() # SB3 logger, not Python logger
            if sb3_logs:
                # Filter for relevant metrics if possible, or log all scalars
                # Example: time/fps, time/iterations, train/actor_loss, etc.
                # We are interested in losses and learning rate primarily.
                metric_log = {
                    'run_id': self.run_id,
                    'session_id': self.session_id,
                    'episode_id': -1, # Or use current episode_id if available and makes sense
                    'step_in_episode': -1,
                    'total_steps_elapsed': self.num_timesteps,
                    'timestamp_event': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'event_type': 'training_metric',
                    'actor_loss': sb3_logs.get('train/actor_loss'),
                    'critic_loss': sb3_logs.get('train/critic_loss'),
                    'entropy_coefficient': sb3_logs.get('train/ent_coef'),
                    'learning_rate': sb3_logs.get('train/learning_rate'),
                }
                # Remove None values to avoid schema issues if a metric is not always present
                metric_log = {k: v for k, v in metric_log.items() if v is not None}
                if len(metric_log) > 6: # Ensure we have actual metrics
                    self.log_buffer.append(metric_log)
            self.last_metrics_log_step = self.num_timesteps

        # Flush buffer if batch size reached or episode ended
        if len(self.log_buffer) >= self.batch_size or (episode_ended and self.log_buffer):
            self._flush_log_buffer()
        
        return True

    def _on_training_end(self) -> None:
        self._flush_log_buffer()
        logger.info(f"BigQuery Logging Ended. Run ID: {self.run_id}, Session ID: {self.session_id}")

    def _flush_log_buffer(self):
        if not self.log_buffer:
            return

        table_id_date_suffix = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
        table_id = f"entrenamiento_{table_id_date_suffix}"

        logger.debug(f"Flushing {len(self.log_buffer)} log records to BigQuery table {self.dataset_id}.{table_id}")
        
        # Ensure all records have all schema fields, adding None if missing
        processed_buffer = []
        field_names = {field.name for field in TRAINING_LOG_SCHEMA}
        for record in self.log_buffer:
            processed_record = {name: record.get(name) for name in field_names}
            processed_buffer.append(processed_record)

        success = stream_data_to_bigquery(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            table_id=table_id,
            rows_to_insert=processed_buffer,
            client=self.bq_client,
            schema=TRAINING_LOG_SCHEMA
        )
        if success:
            self.log_buffer.clear()
        else:
            # Handle failure: currently logs error. Could implement fallback if needed.
            logger.error(f"Failed to flush log buffer to BigQuery for Run ID: {self.run_id}.")
            # Potentially keep buffer for next attempt, or discard to prevent memory issues.
            # For now, clearing to avoid unbounded growth.
            self.log_buffer.clear()
            logger.warning("Log buffer cleared after failed flush attempt to prevent memory issues.")
