# src/callbacks/bigquery_evaluation_schema.py
"""
Schema de BigQuery para los datos de evaluación del agente RL.
"""

from google.cloud import bigquery

# Schema para tablas de evaluación con formato evaluacion_{FECHA}
EVALUATION_LOG_SCHEMA = [
    # Identificadores únicos
    bigquery.SchemaField("evaluation_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("model_path", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("timestamp_evaluation", "TIMESTAMP", mode="REQUIRED"),
    
    # Configuración de evaluación
    bigquery.SchemaField("config_path", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("num_episodes", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("episode_number", "INTEGER", mode="REQUIRED"),
    
    # Métricas de episodio
    bigquery.SchemaField("total_return_percent", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("initial_value", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("final_value", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("avg_reward_per_step", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("cumulative_reward", "FLOAT", mode="REQUIRED"),
    
    # Estadísticas detalladas del episodio
    bigquery.SchemaField("episode_length_steps", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("total_trades", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("winning_trades", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("losing_trades", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("win_rate_percent", "FLOAT", mode="NULLABLE"),
    
    # Métricas de PnL y riesgo
    bigquery.SchemaField("total_pnl", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("max_drawdown_percent", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("max_profit_percent", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("sharpe_ratio", "FLOAT", mode="NULLABLE"),
    
    # Estadísticas de posiciones
    bigquery.SchemaField("avg_position_duration_steps", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("max_position_duration_steps", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("long_positions_count", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("short_positions_count", "INTEGER", mode="NULLABLE"),
    
    # Información de mercado
    bigquery.SchemaField("market_conditions", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("start_timestamp", "TIMESTAMP", mode="NULLABLE"),
    bigquery.SchemaField("end_timestamp", "TIMESTAMP", mode="NULLABLE"),
    
    # Detalles de configuración del modelo
    bigquery.SchemaField("device_used", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("environment_type", "STRING", mode="NULLABLE"),
    
    # Campo para notas adicionales
    bigquery.SchemaField("notes", "STRING", mode="NULLABLE"),
]

# Schema para datos detallados por paso (opcional, para análisis granular)
EVALUATION_STEP_SCHEMA = [
    # Identificadores
    bigquery.SchemaField("evaluation_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("episode_number", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("step_number", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("timestamp_step", "TIMESTAMP", mode="REQUIRED"),
    
    # Datos del paso
    bigquery.SchemaField("reward", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("action_value", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("position", "INTEGER", mode="REQUIRED"),  # -1, 0, 1
    bigquery.SchemaField("portfolio_value", "FLOAT", mode="REQUIRED"),
    bigquery.SchemaField("market_price", "FLOAT", mode="REQUIRED"),
    
    # Estado del entorno
    bigquery.SchemaField("current_equity", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("unrealized_pnl", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("position_size", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("entry_price", "FLOAT", mode="NULLABLE"),
    
    # Análisis técnico (si disponible)
    bigquery.SchemaField("market_feature_0", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("market_feature_1", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("market_feature_2", "FLOAT", mode="NULLABLE"),
]
