from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import datetime

class BigQueryLogger:
    """
    Gestiona el registro de datos de la operativa en vivo a una tabla de Google BigQuery.
    """
    
    # Esquema de la tabla definido de forma centralizada.
    TABLE_SCHEMA = [
        bigquery.SchemaField("log_timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("run_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("candle_timestamp", "TIMESTAMP", mode="NULLABLE"),
        bigquery.SchemaField("symbol", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("market_price", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("agent_action", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("interpreted_intent", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("trade_executed", "BOOLEAN", mode="NULLABLE"),
        bigquery.SchemaField("position_status", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("position_pnl_roe", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("position_duration", "INTEGER", mode="NULLABLE"),
        bigquery.SchemaField("position_entry_price", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("account_balance", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("account_equity", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("max_equity", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("drawdown_pct", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("consecutive_losses", "INTEGER", mode="NULLABLE"),
    ]

    def __init__(self, project_id: str, dataset_id: str, table_id: str = "live_trading_log"):
        """
        Inicializa el logger para Google BigQuery.

        Args:
            project_id (str): Tu ID de proyecto de Google Cloud.
            dataset_id (str): El ID del dataset en BigQuery.
            table_id (str): El ID de la tabla donde se guardarán los logs.
        """
        self.client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.table_ref = self.client.dataset(self.dataset_id).table(self.table_id)
        
        print(f"BigQueryLogger inicializado para tabla: '{project_id}.{dataset_id}.{table_id}'")
        
        # Asegurarse de que la tabla exista al inicializar.
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self):
        """
        Crea la tabla de logs en BigQuery si no existe.
        """
        try:
            self.client.get_table(self.table_ref)
            print(f"La tabla '{self.table_id}' ya existe en el dataset '{self.dataset_id}'.")
        except NotFound:
            print(f"La tabla '{self.table_id}' no existe. Creándola ahora...")
            table = bigquery.Table(self.table_ref, schema=self.TABLE_SCHEMA)
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="log_timestamp"  # Particionar por la fecha del log
            )
            self.client.create_table(table)
            print(f"Tabla '{self.table_id}' creada exitosamente.")

    def log_step_data(self, step_data: dict):
        """
        Inserta una nueva fila de datos en la tabla de BigQuery.

        Args:
            step_data (dict): Un diccionario con los datos a registrar. 
                              Las claves deben coincidir con los nombres de las columnas del esquema.
        """
        # Añadir el timestamp del log en el momento de la inserción
        step_data["log_timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        errors = self.client.insert_rows_json(self.table_ref, [step_data])
        
        if not errors:
            print(f"Registro insertado correctamente en BigQuery a las {step_data['log_timestamp']}.")
        else:
            print(f"Errores al insertar filas en BigQuery: {errors}")