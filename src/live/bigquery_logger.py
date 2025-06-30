from google.cloud import bigquery

class BigQueryLogger:
    def __init__(self, project_id: str, dataset_id: str):
        """
        Inicializa el logger para Google BigQuery.

        Args:
            project_id (str): Tu ID de proyecto de Google Cloud.
            dataset_id (str): El ID del dataset en BigQuery donde se crearán las tablas.
        """
        self.client = bigquery.Client(project=project_id)
        self.dataset_id = dataset_id
        self.project_id = project_id
        print(f"BigQueryLogger inicializado para el dataset '{dataset_id}'.")

    def log_trade(self, trade_data: dict):
        """
        Registra la información de un trade completado.
        (La lógica se implementará más adelante).
        """
        pass

    def log_decision(self, decision_data: dict):
        """
        Registra la información de una decisión de trading.
        (La lógica se implementará más adelante).
        """
        pass