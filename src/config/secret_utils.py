from google.cloud import secretmanager
import logging

class SecretManagerUtils:
    """
    Clase de utilidad para interactuar con Google Cloud Secret Manager.
    Su única responsabilidad es obtener los valores de los secretos.
    """
    def __init__(self, project_id: str):
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = project_id
        self.logger = logging.getLogger(__name__)

    def get_secret(self, secret_id: str, version: str = "latest") -> str:
        """
        Obtiene el valor de un secreto específico.

        Args:
            secret_id (str): El ID del secreto en Secret Manager.
            version (str): La versión del secreto a obtener (por defecto 'latest').

        Returns:
            str: El valor del secreto decodificado.
        """
        name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
        try:
            response = self.client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            self.logger.error(f"No se pudo acceder al secreto '{secret_id}' en el proyecto '{self.project_id}': {e}")
            # Devolver None o lanzar una excepción podría ser una alternativa.
            # Por seguridad, es mejor que el programa falle si no puede obtener una credencial.
            raise RuntimeError(f"Fallo al obtener el secreto: {secret_id}") from e