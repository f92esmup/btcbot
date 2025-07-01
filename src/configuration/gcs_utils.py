"""
Utilidades para Google Cloud Storage.
Maneja la subida y descarga de archivos del scaler a/desde un bucket de GCS.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import re
from google.cloud import storage
from google.cloud.exceptions import NotFound
import joblib
import tempfile


class GCSUtils:
    """Clase para manejar operaciones de Google Cloud Storage."""
    
    def __init__(self, gcp_config: Dict[str, Any]):
        """
        Inicializa las utilidades de GCS.
        
        Args:
            gcp_config (Dict[str, Any]): Configuración de GCP que incluye:
                - project_id: ID del proyecto de Google Cloud
                - storage: Dict con bucket_name, scaler_blob_name, price_scaler_blob_name
        """
        self.logger = logging.getLogger(__name__)
        
        # Validar configuración
        if not isinstance(gcp_config, dict):
            raise ValueError("gcp_config debe ser un diccionario")
        
        # Obtener configuración de GCS
        self.project_id = gcp_config.get('project_id')
        if not self.project_id:
            raise ValueError("project_id es requerido en gcp_config")
        
        storage_config = gcp_config.get('storage', {})
        self.bucket_name = storage_config.get('bucket_name', 'btcbot-models')
        self.scaler_blob_name = storage_config.get('scaler_blob_name', 'scaler.pkl')
        self.price_scaler_blob_name = storage_config.get('price_scaler_blob_name', 'price_scaler.pkl')
        
        # Cliente de GCS
        self._client = None
        
        self.logger.info(f"GCSUtils inicializado para proyecto: {self.project_id}")
        self.logger.info(f"Bucket: {self.bucket_name}")
        self.logger.info(f"Archivo scaler: {self.scaler_blob_name}")
        self.logger.info(f"Archivo price_scaler: {self.price_scaler_blob_name}")
    
    @property
    def client(self) -> storage.Client:
        """Obtiene el cliente de GCS (lazy loading)."""
        if self._client is None:
            try:
                self._client = storage.Client(project=self.project_id)
                self.logger.info("Cliente de GCS inicializado exitosamente")
            except Exception as e:
                self.logger.error(f"Error al inicializar cliente de GCS: {e}")
                raise
        return self._client
    
    def _get_bucket(self) -> storage.Bucket:
        """Obtiene el bucket de GCS."""
        try:
            bucket = self.client.bucket(self.bucket_name)
            # Verificar que el bucket existe
            if not bucket.exists():
                raise ValueError(f"El bucket '{self.bucket_name}' no existe")
            return bucket
        except Exception as e:
            self.logger.error(f"Error al acceder al bucket '{self.bucket_name}': {e}")
            raise
    
    def upload_scaler(self, scaler_path: str) -> bool:
        """
        Sube el archivo scaler.pkl al bucket de GCS.
        
        Args:
            scaler_path (str): Ruta local del archivo scaler.pkl
            
        Returns:
            bool: True si la subida fue exitosa, False en caso contrario
        """
        self.logger.info(f"Subiendo scaler desde {scaler_path} a GCS...")
        
        try:
            # Verificar que el archivo existe localmente
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Archivo scaler no encontrado: {scaler_path}")
            
            # Obtener bucket y blob
            bucket = self._get_bucket()
            blob = bucket.blob(self.scaler_blob_name)
            
            # Subir el archivo
            blob.upload_from_filename(scaler_path)
            
            self.logger.info(f"Scaler subido exitosamente a gs://{self.bucket_name}/{self.scaler_blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al subir scaler a GCS: {e}")
            return False
    
    def download_scaler(self, local_path: str) -> bool:
        """
        Descarga el archivo scaler.pkl desde GCS al path local.
        
        Args:
            local_path (str): Ruta local donde guardar el scaler
            
        Returns:
            bool: True si la descarga fue exitosa, False en caso contrario
        """
        self.logger.info(f"Descargando scaler desde GCS a {local_path}...")
        
        try:
            # Obtener bucket y blob
            bucket = self._get_bucket()
            blob = bucket.blob(self.scaler_blob_name)
            
            # Verificar que el blob existe
            if not blob.exists():
                self.logger.warning(f"Scaler no existe en GCS: gs://{self.bucket_name}/{self.scaler_blob_name}")
                return False
            
            # Crear directorio local si no existe
            local_dir = Path(local_path).parent
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Descargar el archivo
            blob.download_to_filename(local_path)
            
            self.logger.info(f"Scaler descargado exitosamente desde gs://{self.bucket_name}/{self.scaler_blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al descargar scaler desde GCS: {e}")
            return False
    
    def scaler_exists_in_gcs(self) -> bool:
        """
        Verifica si el scaler existe en GCS.
        
        Returns:
            bool: True si el scaler existe en GCS, False en caso contrario
        """
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(self.scaler_blob_name)
            exists = blob.exists()
            
            self.logger.info(f"Scaler en GCS: {'existe' if exists else 'no existe'}")
            return exists
            
        except Exception as e:
            self.logger.error(f"Error al verificar existencia del scaler en GCS: {e}")
            return False
    
    def price_scaler_exists_in_gcs(self) -> bool:
        """
        Verifica si el price_scaler existe en GCS.
        
        Returns:
            bool: True si el price_scaler existe en GCS, False en caso contrario
        """
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(self.price_scaler_blob_name)
            exists = blob.exists()
            
            self.logger.info(f"Price scaler en GCS: {'existe' if exists else 'no existe'}")
            return exists
            
        except Exception as e:
            self.logger.error(f"Error al verificar existencia del price_scaler en GCS: {e}")
            return False
    
    def get_scaler_info(self) -> Optional[dict]:
        """
        Obtiene información del scaler en GCS.
        
        Returns:
            Optional[dict]: Información del scaler o None si no existe
        """
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(self.scaler_blob_name)
            
            if not blob.exists():
                return None
            
            # Recargar para obtener metadatos actualizados
            blob.reload()
            
            info = {
                'name': blob.name,
                'size': blob.size,
                'created': blob.time_created,
                'updated': blob.updated,
                'etag': blob.etag,
                'content_type': blob.content_type
            }
            
            self.logger.info(f"Información del scaler en GCS: {info}")
            return info
            
        except Exception as e:
            self.logger.error(f"Error al obtener información del scaler: {e}")
            return None
    
    def get_price_scaler_info(self) -> Optional[dict]:
        """
        Obtiene información del price_scaler en GCS.
        
        Returns:
            Optional[dict]: Información del price_scaler o None si no existe
        """
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(self.price_scaler_blob_name)
            
            if not blob.exists():
                return None
            
            # Recargar para obtener metadatos actualizados
            blob.reload()
            
            info = {
                'name': blob.name,
                'size': blob.size,
                'created': blob.time_created,
                'updated': blob.updated,
                'etag': blob.etag,
                'content_type': blob.content_type
            }
            
            self.logger.info(f"Información del price_scaler en GCS: {info}")
            return info
            
        except Exception as e:
            self.logger.error(f"Error al obtener información del price_scaler: {e}")
            return None
    
    def load_scaler_from_gcs(self, blob_name: Optional[str] = None):
        """
        Carga el scaler directamente desde GCS sin guardarlo localmente.
        
        Args:
            blob_name (Optional[str]): Nombre específico del blob. Si es None, usa el configurado por defecto.
        
        Returns:
            scaler: Objeto scaler cargado desde GCS
        """
        effective_blob_name = blob_name or self.scaler_blob_name
        self.logger.info(f"Cargando scaler directamente desde GCS: {effective_blob_name}")
        
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(effective_blob_name)
            
            if not blob.exists():
                raise FileNotFoundError(f"Scaler no existe en GCS: gs://{self.bucket_name}/{effective_blob_name}")
            
            # Usar archivo temporal
            with tempfile.NamedTemporaryFile() as temp_file:
                blob.download_to_filename(temp_file.name)
                scaler = joblib.load(temp_file.name)
            
            self.logger.info("Scaler cargado exitosamente desde GCS")
            return scaler
            
        except Exception as e:
            self.logger.error(f"Error al cargar scaler desde GCS: {e}")
            raise
    
    def load_price_scaler_from_gcs(self, blob_name: Optional[str] = None):
        """
        Carga el price_scaler directamente desde GCS sin guardarlo localmente.
        
        Args:
            blob_name (Optional[str]): Nombre específico del blob. Si es None, usa el configurado por defecto.
        
        Returns:
            price_scaler: Objeto price_scaler cargado desde GCS
        """
        effective_blob_name = blob_name or self.price_scaler_blob_name
        self.logger.info(f"Cargando price_scaler directamente desde GCS: {effective_blob_name}")
        
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(effective_blob_name)
            
            if not blob.exists():
                raise FileNotFoundError(f"Price scaler no existe en GCS: gs://{self.bucket_name}/{effective_blob_name}")
            
            # Usar archivo temporal
            with tempfile.NamedTemporaryFile() as temp_file:
                blob.download_to_filename(temp_file.name)
                price_scaler = joblib.load(temp_file.name)
            
            self.logger.info("Price scaler cargado exitosamente desde GCS")
            return price_scaler
            
        except Exception as e:
            self.logger.error(f"Error al cargar price_scaler desde GCS: {e}")
            raise
    
    def save_scaler_to_gcs(self, scaler, blob_name: Optional[str] = None) -> bool:
        """
        Guarda un objeto scaler directamente a GCS sin guardarlo localmente.
        
        Args:
            scaler: Objeto scaler a guardar
            blob_name (Optional[str]): Nombre específico del blob. Si no se proporciona, usa el configurado por defecto.
            
        Returns:
            bool: True si se guardó exitosamente, False en caso contrario
        """
        effective_blob_name = blob_name or self.scaler_blob_name
        self.logger.info(f"Guardando scaler directamente a GCS como {effective_blob_name}...")
        
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(effective_blob_name)
            
            # Usar archivo temporal
            with tempfile.NamedTemporaryFile() as temp_file:
                joblib.dump(scaler, temp_file.name)
                blob.upload_from_filename(temp_file.name)
            
            self.logger.info(f"Scaler guardado exitosamente en gs://{self.bucket_name}/{effective_blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al guardar scaler en GCS: {e}")
            return False
    
    def save_price_scaler_to_gcs(self, price_scaler, blob_name: Optional[str] = None) -> bool:
        """
        Guarda un objeto price_scaler directamente a GCS sin guardarlo localmente.
        
        Args:
            price_scaler: Objeto price_scaler a guardar
            blob_name (Optional[str]): Nombre específico del blob. Si no se proporciona, usa el configurado por defecto.
            
        Returns:
            bool: True si se guardó exitosamente, False en caso contrario
        """
        effective_blob_name = blob_name or self.price_scaler_blob_name
        self.logger.info(f"Guardando price_scaler directamente a GCS como {effective_blob_name}...")
        
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(effective_blob_name)
            
            # Usar archivo temporal
            with tempfile.NamedTemporaryFile() as temp_file:
                joblib.dump(price_scaler, temp_file.name)
                blob.upload_from_filename(temp_file.name)
            
            self.logger.info(f"Price scaler guardado exitosamente en gs://{self.bucket_name}/{effective_blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al guardar price_scaler en GCS: {e}")
            return False
    
    def upload_file_to_gcs(self, local_path: str, gcs_blob_name: str) -> bool:
        """
        Subir un archivo local genérico al bucket de GCS.
        
        Args:
            local_path (str): Ruta local del archivo a subir
            gcs_blob_name (str): Ruta completa del blob en GCS (ej: experiments/RUN_ID/checkpoints/actor.pth)
            
        Returns:
            bool: True si la subida fue exitosa, False en caso contrario
        """
        self.logger.info(f"Subiendo archivo {local_path} a GCS como {gcs_blob_name}...")
        
        try:
            # Verificar que el archivo local existe
            if not os.path.exists(local_path):
                self.logger.error(f"Archivo local no encontrado: {local_path}")
                return False
            
            # Obtener bucket y blob
            bucket = self._get_bucket()
            blob = bucket.blob(gcs_blob_name)
            
            # Subir el archivo
            blob.upload_from_filename(local_path)
            
            self.logger.info(f"Archivo subido exitosamente a gs://{self.bucket_name}/{gcs_blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al subir archivo a GCS: {e}")
            return False
    
    def download_file_from_gcs(self, gcs_blob_name: str, local_path: str) -> bool:
        """
        Descargar un archivo desde GCS a una ruta local.
        
        Args:
            gcs_blob_name (str): Ruta completa del blob en GCS
            local_path (str): Ruta local donde guardar el archivo
            
        Returns:
            bool: True si la descarga fue exitosa, False en caso contrario
        """
        self.logger.info(f"Descargando archivo desde GCS {gcs_blob_name} a {local_path}...")
        
        try:
            # Obtener bucket y blob
            bucket = self._get_bucket()
            blob = bucket.blob(gcs_blob_name)
            
            # Verificar que el blob existe
            if not blob.exists():
                self.logger.warning(f"Archivo no existe en GCS: gs://{self.bucket_name}/{gcs_blob_name}")
                return False
            
            # Crear directorio local si no existe
            local_dir = Path(local_path).parent
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Descargar el archivo
            blob.download_to_filename(local_path)
            
            self.logger.info(f"Archivo descargado exitosamente desde gs://{self.bucket_name}/{gcs_blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al descargar archivo desde GCS: {e}")
            return False
    
    def file_exists_in_gcs(self, gcs_blob_name: str) -> bool:
        """
        Verificar si un archivo específico (blob) existe en GCS.
        
        Args:
            gcs_blob_name (str): Ruta completa del blob en GCS
            
        Returns:
            bool: True si el archivo existe, False en caso contrario
        """
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(gcs_blob_name)
            return blob.exists()
            
        except Exception as e:
            self.logger.error(f"Error al verificar existencia de archivo en GCS: {e}")
            return False
    
    def upload_directory_to_gcs(self, local_directory_path: str, gcs_prefix: str) -> bool:
        """
        Subir recursivamente el contenido de un directorio local a un prefijo específico en GCS.
        
        Args:
            local_directory_path (str): Ruta del directorio local a subir
            gcs_prefix (str): Prefijo en GCS donde subir los archivos (ej: experiments/RUN_ID/tensorboard_logs)
            
        Returns:
            bool: True si la sincronización fue exitosa, False en caso contrario
        """
        self.logger.info(f"Iniciando sincronización de directorio {local_directory_path} a GCS bajo el prefijo {gcs_prefix}...")
        
        try:
            local_path = Path(local_directory_path)
            
            # Verificar que el directorio local existe
            if not local_path.exists():
                self.logger.error(f"Directorio local no encontrado: {local_directory_path}")
                return False
            
            if not local_path.is_dir():
                self.logger.error(f"La ruta especificada no es un directorio: {local_directory_path}")
                return False
            
            files_uploaded = 0
            files_failed = 0
            
            # Iterar sobre todos los archivos recursivamente
            for file_path in local_path.rglob('*'):
                if file_path.is_file():
                    # Construir la ruta relativa desde el directorio base
                    relative_path = file_path.relative_to(local_path)
                    
                    # Construir el nombre del blob en GCS
                    gcs_blob_name = f"{gcs_prefix}/{relative_path.as_posix()}"
                    
                    # Subir el archivo
                    if self.upload_file_to_gcs(str(file_path), gcs_blob_name):
                        files_uploaded += 1
                    else:
                        files_failed += 1
                        self.logger.warning(f"Fallo al subir archivo: {file_path}")
            
            self.logger.info(f"Sincronización completada. Archivos subidos: {files_uploaded}, Fallos: {files_failed}")
            
            return files_failed == 0
            
        except Exception as e:
            self.logger.error(f"Error durante la sincronización de directorio a GCS: {e}")
            return False
    



