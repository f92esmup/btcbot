"""
Utilidades para Google Cloud Storage.
Maneja la subida y descarga de archivos del scaler a/desde un bucket de GCS.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import re
from google.cloud import storage
from google.cloud.exceptions import NotFound
import joblib
import tempfile
from .config import config


class GCSUtils:
    """Clase para manejar operaciones de Google Cloud Storage."""
    
    def __init__(self):
        """Inicializa las utilidades de GCS."""
        self.logger = logging.getLogger(__name__)
        
        # Obtener configuración de GCS
        self.project_id = config.project_id
        self.bucket_name = config.gcs_bucket_name
        self.scaler_blob_name = config.gcs_scaler_blob_name
        
        # Cliente de GCS
        self._client = None
        
        self.logger.info(f"GCSUtils inicializado para proyecto: {self.project_id}")
        self.logger.info(f"Bucket: {self.bucket_name}")
        self.logger.info(f"Archivo scaler: {self.scaler_blob_name}")
    
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
    
    def load_scaler_from_gcs(self):
        """
        Carga el scaler directamente desde GCS sin guardarlo localmente.
        
        Returns:
            scaler: Objeto scaler cargado desde GCS
        """
        self.logger.info("Cargando scaler directamente desde GCS...")
        
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(self.scaler_blob_name)
            
            if not blob.exists():
                raise FileNotFoundError(f"Scaler no existe en GCS: gs://{self.bucket_name}/{self.scaler_blob_name}")
            
            # Usar archivo temporal
            with tempfile.NamedTemporaryFile() as temp_file:
                blob.download_to_filename(temp_file.name)
                scaler = joblib.load(temp_file.name)
            
            self.logger.info("Scaler cargado exitosamente desde GCS")
            return scaler
            
        except Exception as e:
            self.logger.error(f"Error al cargar scaler desde GCS: {e}")
            raise
    
    def save_scaler_to_gcs(self, scaler) -> bool:
        """
        Guarda un objeto scaler directamente a GCS sin guardarlo localmente.
        
        Args:
            scaler: Objeto scaler a guardar
            
        Returns:
            bool: True si se guardó exitosamente, False en caso contrario
        """
        self.logger.info("Guardando scaler directamente a GCS...")
        
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(self.scaler_blob_name)
            
            # Usar archivo temporal
            with tempfile.NamedTemporaryFile() as temp_file:
                joblib.dump(scaler, temp_file.name)
                blob.upload_from_filename(temp_file.name)
            
            self.logger.info(f"Scaler guardado exitosamente en gs://{self.bucket_name}/{self.scaler_blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al guardar scaler en GCS: {e}")
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
    
    def find_latest_checkpoint_gcs_info(self, gcs_checkpoint_folder_prefix: str) -> Optional[Tuple[str, int]]:
        """
        Buscar en la carpeta especificada en GCS el conjunto de archivos de checkpoint 
        con el número de episodio más alto.
        
        Args:
            gcs_checkpoint_folder_prefix (str): Prefijo de la carpeta en GCS donde buscar checkpoints
                                              (ej: "experiments/RUN_ID/checkpoints/")
        
        Returns:
            Optional[Tuple[str, int]]: Tupla con el prefijo completo del checkpoint más reciente
                                     y el número de episodio, o None si no se encuentra ningún checkpoint
        """
        self.logger.info(f"Buscando checkpoints en GCS bajo el prefijo: {gcs_checkpoint_folder_prefix}")
        
        try:
            # Asegurar que el prefijo termine con '/'
            if not gcs_checkpoint_folder_prefix.endswith('/'):
                gcs_checkpoint_folder_prefix += '/'
            
            # Obtener bucket y listar blobs
            bucket = self._get_bucket()
            blobs = list(bucket.list_blobs(prefix=gcs_checkpoint_folder_prefix))
            
            if not blobs:
                self.logger.info(f"No se encontraron archivos bajo el prefijo: {gcs_checkpoint_folder_prefix}")
                return None
            
            # Patrón regex para extraer el nombre base del checkpoint y el número de episodio
            # Buscaremos archivos como "checkpoint_episode_123_metadata.pkl"
            metadata_pattern = re.compile(r"(checkpoint_episode_(\d+))_metadata\.pkl$")
            
            latest_episode_number = -1
            latest_checkpoint_base_name = None
            
            # Iterar sobre los blobs buscando archivos de metadata
            for blob in blobs:
                blob_name = blob.name
                # Obtener solo el nombre del archivo (sin el prefijo de carpeta)
                filename = blob_name.replace(gcs_checkpoint_folder_prefix, '')
                
                match = metadata_pattern.search(filename)
                if match:
                    checkpoint_base_name = match.group(1)  # "checkpoint_episode_123"
                    episode_number = int(match.group(2))    # 123
                    
                    self.logger.debug(f"Encontrado checkpoint: {checkpoint_base_name} (episodio {episode_number})")
                    
                    if episode_number > latest_episode_number:
                        latest_episode_number = episode_number
                        latest_checkpoint_base_name = checkpoint_base_name
            
            if latest_checkpoint_base_name is None:
                self.logger.info("No se encontraron checkpoints válidos en GCS")
                return None
            
            # Construir el prefijo completo del checkpoint más reciente
            latest_checkpoint_full_prefix = f"{gcs_checkpoint_folder_prefix}{latest_checkpoint_base_name}"
            
            self.logger.info(f"Checkpoint más reciente encontrado: {latest_checkpoint_base_name} (episodio {latest_episode_number})")
            self.logger.info(f"Prefijo completo: {latest_checkpoint_full_prefix}")
            
            return (latest_checkpoint_full_prefix, latest_episode_number)
            
        except Exception as e:
            self.logger.error(f"Error al buscar checkpoints en GCS: {e}")
            return None


# Instancia global para facilitar el uso
gcs_utils = GCSUtils()
