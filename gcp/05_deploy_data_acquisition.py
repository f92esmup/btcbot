"""
Script para desplegar el servicio de adquisición de datos en Cloud Run.
"""
import argparse
import os
import tempfile
import shutil
import zipfile
import subprocess
from common import config, clients

def create_data_acquisition_service():
    """
    Crea un servicio de Cloud Run para la adquisición de datos.
    """
    # Crear una estructura temporal para el servicio
    with tempfile.TemporaryDirectory() as temp_dir:
        # Crea el Dockerfile
        with open(os.path.join(temp_dir, "Dockerfile"), "w") as f:
            f.write("""FROM python:3.12-slim

WORKDIR /app

# Copiar solo los archivos necesarios
COPY requirements.txt /app/
COPY src/ /app/src/
COPY scripts/download_data.py /app/scripts/

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Variables de entorno
ENV PYTHONPATH=/app

# Puerto para Cloud Run
ENV PORT 8080

# Exponer el puerto
EXPOSE ${PORT}

# Instalar gunicorn para servir la aplicación
RUN pip install gunicorn flask

# Copiar la aplicación web
COPY app.py /app/

# Ejecutar con gunicorn
CMD exec gunicorn --bind :${PORT} --workers 1 --threads 8 --timeout 0 app:app
""")
        
        # Crear un archivo app.py para la API Flask
        with open(os.path.join(temp_dir, "app.py"), "w") as f:
            f.write("""import os
import json
import logging
import subprocess
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify
from google.cloud import storage
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('btcbot-data-acquisition')

app = Flask(__name__)

# Obtener el ID del proyecto
PROJECT_ID = os.environ.get('PROJECT_ID', 'btcbot276299')
RAW_DATA_BUCKET = f"btcbot-raw-data-{PROJECT_ID}"

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "ok", "service": "btcbot-data-acquisition"})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/download', methods=['POST'])
def download_data():
    try:
        # Obtener parámetros
        data = request.get_json() or {}
        symbol = data.get('symbol', 'BTCUSDT')
        interval = data.get('interval', '1h')
        start_date = data.get('start_date', '2020-01-01')
        end_date = data.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        
        logger.info(f"Iniciando descarga de datos: {symbol} {interval} desde {start_date} hasta {end_date}")
        
        # Crear directorio temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            # Ejecutar el script de descarga
            cmd = [
                "python", "/app/scripts/download_data.py",
                "--symbol", symbol,
                "--interval", interval,
                "--start_date", start_date,
                "--end_date", end_date,
                "--output_dir", temp_dir
            ]
            
            logger.info(f"Ejecutando comando: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error en la descarga: {result.stderr}")
                return jsonify({
                    "status": "error",
                    "message": f"Error downloading data: {result.stderr}"
                }), 500
            
            # Subir a GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(RAW_DATA_BUCKET)
            
            # Encontrar el archivo CSV generado
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
            if not csv_files:
                return jsonify({
                    "status": "error",
                    "message": "No CSV files were generated"
                }), 500
            
            filename = csv_files[0]
            local_path = os.path.join(temp_dir, filename)
            
            # Subir el archivo
            blob = bucket.blob(f"raw/{filename}")
            blob.upload_from_filename(local_path)
            
            gcs_uri = f"gs://{RAW_DATA_BUCKET}/raw/{filename}"
            logger.info(f"Archivo subido a: {gcs_uri}")
            
            return jsonify({
                "status": "success",
                "message": "Data downloaded and uploaded to GCS",
                "file": filename,
                "gcs_uri": gcs_uri
            })
    
    except Exception as e:
        logger.exception("Error al procesar la solicitud")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
""")
        
        # Copiar los archivos del proyecto
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        shutil.copy(os.path.join(project_dir, "requirements.txt"), temp_dir)
        
        # Copiar el directorio src
        src_dir = os.path.join(project_dir, "src")
        dst_src_dir = os.path.join(temp_dir, "src")
        shutil.copytree(src_dir, dst_src_dir)
        
        # Copiar el script de descarga
        scripts_dir = os.path.join(project_dir, "scripts")
        os.makedirs(os.path.join(temp_dir, "scripts"), exist_ok=True)
        shutil.copy(os.path.join(scripts_dir, "download_data.py"), os.path.join(temp_dir, "scripts"))
        
        # Construir la imagen y subir a Artifact Registry
        service_name = config.DATA_ACQUISITION_SERVICE_NAME
        image_name = f"{config.REGION}-docker.pkg.dev/{config.PROJECT_ID}/{config.ARTIFACT_REPO}/{service_name}"
        
        # Asegurarse de que el repositorio existe
        subprocess.run(
            ["python", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_build_docker_image.py")],
            check=True
        )
        
        # Construir la imagen
        subprocess.run(
            ["docker", "build", "-t", f"{image_name}:latest", temp_dir],
            check=True
        )
        
        # Subir la imagen
        subprocess.run(
            ["docker", "push", f"{image_name}:latest"],
            check=True
        )
        
        # Desplegar en Cloud Run
        subprocess.run([
            "gcloud", "run", "deploy", service_name,
            f"--image={image_name}:latest",
            f"--project={config.PROJECT_ID}",
            f"--region={config.REGION}",
            "--platform=managed",
            "--memory=2Gi",
            "--timeout=1800",
            "--cpu=1",
            "--min-instances=0",
            "--max-instances=1",
            "--service-account", config.SERVICE_ACCOUNT_EMAIL,
            "--set-env-vars", ",".join([
                f"PROJECT_ID={config.PROJECT_ID}",
                f"RAW_DATA_BUCKET={config.RAW_DATA_BUCKET}",
                f"BINANCE_API_KEY_SECRET_NAME={config.BINANCE_API_KEY_SECRET_NAME}",
                f"BINANCE_API_SECRET_SECRET_NAME={config.BINANCE_API_SECRET_SECRET_NAME}",
                f"DEFAULT_SYMBOL={config.DEFAULT_SYMBOL}",
                f"DEFAULT_INTERVAL={config.DEFAULT_INTERVAL}"
            ]),
            "--allow-unauthenticated"
        ], check=True)
        
        print(f"Servicio {service_name} desplegado exitosamente.")

def setup_scheduler(service_url):
    """
    Configura un Cloud Scheduler para ejecutar periódicamente el servicio.
    
    Args:
        service_url: URL del servicio de Cloud Run.
    """
    # Nombre del job
    job_name = "btcbot-data-acquisition-daily"
    
    # Eliminar el job si ya existe
    try:
        subprocess.run([
            "gcloud", "scheduler", "jobs", "delete", job_name,
            f"--project={config.PROJECT_ID}",
            f"--location={config.REGION}",
            "--quiet"
        ])
    except:
        pass
    
    # Crear el nuevo job
    subprocess.run([
        "gcloud", "scheduler", "jobs", "create", "http", job_name,
        f"--project={config.PROJECT_ID}",
        f"--location={config.REGION}",
        "--schedule=0 0 * * *",  # Ejecutar a medianoche todos los días
        f"--uri={service_url}/download",
        "--http-method=POST",
        "--message-body='{\"symbol\": \"BTCUSDT\", \"interval\": \"1h\"}'",
        "--headers=Content-Type=application/json",
        "--time-zone=UTC",
        "--attempt-deadline=30m"
    ], check=True)
    
    print(f"Scheduler {job_name} configurado para ejecutarse diariamente.")

def main():
    """
    Función principal para desplegar el servicio de adquisición de datos.
    """
    # Desplegar el servicio
    create_data_acquisition_service()
    
    # Obtener la URL del servicio
    result = subprocess.run([
        "gcloud", "run", "services", "describe", config.DATA_ACQUISITION_SERVICE_NAME,
        f"--project={config.PROJECT_ID}",
        f"--region={config.REGION}",
        "--format=value(status.url)"
    ], capture_output=True, text=True, check=True)
    
    service_url = result.stdout.strip()
    
    # Configurar el scheduler
    setup_scheduler(service_url)
    
    print(f"Servicio de adquisición de datos desplegado en: {service_url}")
    print("Configuración completada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Desplegar servicio de adquisición de datos en Cloud Run")
    args = parser.parse_args()
    main()
