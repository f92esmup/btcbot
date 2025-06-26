# launch_vertex_job.py

import argparse
import logging
import sys
import yaml
from datetime import datetime
from google.cloud import aiplatform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Lanzador de trabajos de entrenamiento en Vertex AI.")
    parser.add_argument("--config-file", type=str, required=True, 
                       help="Ruta al archivo de configuración del trabajo (ej. relayonvertex.yaml).")
    parser.add_argument("--display-name", type=str, default=None, 
                       help="Nombre para mostrar en la consola de Vertex AI.")
    parser.add_argument("--sync", action="store_true", 
                       help="Si se especifica, esperar a que el trabajo termine.")
    return parser.parse_args()


def main():
    """Función principal para lanzar trabajos en Vertex AI."""
    args = parse_arguments()
    logger.info("🚀 Iniciando lanzamiento de trabajo en Vertex AI.")
    
    # Cargar la especificación del trabajo desde el archivo YAML
    try:
        with open(args.config_file, 'r') as f:
            job_spec = yaml.safe_load(f)
        logger.info(f"✅ Especificación cargada desde '{args.config_file}'.")
    except Exception as e:
        logger.error(f"❌ Error leyendo el archivo de configuración: {e}")
        sys.exit(1)
    
    # Inicializar Vertex AI SDK
    aiplatform.init()
    
    # Generar nombre para mostrar
    display_name = args.display_name or f"btcbot_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Crear el trabajo personalizado
    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=job_spec['workerPoolSpecs'],
        base_output_directory=job_spec.get('baseOutputDirectory', {}).get('outputUriPrefix'),
    )
    
    # Enviar el trabajo a Vertex AI
    try:
        job.run(service_account=job_spec.get('serviceAccount'), sync=args.sync)
        logger.info("✅ ¡Trabajo enviado a Vertex AI exitosamente!")
        logger.info(f"🔗 Monitoréalo en: {job._dashboard_uri()}")
    except Exception as e:
        logger.error(f"❌ Falló el envío del trabajo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
