#!/usr/bin/env python3
"""
Utilidad para probar sin el pipeline de Kubeflow.
Este script serializa correctamente los parámetros de PipelineParameterChannel.
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Asegurar que la carpeta src está en el path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def safe_serialize(obj):
    """Serializa cualquier objeto a formato JSON, convirtiendo objetos especiales a strings."""
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return obj.to_dict()
    if hasattr(obj, '__str__'):
        return str(obj)
    return repr(obj)

def main():
    """Compila el pipeline de forma segura para ejecutarlo localmente."""
    parser = argparse.ArgumentParser(description='Compila el pipeline para Bitcoin Trading Bot')
    parser.add_argument('--output-file', type=str, default='pipeline_test.json',
                      help='Ruta al archivo JSON de salida')
    parser.add_argument('--project-id', type=str, default=None,
                      help='ID del proyecto de Google Cloud')
    parser.add_argument('--gcs-bucket', type=str, default=None,
                      help='Bucket de Google Cloud Storage')
    args = parser.parse_args()

    # Obtener el project_id de terraform.tfvars si no se especifica
    if not args.project_id:
        try:
            with open('../terraform/terraform.tfvars', 'r') as f:
                for line in f:
                    if line.startswith('project_id'):
                        args.project_id = line.split('=')[1].strip().strip('"')
                        break
        except:
            args.project_id = "local-project"

    # Establecer un bucket por defecto basado en el project_id si no se especifica
    if not args.gcs_bucket:
        args.gcs_bucket = f"{args.project_id}-btc-artifacts"

    print(f"🔧 Compilando pipeline para proyecto: {args.project_id}")
    print(f"🪣 Usando bucket GCS: {args.gcs_bucket}")

    # Importar la definición del pipeline solo cuando sea necesario
    sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from pipeline_definition import compile_pipeline

    # Usar la función de compilación segura
    try:
        # Modificar temporalmente json.dumps para usar nuestra función segura
        original_dumps = json.dumps
        
        def safe_dumps(*args, **kwargs):
            """Wrapper alrededor de json.dumps que maneja objetos especiales."""
            if 'default' not in kwargs:
                kwargs['default'] = safe_serialize
            return original_dumps(*args, **kwargs)
        
        # Reemplazar temporalmente json.dumps
        json.dumps = safe_dumps
        
        # Compilar el pipeline
        compile_pipeline(output_file=args.output_file)
        
        # Restaurar json.dumps
        json.dumps = original_dumps
        
        print(f"✅ Pipeline compilado exitosamente en {args.output_file}")
    except Exception as e:
        print(f"❌ Error compilando el pipeline: {e}")
        sys.exit(1)

    print("🏃 Para ejecutar una prueba local, usa: ./scripts/run_test_training.sh --local")
    print("🚀 Para ejecutar en Vertex AI, usa: ./scripts/run_test_training.sh")

if __name__ == '__main__':
    main()
