#!/usr/bin/env python
"""
Script para preprocesar datos históricos y generar secuencias para el entrenamiento.
Optimizado para ejecutarse como un componente de Vertex AI Pipelines.
"""
import argparse
import logging
import os
import json

from src.data.preprocessor_cloud import DataPreprocessorCloud

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Función principal para preprocesar datos."""
    parser = argparse.ArgumentParser(description="Preprocesa datos históricos para entrenamiento")
    
    # Parámetros básicos
    parser.add_argument("--input-file-gcs", type=str, required=True,
                        help="Ruta completa al archivo CSV en GCS")
    parser.add_argument("--output-gcs-path", type=str, required=False,
                        default=None,
                        help="Ruta GCS completa para el archivo de salida (default: se genera automáticamente)")
    parser.add_argument("--sequence-length", type=int, required=False,
                        default=int(os.getenv("SEQUENCE_LENGTH_L", "96")),
                        help="Longitud de la secuencia para el Transformer (default: desde variable SEQUENCE_LENGTH_L o 96)")
    parser.add_argument("--norm-window-multiplier", type=int, required=False,
                        default=int(os.getenv("NORM_WINDOW_MULTIPLIER", "2")),
                        help="Multiplicador para la ventana de normalización (default: desde NORM_WINDOW_MULTIPLIER o 2)")
    
    # Parámetros de GCP
    parser.add_argument("--project-id", type=str, required=False,
                        default=os.getenv("GCP_PROJECT_ID"),
                        help="ID del proyecto GCP (default: desde variable GCP_PROJECT_ID)")
    parser.add_argument("--raw-data-bucket", type=str, required=False,
                        default=os.getenv("RAW_DATA_BUCKET"),
                        help="Nombre del bucket para datos crudos (default: desde variable RAW_DATA_BUCKET)")
    parser.add_argument("--processed-data-bucket", type=str, required=False,
                        default=os.getenv("PROCESSED_DATA_BUCKET"),
                        help="Nombre del bucket para datos procesados (default: desde variable PROCESSED_DATA_BUCKET)")
    
    # Parámetros para indicadores técnicos (opcional, como JSON)
    parser.add_argument("--indicators-config", type=str, required=False,
                        default=os.getenv("INDICATORS_CONFIG", None),
                        help="Configuración JSON para indicadores técnicos (default: configuración interna)")
    parser.add_argument("--ohlcv-config", type=str, required=False,
                        default=os.getenv("OHLCV_CONFIG", None),
                        help="Configuración JSON para procesamiento OHLCV (default: configuración interna)")
    parser.add_argument("--feature-columns", type=str, required=False,
                        default=os.getenv("FEATURE_COLUMNS", None),
                        help="Lista JSON de columnas de características finales (default: configuración interna)")
    
    args = parser.parse_args()
    
    # Verificar parámetros obligatorios
    if not args.project_id:
        raise ValueError("Se requiere --project-id o la variable de entorno GCP_PROJECT_ID")
    if not args.raw_data_bucket:
        raise ValueError("Se requiere --raw-data-bucket o la variable de entorno RAW_DATA_BUCKET")
    if not args.processed_data_bucket:
        raise ValueError("Se requiere --processed-data-bucket o la variable de entorno PROCESSED_DATA_BUCKET")
    
    # Parsear configuraciones JSON opcionales
    indicators_config_dict = None
    if args.indicators_config:
        try:
            indicators_config_dict = json.loads(args.indicators_config)
        except json.JSONDecodeError:
            logger.warning("Error decodificando indicators-config JSON. Usando configuración por defecto.")
    
    ohlcv_config_dict = None
    if args.ohlcv_config:
        try:
            ohlcv_config_dict = json.loads(args.ohlcv_config)
        except json.JSONDecodeError:
            logger.warning("Error decodificando ohlcv-config JSON. Usando configuración por defecto.")
    
    final_feature_columns = None
    if args.feature_columns:
        try:
            final_feature_columns = json.loads(args.feature_columns)
        except json.JSONDecodeError:
            logger.warning("Error decodificando feature-columns JSON. Usando configuración por defecto.")
    
    try:
        logger.info(f"Iniciando preprocesamiento de datos desde {args.input_file_gcs}")
        logger.info(f"Configuración: L={args.sequence_length}, norm_mult={args.norm_window_multiplier}")
        
        # Inicializar el preprocesador
        preprocessor = DataPreprocessorCloud(
            project_id=args.project_id,
            raw_data_bucket=args.raw_data_bucket,
            processed_data_bucket=args.processed_data_bucket,
            sequence_length_L=args.sequence_length,
            norm_window_multiplier=args.norm_window_multiplier,
            indicators_config_dict=indicators_config_dict,
            ohlcv_config_dict=ohlcv_config_dict,
            final_market_feature_columns=final_feature_columns,
            use_float32=True
        )
        
        # Procesar los datos
        output_npz_path = preprocessor.process_data(
            raw_data_gcs_path=args.input_file_gcs,
            output_gcs_prefix=None if not args.output_gcs_path else os.path.dirname(args.output_gcs_path)
        )
        
        if output_npz_path:
            logger.info(f"Preprocesamiento completado exitosamente. Secuencias guardadas en: {output_npz_path}")
            
            # Para integrarse con Vertex AI Pipelines
            output_file = args.output_gcs_path if args.output_gcs_path else output_npz_path
            # Si el output_file ya viene predefinido por el pipeline, guardar un archivo de output.txt
            if args.output_gcs_path and args.output_gcs_path != output_npz_path:
                logger.info(f"Actualizando ruta de salida del Pipeline: {output_file}")
                with open('/tmp/output.txt', 'w') as f:
                    f.write(output_npz_path)
            
            print(f"SUCCESS:{output_npz_path}")
            return 0
        else:
            logger.error("Error en el preprocesamiento de datos")
            return 1
            
    except Exception as e:
        logger.exception(f"Error no controlado: {e}")
        return 1

if __name__ == "__main__":
    exit(main())