#!/usr/bin/env python3
"""
Script dedicado para la creación de datasets.

Este script separa completamente la generación de datos del entrenamiento,
creando artefactos inmutables y versionados que pueden ser reutilizados
por múltiples entrenamientos.

Uso:
    python create_dataset.py --symbol BTCUSDT --interval 1m --start-date 2025-01-01
    python create_dataset.py --symbol BTCUSDT --interval 1m --start-date 2025-01-01 --end-date 2025-06-30
"""

import argparse
import logging
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from src.data.pipeline import DataPipeline
from src.training.run_manager import RunManager
from src.utils.validation import validate_date_format
from src.configuration.secret_utils import SecretManagerUtils


def parse_arguments() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos específicos para la creación de datasets."""
    parser = argparse.ArgumentParser(
        description='Script dedicado para la creación de datasets inmutables y versionados',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  %(prog)s --symbol BTCUSDT --interval 1m --start-date 2025-01-01
  %(prog)s --symbol BTCUSDT --interval 1m --start-date 2025-01-01 --end-date 2025-06-30
  %(prog)s --symbol ETHUSDT --interval 4h --start-date 2024-01-01
        """
    )
    
    # Argumentos requeridos
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Símbolo del par de trading (ej: BTCUSDT, ETHUSDT)'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        required=True,
        help='Intervalo de tiempo para las velas (ej: 1m, 5m, 15m, 1h, 4h, 1d)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Fecha de inicio en formato YYYY-MM-DD (ej: 2025-01-01)'
    )
    
    # Argumentos opcionales
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='Fecha de fin en formato YYYY-MM-DD (opcional, si no se especifica se usa hasta ahora)'
    )
    
    return parser.parse_args()


def generate_data_run_id(symbol: str, interval: str, start_date: str, end_date: Optional[str] = None) -> str:
    """
    Genera un identificador único para el data_run.
    
    Args:
        symbol: Símbolo del par de trading
        interval: Intervalo de tiempo
        start_date: Fecha de inicio
        end_date: Fecha de fin (opcional)
        
    Returns:
        str: ID único del data_run (ej: BTCUSDT_1m_20250101_20250703-153000)
    """
    timestamp = datetime.now().strftime("%H%M%S")
    date_str = start_date.replace("-", "")
    
    if end_date:
        end_date_str = end_date.replace("-", "")
        return f"{symbol}_{interval}_{date_str}_{end_date_str}-{timestamp}"
    else:
        today_str = datetime.now().strftime("%Y%m%d")
        return f"{symbol}_{interval}_{date_str}_{today_str}-{timestamp}"


def load_system_config() -> Dict[str, Any]:
    """
    Carga la configuración principal del sistema desde config.yaml.
    
    Returns:
        Dict[str, Any]: Configuración completa del sistema
        
    Raises:
        SystemExit: Si no se puede cargar la configuración
    """
    config_path = Path('src/configuration/config.yaml')
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuración del sistema cargada desde: {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"No se encontró el archivo de configuración: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error al parsear el archivo YAML: {e}")
        sys.exit(1)


def create_data_run_metadata(symbol: str, interval: str, start_date: str, 
                           end_date: Optional[str], data_run_id: str) -> Dict[str, Any]:
    """
    Crea el diccionario de metadatos para el data_run.
    
    Args:
        symbol: Símbolo del par de trading
        interval: Intervalo de tiempo
        start_date: Fecha de inicio
        end_date: Fecha de fin (opcional)
        data_run_id: Identificador único del data_run
        
    Returns:
        Dict[str, Any]: Metadatos del data_run
    """
    metadata = {
        'data_run_info': {
            'data_run_id': data_run_id,
            'creation_timestamp': datetime.now().isoformat(),
            'created_by': 'create_dataset.py',
            'description': f'Dataset inmutable para {symbol} ({interval}) desde {start_date}'
        },
        'experiment_parameters': {
            'symbol': symbol,
            'interval': interval,
            'start_date': start_date,
            'end_date': end_date,
            'data_source': 'Binance API'
        },
        'data_pipeline_version': {
            'script_version': '1.0.0',
            'pipeline_modules': [
                'src.data.Adquisicion',
                'src.data.indicadores', 
                'src.data.normalization'
            ]
        }
    }
    
    return metadata


def setup_logging() -> logging.Logger:
    """Configura el sistema de logging para el script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def main() -> None:
    """Función principal del script de creación de datasets."""
    # Parsear argumentos primero (para que --help funcione sin logging)
    args = parse_arguments()
    
    # Configurar logging
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("🔧 INICIANDO CREACIÓN DE DATASET INMUTABLE")
    logger.info("=" * 60)
    
    # Validar formato de fechas
    if not validate_date_format(args.start_date):
        logger.error(f"Formato de fecha de inicio inválido: {args.start_date}. Use YYYY-MM-DD")
        sys.exit(1)
        
    if args.end_date and not validate_date_format(args.end_date):
        logger.error(f"Formato de fecha de fin inválido: {args.end_date}. Use YYYY-MM-DD")
        sys.exit(1)
    
    # Mostrar parámetros del experimento
    logger.info("📋 PARÁMETROS DEL DATASET:")
    logger.info(f"  • Símbolo: {args.symbol}")
    logger.info(f"  • Intervalo: {args.interval}")
    logger.info(f"  • Fecha inicio: {args.start_date}")
    logger.info(f"  • Fecha fin: {args.end_date if args.end_date else 'Hasta ahora'}")
    
    # Generar data_run_id único
    data_run_id = generate_data_run_id(args.symbol, args.interval, args.start_date, args.end_date)
    logger.info(f"  • Data Run ID: {data_run_id}")
    
    # Definir ruta base para el data_run
    data_run_path = f"data_runs/{data_run_id}"
    logger.info(f"  • Ruta de salida: {data_run_path}")
    
    # Cargar configuración del sistema
    logger.info("\n📦 CARGANDO CONFIGURACIÓN DEL SISTEMA:")
    system_config = load_system_config()
    
    # Crear metadatos del data_run
    logger.info("\n📝 CREANDO METADATOS DEL DATASET:")
    data_run_metadata = create_data_run_metadata(
        args.symbol, args.interval, args.start_date, args.end_date, data_run_id
    )
    
    # Configurar RunManager para el data_run
    logger.info("\n🗂️  CONFIGURANDO GESTOR DE ARCHIVOS:")
    
    # Determinar modo de almacenamiento basado en la configuración
    storage_mode = system_config.get('normalization', {}).get('storage_mode', 'local')
    gcp_config = system_config.get('gcp', {}) if storage_mode == 'gcp' else None
    
    run_manager = RunManager(
        storage_mode=storage_mode,
        gcp_config=gcp_config
    )
    
    # Guardar metadatos del data_run usando el RunManager
    logger.info("💾 GUARDANDO METADATOS DEL DATASET:")
    metadata_filename = "data_run_metadata.yaml"
    
    if storage_mode == "gcp":
        # Para GCP, guardamos temporalmente y subimos usando RunManager
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            yaml.dump(data_run_metadata, temp_file, default_flow_style=False, allow_unicode=True)
            temp_path = temp_file.name
        
        try:
            data_run_prefix = run_manager._get_data_run_prefix(data_run_id)
            gcs_blob_name = f"{data_run_prefix}/{metadata_filename}"
            if run_manager.gcs_utils.upload_file_to_gcs(temp_path, gcs_blob_name):
                logger.info(f"  ✅ Metadatos guardados en GCS: gs://{run_manager.gcs_bucket_name}/{gcs_blob_name}")
            else:
                logger.error("  ❌ Error guardando metadatos en GCS")
                sys.exit(1)
        finally:
            os.unlink(temp_path)
    else:
        # Para almacenamiento local usando el prefix del RunManager
        data_run_prefix = run_manager._get_data_run_prefix(data_run_id)
        metadata_path = Path(data_run_prefix) / metadata_filename
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_run_metadata, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"  ✅ Metadatos guardados en: {metadata_path}")
    
    # Configurar credenciales de API si es necesario
    api_key = None
    api_secret = None
    
    if system_config.get('gcp'):
        logger.info("\n🔐 CONFIGURANDO CREDENCIALES DE API:")
        try:
            secret_manager = SecretManagerUtils(system_config['gcp'])
            
            # Determinar qué credenciales usar basado en el modo de trading
            trading_config = system_config.get('trading', {})
            is_testnet = trading_config.get('testnet', True)
            
            if is_testnet:
                api_key = secret_manager.get_secret(system_config['gcp']['secrets']['testnet_binance_api_key_futures'])
                api_secret = secret_manager.get_secret(system_config['gcp']['secrets']['testnet_binance_api_secret_futures'])
                logger.info("  ✅ Credenciales de testnet cargadas")
            else:
                api_key = secret_manager.get_secret(system_config['gcp']['secrets']['binance_api_key_futures'])
                api_secret = secret_manager.get_secret(system_config['gcp']['secrets']['binance_api_secret_futures'])
                logger.info("  ✅ Credenciales de producción cargadas")
                
        except Exception as e:
            logger.warning(f"  ⚠️  No se pudieron cargar las credenciales de API: {e}")
            logger.warning("  ⚠️  Se continuará sin credenciales (puede afectar límites de API)")
    
    # Crear y ejecutar el pipeline de datos
    logger.info("\n🚀 EJECUTANDO PIPELINE DE DATOS:")
    logger.info("-" * 40)
    
    try:
        # Configurar GCSUtils si es necesario
        gcs_utils = None
        if storage_mode == "gcp":
            from src.configuration.gcs_utils import GCSUtils
            gcs_utils = GCSUtils(gcp_config)
        
        # Instanciar el pipeline de datos
        data_pipeline = DataPipeline(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date,
            run_id=data_run_id,
            base_path=data_run_path,
            full_config=system_config,
            save_artifacts=True,  # Siempre guardar artefactos en creación de datasets
            api_key=api_key,
            api_secret=api_secret,
            gcs_utils=gcs_utils
        )
        
        # Ejecutar el pipeline
        normalized_dataframe, price_scaler = data_pipeline.run()
        
        # Guardar el DataFrame normalizado
        logger.info("💾 GUARDANDO DATAFRAME NORMALIZADO:")
        if storage_mode == "gcp":
            # Para GCP, guardamos temporalmente y subimos
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
                normalized_dataframe.to_pickle(temp_file.name)
                temp_path = temp_file.name
            
            try:
                gcs_blob_name = f"{data_run_id}/normalized_dataframe.pkl"
                if run_manager.gcs_utils.upload_file_to_gcs(temp_path, gcs_blob_name):
                    logger.info(f"  ✅ DataFrame guardado en GCS: gs://{run_manager.gcs_bucket_name}/{gcs_blob_name}")
                else:
                    logger.error("  ❌ Error guardando DataFrame en GCS")
            finally:
                os.unlink(temp_path)
        else:
            # Para almacenamiento local
            dataframe_path = Path(data_run_path) / "normalized_dataframe.pkl"
            normalized_dataframe.to_pickle(dataframe_path)
            logger.info(f"  ✅ DataFrame guardado en: {dataframe_path}")
        
        logger.info("-" * 40)
        logger.info("🎉 DATASET CREADO EXITOSAMENTE")
        logger.info("=" * 60)
        
        # Mostrar resumen final
        logger.info("📊 RESUMEN DEL DATASET CREADO:")
        logger.info(f"  • Data Run ID: {data_run_id}")
        logger.info(f"  • Ubicación: {run_manager._get_data_run_prefix(data_run_id)}")
        logger.info(f"  • Forma del DataFrame: {normalized_dataframe.shape}")
        logger.info(f"  • Rango temporal: {normalized_dataframe.index.min()} → {normalized_dataframe.index.max()}")
        logger.info(f"  • Modo de almacenamiento: {storage_mode}")
        
        logger.info("\n📁 ARTEFACTOS GENERADOS:")
        logger.info(f"  • {metadata_filename} - Metadatos del dataset")
        logger.info("  • normalized_dataframe.pkl - Datos normalizados")
        logger.info("  • scaler.pkl - Escalador de características")
        logger.info("  • price_scaler.pkl - Escalador de precios")
        
        logger.info(f"\n✨ El dataset '{data_run_id}' está listo para ser utilizado en entrenamientos!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"\n❌ ERROR DURANTE LA EJECUCIÓN DEL PIPELINE:")
        logger.error(f"   {str(e)}")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
