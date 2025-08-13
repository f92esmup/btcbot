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

import sys
sys.path.append('.')

import logging

# Import del pipeline principal
from src.data import (
    ArtifactManager,
    BinanceDataSource,
    DataPipeline
)
from src.config import (
    parse_dataset_arguments, 
    setup_logging,
    validate_date_format,
    load_system_config,
    generate_data_run_id,
    create_data_run_metadata
)

def main() -> None:
    """Función principal del script de creación de datasets."""
    # Parsear argumentos primero (para que --help funcione sin logging)
    args = parse_dataset_arguments()

    # Configuración global de logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    
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

    # Cargar configuración del sistema
    logger.info("\n📦 CARGANDO CONFIGURACIÓN DEL SISTEMA:")
    system_config = load_system_config(args.config_path)

    # Mostrar parámetros del experimento
    logger.info("📋 PARÁMETROS DEL DATASET:")
    logger.info(f"  • Símbolo: {args.symbol}")
    logger.info(f"  • Intervalo: {args.interval}")
    logger.info(f"  • Fecha inicio: {args.start_date}")
    logger.info(f"  • Fecha fin: {args.end_date if args.end_date else 'Hasta ahora'}")
    logger.info(f"  • Ruta de configuración: {args.config_path}")

    # Generar data_run_id único
    data_run_id = generate_data_run_id(args.symbol, args.interval, args.start_date, args.end_date)
    logger.info(f"  • Data Run ID: {data_run_id}")
    
    # Definir ruta base para el data_run
    data_run_path = system_config.base.dir.data_runs / data_run_id
    logger.info(f"  • Ruta de salida: {data_run_path}")

    # Crear metadatos del data_run 
    logger.info("\n📝 CREANDO METADATOS DEL DATASET:")
    data_run_metadata = create_data_run_metadata(
        args.symbol, args.interval, args.start_date, args.end_date, data_run_id
    )

    # Configurar gestores especializados para el data_run
    logger.info("\n🗂️  CONFIGURANDO GESTORES DE ARCHIVOS:")
    
    artifact_manager = ArtifactManager(
        config=system_config  # Pasar configuración completa como un objeto pydantic
    )

    # Crear y ejecutar el pipeline de datos
    logger.info("\n🚀 EJECUTANDO PIPELINE DE DATOS:")
    logger.info("-" * 40)
    
    try:
        # Crear fuente de datos de Binance (Dependency Injection)
        binance_data_source = BinanceDataSource(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            config=system_config,
            end_date=args.end_date
        )
        
        # Instanciar el pipeline de datos con la fuente de datos inyectada
        data_pipeline = DataPipeline(
            data_source=binance_data_source,  # Inyección de dependencia
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date,
            run_id=data_run_id,
            base_path=data_run_path,
            config=system_config
        )
        
        # Ejecutar el pipeline
        normalized_dataframe, scaler, price_scaler = data_pipeline.run()
        
        # Guardar todos los artefactos usando el ArtifactManager
        logger.info("💾 GUARDANDO ARTEFACTOS DEL DATASET:")
        artifact_save_success = artifact_manager.save_all_artifacts(
            data_run_id=data_run_id,
            normalized_dataframe=normalized_dataframe,
            scaler=scaler,
            price_scaler=price_scaler,
            dataset_metadata=data_run_metadata
        )

        if not artifact_save_success:
            logger.error("  ❌ Error guardando artefactos del dataset")
            sys.exit(1)
        
        logger.info("  ✅ Todos los artefactos guardados exitosamente")
        
        logger.info("-" * 40)
        logger.info("🎉 DATASET CREADO EXITOSAMENTE")
        logger.info("=" * 60)
        
        # Mostrar resumen final
        logger.info("📊 RESUMEN DEL DATASET CREADO:")
        logger.info(f"  • Data Run ID: {data_run_id}")
        logger.info(f"  • Ubicación: {data_run_path}")
        logger.info(f"  • Forma del DataFrame: {normalized_dataframe.shape}")
        logger.info(f"  • Rango temporal: {normalized_dataframe.index.min()} → {normalized_dataframe.index.max()}")
        
        logger.info("\n📁 ARTEFACTOS GENERADOS:")
        logger.info(f"  • {system_config.base.artifacts.dataset_metadata} - Metadatos del dataset")
        logger.info(f"  • {system_config.base.artifacts.normalized_dataframe} - Datos normalizados")
        logger.info(f"  • {system_config.base.artifacts.scaler} - Escalador de características")
        logger.info(f"  • {system_config.base.artifacts.price_scaler} - Escalador de precios")

        logger.info(f"\n✨ El dataset '{data_run_id}' está listo para ser utilizado en entrenamientos!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"\n❌ ERROR DURANTE LA EJECUCIÓN DEL PIPELINE:")
        logger.error(f"   {str(e)}")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
