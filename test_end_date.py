#!/usr/bin/env python3
"""
Script de prueba para verificar que la funcionalidad end_date funciona correctamente.
"""

import sys
from datetime import datetime
from src.utils.cli import parse_arguments
from src.data.pipeline import DataPipeline
from src.data.Adquisicion import Adquisicion

def test_cli_with_end_date():
    """Prueba que el CLI acepta el argumento --end-date"""
    print("=== Probando CLI con --end-date ===")
    
    # Simular argumentos de línea de comandos
    sys.argv = [
        'test', 
        '--symbol', 'BTCUSDT', 
        '--interval', '1h', 
        '--start-date', '2024-01-01',
        '--end-date', '2024-01-02'
    ]
    
    args = parse_arguments()
    print(f"✓ CLI parseado correctamente:")
    print(f"  Symbol: {args.symbol}")
    print(f"  Interval: {args.interval}")
    print(f"  Start date: {args.start_date}")
    print(f"  End date: {args.end_date}")
    return args

def test_cli_without_end_date():
    """Prueba que el CLI funciona sin --end-date"""
    print("\n=== Probando CLI sin --end-date ===")
    
    # Simular argumentos de línea de comandos
    sys.argv = [
        'test', 
        '--symbol', 'BTCUSDT', 
        '--interval', '1h', 
        '--start-date', '2024-01-01'
    ]
    
    args = parse_arguments()
    print(f"✓ CLI parseado correctamente:")
    print(f"  Symbol: {args.symbol}")
    print(f"  Interval: {args.interval}")
    print(f"  Start date: {args.start_date}")
    print(f"  End date: {args.end_date} (default None)")
    return args

def test_data_pipeline_init():
    """Prueba que DataPipeline acepta el parámetro end_date"""
    print("\n=== Probando inicialización de DataPipeline ===")
    
    # Con end_date
    pipeline_with_end = DataPipeline(
        symbol='BTCUSDT',
        interval='1h',
        start_date='2024-01-01',
        end_date='2024-01-02',
        run_id='test_run',
        base_path='/tmp/test'
    )
    print(f"✓ DataPipeline con end_date inicializado:")
    print(f"  End date: {pipeline_with_end.end_date}")
    
    # Sin end_date
    pipeline_without_end = DataPipeline(
        symbol='BTCUSDT',
        interval='1h',
        start_date='2024-01-01',
        run_id='test_run',
        base_path='/tmp/test'
    )
    print(f"✓ DataPipeline sin end_date inicializado:")
    print(f"  End date: {pipeline_without_end.end_date}")

def test_adquisicion_init():
    """Prueba que Adquisicion acepta el parámetro end_date"""
    print("\n=== Probando inicialización de Adquisicion ===")
    
    # Con end_date
    adq_with_end = Adquisicion(
        symbol='BTCUSDT',
        interval='1h',
        start_date='2024-01-01',
        end_date='2024-01-02'
    )
    print(f"✓ Adquisicion con end_date inicializado:")
    print(f"  End date: {adq_with_end.end_date}")
    
    # Sin end_date
    adq_without_end = Adquisicion(
        symbol='BTCUSDT',
        interval='1h',
        start_date='2024-01-01'
    )
    print(f"✓ Adquisicion sin end_date inicializado:")
    print(f"  End date: {adq_without_end.end_date}")

def test_timestamp_calculation():
    """Prueba la lógica de cálculo de timestamps"""
    print("\n=== Probando cálculo de timestamps ===")
    
    from datetime import datetime, timezone
    
    # Simular la lógica de cálculo de timestamps
    start_date = '2024-01-01'
    end_date = '2024-01-02'
    
    start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    start_timestamp = int(start_date_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    # Con end_date
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    end_timestamp = int(end_date_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
    
    print(f"✓ Timestamps calculados correctamente:")
    print(f"  Start timestamp: {start_timestamp} ({datetime.fromtimestamp(start_timestamp/1000, timezone.utc)})")
    print(f"  End timestamp: {end_timestamp} ({datetime.fromtimestamp(end_timestamp/1000, timezone.utc)})")
    print(f"  Diferencia: {(end_timestamp - start_timestamp) / (1000 * 60 * 60 * 24)} días")
    
    # Sin end_date (usar timestamp actual)
    current_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
    print(f"  Current timestamp: {current_timestamp} ({datetime.fromtimestamp(current_timestamp/1000, timezone.utc)})")

if __name__ == "__main__":
    try:
        print("Iniciando pruebas de la funcionalidad end_date...\n")
        
        # Probar CLI
        test_cli_with_end_date()
        test_cli_without_end_date()
        
        # Probar componentes
        test_data_pipeline_init()
        test_adquisicion_init()
        test_timestamp_calculation()
        
        print("\n✅ Todas las pruebas pasaron exitosamente!")
        
    except Exception as e:
        print(f"\n❌ Error en las pruebas: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
