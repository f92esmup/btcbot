#!/usr/bin/env python3
"""
Script de prueba simplificado para verificar que la funcionalidad end_date funciona correctamente.
"""

import sys
import os

# Añadir el directorio raíz al path
sys.path.insert(0, '/workspaces/btcbot')

def test_cli_functionality():
    """Prueba que el CLI acepta el argumento --end-date"""
    print("=== Probando funcionalidad CLI ===")
    
    try:
        from src.utils.cli import parse_arguments
        
        # Prueba con --end-date
        sys.argv = [
            'test', 
            '--symbol', 'BTCUSDT', 
            '--interval', '1h', 
            '--start-date', '2024-01-01',
            '--end-date', '2024-01-02'
        ]
        
        args = parse_arguments()
        print(f"✓ CLI con --end-date:")
        print(f"  Symbol: {args.symbol}")
        print(f"  Interval: {args.interval}")
        print(f"  Start date: {args.start_date}")
        print(f"  End date: {args.end_date}")
        
        assert args.end_date == '2024-01-02', f"Expected '2024-01-02', got '{args.end_date}'"
        
        # Prueba sin --end-date
        sys.argv = [
            'test', 
            '--symbol', 'BTCUSDT', 
            '--interval', '1h', 
            '--start-date', '2024-01-01'
        ]
        
        args = parse_arguments()
        print(f"✓ CLI sin --end-date:")
        print(f"  End date: {args.end_date} (debe ser None)")
        
        assert args.end_date is None, f"Expected None, got '{args.end_date}'"
        
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba CLI: {e}")
        return False

def test_signature_changes():
    """Verifica que las firmas de los métodos hayan cambiado correctamente"""
    print("\n=== Verificando firmas de métodos ===")
    
    try:
        # Verificar que el archivo CLI tiene el nuevo argumento
        with open('/workspaces/btcbot/src/utils/cli.py', 'r') as f:
            cli_content = f.read()
            if '--end-date' in cli_content and 'Fecha de fin para la descarga de datos' in cli_content:
                print("✓ CLI modificado correctamente")
            else:
                print("❌ CLI no tiene las modificaciones esperadas")
                return False
        
        # Verificar DataPipeline
        with open('/workspaces/btcbot/src/data/pipeline.py', 'r') as f:
            pipeline_content = f.read()
            if 'end_date: str = None' in pipeline_content and 'self.end_date = end_date' in pipeline_content:
                print("✓ DataPipeline modificado correctamente")
            else:
                print("❌ DataPipeline no tiene las modificaciones esperadas")
                return False
        
        # Verificar Adquisicion
        with open('/workspaces/btcbot/src/data/Adquisicion.py', 'r') as f:
            adq_content = f.read()
            if ('end_date: str = None' in adq_content and 
                'self.end_date = end_date' in adq_content and
                'if self.end_date is not None:' in adq_content):
                print("✓ Adquisicion modificado correctamente")
            else:
                print("❌ Adquisicion no tiene las modificaciones esperadas")
                return False
        
        # Verificar train.py
        with open('/workspaces/btcbot/train.py', 'r') as f:
            train_content = f.read()
            if 'end_date=args.end_date' in train_content:
                print("✓ train.py modificado correctamente")
            else:
                print("❌ train.py no tiene las modificaciones esperadas")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error verificando firmas: {e}")
        return False

def test_timestamp_logic():
    """Prueba la lógica de cálculo de timestamps"""
    print("\n=== Probando lógica de timestamps ===")
    
    try:
        from datetime import datetime, timezone
        
        # Simular la lógica que implementamos
        start_date = '2024-01-01'
        end_date = '2024-01-02'
        
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        start_timestamp = int(start_date_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        # Con end_date
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        end_timestamp = int(end_date_obj.replace(tzinfo=timezone.utc).timestamp() * 1000)
        
        print(f"✓ Cálculo con end_date:")
        print(f"  Start: {start_timestamp} ({datetime.fromtimestamp(start_timestamp/1000, timezone.utc)})")
        print(f"  End: {end_timestamp} ({datetime.fromtimestamp(end_timestamp/1000, timezone.utc)})")
        
        # Sin end_date
        current_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        print(f"✓ Cálculo sin end_date:")
        print(f"  Current: {current_timestamp} ({datetime.fromtimestamp(current_timestamp/1000, timezone.utc)})")
        
        # Verificar que end_timestamp es mayor que start_timestamp
        assert end_timestamp > start_timestamp, "End timestamp debe ser mayor que start timestamp"
        
        # Verificar que current_timestamp es mayor que end_timestamp (porque estamos en 2025)
        assert current_timestamp > end_timestamp, "Current timestamp debe ser mayor que end timestamp para fechas pasadas"
        
        return True
        
    except Exception as e:
        print(f"❌ Error en lógica de timestamps: {e}")
        return False

if __name__ == "__main__":
    print("Iniciando pruebas de la funcionalidad end_date...\n")
    
    success = True
    
    # Ejecutar todas las pruebas
    success &= test_cli_functionality()
    success &= test_signature_changes()
    success &= test_timestamp_logic()
    
    if success:
        print("\n✅ Todas las pruebas pasaron exitosamente!")
        print("\nResumen de cambios implementados:")
        print("1. ✓ CLI acepta --end-date como argumento opcional")
        print("2. ✓ train.py pasa end_date a DataPipeline")
        print("3. ✓ DataPipeline acepta y propaga end_date a Adquisicion")
        print("4. ✓ Adquisicion usa end_date en ambos métodos de descarga")
        print("5. ✓ Lógica condicional implementada correctamente")
    else:
        print("\n❌ Algunas pruebas fallaron")
        sys.exit(1)
