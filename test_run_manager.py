#!/usr/bin/env python3
"""
Script de prueba para verificar la nueva API del RunManager refactorizado.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Agregar el directorio raíz al path para importar módulos
sys.path.insert(0, '/workspaces/btcbot')

from src.training.run_manager import RunManager

def test_basic_api():
    """Prueba básica de la nueva API del RunManager."""
    print("🧪 PRUEBA 1: Verificación Básica de la Nueva API")
    print("=" * 60)
    
    try:
        # 1. Crear RunManager con la nueva interfaz
        print("1️⃣ Creando RunManager en modo local...")
        run_manager = RunManager(storage_mode="local")
        print("   ✅ RunManager creado exitosamente")
        
        # 2. Probar helpers de rutas
        print("\n2️⃣ Probando helpers de construcción de rutas...")
        
        test_data_run_id = "data_20250703_123456"
        test_training_run_id = "training_20250703_123456"
        
        data_prefix = run_manager._get_data_run_prefix(test_data_run_id)
        training_prefix = run_manager._get_training_run_prefix(test_training_run_id)
        
        print(f"   📁 Data run prefix: {data_prefix}")
        print(f"   📁 Training run prefix: {training_prefix}")
        
        # Verificar que las rutas tienen el formato esperado
        assert data_prefix == f"data_runs/{test_data_run_id}", f"Esperado: data_runs/{test_data_run_id}, Obtenido: {data_prefix}"
        assert training_prefix == f"training_runs/{test_training_run_id}", f"Esperado: training_runs/{test_training_run_id}, Obtenido: {training_prefix}"
        print("   ✅ Helpers de rutas funcionan correctamente")
        
        print("\n🎉 PRUEBA 1 COMPLETADA EXITOSAMENTE")
        return True
        
    except Exception as e:
        print(f"   ❌ Error en prueba básica: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_operations():
    """Prueba las operaciones de configuración."""
    print("\n🧪 PRUEBA 2: Operaciones de Configuración")
    print("=" * 60)
    
    try:
        # Crear RunManager
        run_manager = RunManager(storage_mode="local")
        test_training_run_id = "test_training_20250703_config"
        
        # Crear configuración de prueba
        test_config = {
            'run_info': {
                'run_id': test_training_run_id,
                'timestamp': '2025-07-03T12:34:56',
                'storage_mode': 'local',
                'operation_mode': 'new_training'
            },
            'command_line_args': {
                'episodes': 1000,
                'data_run_id': 'test_data_run'
            },
            'config': {
                'agent': {'learning_rate': 0.001},
                'environment': {'initial_balance': 10000}
            }
        }
        
        print("1️⃣ Guardando configuración de training run...")
        run_manager.save_run_config(test_training_run_id, test_config)
        
        # Verificar que el archivo se creó
        expected_path = Path(f"training_runs/{test_training_run_id}/config_run.yaml")
        assert expected_path.exists(), f"Archivo de configuración no encontrado en: {expected_path}"
        print(f"   ✅ Configuración guardada en: {expected_path}")
        
        print("\n2️⃣ Cargando configuración con método estático...")
        loaded_config = RunManager.load_training_run_config(test_training_run_id, storage_mode="local")
        
        assert loaded_config is not None, "No se pudo cargar la configuración"
        assert loaded_config['run_info']['run_id'] == test_training_run_id, "Run ID no coincide"
        print("   ✅ Configuración cargada correctamente")
        
        # Limpiar archivos de prueba
        shutil.rmtree(f"training_runs/{test_training_run_id}", ignore_errors=True)
        print("   🧹 Archivos de prueba limpiados")
        
        print("\n🎉 PRUEBA 2 COMPLETADA EXITOSAMENTE")
        return True
        
    except Exception as e:
        print(f"   ❌ Error en pruebas de configuración: {e}")
        import traceback
        traceback.print_exc()
        # Limpiar en caso de error
        shutil.rmtree(f"training_runs/{test_training_run_id}", ignore_errors=True)
        return False

if __name__ == "__main__":
    print("🚀 INICIANDO PRUEBAS DEL RUNMANAGER REFACTORIZADO")
    print("=" * 60)
    
    # Ejecutar pruebas
    results = []
    results.append(test_basic_api())
    results.append(test_config_operations())
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, result in enumerate(results, 1):
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"Prueba {i}: {status}")
    
    print(f"\nResultado: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON!")
        sys.exit(0)
    else:
        print("❌ Algunas pruebas fallaron")
        sys.exit(1)
