#!/usr/bin/env python3
"""
Script para ejecutar un entrenamiento de prueba directo sin usar el pipeline.
Se centra solo en el componente de entrenamiento con datos de prueba.
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path

# Asegurar que el paquete src está en el path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """Ejecutar un test de entrenamiento directo."""
    parser = argparse.ArgumentParser(description='Ejecutar un test de entrenamiento del agente')
    parser.add_argument('--project-id', type=str, default='local-project',
                        help='ID del proyecto (no usado en local)')
    parser.add_argument('--training-steps', type=int, default=1000,
                        help='Número de pasos de entrenamiento')
    parser.add_argument('--temp-dir', type=str, default='./tmp',
                        help='Directorio temporal para datos de prueba')
    args = parser.parse_args()
    
    # Crear directorios para datos simulados
    temp_dir = Path(args.temp_dir)
    train_data_dir = temp_dir / 'data' / 'processed' / 'train'
    model_dir = temp_dir / 'models'
    metrics_dir = temp_dir / 'metrics'
    
    for directory in [train_data_dir, model_dir, metrics_dir]:
        os.makedirs(directory, exist_ok=True)
    
    print(f"🏠 Directorio de datos: {temp_dir}")
    
    # Crear datos simulados para pruebas
    print("\n🔄 Creando datos simulados para prueba...")
    try:
        import numpy as np
        import pandas as pd
        
        # Crear datos simulados de precios de BTC
        date_range = pd.date_range('2022-01-01', '2022-01-31', freq='1h')
        n_samples = len(date_range)
        
        # Generar precios simulados
        prices = np.linspace(40000, 45000, n_samples) + np.random.normal(0, 1000, n_samples)
        
        # Crear DataFrame con datos OHLCV
        df = pd.DataFrame({
            'timestamp': date_range,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.02, n_samples)),
            'low': prices * (1 - np.random.uniform(0, 0.02, n_samples)),
            'close': prices * (1 + np.random.normal(0, 0.01, n_samples)),
            'volume': np.random.uniform(100, 1000, n_samples)
        })
        
        # Guardar como parquet
        df.to_parquet(train_data_dir / 'btc_data.parquet', index=False)
        print(f"✅ Datos simulados guardados en {train_data_dir / 'btc_data.parquet'}")
        
        # Crear archivo de metadata
        with open(train_data_dir / 'metadata.json', 'w') as f:
            json.dump({
                'file_path': str(train_data_dir / 'btc_data.parquet'),
                'rows': len(df),
                'columns': list(df.columns)
            }, f)
    except Exception as e:
        print(f"❌ Error creando datos simulados: {e}")
        return
    
    # Definir parámetros del entorno
    env_params = {
        "project_id": args.project_id,
        "gcs_processed_data_uri": str(train_data_dir),
        "initial_balance_usd": 10000.0,
        "max_position_btc": 1.0,
        "commission_rate": 0.0004,
        "max_leverage": 20,
        "random_episode_start": True,
        "episode_steps": 100,
        "slippage_model": "atr_based",
        "slippage_factor": 0.05
    }
    
    # Definir parámetros del transformador
    transformer_params = {
        "n_heads": 2,
        "n_layers": 2,
        "d_model": 32
    }
    
    # Ejecutar componente de entrenamiento
    print("\n🧠 Ejecutando entrenamiento del agente...")
    try:
        # Importar el componente de entrenamiento
        from src.components import run_train_agent
        
        # Crear directorios adicionales
        tensorboard_dir = temp_dir / 'tensorboard'
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        # Configurar argumentos con los nombres correctos
        training_args = [
            '--project_id', args.project_id,
            '--input_data_uri', str(train_data_dir),
            '--output_model_dir', str(model_dir),
            '--output_tensorboard_dir', str(tensorboard_dir),
            '--gcs_bucket', 'local-bucket',  # Valor ficticio para pruebas locales
            '--total_timesteps', str(args.training_steps),
            '--trained_model_output_path', str(model_dir / 'metadata.json'),
            '--tensorboard_log_output_path', str(tensorboard_dir / 'tb_logs.json'),
            '--training_metrics_output_path', str(metrics_dir / 'training_metrics.json'),
            '--initial_balance_usd', '10000',
            '--max_position_btc', '1.0',
            '--commission_rate', '0.0004',
            '--random_episode_start', 'True',
            '--episode_steps', '100',
            '--d_model', '32',
            '--n_heads', '2',
            '--n_encoder_layers', '2',
        ]
        
        print(f"Ejecutando con argumentos: {' '.join(training_args)}")
        
        # Guardar sys.argv original y reemplazarlo
        original_argv = sys.argv
        sys.argv = [sys.argv[0]] + training_args
        
        try:
            # Llamada al método main() sin parámetros
            run_train_agent.main()
        finally:
            # Restaurar sys.argv
            sys.argv = original_argv
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Test de entrenamiento completado!")

if __name__ == "__main__":
    main()
