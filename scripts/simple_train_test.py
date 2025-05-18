#!/usr/bin/env python3
"""
Script simplificado para ejecutar un entrenamiento básico del agente sin KFP.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import argparse

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Asegurar que el paquete src está en el path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """Ejecutar un test de entrenamiento simplificado."""
    parser = argparse.ArgumentParser(description='Ejecutar un test básico de entrenamiento')
    parser.add_argument('--steps', type=int, default=100,
                        help='Número de pasos de entrenamiento')
    parser.add_argument('--temp-dir', type=str, default='./tmp',
                        help='Directorio temporal para datos de prueba')
    args = parser.parse_args()
    
    # Crear directorios para datos simulados
    temp_dir = Path(args.temp_dir)
    data_dir = temp_dir / 'data' / 'processed' / 'train'
    model_dir = temp_dir / 'models'
    
    for directory in [data_dir, model_dir]:
        os.makedirs(directory, exist_ok=True)
    
    print(f"🏠 Directorio de datos: {temp_dir}")
    
    # Crear datos simulados para pruebas
    print("\n🔄 Creando datos simulados para prueba...")
    try:
        # Crear datos simulados de precios de BTC
        date_range = pd.date_range('2022-01-01', '2022-01-31', freq='1h')
        n_samples = len(date_range)
        
        # Generar precios simulados
        prices = np.linspace(40000, 45000, n_samples) + np.random.normal(0, 1000, n_samples)
        
        # Crear DataFrame con datos OHLCV y algunos indicadores técnicos básicos
        df = pd.DataFrame({
            'timestamp': date_range,
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.02, n_samples)),
            'low': prices * (1 - np.random.uniform(0, 0.02, n_samples)),
            'close': prices * (1 + np.random.normal(0, 0.01, n_samples)),
            'volume': np.random.uniform(100, 1000, n_samples),
            'sma_20': np.convolve(prices, np.ones(20)/20, mode='same'),
            'sma_50': np.convolve(prices, np.ones(50)/50, mode='same'),
            'rsi_14': np.random.uniform(30, 70, n_samples),
            'atr_14': np.random.uniform(100, 500, n_samples),
            'macd': np.random.normal(0, 200, n_samples),
            'macd_signal': np.random.normal(0, 200, n_samples),
            'macd_hist': np.random.normal(0, 50, n_samples),
        })
        
        # Normalizar los datos
        for col in df.columns:
            if col not in ['timestamp']:
                df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()
        
        # Guardar como parquet
        df.to_parquet(data_dir / 'btc_data.parquet', index=False)
        print(f"✅ Datos simulados guardados en {data_dir / 'btc_data.parquet'}")
        
    except Exception as e:
        print(f"❌ Error creando datos simulados: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Importar directamente módulos necesarios
    print("\n🧠 Creando entorno de trading y modelo...")
    try:
        from src.environments.trading_env import TradingEnvironment
        from src.agent.custom_transformer_extractor import CustomTransformerFeatureExtractor
        from stable_baselines3 import SAC
        from stable_baselines3.common.policies import ActorCriticPolicy
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
        
        # Configurar entorno de trading
        env_config = {
            "project_id": "local-project",
            "gcs_processed_data_uri": str(data_dir),
            "initial_balance_usd": 10000.0,
            "max_position_btc": 1.0,
            "commission_rate": 0.0004,
            "max_leverage": 5,
            "random_episode_start": True,
            "episode_steps": 100,
            "slippage_model": "atr_based",
            "slippage_factor": 0.05
        }
        
        print(f"Creando entorno con config: {env_config}")
        
        # Crear entorno vectorizado
        def make_env():
            env = TradingEnvironment(**env_config)
            return env
        
        print("Creando entorno vectorizado...")
        env = DummyVecEnv([make_env])
        env = VecMonitor(env)
        print("Entorno creado correctamente.")
        
        # Configurar feature extractor personalizado
        policy_kwargs = {
            "features_extractor_class": CustomTransformerFeatureExtractor,
            "features_extractor_kwargs": {
                "d_model": 32,      # Dimensión del modelo transformer
                "n_heads": 2,       # Número de cabezas de atención
                "n_encoder_layers": 2,  # Número de capas del encoder
                "dim_feedforward": 64,  # Dimensión de la capa feedforward
                "dropout": 0.1,     # Dropout
                "activation": "gelu", # Función de activación
                "features_dim": 64   # Dimensión de características de salida
            },
            "net_arch": [64, 64]  # Arquitectura de la red (actor y crítico)
        }
        
        # Crear modelo SAC
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.0003,
            buffer_size=10000,
            learning_starts=100,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(temp_dir / "tensorboard"),
            verbose=1
        )
        
        # Entrenar modelo
        print(f"\n🏃‍♂️ Entrenando modelo SAC por {args.steps} pasos...")
        model.learn(total_timesteps=args.steps)
        
        # Guardar modelo entrenado
        model_path = model_dir / "sac_model.zip"
        model.save(model_path)
        print(f"\n✅ Modelo guardado en {model_path}")
        
        # Probar con algunas inferencias
        print("\n🧪 Probando inferencias con el modelo entrenado...")
        obs = env.reset()
        for i in range(10):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            print(f"Step {i+1}: Action={action[0]:.4f}, Reward={rewards[0]:.4f}")
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Test de entrenamiento completado!")

if __name__ == "__main__":
    main()
