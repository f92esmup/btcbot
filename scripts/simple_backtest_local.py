#!/usr/bin/env python3
"""
Script for basic backtesting of a trained model.

This script loads a trained model and performs backtesting on test data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from stable_baselines3 import SAC
import gymnasium as gym

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.trading_env import TradingEnvironment
from scripts.simple_train_test_local import LocalTradingEnvironment, create_dummy_ohlcv_data

def create_test_data(data_dir):
    """Create and save local test data for backtesting."""
    os.makedirs(os.path.join(data_dir, 'data/processed/test'), exist_ok=True)
    
    # Create dummy data with different seed
    df = create_dummy_ohlcv_data(seed=43)
    
    # Save to parquet file
    output_path = os.path.join(data_dir, 'data/processed/test/btc_data.parquet')
    df.to_parquet(output_path)
    
    print(f"✅ Datos de prueba guardados en {output_path}")
    return output_path

def run_backtest(model_path, data_path, data_dir='tmp'):
    """Run a backtest with the trained model."""
    try:
        # Environment configuration
        env_config = {
            'project_id': 'local-project',
            'gcs_processed_data_uri': data_path,
            'initial_balance_usd': 10000.0,
            'max_position_btc': 1.0,
            'commission_rate': 0.0004,
            'max_leverage': 5,
            'random_episode_start': False,  # Sequential for backtesting
            'episode_steps': 500,
            'sequence_length': 60,  # We'll pop this in LocalTradingEnvironment
            'slippage_model': 'atr_based',
            'slippage_factor': 0.05
        }
        
        # Create environment
        env = LocalTradingEnvironment(**env_config)
        
        # Load trained model
        model = SAC.load(model_path)
        
        # Run backtest
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        
        # Store history
        balance_history = []
        position_history = []
        action_history = []
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            # Record history
            balance_history.append(env.current_balance_usd)
            position_history.append(env.current_position_btc)
            action_history.append(action)
        
        # Create results directory
        results_dir = os.path.join(data_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Plot results
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Balance
        plt.subplot(3, 1, 1)
        plt.plot(balance_history)
        plt.title('Account Balance (USD)')
        plt.grid(True)
        
        # Plot 2: Position
        plt.subplot(3, 1, 2)
        plt.plot(position_history)
        plt.title('Position Size (BTC)')
        plt.grid(True)
        
        # Plot 3: Actions
        plt.subplot(3, 1, 3)
        plt.plot(action_history)
        plt.title('Actions')
        plt.grid(True)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'backtest_results.png'))
        
        # Save CSV
        results_df = pd.DataFrame({
            'balance': balance_history,
            'position': position_history,
            'action': action_history
        })
        results_df.to_csv(os.path.join(results_dir, 'portfolio_history.csv'))
        
        print(f"✅ Backtest completado con recompensa total: {total_reward}")
        print(f"Balance final: {balance_history[-1]} USD")
        print(f"Resultados guardados en {results_dir}")
        
        return results_dir
        
    except Exception as e:
        print(f"❌ Error en backtesting: {str(e)}")
        raise

def main():
    """Main function to run the backtesting."""
    try:
        print("🏠 Directorio de datos: tmp")
        data_dir = 'tmp'
        
        # Check if model exists
        model_path = os.path.join(data_dir, 'models/simple_sac_model')
        if not os.path.exists(model_path + '.zip'):
            print("❌ Modelo no encontrado. Primero debes entrenar el modelo.")
            return
        
        print("🔄 Creando datos de prueba para backtesting...")
        data_path = create_test_data(data_dir)
        
        print("📊 Ejecutando backtesting...")
        results_dir = run_backtest(model_path, data_path, data_dir)
        
        print(f"✅ Resultados guardados en {results_dir}")
        
    except Exception as e:
        print(f"❌ Error en el proceso de backtesting: {str(e)}")
    finally:
        print("✅ Test de backtesting completado!")

if __name__ == "__main__":
    main()
