#!/usr/bin/env python3
"""
Script for simple training test of the Bitcoin trading bot.

This script creates simulated data and performs a quick training test
using a simplified version of the full training pipeline.
"""

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging
import torch
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.trading_env import TradingEnvironment

def create_dummy_ohlcv_data(
    n_samples=1000,
    start_price=20000,
    volatility=0.02,
    trend=0.0001,
    volume_mean=100,
    volume_std=20,
    seed=42
):
    """
    Create dummy OHLCV data for testing purposes.
    """
    np.random.seed(seed)
    
    # Create time index
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Generate price movement
    returns = np.random.normal(trend, volatility, n_samples)
    prices = start_price * np.cumprod(1 + returns)
    
    # Ensure no zero or negative prices
    prices = np.maximum(prices, 1.0)
    
    # Generate OHLCV data
    data = {
        'open_time': dates,
        'open': prices,
        'close': prices * np.exp(np.random.normal(0, 0.005, n_samples)),
        'volume': np.abs(np.random.normal(volume_mean, volume_std, n_samples)) + 1  # ensure volume > 0
    }
    
    # Ensure high and low are correct
    data['high'] = np.maximum(
        data['open'], 
        data['close']
    ) * np.exp(np.abs(np.random.normal(0, 0.01, n_samples)))
    
    data['low'] = np.minimum(
        data['open'], 
        data['close']
    ) * np.exp(-np.abs(np.random.normal(0, 0.01, n_samples)))
    
    # Ensure all prices are positive
    for col in ['open', 'high', 'low', 'close']:
        data[col] = np.maximum(data[col], 1.0)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('open_time', inplace=True)
    
    return df

def create_local_data(data_dir):
    """Create and save local test data."""
    os.makedirs(os.path.join(data_dir, 'data/processed/train'), exist_ok=True)
    
    # Create dummy data
    df = create_dummy_ohlcv_data()
    
    # Save to parquet file
    output_path = os.path.join(data_dir, 'data/processed/train/btc_data.parquet')
    df.to_parquet(output_path)
    
    print(f"✅ Datos simulados guardados en {output_path}")
    return output_path

class LocalTradingEnvironment(TradingEnvironment):
    """Modified TradingEnvironment that loads data from local files."""
    
    def __init__(self, **kwargs):
        # Save sequence_length before passing to parent
        self.sequence_length = kwargs.pop('sequence_length', 60)
        super().__init__(**kwargs)
    
    def _load_market_data(self):
        """
        Load market data from a local file instead of GCS.
        """
        try:
            self.logger.info(f"Loading market data from local: {self.gcs_processed_data_uri}")
            
            if self.gcs_processed_data_uri.endswith('.parquet'):
                # Local file path
                df = pd.read_parquet(self.gcs_processed_data_uri)
                
                # Get the market data array
                # This is a simplified version - just basic features
                self.market_data = np.zeros((len(df), self.sequence_length, 20))
                for i in range(len(df) - self.sequence_length + 1):
                    # Take a window of the dataframe
                    window = df.iloc[i:i+self.sequence_length]
                    
                    # Extract OHLCV
                    ohlcv = window[['open', 'high', 'low', 'close', 'volume']].values
                    
                    # Calculate basic features
                    features = np.zeros((len(window), 20))
                    features[:, :5] = ohlcv  # First 5 are OHLCV
                    
                    # Add ATR as a simple placeholder (required by the environment)
                    # Here we'll just use a simple approximation (high-low)
                    features[:, 8] = window['high'].values - window['low'].values
                    
                    # Add to market_data
                    self.market_data[i] = features
                
                # Set timestamps and feature names
                self.timestamps = df.index.to_numpy()
                self.feature_names = ['open', 'high', 'low', 'close', 'volume', 
                                     'log_return', 'hl_range', 'body_size_rel', 'atr', 'rsi',
                                     'macd', 'macd_signal', 'macd_histogram', 'bb_width', 'sma_cross',
                                     'stoch_k', 'adx', 'volume_ratio', 'mfi', 'obv']
                
                # Set counts
                self.n_samples = len(self.market_data)
                self.n_features = self.market_data.shape[2]
                
                self.logger.info(f"Loaded market data with shape: {self.market_data.shape}")
            else:
                # Local directory path
                # Find all parquet files in the directory
                import glob
                parquet_files = glob.glob(os.path.join(self.gcs_processed_data_uri, '*.parquet'))
                
                if not parquet_files:
                    raise ValueError(f"No parquet files found in {self.gcs_processed_data_uri}")
                
                # Just use the first file
                df = pd.read_parquet(parquet_files[0])
                
                # Same processing as above
                self.market_data = np.zeros((len(df), self.sequence_length, 20))
                for i in range(len(df) - self.sequence_length + 1):
                    window = df.iloc[i:i+self.sequence_length]
                    ohlcv = window[['open', 'high', 'low', 'close', 'volume']].values
                    features = np.zeros((len(window), 20))
                    features[:, :5] = ohlcv
                    self.market_data[i] = features
                
                self.timestamps = df.index.to_numpy()
                self.feature_names = ['open', 'high', 'low', 'close', 'volume'] + ['feature_' + str(i) for i in range(15)]
                self.n_samples = len(self.market_data)
                self.n_features = self.market_data.shape[2]
                
                self.logger.info(f"Loaded market data with shape: {self.market_data.shape}")
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {str(e)}")
            raise

def main():
    """Main function to run the simple training test."""
    try:
        print("🏠 Directorio de datos: tmp")
        
        # Create data directory structure
        data_dir = 'tmp'
        
        print("🔄 Creando datos simulados para prueba...")
        data_path = create_local_data(data_dir)
        
        print("🧠 Creando entorno de trading y modelo...")
        
        # Environment configuration
        env_config = {
            'project_id': 'local-project',
            'gcs_processed_data_uri': data_path,  # Use the local path
            'initial_balance_usd': 10000.0,
            'max_position_btc': 1.0,
            'commission_rate': 0.0004,
            'max_leverage': 5,
            'random_episode_start': True,
            'episode_steps': 100,
            'sequence_length': 60,  # We'll pop this in LocalTradingEnvironment
            'slippage_model': 'atr_based',
            'slippage_factor': 0.05
        }
        
        print(f"Creando entorno con config: {env_config}")
        
        # Create vectorized environment
        print("Creando entorno vectorizado...")
        def make_env():
            env = LocalTradingEnvironment(**env_config)
            return env
        
        env = DummyVecEnv([make_env])
        
        # Create SAC model
        print("Creando modelo SAC...")
        policy_kwargs = {
            'net_arch': dict(pi=[64, 64], qf=[64, 64])
        }
        
        model = SAC(
            'MultiInputPolicy',
            env,
            learning_rate=0.0003,
            buffer_size=10000,
            batch_size=64,
            gamma=0.99,
            tau=0.005,
            ent_coef='auto',
            policy_kwargs=policy_kwargs,
            train_freq=(1, 'episode'),
            gradient_steps=1,
            verbose=1
        )
        
        # Train for a few steps
        print("🏋️‍♂️ Entrenando por 1000 timesteps...")
        model.learn(total_timesteps=1000, progress_bar=False)
        
        # Save model
        print("💾 Guardando modelo...")
        model_path = os.path.join(data_dir, 'models/simple_sac_model')
        model.save(model_path)
        print(f"✅ Modelo guardado en {model_path}")
        
        # Close environment
        env.close()
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {str(e)}")
        raise
    finally:
        print("✅ Test de entrenamiento completado!")

if __name__ == "__main__":
    main()
