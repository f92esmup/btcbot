#!/usr/bin/env python3
"""
Script para prueba básica con un entorno simplificado
"""

import os
import sys
import gymnasium as gym
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import argparse
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# Asegurar que el paquete src está en el path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Crear un entorno simple de trading basado en gym
class SimpleTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000, commission=0.0004, window_size=60):
        super(SimpleTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission
        self.window_size = window_size
        self.current_step = 0
        self.position = 0
        self.trades = []
        
        # Escalar los datos para normalización
        self.scaler = StandardScaler()
        price_features = ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm', 
                         'sma_20_norm', 'sma_50_norm', 'rsi_14_norm', 'macd_norm']
        self.df[price_features] = self.scaler.fit_transform(self.df[price_features])
        
        # Espacio de observación: ventana de precios + posición + balance
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(price_features) * window_size + 2,)
        )
        
        # Espacio de acción: comprar, vender, mantener [-1, 1]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Iniciar desde un punto aleatorio que deje suficiente historia
        self.current_step = self.window_size + np.random.randint(0, len(self.df) - self.window_size - 100)
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        return self._get_observation(), {}
    
    def _get_observation(self):
        # Extraer la ventana de observación y aplanarla
        frame = self.df.iloc[self.current_step - self.window_size:self.current_step]
        price_features = ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm', 
                         'sma_20_norm', 'sma_50_norm', 'rsi_14_norm', 'macd_norm']
        obs = frame[price_features].values.flatten()
        
        # Añadir estado del portafolio
        portfolio_obs = np.array([
            self.position,  # Posición actual
            self.balance / self.initial_balance  # Balance normalizado
        ])
        
        return np.concatenate([obs, portfolio_obs])
    
    def step(self, action):
        # Ejecutar la acción
        action_value = action[0]  # Tomar el valor de la acción (entre -1 y 1)
        
        # Obtener precio actual
        current_price = self.df.iloc[self.current_step]['close']
        
        # Determinar acción: -1 = vender, 1 = comprar, entre medias = mantener con ajustes
        new_position = np.clip(action_value, -1.0, 1.0)
        position_delta = new_position - self.position
        
        # Calcular costos de transacción (simplificado)
        if abs(position_delta) > 0.1:  # Umbral mínimo para operación
            cost = abs(position_delta) * current_price * self.commission
            self.balance -= cost
            
            # Registrar operación
            trade = {
                'step': self.current_step,
                'price': current_price,
                'action': position_delta,
                'cost': cost
            }
            self.trades.append(trade)
        
        # Actualizar posición
        self.position = new_position
        
        # Avanzar al siguiente paso
        self.current_step += 1
        
        # Verificar si el episodio ha terminado
        done = self.current_step >= len(self.df) - 1
        
        # Calcular recompensa: cambio en el valor del portafolio
        old_value = self.balance + self.position * current_price
        new_price = self.df.iloc[self.current_step]['close']
        new_value = self.balance + self.position * new_price
        reward = (new_value - old_value) / self.initial_balance  # Normalizar
        
        # Información adicional
        info = {
            'balance': self.balance,
            'position': self.position,
            'trades': len(self.trades)
        }
        
        return self._get_observation(), reward, done, False, info
    
    def render(self, mode='human'):
        # Simplemente imprimir estado actual
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position:.2f}")
        return

def main():
    parser = argparse.ArgumentParser(description="Ejecutar prueba de entrenamiento simple")
    parser.add_argument('--steps', type=int, default=1000, help="Número de pasos de entrenamiento")
    args = parser.parse_args()
    
    # Crear directorio para datos simulados
    tmp_dir = Path('./tmp')
    models_dir = tmp_dir / 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Generar datos de mercado simulados
    print("🔄 Generando datos de mercado simulados...")
    date_range = pd.date_range(start='2022-01-01', end='2022-02-01', freq='1h')
    n_samples = len(date_range)
    
    # Generar precios simulados con tendencia y volatilidad
    trend = np.linspace(0, 0.3, n_samples) + np.random.normal(0, 0.01, n_samples).cumsum()
    noise = np.random.normal(0, 0.01, n_samples)
    prices = 40000 * (1 + trend + noise)
    
    # Crear DataFrame con datos simulados
    df = pd.DataFrame({
        'timestamp': date_range,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.02, n_samples)),
        'low': prices * (1 - np.random.uniform(0, 0.02, n_samples)),
        'close': prices * (1 + np.random.normal(0, 0.01, n_samples)),
        'volume': np.random.uniform(100, 1000, n_samples)
    })
    
    # Calcular indicadores técnicos simples
    df['sma_20'] = df['close'].rolling(window=20).mean().fillna(method='bfill')
    df['sma_50'] = df['close'].rolling(window=50).mean().fillna(method='bfill')
    df['rsi_14'] = np.random.uniform(30, 70, n_samples)  # Simulado
    df['macd'] = np.random.normal(0, 200, n_samples)  # Simulado
    
    # Normalizar columnas (añadir versiones normalizadas)
    for col in ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_50', 'rsi_14', 'macd']:
        df[f'{col}_norm'] = df[col]  # La normalización real se hace en el entorno
    
    # Crear entorno de trading
    print("🏠 Creando entorno de trading simplificado...")
    
    def make_env():
        return SimpleTradingEnv(df, window_size=60)
    
    env = DummyVecEnv([make_env])
    
    # Crear y entrenar modelo
    print("🧠 Creando y entrenando modelo SAC...")
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0003,
        buffer_size=10000,
        learning_starts=100,
        batch_size=64,
        verbose=1
    )
    
    model.learn(total_timesteps=args.steps)
    
    # Guardar modelo
    model_path = models_dir / "simple_sac_model.zip"
    model.save(model_path)
    print(f"✅ Modelo guardado en {model_path}")
    
    # Evaluar modelo
    print("🧪 Evaluando modelo...")
    obs = env.reset()
    total_reward = 0
    
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done.any():
            obs = env.reset()
    
    print(f"✅ Recompensa total de evaluación: {total_reward[0]:.4f}")
    print("✅ Prueba completada con éxito!")

if __name__ == "__main__":
    main()
