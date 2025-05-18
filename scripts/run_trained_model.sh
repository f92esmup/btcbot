#!/bin/bash
# Ejecuta el modelo directamente, saltando Terraform si da problemas

set -e
set -x  # Modo verbose para depuración

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📂 Directorio del script: $SCRIPT_DIR"
echo "📂 Directorio del proyecto: $PROJECT_ROOT"

# Verificar si el modelo entrenado existe
MODEL_PATH="$PROJECT_ROOT/tmp/models/simple_sac_model.zip"
if [ ! -f "$MODEL_PATH" ]; then
    echo "🚫 No se encontró un modelo entrenado en $MODEL_PATH"
    echo "🏃 Ejecutando entrenamiento básico primero..."
    python3 "$SCRIPT_DIR/basic_trading_test.py" --steps 1000
fi

# Verificar que el modelo existe después del entrenamiento
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error crítico: No se pudo crear el modelo $MODEL_PATH"
    exit 1
fi

echo "✅ Modelo encontrado: $MODEL_PATH"

# Ejecutar modelo de trading
echo "🚀 Ejecutando modelo de trading Bitcoin..."
echo "📈 Usando modelo: $MODEL_PATH"

# Directorio de salida para resultados
RESULTS_DIR="$PROJECT_ROOT/tmp/results"
mkdir -p "$RESULTS_DIR"

echo "📁 Carpeta de resultados: $RESULTS_DIR"

# Ejecutar script de backtesting simplificado
python3 - << EOF
#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from pathlib import Path
import sys

# Añadir directorio raíz al path
sys.path.append('${PROJECT_ROOT}')

# Configuración
MODEL_PATH = '${MODEL_PATH}'
RESULTS_DIR = '${RESULTS_DIR}'

print(f"📊 Cargando modelo desde {MODEL_PATH}")
print(f"📁 Resultados se guardarán en {RESULTS_DIR}")

# Verificar que el directorio existe
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"✅ Directorio de resultados existe: {os.path.isdir(RESULTS_DIR)}")

print("📊 Generando datos de prueba...")

# Generar datos de prueba (los mismos que usamos para entrenar pero extendidos)
date_range = pd.date_range(start='2022-01-01', end='2022-06-01', freq='1h')
n_samples = len(date_range)

# Generar precios simulados con tendencia y volatilidad
trend = np.linspace(0, 0.5, n_samples) + np.random.normal(0, 0.02, n_samples).cumsum()
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

# Normalizar columnas
for col in ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_50', 'rsi_14', 'macd']:
    col_mean = df[col].mean()
    col_std = df[col].std()
    df[f'{col}_norm'] = (df[col] - col_mean) / col_std

# Configurar parámetros de simulación
initial_balance = 10000
test_length = 1000  # Número de pasos a simular
window_size = 60
position = 0
balance = initial_balance
portfolio_values = []
trades = []

# Cargar modelo
print("🤖 Cargando modelo SAC...")
model = SAC.load(MODEL_PATH)

# Función para obtener observación
def get_observation(step):
    # Extraer la ventana de observación y aplanarla
    frame = df.iloc[step - window_size:step]
    price_features = ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm', 
                     'sma_20_norm', 'sma_50_norm', 'rsi_14_norm', 'macd_norm']
    obs = frame[price_features].values.flatten()
    
    # Añadir estado del portafolio
    portfolio_obs = np.array([
        position,
        balance / initial_balance
    ])
    
    return np.concatenate([obs, portfolio_obs])

# Iniciar simulación
print("🏃 Ejecutando simulación de trading...")
start_step = window_size + 100  # Dejar espacio para la ventana inicial
commission = 0.0004

for step in range(start_step, start_step + test_length):
    # Obtener precio actual
    current_price = df.iloc[step]['close']
    
    # Obtener observación
    obs = get_observation(step)
    
    # Predecir acción
    action, _ = model.predict(obs, deterministic=True)
    action_value = action[0]
    
    # Determinar nueva posición
    new_position = np.clip(action_value, -1.0, 1.0)
    position_delta = new_position - position
    
    # Calcular costos de transacción
    if abs(position_delta) > 0.1:  # Umbral mínimo para operación
        cost = abs(position_delta) * current_price * commission
        balance -= cost
        
        # Registrar operación
        trade = {
            'step': step,
            'timestamp': df.iloc[step]['timestamp'],
            'price': current_price,
            'action': position_delta,
            'cost': cost,
            'new_position': new_position,
            'balance': balance
        }
        trades.append(trade)
    
    # Actualizar posición
    position = new_position
    
    # Calcular valor del portafolio
    portfolio_value = balance + position * current_price
    portfolio_values.append({
        'step': step,
        'timestamp': df.iloc[step]['timestamp'],
        'price': current_price,
        'position': position,
        'balance': balance,
        'portfolio_value': portfolio_value
    })

# Crear DataFrame de resultados
print("📊 Analizando resultados...")
results_df = pd.DataFrame(portfolio_values)
trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

# Calcular métricas
initial_value = initial_balance
final_value = results_df['portfolio_value'].iloc[-1]
pnl = final_value - initial_value
pnl_pct = (pnl / initial_value) * 100
n_trades = len(trades)

print(f"\n📈 Resultados de la simulación:")
print(f"💰 Balance inicial: ${initial_value:.2f}")
print(f"💰 Balance final: ${final_value:.2f}")
print(f"💹 P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
print(f"🔄 Número de operaciones: {n_trades}")

# Guardar resultados
results_df.to_csv(f"{RESULTS_DIR}/portfolio_history.csv", index=False)
if not trades_df.empty:
    trades_df.to_csv(f"{RESULTS_DIR}/trades.csv", index=False)

# Generar gráfico
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(results_df['timestamp'], df.iloc[results_df['step']]['close'], label='BTC Price')
plt.title('Bitcoin Price')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(results_df['timestamp'], results_df['portfolio_value'], label='Portfolio Value', color='green')
plt.fill_between(results_df['timestamp'], initial_value, results_df['portfolio_value'], 
                where=(results_df['portfolio_value'] >= initial_value), color='green', alpha=0.3)
plt.fill_between(results_df['timestamp'], initial_value, results_df['portfolio_value'], 
                where=(results_df['portfolio_value'] < initial_value), color='red', alpha=0.3)
plt.title('Portfolio Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/backtest_results.png")
print(f"✅ Gráficos guardados en {RESULTS_DIR}/backtest_results.png")

# También crear un gráfico para las posiciones
plt.figure(figsize=(12, 6))
plt.plot(results_df['timestamp'], results_df['position'], label='Position Size')
plt.title('Bitcoin Position Size Over Time')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.fill_between(results_df['timestamp'], 0, results_df['position'], 
                where=(results_df['position'] >= 0), color='green', alpha=0.3)
plt.fill_between(results_df['timestamp'], 0, results_df['position'], 
                where=(results_df['position'] < 0), color='red', alpha=0.3)
plt.grid(True)
plt.legend()
plt.savefig(f"{RESULTS_DIR}/positions.png")
print(f"✅ Gráfico de posiciones guardado en {RESULTS_DIR}/positions.png")
EOF

echo "✅ Ejecución completa!"
echo "📁 Resultados disponibles en: $RESULTS_DIR"
