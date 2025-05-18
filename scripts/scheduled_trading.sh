#!/bin/bash
# scheduled_trading.sh - Script para ejecutar el trading automáticamente
#
# Este script está diseñado para ser ejecutado periódicamente mediante un cron job
# y realizar operaciones de trading según el modelo entrenado.

set -e  # Exit immediately if a command exits with a non-zero status

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuración
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/trading_${TIMESTAMP}.log"
MODEL_GCS_PATH="gs://btcbot276299-btc-models/simple_sac_model.zip"
MODEL_LOCAL_PATH="$PROJECT_ROOT/tmp/models/simple_sac_model.zip"
RESULTS_DIR="$PROJECT_ROOT/tmp/results/${TIMESTAMP}"

# Crear directorios necesarios
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$(dirname "$MODEL_LOCAL_PATH")"

# Iniciar logging
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================================"
echo "🚀 BITCOIN TRADING BOT - EJECUCIÓN PROGRAMADA"
echo "🕒 $(date)"
echo "========================================================"

# Verificar credenciales de GCP
if ! gcloud auth print-access-token &>/dev/null; then
    echo "❌ Error: No se encontraron credenciales de GCP válidas."
    echo "   Ejecuta 'gcloud auth login' y vuelve a intentarlo."
    exit 1
fi

# Descargar el modelo desde GCS
echo "📥 Descargando modelo desde GCS..."
if ! gsutil cp "$MODEL_GCS_PATH" "$MODEL_LOCAL_PATH"; then
    echo "❌ Error: No se pudo descargar el modelo desde $MODEL_GCS_PATH"
    exit 1
fi

echo "✅ Modelo descargado correctamente en $MODEL_LOCAL_PATH"

# Ejecutar el trading
echo "🏃 Ejecutando trading con el modelo..."
python3 - << EOF
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from stable_baselines3 import SAC
import json
import requests
from pathlib import Path

# Añadir directorio raíz al path
sys.path.append('${PROJECT_ROOT}')

# Configuración
MODEL_PATH = '${MODEL_LOCAL_PATH}'
RESULTS_DIR = '${RESULTS_DIR}'
BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET', '')

print(f"📊 Usando modelo: {MODEL_PATH}")
print(f"📁 Guardando resultados en: {RESULTS_DIR}")

# Crear directorios si no existen
os.makedirs(RESULTS_DIR, exist_ok=True)

# Función para obtener datos de Binance
def get_binance_data(symbol='BTCUSDT', interval='1h', limit=100):
    try:
        print(f"📈 Obteniendo datos de Binance para {symbol}...")
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        
        # Convertir tipos de datos
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        print(f"✅ Datos obtenidos: {len(df)} registros")
        return df
    except Exception as e:
        print(f"❌ Error al obtener datos de Binance: {str(e)}")
        # Crear datos simulados como alternativa
        print("⚠️ Usando datos simulados como alternativa...")
        return generate_simulated_data(limit)

# Función para generar datos simulados
def generate_simulated_data(n_samples=100):
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=n_samples)
    date_range = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Generar precios simulados con tendencia y volatilidad similar a BTC
    last_price = 60000  # Último precio conocido de BTC
    prices = [last_price]
    for i in range(1, n_samples):
        # Simular movimiento browniano con tendencia
        change = np.random.normal(0, 0.01) * prices[-1]
        new_price = prices[-1] + change
        prices.append(new_price)
    
    prices = np.array(prices)
    
    df = pd.DataFrame({
        'timestamp': date_range,
        'open': prices * (1 + np.random.normal(0, 0.002, n_samples)),
        'high': prices * (1 + np.random.uniform(0.001, 0.02, n_samples)),
        'low': prices * (1 - np.random.uniform(0.001, 0.02, n_samples)),
        'close': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'volume': np.random.uniform(100, 1000, n_samples) * prices / 1000
    })
    
    return df

# Obtener datos
data = get_binance_data(limit=200)

# Calcular indicadores técnicos
print("📊 Calculando indicadores técnicos...")
data['sma_20'] = data['close'].rolling(window=20).mean().fillna(method='bfill')
data['sma_50'] = data['close'].rolling(window=50).mean().fillna(method='bfill')
data['rsi_14'] = 50 + np.random.normal(0, 10, len(data))  # Simplificado
data['macd'] = data['sma_20'] - data['sma_50']  # Simplificado

# Normalizar columnas
for col in ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_50', 'rsi_14', 'macd']:
    col_mean = data[col].mean()
    col_std = data[col].std()
    data[f'{col}_norm'] = (data[col] - col_mean) / col_std

# Cargar modelo
print("🤖 Cargando modelo...")
model = SAC.load(MODEL_PATH)

# Configuración de trading
window_size = 60
position = 0
balance = 10000
commission = 0.0004
current_btc = 0
last_action = 'HOLD'
portfolio_history = []
trade_history = []

# Función para obtener observación
def get_observation(step):
    # Extraer la ventana de observación
    frame = data.iloc[step - window_size:step]
    price_features = ['open_norm', 'high_norm', 'low_norm', 'close_norm', 'volume_norm', 
                     'sma_20_norm', 'sma_50_norm', 'rsi_14_norm', 'macd_norm']
    obs = frame[price_features].values.flatten()
    
    # Añadir estado del portafolio
    portfolio_obs = np.array([
        position,
        balance / 10000  # Normalizado al balance inicial
    ])
    
    return np.concatenate([obs, portfolio_obs])

# Ejecutar el trading con los datos más recientes
print("🚀 Ejecutando trading...")
current_step = len(data) - 1
if current_step >= window_size:
    current_price = data.iloc[current_step]['close']
    
    # Obtener observación
    obs = get_observation(current_step)
    
    # Predecir acción
    action, _ = model.predict(obs, deterministic=True)
    action_value = action[0]
    
    # Determinar nueva posición
    new_position = np.clip(action_value, -1.0, 1.0)
    position_delta = new_position - position
    
    # Interpretar acción
    if abs(position_delta) > 0.1:
        cost = abs(position_delta) * current_price * commission
        balance -= cost
        
        if position_delta > 0:
            action_type = "BUY"
        else:
            action_type = "SELL"
        
        # Registrar operación
        trade = {
            'timestamp': data.iloc[current_step]['timestamp'],
            'price': current_price,
            'action': action_type,
            'position_delta': position_delta,
            'cost': cost,
            'new_position': new_position,
            'balance': balance
        }
        trade_history.append(trade)
        last_action = action_type
    else:
        last_action = "HOLD"
    
    # Actualizar posición
    position = new_position
    
    # Calcular valor del portafolio
    portfolio_value = balance + position * current_price
    
    # Guardar estado actual
    portfolio_history.append({
        'timestamp': data.iloc[current_step]['timestamp'],
        'price': current_price,
        'position': position,
        'balance': balance,
        'portfolio_value': portfolio_value,
        'last_action': last_action
    })
    
    # Mostrar estado actual
    print(f"📅 Fecha: {data.iloc[current_step]['timestamp']}")
    print(f"💰 Precio BTC: ${current_price:.2f}")
    print(f"🔄 Acción: {last_action}")
    print(f"📊 Posición: {position:.4f}")
    print(f"💵 Balance: ${balance:.2f}")
    print(f"📈 Valor portafolio: ${portfolio_value:.2f}")
    
    # Guardar resultados
    results = {
        'timestamp': datetime.now().isoformat(),
        'price': float(current_price),
        'position': float(position),
        'balance': float(balance),
        'portfolio_value': float(portfolio_value),
        'last_action': last_action
    }
    
    with open(f"{RESULTS_DIR}/latest_trading_result.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    if trade_history:
        trades_df = pd.DataFrame(trade_history)
        trades_df.to_csv(f"{RESULTS_DIR}/trades.csv", index=False)
    
    print(f"✅ Resultados guardados en {RESULTS_DIR}/latest_trading_result.json")
    
    # Generar gráfico
    if len(portfolio_history) > 1:
        ph_df = pd.DataFrame(portfolio_history)
        
        plt.figure(figsize=(10, 6))
        plt.plot(ph_df['timestamp'], ph_df['portfolio_value'], label='Valor del Portafolio')
        plt.title('Evolución del Valor del Portafolio')
        plt.xlabel('Fecha')
        plt.ylabel('Valor ($)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{RESULTS_DIR}/portfolio_value.png")
        print(f"✅ Gráfico guardado en {RESULTS_DIR}/portfolio_value.png")
else:
    print("❌ No hay suficientes datos para ejecutar el trading (se necesitan al menos window_size períodos)")

print("✅ Ejecución completada")
EOF

echo "========================================================"
echo "✅ EJECUCIÓN COMPLETADA"
echo "📁 Resultados disponibles en: $RESULTS_DIR"
echo "📝 Log completo en: $LOG_FILE"
echo "========================================================"
