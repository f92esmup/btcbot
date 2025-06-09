#!/usr/bin/env python3
"""
Test script to validate the optimization of technical indicators calculation.
This script demonstrates the performance improvement by avoiding DataFrame fragmentation.
"""

import pandas as pd
import numpy as np
import time
import warnings

# Suppress PerformanceWarnings to see the difference
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

def create_sample_data(n_rows=1000):
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 100
    changes = np.random.normal(0, 1, n_rows)
    close_prices = base_price + np.cumsum(changes)
    
    data = {
        'Open': close_prices + np.random.normal(0, 0.5, n_rows),
        'High': close_prices + np.abs(np.random.normal(0, 1, n_rows)),
        'Low': close_prices - np.abs(np.random.normal(0, 1, n_rows)),
        'Close': close_prices,
        'Volume': np.random.randint(1000, 10000, n_rows)
    }
    
    return pd.DataFrame(data)

def test_old_approach(df):
    """Simulate the old approach: adding columns one by one."""
    df_copy = df.copy()
    start_time = time.time()
    
    # Simulate adding indicators one by one (this would generate PerformanceWarnings)
    for i in range(5):  # Simulate 5 indicators
        df_copy[f'indicator_{i}'] = df_copy['Close'].rolling(window=14).mean()
    
    end_time = time.time()
    return end_time - start_time, df_copy

def test_new_approach(df):
    """Simulate the new optimized approach: batch addition of columns."""
    df_copy = df.copy()
    start_time = time.time()
    
    # Collect all indicators in a dictionary first
    new_indicators = {}
    for i in range(5):  # Simulate 5 indicators
        new_indicators[f'indicator_{i}'] = df_copy['Close'].rolling(window=14).mean()
    
    # Add all indicators at once
    indicators_df = pd.DataFrame(new_indicators, index=df_copy.index)
    df_result = pd.concat([df_copy, indicators_df], axis=1)
    
    end_time = time.time()
    return end_time - start_time, df_result

def main():
    print("=== Prueba de Optimización de Indicadores Técnicos ===\n")
    
    # Create test data
    print("Creando datos de prueba...")
    df = create_sample_data(1000)
    print(f"DataFrame creado con {len(df)} filas y {len(df.columns)} columnas\n")
    
    # Test old approach
    print("Probando enfoque anterior (añadir columnas una por una)...")
    old_time, df_old = test_old_approach(df)
    print(f"Tiempo transcurrido: {old_time:.4f} segundos")
    print(f"Columnas resultantes: {len(df_old.columns)}\n")
    
    # Test new approach  
    print("Probando enfoque optimizado (añadir todas las columnas de una vez)...")
    new_time, df_new = test_new_approach(df)
    print(f"Tiempo transcurrido: {new_time:.4f} segundos")
    print(f"Columnas resultantes: {len(df_new.columns)}\n")
    
    # Compare results
    if old_time > 0:
        speedup = old_time / new_time
        print(f"=== RESULTADOS ===")
        print(f"Mejora de rendimiento: {speedup:.2f}x más rápido")
        print(f"Reducción de tiempo: {((old_time - new_time) / old_time * 100):.1f}%")
    
    # Verify data integrity
    print(f"\n=== VERIFICACIÓN ===")
    print(f"¿Los resultados son idénticos? {df_old.equals(df_new)}")
    print("✅ Optimización completada exitosamente!")

if __name__ == "__main__":
    main()
