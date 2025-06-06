#!/usr/bin/env python3
"""
Script de prueba para verificar la funcionalidad del price_scaler.
"""

import pandas as pd
import numpy as np
from src.data.normalization import Normalization
from src.configuration.config import config
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Crear datos de muestra para prueba."""
    np.random.seed(42)
    
    # Simular datos OHLCV
    n_samples = 1000
    base_price = 50000
    
    data = {
        'Open': np.random.normal(base_price, 1000, n_samples),
        'High': np.random.normal(base_price + 500, 1200, n_samples),
        'Low': np.random.normal(base_price - 500, 800, n_samples),
        'Close': np.random.normal(base_price, 1100, n_samples),
        'Volume': np.random.exponential(1000000, n_samples)
    }
    
    # Asegurar que High >= max(Open, Close) y Low <= min(Open, Close)
    for i in range(n_samples):
        max_oc = max(data['Open'][i], data['Close'][i])
        min_oc = min(data['Open'][i], data['Close'][i])
        
        if data['High'][i] < max_oc:
            data['High'][i] = max_oc + np.random.uniform(0, 100)
        if data['Low'][i] > min_oc:
            data['Low'][i] = min_oc - np.random.uniform(0, 100)
    
    # Agregar algunos indicadores técnicos de ejemplo
    data['SMA_20'] = np.random.normal(base_price, 800, n_samples)
    data['RSI'] = np.random.uniform(20, 80, n_samples)
    
    return pd.DataFrame(data)

def test_price_scaler_functionality():
    """Probar la funcionalidad completa del price_scaler."""
    logger.info("=== Iniciando prueba del price_scaler ===")
    
    # 1. Crear datos de muestra
    logger.info("1. Creando datos de muestra...")
    df = create_sample_data()
    logger.info(f"Datos creados: {df.shape}")
    logger.info(f"Rango original de Close: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
    
    # 2. Crear instancia de Normalization
    logger.info("2. Creando instancia de Normalization...")
    normalizer = Normalization(df)
    
    # 3. Ejecutar normalización completa
    logger.info("3. Ejecutando normalización...")
    normalized_df, scaler = normalizer.main()
    
    logger.info(f"Normalización completada: {normalized_df.shape}")
    logger.info(f"Rango normalizado de Close: {normalized_df['Close'].min():.6f} - {normalized_df['Close'].max():.6f}")
    
    # 4. Verificar que el price_scaler fue creado
    logger.info("4. Verificando price_scaler...")
    if normalizer.price_scaler is not None:
        logger.info("✓ Price_scaler creado exitosamente")
        close_min = normalizer.price_scaler.data_min_[0]
        close_max = normalizer.price_scaler.data_max_[0]
        logger.info(f"Price_scaler - Rango: {close_min:.2f} - {close_max:.2f}")
    else:
        logger.error("✗ Price_scaler no fue creado")
        return False
    
    # 5. Probar carga del price_scaler
    logger.info("5. Probando carga del price_scaler...")
    try:
        loaded_price_scaler = Normalization.load_price_scaler()
        logger.info("✓ Price_scaler cargado exitosamente")
        
        # Verificar que los parámetros coinciden
        original_min = normalizer.price_scaler.data_min_[0]
        loaded_min = loaded_price_scaler.data_min_[0]
        
        if abs(original_min - loaded_min) < 1e-6:
            logger.info("✓ Parámetros del price_scaler coinciden")
        else:
            logger.error(f"✗ Parámetros no coinciden: {original_min} vs {loaded_min}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error al cargar price_scaler: {e}")
        return False
    
    # 6. Probar transformación con price_scaler
    logger.info("6. Probando transformación con price_scaler...")
    try:
        # Tomar algunos precios de muestra
        sample_prices = df['Close'].iloc[:5].values.reshape(-1, 1)
        transformed_prices = loaded_price_scaler.transform(sample_prices)
        inverse_prices = loaded_price_scaler.inverse_transform(transformed_prices)
        
        logger.info("Precios originales vs transformados vs inversos:")
        for i in range(len(sample_prices)):
            orig = sample_prices[i][0]
            trans = transformed_prices[i][0]
            inv = inverse_prices[i][0]
            logger.info(f"  {orig:.2f} -> {trans:.6f} -> {inv:.2f}")
        
        # Verificar que la transformación inversa funciona
        max_diff = np.max(np.abs(sample_prices.flatten() - inverse_prices.flatten()))
        if max_diff < 1e-6:
            logger.info("✓ Transformación inversa funciona correctamente")
        else:
            logger.error(f"✗ Error en transformación inversa, diferencia máxima: {max_diff}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error en transformación: {e}")
        return False
    
    # 7. Verificar información de almacenamiento
    logger.info("7. Verificando información de almacenamiento...")
    try:
        storage_info = normalizer.get_scaler_storage_info()
        logger.info("Información de almacenamiento:")
        for key, value in storage_info.items():
            logger.info(f"  {key}: {value}")
            
        if storage_info.get('price_scaler_exists', False):
            logger.info("✓ Price_scaler existe en almacenamiento")
        else:
            logger.warning("⚠ Price_scaler no confirmado en almacenamiento")
            
    except Exception as e:
        logger.error(f"✗ Error al obtener información de almacenamiento: {e}")
        return False
    
    logger.info("=== Prueba del price_scaler completada exitosamente ===")
    return True

if __name__ == "__main__":
    success = test_price_scaler_functionality()
    if success:
        print("\n🎉 Todas las pruebas pasaron exitosamente!")
    else:
        print("\n❌ Algunas pruebas fallaron.")
        exit(1)
