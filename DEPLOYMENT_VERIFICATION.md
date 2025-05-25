# Guía de Verificación de Despliegue

Este documento proporciona un flujo de trabajo paso a paso para verificar que los despliegues del BTCBot están configurados y funcionan correctamente antes de pasar a producción.

## 📋 Lista de Verificación Previa al Despliegue

### 1. Configuración del Modelo

- [ ] **Ruta del Modelo**: Verificar que `agent.live_model_path` en `src/config.yaml` apunta al modelo correcto:
  ```yaml
  live_model_path: "models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000_steps.zip"
  ```

- [ ] **Existencia del Modelo**: Confirmar que el modelo existe en Google Cloud Storage:
  ```bash
  gsutil ls gs://lofty-complex-460416-r6/models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000_steps.zip
  ```

### 2. Variables de Entorno

- [ ] **Variables Básicas**: Comprobar que las siguientes variables están configuradas:
  ```
  GCP_PROJECT_ID=lofty-complex-460416-r6
  GCP_REGION=europe-southwest1
  BIGQUERY_LOG_DATASET_ID=btcbot_logs
  ```

- [ ] **Modo de Trading**: Inicialmente usar TESTNET para verificación:
  ```
  LIVE_TRADING_MODE=TESTNET
  ```

- [ ] **Secretos de Binance**: Confirmar que los secretos necesarios están disponibles:
  ```
  SECRET_NAME_TESTNET_BINANCE_API_KEY_FUTURES
  SECRET_NAME_TESTNET_BINANCE_API_SECRET_FUTURES
  ```

### 3. Permisos y Accesos

- [ ] **Permisos de GCS**: Verificar acceso a Google Cloud Storage
- [ ] **Permisos de Secret Manager**: Confirmar acceso a los secretos
- [ ] **Permisos de BigQuery**: Validar permisos para escribir en el dataset
- [ ] **IP en Whitelist**: Confirmar que la IP `34.175.215.35` está en la whitelist de Binance

## 🚀 Procedimiento de Verificación en Entorno Local

### 1. Ejecución Inicial

```bash
cd /Users/f92esmup/btcbot
python scripts/run_live_trader.py --config_path src/config.yaml
```

### 2. Verificaciones de Funcionamiento

- [ ] **Carga del Modelo**: El log debe mostrar que el modelo se carga correctamente desde GCS
- [ ] **Conexión WebSocket**: Debe establecer conexión con Binance TESTNET
- [ ] **Configuración de Leverage**: Debe configurar el leverage correctamente (por defecto 10x)
- [ ] **Procesamiento de Datos**: Debe procesar datos de mercado correctamente
- [ ] **Decisiones de Trading**: Debe tomar decisiones basadas en el modelo

### 3. Validación de Operaciones

Ejecute una prueba completa que incluya al menos:
- [ ] **Apertura de Posición**: El bot debe ser capaz de abrir una posición LONG o SHORT
- [ ] **Mantenimiento de Posición**: Debe mantener la posición mientras sea apropiado
- [ ] **Cierre de Posición**: Debe cerrar la posición cuando el modelo lo indique

## 📊 Análisis de Resultados

Después de ejecutar durante un período de prueba (mínimo 24 horas):

### 1. Verificación de Logs

Consulta en BigQuery:

```sql
SELECT 
  timestamp_decision_madrid,
  symbol,
  model_action_value,
  action_threshold,
  current_position_side_bq,
  trade_action_taken,
  realized_pnl_trade_bq,
  current_equity_after_action_bq
FROM 
  `lofty-complex-460416-r6.btcbot_logs.LiveTrading_YYYYMMDD` 
ORDER BY 
  timestamp_decision_madrid DESC
LIMIT 100;
```

### 2. Métricas a Evaluar

- [ ] **Frecuencia de Decisiones**: Verificar que el bot toma decisiones a la frecuencia esperada
- [ ] **Comportamiento del Modelo**: Comprobar que los valores del modelo tienen sentido
- [ ] **Ejecución de Órdenes**: Confirmar que las órdenes se ejecutan cuando corresponde
- [ ] **PnL Realizado**: Verificar que el PnL realizado se registra correctamente
- [ ] **Equity**: Confirmar que el equity se actualiza correctamente

### 3. Análisis de Errores

Revisar los logs en busca de:
- [ ] **Errores de Conexión**: Problemas con WebSocket o API
- [ ] **Errores de Modelo**: Problemas al cargar o ejecutar el modelo
- [ ] **Errores de Ejecución**: Problemas al ejecutar órdenes
- [ ] **Errores de Datos**: Problemas en la obtención o procesamiento de datos

## 🚀 Procedimiento para Transición a Producción

Si todas las verificaciones anteriores son exitosas:

1. **Actualizar Modo de Trading**:
   - Cambiar de TESTNET a REAL en las variables de entorno

2. **Escalar Recursos** (si es necesario):
   - Aumentar CPU/memoria para garantizar el rendimiento

3. **Monitoreo Inicial**:
   - Vigilar estrechamente los primeros ciclos de trading
   - Verificar las primeras ejecuciones de órdenes reales
   - Confirmar que el logging funciona correctamente

4. **Etiquetar Despliegue**:
   ```bash
   git tag -a "v1.x.x-production" -m "Despliegue verificado en producción con modelo XXX"
   git push --tags
   ```

## 🛑 Procedimiento de Rollback

Si se encuentran problemas graves en producción:

1. **Detener el Bot**:
   - En GKE: `kubectl scale deployment live-trader-deployment --replicas=0 -n btcbot`
   - Local: Detener el proceso

2. **Cerrar Posiciones Manualmente** (si es necesario):
   - Usar la interfaz de Binance para cerrar cualquier posición abierta

3. **Revertir Configuración**:
   - Volver a la configuración anterior conocida
   - Actualizar `src/config.yaml` si es necesario

4. **Reiniciar en TESTNET**:
   - Cambiar a TESTNET para verificar la corrección
   - Repetir el proceso de verificación
