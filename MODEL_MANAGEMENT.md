# Gestión de Modelos y Despliegues para BTCBot

Este documento proporciona directrices detalladas sobre cómo gestionar los modelos de IA del BTCBot y administrar sus despliegues en diferentes entornos.

## Modelos de IA

### 📂 Estructura de Almacenamiento

Los modelos entrenados se almacenan en Google Cloud Storage siguiendo esta estructura:

```
gs://<bucket>/models/sac_transformer_trading_agent/<nombre_modelo>.zip
```

Donde `<nombre_modelo>` sigue el formato: `sac_transformer_trading_agent_final_<pasos>.zip`

### 🔄 Ciclo de Vida del Modelo

1. **Entrenamiento**: Los modelos se entrenan con `scripts/train_rl_agent.py`
2. **Evaluación**: Se evalúan con `scripts/evaluate_rl_agent.py` 
3. **Despliegue**: Se configuran en `src/config.yaml` bajo `agent.live_model_path`
4. **Monitoreo**: Su comportamiento en producción se monitorea a través de logs en BigQuery

### ✅ Modelo de Producción Verificado

El siguiente modelo ha sido verificado para uso en producción:

- **Ruta**: `models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000_steps.zip`
- **Verificado**: ✅ 25 Mayo 2025
- **Entorno**: Binance TESTNET
- **Comportamiento**: Capaz de cerrar posiciones short correctamente

## Configuración del Trading en Vivo

### 📝 Archivo de Configuración

La configuración del sistema se centraliza en `src/config.yaml`. Para el trading en vivo, las siguientes opciones son críticas:

```yaml
agent:
  # otras configuraciones...
  live_model_path: "models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000_steps.zip"

live_trading:
  # Configuración del websocket, retrasos, y opciones de logging
```

### 🚀 Iniciar Trading en Vivo

Para ejecutar el bot de trading en vivo manualmente:

```bash
python scripts/run_live_trader.py --config_path src/config.yaml
```

**Importante**: Use `--config_path` (no `--config`) como argumento.

### 🔐 Variables de Entorno

Las siguientes variables de entorno deben configurarse en cada entorno de ejecución:

```
GCP_PROJECT_ID=lofty-complex-460416-r6
GCP_REGION=europe-southwest1
BIGQUERY_LOG_DATASET_ID=btcbot_logs
LIVE_TRADING_MODE=TESTNET  # Usar TESTNET para pruebas, REAL para trading real
SECRET_NAME_BINANCE_API_KEY_FUTURES=BINANCE_API_KEY_FUTURES
SECRET_NAME_BINANCE_API_SECRET_FUTURES=BINANCE_API_SECRET_FUTURES
SECRET_NAME_TESTNET_BINANCE_API_KEY_FUTURES=TESTNET_BINANCE_API_KEY_FUTURES
SECRET_NAME_TESTNET_BINANCE_API_SECRET_FUTURES=TESTNET_BINANCE_API_SECRET_FUTURES
```

## Despliegue y Monitoreo

### 📊 Logs de Trading

Los eventos de trading en vivo se registran en varias ubicaciones:

1. **Console/Archivos**: Logs detallados en tiempo real
2. **BigQuery**: Tabla `LiveTrading_{FECHA}` (ej. `LiveTrading_20250525`) en el dataset configurado
3. **Google Cloud Storage**: Archivos CSV de respaldo

### 📈 Monitoreo de Rendimiento

Para monitorear el rendimiento del bot:

#### En GKE:
```bash
kubectl logs -f -n btcbot deployment/live-trader-deployment
```

#### Métricas en BigQuery:
```sql
SELECT 
  timestamp_decision_madrid,
  symbol,
  model_action_value,
  action_threshold,
  current_position_side_bq,
  realized_pnl_trade_bq,
  current_equity_after_action_bq
FROM 
  `lofty-complex-460416-r6.btcbot_logs.LiveTrading_20250525` 
ORDER BY 
  timestamp_decision_madrid DESC
LIMIT 100;
```

## Proceso de Actualización del Modelo

Para actualizar el modelo en uso:

1. **Entrenamiento**: Entrene un nuevo modelo:
```bash
python scripts/train_rl_agent.py --config src/config.yaml --timesteps 1000000
```

2. **Evaluación**: Evalúe el nuevo modelo:
```bash
python scripts/evaluate_rl_agent.py --config src/config.yaml --model-path models/sac_transformer_trading_agent_final_1000000_steps.zip --episodes 10
```

3. **Actualización**: Modifique `src/config.yaml`:
```yaml
agent:
  # otras configuraciones...
  live_model_path: "models/sac_transformer_trading_agent/sac_transformer_trading_agent_final_1000000_steps.zip"
```

4. **Despliegue**: Aplique la nueva configuración:
   - En GKE: Actualice el ConfigMap y reinicie el despliegue
   - Local: Reinicie la aplicación con la nueva configuración

5. **Versión**: Etiquete la versión con el hash de Git para seguimiento:
```bash
git add src/config.yaml
git commit -m "feat: Actualizado modelo a sac_transformer_trading_agent_final_1000000_steps"
git tag -a "v1.2.0-model-1000000" -m "Modelo de producción - 1M steps"
git push && git push --tags
```

## Rollback

En caso de problemas con un modelo nuevo, revierta a un modelo anterior:

1. Actualice `src/config.yaml` con la ruta del modelo anterior verificado
2. Aplique los cambios y reinicie el sistema
3. Etiquete la versión como rollback:
```bash
git add src/config.yaml
git commit -m "fix: Rollback a modelo estable anterior"
git tag -a "v1.2.1-rollback" -m "Rollback a modelo estable"
git push && git push --tags
```
