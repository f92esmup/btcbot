# 📊 Guía Completa de TensorBoard para BTCBot

Esta guía explica cómo utilizar TensorBoard para visualizar las métricas de entrenamiento del agente SAC y el rendimiento del entorno de trading.

## 🚀 Configuración y Inicio

### Instalación
TensorBoard ya está incluido en `requirements.txt`:
```bash
pip install tensorboard>=2.8.0
```

### Inicio de TensorBoard
```bash
# Ejecutar desde el directorio del proyecto
tensorboard --logdir=runs --port=6006

# O especificar un experimento específico
tensorboard --logdir=runs/SAC_BTCUSDT_1h_20250603_143022 --port=6006
```

Luego acceder a: http://localhost:6006

## 📈 Métricas Disponibles

### 1. Métricas del Agente SAC

#### **Agent/Losses**
- **actor_loss**: Pérdida del actor (política) por paso de aprendizaje
- **critic1_loss**: Pérdida del primer crítico por paso de aprendizaje  
- **critic2_loss**: Pérdida del segundo crítico por paso de aprendizaje
- **alpha_loss**: Pérdida del parámetro de temperatura alpha por paso de aprendizaje

#### **Agent/Values**
- **alpha_value**: Valor actual del parámetro de temperatura alpha
- **buffer_size**: Tamaño actual del buffer de experiencias

### 2. Métricas por Episodio

#### **Episode/Returns**
- **episode_return**: Return total del episodio
- **episode_profit_pct**: Ganancia/pérdida porcentual del episodio
- **episode_length**: Número de pasos en el episodio

#### **Episode/Trading_Stats**
- **num_trades**: Número total de trades realizados en el episodio
- **profitable_trades**: Número de trades rentables
- **win_rate**: Porcentaje de trades ganadores (profitable_trades / num_trades)

#### **Episode/Environment**
- **final_balance**: Balance final al terminar el episodio
- **final_equity**: Equity final (balance + PnL no realizado)
- **max_drawdown**: Máximo drawdown alcanzado durante el episodio

### 3. Métricas por Trade Individual

#### **Trades/Performance**
- **trade_pnl**: PnL absoluto de cada trade individual
- **trade_roe**: Return on Equity (ROE) de cada trade individual
- **trade_duration**: Duración de cada trade en pasos
- **trade_margin_used**: Margen utilizado en cada trade

#### **Trades/Direction**
- **trade_long_count**: Contador de trades largos
- **trade_short_count**: Contador de trades cortos

### 4. Métricas de Evaluación

#### **Evaluation/Performance**
- **eval_mean_return**: Return promedio en evaluación
- **eval_mean_profit**: Ganancia promedio en evaluación
- **eval_mean_win_rate**: Win rate promedio en evaluación
- **eval_mean_trades**: Número promedio de trades en evaluación

## 🎯 Cómo Interpretar las Métricas

### **Entrenamiento del Agente**
- **Actor Loss**: Debe tender a estabilizarse y no crecer indefinidamente
- **Critic Losses**: Deben disminuir gradualmente durante el entrenamiento
- **Alpha Value**: Se ajusta automáticamente, valores típicos entre 0.1-1.0
- **Buffer Size**: Debe crecer hasta el límite máximo configurado

### **Rendimiento de Trading**
- **Episode Return**: Debe mostrar tendencia creciente con el tiempo
- **Win Rate**: Un buen agente típicamente mantiene >50% win rate
- **Max Drawdown**: Debe mantenerse dentro de límites aceptables (<50%)
- **Trade ROE**: Distribución de ROEs debe estar centrada en valores positivos

### **Convergencia del Entrenamiento**
- **Pérdidas estabilizándose**: Las pérdidas del actor y críticos se estabilizan
- **Returns crecientes**: Los returns por episodio muestran tendencia ascendente
- **Win rate estable**: El win rate se estabiliza en valores razonables
- **Reducción de varianza**: Las métricas muestran menos volatilidad

## 📊 Dashboards Recomendados

### **Dashboard Principal de Entrenamiento**
1. `Agent/Losses/*` - Monitorear convergencia del modelo
2. `Episode/Returns/episode_return` - Progreso del rendimiento
3. `Episode/Trading_Stats/win_rate` - Calidad de las decisiones
4. `Episode/Environment/max_drawdown` - Control de riesgo

### **Dashboard de Análisis de Trades**
1. `Trades/Performance/trade_roe` - Distribución de rendimientos
2. `Trades/Performance/trade_duration` - Patrones temporales
3. `Trades/Direction/*` - Balance entre estrategias long/short
4. `Trades/Performance/trade_margin_used` - Uso del capital

### **Dashboard de Evaluación**
1. `Evaluation/Performance/*` - Todas las métricas de evaluación
2. Comparar con métricas de entrenamiento para detectar overfitting

## 🔧 Configuración Avanzada

### **Filtros Personalizados**
```python
# En train.py, puedes agregar filtros personalizados:
writer.add_scalar('Custom/Sharpe_Ratio', sharpe_ratio, global_step)
writer.add_scalar('Custom/Max_Consecutive_Losses', max_losses, global_step)
```

### **Histogramas de Distribuciones**
```python
# Para visualizar distribuciones de acciones o recompensas
writer.add_histogram('Actions/Distribution', actions_batch, global_step)
writer.add_histogram('Rewards/Distribution', rewards_batch, global_step)
```

### **Comparación de Experimentos**
```bash
# Ejecutar múltiples experimentos y compararlos
tensorboard --logdir=runs --reload_interval=1
```

## 🎮 Comandos de Ejemplo

### **Entrenamiento con Logging Completo**
```bash
python train.py --symbol BTCUSDT --interval 1h --start-date 2024-01-01 --episodes 1000
```

### **Monitoreo en Tiempo Real**
```bash
# Terminal 1: Entrenar el modelo
python train.py --symbol BTCUSDT --interval 4h --start-date 2024-01-01

# Terminal 2: Iniciar TensorBoard
tensorboard --logdir=runs --reload_interval=1
```

## 🔍 Solución de Problemas

### **No se muestran métricas**
1. Verificar que el directorio `runs/` existe
2. Comprobar que no hay errores en el logging
3. Refrescar el navegador o reiniciar TensorBoard

### **Métricas inconsistentes**
1. Verificar que `global_step` se incrementa correctamente
2. Comprobar que las métricas se loggean en el momento adecuado
3. Revisar la configuración del experimento

### **Rendimiento lento**
1. Reducir la frecuencia de logging si es necesario
2. Usar `--reload_interval` más alto en TensorBoard
3. Limpiar experimentos antiguos del directorio `runs/`

## 📝 Logs de Hiperparámetros

El sistema automáticamente loggea todos los hiperparámetros importantes:
- Configuración del agente SAC
- Parámetros del entorno de trading
- Configuración de red neuronal
- Parámetros de entrenamiento

Estos aparecen en la pestaña "HParams" de TensorBoard para comparar experimentos.

## 🎯 Mejores Prácticas

1. **Nombrado Consistente**: Usar nombres descriptivos para experimentos
2. **Logging Frecuente**: Loggear métricas importantes cada paso/episodio
3. **Limpieza Regular**: Eliminar experimentos antiguos innecesarios
4. **Comparación**: Ejecutar múltiples experimentos para validar resultados
5. **Documentación**: Mantener notas sobre configuraciones exitosas

---

**Nota**: Esta integración de TensorBoard proporciona visibilidad completa del proceso de entrenamiento y permite optimizar tanto el agente como los parámetros del entorno para obtener el mejor rendimiento en trading.
