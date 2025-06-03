# 📊 Guía Completa de TensorBoard para BTCBot

## 🎯 Introducción

TensorBoard está completamente integrado en el sistema BTCBot para proporcionar visualización en tiempo real y análisis detallado del entrenamiento del agente SAC (Soft Actor-Critic) con arquitectura Transformer. Esta guía explica cómo usar todas las funcionalidades de monitoreo disponibles.

## 🚀 Inicio Rápido

### 1. Entrenar con TensorBoard
```bash
# Entrenar el agente con logging de TensorBoard
python train.py --symbol BTCUSDT --interval 1h --start-date 2024-01-01 --episodes 1000

# El sistema automáticamente creará logs en: runs/btcbot_YYYYMMDD_HHMMSS/
```

### 2. Visualizar en TensorBoard
```bash
# Abrir TensorBoard (desde otra terminal)
tensorboard --logdir=runs/

# Acceder en el navegador
# http://localhost:6006
```

## 📈 Métricas Disponibles

### 🤖 Métricas del Agente (por paso de aprendizaje)

#### **Actor Loss**
- **Descripción**: Pérdida del actor (política) durante el entrenamiento
- **Interpretación**: 
  - Valores decrecientes indican mejora en la política
  - Fluctuaciones normales durante exploración
- **Gráfico**: `Agent/actor_loss`

#### **Critic Losses**
- **Critic 1 Loss**: `Agent/critic1_loss`
- **Critic 2 Loss**: `Agent/critic2_loss`
- **Interpretación**:
  - Pérdidas de los críticos (funciones de valor)
  - Deberían decrecer y estabilizarse
  - Divergencia puede indicar problemas de entrenamiento

#### **Alpha (Temperature)**
- **Alpha Value**: `Agent/alpha_value`
- **Alpha Loss**: `Agent/alpha_loss`
- **Interpretación**:
  - Controla exploración vs explotación
  - Se ajusta automáticamente durante entrenamiento
  - Valores altos = más exploración

#### **Buffer Size**
- **Descripción**: Tamaño actual del buffer de experiencias
- **Gráfico**: `Agent/buffer_size`
- **Interpretación**: Crecimiento hasta el máximo configurado

### 📊 Métricas de Episodio

#### **Returns y Rendimiento**
- **Episode Return**: `Episode/return` - Retorno total del episodio
- **Episode Profit %**: `Episode/profit_percentage` - Ganancia porcentual
- **Episode Length**: `Episode/length` - Duración en pasos

#### **Estadísticas de Trading**
- **Total Trades**: `Episode/total_trades` - Número de operaciones
- **Win Rate**: `Episode/win_rate` - Porcentaje de trades ganadores
- **Average Trade ROE**: `Episode/avg_trade_roe` - ROE promedio por trade
- **Average Trade Duration**: `Episode/avg_trade_duration` - Duración promedio

#### **Métricas de Balance**
- **Final Balance**: `Episode/final_balance` - Balance final en USDT
- **Final Equity**: `Episode/final_equity` - Equity final
- **Max Drawdown**: `Episode/max_drawdown` - Máximo drawdown del episodio
- **Max Equity**: `Episode/max_equity` - Máximo equity alcanzado

### 💰 Métricas por Trade Individual

#### **Performance del Trade**
- **Trade PNL**: `Trades/pnl_abs` - PNL absoluto en USDT
- **Trade ROE**: `Trades/roe` - Return on Equity
- **Trade Duration**: `Trades/duration_steps` - Duración en pasos

#### **Detalles Operativos**
- **Margin Used**: `Trades/margin_used` - Margen utilizado
- **Trade Direction**: `Trades/direction` - Dirección (1=LARGO, -1=CORTO)

### 🏦 Métricas del Entorno

#### **Estado Financiero**
- **Balance**: `Environment/balance` - Balance actual
- **Equity**: `Environment/equity` - Equity actual
- **Drawdown**: `Environment/drawdown` - Drawdown desde máximo

#### **Estado de Posición**
- **Position Type**: `Environment/position_type` - Tipo de posición actual
- **Position PNL ROE**: `Environment/position_pnl_roe` - ROE no realizado
- **Steps in Position**: `Environment/steps_in_position` - Pasos en posición

### 🎯 Métricas de Evaluación

Durante las evaluaciones periódicas:
- **Eval Mean Return**: `Evaluation/mean_return`
- **Eval Std Return**: `Evaluation/std_return`
- **Eval Mean Profit**: `Evaluation/mean_profit_percentage`
- **Eval Win Rate**: `Evaluation/win_rate`

## 📋 Interpretación de Métricas

### ✅ Señales de Entrenamiento Saludable

1. **Actor Loss**: Decrece gradualmente con fluctuaciones
2. **Critic Losses**: Se estabilizan en valores bajos
3. **Episode Return**: Tendencia creciente a largo plazo
4. **Win Rate**: Mejora progresiva (>50% es bueno)
5. **Drawdown**: Se mantiene dentro de límites aceptables

### ⚠️ Señales de Problemas

1. **Critic Losses**: Crecimiento sostenido o valores muy altos
2. **Alpha**: Valores extremos (muy altos o muy bajos)
3. **Episode Return**: Tendencia decreciente persistente
4. **Win Rate**: Consistentemente <40%
5. **Drawdown**: Frecuentemente cerca del límite máximo

### 🔧 Ajustes Recomendados

#### Si el agente no aprende:
- Reducir `learning_rate` en `config.yaml`
- Aumentar `batch_size`
- Ajustar `target_update_tau`

#### Si hay overfitting:
- Aumentar exploración (alpha)
- Reducir `network_hidden_size`
- Implementar regularización

#### Si trading es muy agresivo:
- Ajustar `porcentaje_max_inversion_por_trade`
- Modificar función de recompensa
- Revisar `zona_muerta_mantener`

## 🎨 Visualizaciones Avanzadas

### Comparar Experimentos
```bash
# Entrenar múltiples configuraciones
python train.py --symbol BTCUSDT --episodes 500  # Experimento 1
# Modificar config.yaml
python train.py --symbol BTCUSDT --episodes 500  # Experimento 2

# TensorBoard mostrará ambos experimentos
tensorboard --logdir=runs/
```

### Filtrar Métricas
- Usar regex en TensorBoard: `Episode/.*` para ver solo métricas de episodio
- Agrupar por categorías: `Agent|Episode|Trades`

### Escalas Personalizadas
- Cambiar a escala logarítmica para losses
- Suavizar gráficos con el slider "Smoothing"

## 🔍 Análisis de Patrones

### Convergencia del Entrenamiento
1. **Verificar**: Actor/Critic losses se estabilizan
2. **Observar**: Episode returns mejoran consistentemente
3. **Validar**: Win rate se mantiene estable

### Calidad del Trading
1. **Analizar**: Distribución de ROE por trade
2. **Evaluar**: Relación entre duración y rentabilidad
3. **Monitorear**: Uso eficiente del margen

### Gestión de Riesgo
1. **Vigilar**: Máximo drawdown por episodio
2. **Controlar**: Frecuencia de trades perdedores
3. **Optimizar**: Balance entre riesgo y retorno

## 📁 Estructura de Archivos

```
runs/
├── btcbot_20250603_143022/     # Experimento con timestamp
│   ├── events.out.tfevents.*   # Datos de TensorBoard
│   └── hyperparameters.json    # Hiperparámetros del experimento
├── btcbot_20250603_151445/     # Otro experimento
└── ...
```

## 🛠️ Configuración Avanzada

### Personalizar Logging
En `train.py`, puedes modificar:
```python
# Frecuencia de logging
if step % 100 == 0:  # Log cada 100 pasos en lugar de cada paso
    writer.add_scalar('Agent/actor_loss', actor_loss, global_step)
```

### Métricas Personalizadas
```python
# Agregar nuevas métricas
writer.add_scalar('Custom/my_metric', value, global_step)
writer.add_histogram('Custom/action_distribution', actions, global_step)
```

### Escalabilidad
Para entrenamientos largos, considera:
- Logging menos frecuente para pasos individuales
- Agregación de métricas en ventanas
- Limpieza periódica de logs antiguos

## 🎯 Ejemplos de Uso

### Sesión de Entrenamiento Típica
```bash
# Terminal 1: Entrenar
python train.py --symbol BTCUSDT --interval 4h --start-date 2024-01-01 --episodes 2000

# Terminal 2: Monitorear
tensorboard --logdir=runs/ --port=6006

# Terminal 3: Análisis adicional (opcional)
python -c "
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
# Análisis personalizado de métricas
"
```

### Comparar Hiperparámetros
1. Entrenar con `learning_rate: 0.001`
2. Modificar a `learning_rate: 0.0001`
3. Entrenar nuevamente
4. Comparar en TensorBoard

## 📊 Dashboard Recomendado

Para monitoreo en tiempo real, organiza las pestañas de TensorBoard así:

1. **SCALARS - Entrenamiento**: Actor/Critic losses, Alpha
2. **SCALARS - Performance**: Episode returns, Win rate
3. **SCALARS - Trading**: Trade metrics, Balance evolution
4. **SCALARS - Risk**: Drawdown, Position metrics

## 🔄 Integración Continua

### Automatización
```bash
#!/bin/bash
# Script para entrenamiento automatizado con TensorBoard
python train.py --symbol BTCUSDT --episodes 1000 &
sleep 10
tensorboard --logdir=runs/ --host=0.0.0.0 --port=6006 &
echo "Training started. TensorBoard available at http://localhost:6006"
```

---

## 📞 Soporte

Si encuentras problemas:
1. Verificar que TensorBoard esté instalado: `pip install tensorboard`
2. Comprobar permisos de escritura en directorio `runs/`
3. Revisar logs de error en la consola
4. Confirmar que el puerto 6006 esté disponible

**¡Feliz entrenamiento y que tengas visualizaciones exitosas! 🚀📈**
