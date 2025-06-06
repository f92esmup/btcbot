# Métricas Financieras Avanzadas en evaluate_agent

## Resumen de la Implementación

Se han agregado tres métricas financieras avanzadas a la función `evaluate_agent` en `train.py`:

### 1. Máximo Drawdown
- **Descripción**: Mide la mayor pérdida desde un pico hasta un valle en la serie de equity
- **Fórmula**: `max((peak - current_equity) / peak)` donde peak es el máximo acumulado
- **Interpretación**: 
  - 0% = Sin pérdidas desde picos
  - 10% = Pérdida máxima del 10% desde el pico más alto
  - Valores menores son mejores

### 2. Sharpe Ratio
- **Descripción**: Mide el rendimiento ajustado por riesgo
- **Fórmula**: `(return_promedio - tasa_libre_riesgo) / desviación_estándar_returns`
- **Interpretación**:
  - \> 1.0 = Muy bueno
  - 0.5-1.0 = Aceptable
  - < 0.5 = Pobre
  - Valores mayores son mejores

### 3. Sortino Ratio
- **Descripción**: Similar al Sharpe Ratio pero solo considera la volatilidad negativa (downside risk)
- **Fórmula**: `(return_promedio - tasa_libre_riesgo) / desviación_estándar_returns_negativos`
- **Interpretación**:
  - \> 2.0 = Excelente
  - 1.0-2.0 = Bueno
  - < 1.0 = Necesita mejora
  - Valores mayores son mejores

## Funciones Implementadas

### `calculate_max_drawdown(equity_series: list) -> float`
```python
# Ejemplo de uso
equity_series = [1000, 1100, 1050, 900, 950, 1200, 1000]
max_dd = calculate_max_drawdown(equity_series)
print(f"Max Drawdown: {max_dd:.2%}")  # Max Drawdown: 18.18%
```

### `calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.0) -> float`
```python
# Ejemplo de uso
returns = [0.05, -0.02, 0.08, 0.03, -0.01, 0.06]
sharpe = calculate_sharpe_ratio(returns)
print(f"Sharpe Ratio: {sharpe:.4f}")  # Sharpe Ratio: 0.8736
```

### `calculate_sortino_ratio(returns: list, risk_free_rate: float = 0.0) -> float`
```python
# Ejemplo de uso
returns = [0.05, -0.02, 0.08, 0.03, -0.01, 0.06]
sortino = calculate_sortino_ratio(returns)
print(f"Sortino Ratio: {sortino:.4f}")  # Sortino Ratio: 6.3333
```

## Integración en evaluate_agent

Las métricas se calculan automáticamente durante la evaluación y se incluyen en:

### 1. Logs de Consola
```
Métricas de evaluación avanzadas:
  - Máximo Drawdown: 18.18%
  - Sharpe Ratio: 0.8736
  - Sortino Ratio: 6.3333
```

### 2. TensorBoard
Las métricas se registran en TensorBoard bajo el namespace `Evaluation/`:
- `Evaluation/Max_Drawdown_Pct`
- `Evaluation/Sharpe_Ratio` 
- `Evaluation/Sortino_Ratio`

### 3. Diccionario de Retorno
```python
eval_metrics = evaluate_agent(agent, env, eval_episodes, logger, writer, episode)
print(f"Max Drawdown: {eval_metrics['max_drawdown']:.2%}")
print(f"Sharpe Ratio: {eval_metrics['sharpe_ratio']:.4f}")
print(f"Sortino Ratio: {eval_metrics['sortino_ratio']:.4f}")
```

## Casos Especiales Manejados

1. **Series vacías**: Retorna 0.0 para evitar errores
2. **Sin volatilidad**: Retorna 0.0 para Sharpe/Sortino cuando desviación = 0
3. **Solo retornos positivos**: Sortino Ratio retorna infinity si promedio > 0
4. **División por cero**: Protegido en todos los cálculos

## Beneficios de la Implementación

1. **Mejor evaluación del rendimiento**: Las métricas proporcionan una visión más completa del desempeño del agente
2. **Gestión de riesgo**: El drawdown y Sortino ratio ayudan a evaluar el riesgo de las estrategias
3. **Comparación de modelos**: Facilita la comparación objetiva entre diferentes versiones del agente
4. **Monitoreo en tiempo real**: Las métricas se registran en TensorBoard para análisis visual
5. **Toma de decisiones**: Ayuda a identificar cuándo un modelo está mejorando realmente vs. solo teniendo suerte

## Uso en Entrenamiento

Durante el entrenamiento, estas métricas aparecerán automáticamente:
- En los logs cada vez que se ejecute una evaluación (cada `eval_frequency` episodios)
- En TensorBoard para visualización gráfica
- En los logs finales de resumen del entrenamiento

La implementación es completamente retrocompatible y no afecta el funcionamiento existente del sistema.
