# Guía de Entrenamiento del Agente SAC con Transformer

## Resumen

Este documento explica cómo entrenar el agente SAC (Soft Actor-Critic) con arquitectura Transformer para trading de futuros de Bitcoin. El sistema está completamente implementado y listo para usar.

## Arquitectura Implementada

### Agente SAC con Transformer
- **Algoritmo**: Soft Actor-Critic (SAC) con temperatura alfa aprendible
- **Arquitectura**: Transformer Encoder para procesar secuencias de mercado + MLP heads
- **Redes**: Actor (política) y doble Crítico con redes objetivo
- **Replay Buffer**: Buffer de experiencia con muestreo eficiente

### Configuración Centralizada
Todos los parámetros están en `src/configuration/config.yaml`:

```yaml
agent:
  algorithm: "SAC"
  replay_buffer_size: 1000000
  batch_size: 256
  
  hiperparametros_sac:
    gamma: 0.99                   # Factor de descuento
    tau: 0.005                    # Actualización suave de redes objetivo
    actor_learning_rate: 0.0003   # LR del actor
    critic_learning_rate: 0.0003  # LR del crítico
    learn_alpha: true             # Alpha aprendible
    alpha_learning_rate: 0.0003   # LR de alpha
    target_entropy: -1.0          # Entropía objetivo
    
  transformer:
    d_model: 128                  # Dimensión del modelo
    n_head: 4                     # Cabezales de atención
    num_encoder_layers: 3         # Capas del encoder
    dim_feedforward: 256          # Dimensión FFN
    dropout_rate: 0.1             # Dropout
    
  mlp_heads:
    hidden_dims: [256, 256]       # Capas ocultas del MLP
```

## Instalación

1. **Instalar dependencias**:
```bash
pip install torch>=2.0.0
pip install -r requirements.txt
```

2. **Verificar instalación**:
```bash
python test_training_setup.py
```

## Entrenamiento

### Comando Básico
```bash
python train.py --symbol BTCUSDT --interval 1h --start-date 2024-01-01
```

### Parámetros de Entrenamiento
```bash
python train.py \
  --symbol BTCUSDT \
  --interval 1h \
  --start-date 2024-01-01 \
  --episodes 1000 \
  --eval-frequency 50 \
  --eval-episodes 5 \
  --save-frequency 100
```

### Parámetros Disponibles

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--symbol` | Par de trading (ej: BTCUSDT) | Requerido |
| `--interval` | Intervalo temporal (1m, 5m, 1h, 1d, etc.) | Requerido |
| `--start-date` | Fecha inicio (YYYY-MM-DD) | Requerido |
| `--episodes` | Episodios de entrenamiento | 1000 |
| `--eval-frequency` | Frecuencia de evaluación | 50 |
| `--eval-episodes` | Episodios para evaluación | 5 |
| `--save-frequency` | Frecuencia de guardado | 100 |
| `--no-cuda` | Forzar uso de CPU | false |

## Proceso de Entrenamiento

El script `train.py` ejecuta las siguientes fases:

### Fase 1: Adquisición de Datos
- Descarga datos históricos desde Binance
- Valida y limpia los datos
- Almacena en formato optimizado

### Fase 2: Cálculo de Indicadores
- RSI, MACD, Bollinger Bands, etc.
- Medias móviles y volatilidad
- Análisis de volumen

### Fase 3: Normalización
- Normalización MinMax [0,1]
- Preserva relaciones temporales
- Guarda scaler para inferencia

### Fase 4: Entrenamiento SAC
- Crea entorno de trading realista
- Inicializa agente con configuración
- Ejecuta entrenamiento con evaluación periódica
- Guarda mejores modelos automáticamente

## Entorno de Trading

### Características
- **Apalancamiento**: Configurable (default 10x)
- **Comisiones**: Realistas (0.04% taker)
- **Slippage**: Simulado (0.005%)
- **Gestión de Riesgo**: Máximo drawdown y stop-loss

### Espacios
- **Observación**: Ventana de mercado (24 períodos) + estado del portfolio (4 features)
- **Acción**: Continua [-1, 1] (vender fuerte, mantener, comprar fuerte)

### Recompensas
- Basadas en cambios de equity
- Penalización por trades frecuentes
- Bonus por gestión de riesgo

## Monitoreo del Entrenamiento

### Métricas Registradas
- **Return promedio**: Rendimiento por episodio
- **Profit %**: Ganancia/pérdida porcentual
- **Win Rate**: Porcentaje de trades exitosos
- **Losses**: Actor, Critic y Alpha loss
- **Alpha**: Valor del parámetro de temperatura

### Logs de Ejemplo
```
Episodio 50/1000 | Return: 15.23 | Profit: 2.34% | Avg Return (10): 12.45
Alpha: 0.1234 | Actor Loss: 0.0123 | Critic Loss: 0.0456

=== Evaluación en episodio 50 ===
Métricas de evaluación:
  - Return promedio: 18.45 ± 3.21
  - Profit promedio: 2.89% ± 1.12%
  - Win rate: 65.00%
  - Nuevo mejor modelo guardado: models/best_model.pth
```

## Modelos Guardados

### Estructura de Archivos
```
models/
├── best_model.pth              # Mejor modelo (mayor return)
├── final_model.pth             # Modelo final del entrenamiento
├── checkpoint_episode_100.pth  # Checkpoints periódicos
├── checkpoint_episode_200.pth
└── ...
```

### Cargar Modelo
```python
from src.agente.agent import TransformerSACAgent

agent = TransformerSACAgent(...)
agent.load("models/best_model.pth")
```

## Configuración del Hardware

### CPU
- Mínimo: 4 cores
- Recomendado: 8+ cores
- RAM: 8GB mínimo, 16GB recomendado

### GPU (Opcional)
- CUDA compatible
- 6GB+ VRAM recomendado
- Acelera entrenamiento significativamente

### Almacenamiento
- 5GB para datos históricos
- 1GB para modelos y checkpoints

## Troubleshooting

### Error: "No module named 'torch'"
```bash
pip install torch>=2.0.0
```

### Error: "CUDA out of memory"
```bash
python train.py --no-cuda [otros parámetros]
```

### Error: "Insufficient historical data"
- Usar fecha de inicio más antigua
- Verificar disponibilidad de datos en Binance

### Rendimiento Lento
- Reducir `batch_size` en config.yaml
- Reducir `replay_buffer_size`
- Usar GPU si está disponible

## Próximos Pasos

1. **Ejecutar entrenamiento de prueba**: 100 episodios para verificar funcionamiento
2. **Entrenar modelo completo**: 1000+ episodios para convergencia
3. **Optimizar hiperparámetros**: Ajustar learning rates, architecture
4. **Backtesting**: Evaluar en datos out-of-sample
5. **Despliegue**: Integrar con trading en vivo

## Soporte

Para preguntas o problemas:
1. Verificar logs de entrenamiento
2. Ejecutar `python test_training_setup.py` para diagnosticar
3. Revisar configuración en `config.yaml`
4. Consultar documentación de cada módulo
