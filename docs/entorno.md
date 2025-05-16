El Módulo 3: Entorno de Trading es donde la estrategia cobra vida y el agente aprende. Tu descripción es exhaustiva y cubre todos los elementos esenciales para una simulación de trading de futuros realista y robusta, compatible con `gymnasium`.

La idea de modularizar la lógica del broker en una clase `SimulatedBroker` es excelente, ya que facilitará enormemente la transición a un entorno de trading en vivo simplemente intercambiando esa clase por una que interactúe con la API real de Binance, manteniendo intacta la lógica del `TradingEnvironment`.

Vamos a consolidar y detallar la definición de este módulo basándonos en tus especificaciones.

**Módulo 3: Entorno de Trading (Trading Environment)**

**1. Responsabilidades y Funcionalidades Exactas:**

  * **Implementación de la Interfaz `gymnasium.Env`:**
      * Clase principal `TradingEnvironment` que hereda de `gymnasium.Env`.
      * Implementación de métodos clave: `__init__`, `reset`, `step`, `render` (inicialmente puede ser un print del estado/equity, o una visualización más avanzada si se desea más adelante), `close`.
      * Definición precisa de `observation_space` y `action_space`.
  * **Gestión Integral del Estado de la Cartera de Futuros:**
      * Mantenimiento y actualización en cada paso del estado de la cuenta simulada:
          * `initial_equity_episode`: Equity al inicio del episodio actual (para cálculo de drawdown).
          * `current_equity`: Valor total actual de la cuenta.
          * `balance`: Efectivo o colateral disponible.
          * `active_position_side`: Entero (-1 para Corto, 0 para Neutral, 1 para Largo).
          * `active_position_size_contracts`: Tamaño de la posición en contratos o unidades de BTC.
          * `active_position_entry_price`: Precio de entrada de la posición activa.
          * `unrealized_pnl`: Ganancia o pérdida flotante de la posición activa.
          * `margin_used`: Margen retenido para la posición actual.
          * `available_margin`: `current_equity - margin_used` (aproximación, Binance tiene cálculos más detallados).
          * `configured_leverage`: Apalancamiento fijo (ej. 10x).
          * `steps_in_current_position`: Contador de pasos desde que se abrió la posición actual.
  * **Simulación Detallada de Ejecución de Órdenes (vía `SimulatedBroker`):**
      * **Comisiones (Taker Fee):** Aplicar una comisión configurable (ej., 0.04% de Binance) sobre el valor nocional de cada operación (apertura y cierre).
      * **Modelado de Slippage:** Ajustar el precio de ejecución en contra del agente en `slippage_atr_multiplier * ATR(14)` del KLine actual (configurable, ej. 0.1). El ATR se tomará de los datos de mercado.
          * Compra (Largo): `execution_price = market_price_ask_simulated = current_market_close_price + (slippage_factor * ATR)`.
          * Venta (Corto o cierre de Largo): `execution_price = market_price_bid_simulated = current_market_close_price - (slippage_factor * ATR)`.
      * **Mínimos de Orden:** Considerar (y potencialmente rechazar órdenes) si el tamaño de la posición calculada no cumple con los mínimos de Binance (ej., 0.001 BTC para futuros BTCUSDT).
  * **Lógica de "Una Operación a la Vez":** El entorno solo permite una posición (Larga o Corta) o estar Neutral. No se permite promediar ni abrir múltiples posiciones.
  * **Dimensionamiento de Posición Fijo (Porcentual al Equity):**
      * Cuando se abre una nueva posición, su valor nocional es `current_equity * position_size_pct_equity` (ej. 5% del equity).
      * `position_size_contracts = (current_equity * position_size_pct_equity * configured_leverage) / execution_price`.
  * **Apalancamiento Fijo:** Configurable (ej. 10x). Afecta el margen y el precio de liquidación.
  * **Definición del Espacio de Observación (`observation_space`):**
      * Será un `gymnasium.spaces.Dict` para manejar la heterogeneidad de los datos de forma clara:
          * `'market_features'`: `gymnasium.spaces.Box` de forma `(L, N_features_mercado)` (ej. `(96, 20)`), conteniendo la secuencia de datos de mercado preprocesados del Módulo 2.
          * `'portfolio_features'`: `gymnasium.spaces.Box` de forma `(8,)`, conteniendo las 8 características de cartera normalizadas:
            1.  `Estado Posición`: {-1, 0, 1}.
            2.  `Tamaño Posición Normalizado`: ej., `(size_contracts * entry_price) / initial_equity_episode` o similar.
            3.  `Precio Entrada Normalizado`: ej., `(entry_price - current_price) / current_price` o `entry_price / current_price - 1`.
            4.  `P&L No Realizado Normalizado`: ej., `unrealized_pnl / current_equity`.
            5.  `Retorno Log Equity (último paso)`: `log(equity_t / equity_{t-1})`. (Esto es la recompensa del paso anterior, actuando como una característica de estado).
            6.  `Ratio de Margen Disponible`: ej., `available_margin / current_equity`.
            7.  `Pasos en Posición Normalizados`: ej., `steps_in_current_position / L` (o algún máximo razonable).
            8.  `Apalancamiento Configurado`: Constante (ej. 10.0), pero incluido para consistencia.
  * **Definición del Espacio de Acciones (`action_space`):**
      * Un `gymnasium.spaces.Box` continuo de una dimensión: `Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)`.
  * **Interpretación de la Acción del Agente (`action_signal`):**
      * `action_signal > action_threshold` (ej. \> 0.15): Intento de Abrir Largo.
          * Si Neutral: Abrir Largo.
          * Si Corto: Cerrar Corto y luego Abrir Largo (dos transacciones).
          * Si Largo: Mantener Largo (no se reabre ni se aumenta).
      * `action_signal < -action_threshold` (ej. \< -0.15): Intento de Abrir Corto.
          * Si Neutral: Abrir Corto.
          * Si Largo: Cerrar Largo y luego Abrir Corto (dos transacciones).
          * Si Corto: Mantener Corto.
      * `-action_threshold <= action_signal <= action_threshold`: Intento de Cerrar/Neutralizar.
          * Si Largo o Corto: Cerrar Posición Actual.
          * Si Neutral: Mantener Neutral.
  * **Cálculo de la Función de Recompensa (`reward`):**
      * Por paso `t`: `recompensa_t = log(current_equity_t / current_equity_{t-1})`. Esta se calcula en el método `step` del entorno.
  * **Simulación de Liquidación de Posición:**
      * Ocurre si el precio de mercado se mueve en contra de la posición un porcentaje igual a `(1 / configured_leverage) * liquidation_safety_factor` (ej. un movimiento del 8% para 10x de apalancamiento y factor 0.8).
      * El precio de liquidación se calcula a partir del `active_position_entry_price`.
      * Si el `Low` (para Largos) o `High` (para Cortos) del KLine actual cruza este precio, la posición se cierra forzosamente (considerando el precio de liquidación como precio de ejecución, aplicando comisión). El `equity` se actualiza, y `terminated = True`.
  * **Condiciones de Fin de Episodio (para Entrenamiento):**
      * `terminated = True`:
          * **Drawdown Máximo de Equity:** Si `current_equity <= initial_equity_episode * (1 + equity_drawdown_threshold_episode_end)` (ej. -20% drawdown).
          * **Liquidación de Posición.**
      * `truncated = True`:
          * **Agotamiento del Conjunto de Datos:** Si el entorno alcanza el final de los datos de mercado disponibles (`current_step_index >= len(market_data_full_dataset) - L`).
      * **Reinicio de Episodio (`reset` method):**
          * Reinicia el estado de la cartera (`current_equity = initial_equity_configurado`, sin posiciones).
          * Selecciona un **índice de inicio aleatorio** en el conjunto de datos de entrenamiento para el nuevo episodio, asegurando que haya suficientes datos para al menos una secuencia de longitud `L`.

**2. Clase `SimulatedBroker` (Propuesta de Interfaz):**

Esta clase se alojará en `src/environments/simulated_broker.py`.

  * **`__init__(self, taker_fee_rate: float, slippage_atr_multiplier: float, min_order_size_btc: float = 0.001)`**
  * **`calculate_execution_details(self, desired_action: str, market_close_price: float, atr_value: float, position_to_close_entry_price: float = None, position_to_close_size: float = None)`**
      * `desired_action`: "OPEN\_LONG", "OPEN\_SHORT", "CLOSE\_LONG", "CLOSE\_SHORT".
      * Calcula `execution_price` (con slippage), `potential_pnl` (si es cierre), `commission_to_be_paid` (basado en `market_close_price` o `execution_price` para el nocional).
      * Devuelve un diccionario con estos detalles. NO modifica el estado de la cuenta.
  * **`calculate_position_size_contracts(self, equity: float, position_size_pct_equity: float, leverage: float, execution_price: float)`**
      * Devuelve el tamaño de la posición en contratos/BTC. Verifica el mínimo de orden.
  * **`calculate_margin_required(self, position_size_contracts: float, execution_price: float, leverage: float)`**
      * Devuelve el margen necesario.

El `TradingEnvironment` usará estos métodos para obtener los costos y precios, y luego actualizará su propio estado de cartera (`current_equity`, `balance`, `active_position_*`, etc.). Esto mantiene al `SimulatedBroker` sin estado (o con estado mínimo) y facilita su reemplazo.

**3. Tecnologías Clave:**

  * Python
  * `gymnasium`
  * NumPy
  * Pandas (para manejar el dataset completo del cual se extraen las observaciones)

**4. Parámetros Configurables (`src/environments/environment_config.yaml`):**

```yaml
# --- Configuración General del Entorno ---
env_id: "FuturesTradingEnv-v0" # Identificador para registrar en Gymnasium
initial_equity: 10000.0       # Equity inicial para cada episodio (USDT)
max_episode_steps_use_dataset_length: true # Si true, max_steps = len(dataset) - L
allow_random_episode_start: true # Para el entrenamiento, empezar en puntos aleatorios del dataset

# --- Configuración de Trading y Cartera ---
leverage: 10.0
position_size_pct_equity: 0.05 # 5% del equity actual para el tamaño nocional
taker_fee_rate: 0.0004         # Comisión de Taker (0.04%)
slippage_atr_multiplier: 0.1   # Multiplicador de ATR(14) para el slippage (por lado)
min_order_size_btc: 0.001      # Mínimo tamaño de orden en BTC (o la unidad base)

# --- Lógica de Acción del Agente ---
action_threshold: 0.15    # Umbral para interpretar la señal continua del agente

# --- Lógica de Finalización de Episodio y Liquidación ---
equity_drawdown_threshold_episode_end: -0.20 # -20% drawdown del equity inicial del episodio
liquidation_safety_factor: 0.8 # (1 / leverage) * safety_factor = % de movimiento para liquidación

# --- Características de Observación (Portfolio) ---
# Parámetros para la normalización de las características de la cartera, si se hace dentro del entorno
portfolio_features_normalization:
  max_steps_in_position: 288 # ej. L * 3 (para normalizar 'steps_in_current_position')
  # Otros parámetros de normalización si son necesarios (ej. medias/std para Z-score de P&L o precio entrada)

# --- Carga de Datos de Mercado (del Módulo 2) ---
# El entorno necesitará saber dónde encontrar los datos preprocesados.
# Esto puede ser una ruta fija o buscar un archivo específico.
# Ejemplo:
# processed_data_directory: "data/processed/"
# processed_data_file_identifier: "_L96_market_features.npz" # Parte del nombre del archivo
```

**Función de Recompensa:**

Confirmado: `recompensa_t = log(current_equity_t / current_equity_{t-1})` es una excelente elección y estará integrada en el método `step()` del `TradingEnvironment`.

**Estructura de Archivos:**

  * `src/environments/__init__.py` (para registrar el entorno en Gymnasium)
  * `src/environments/trading_env.py` (Clase `TradingEnvironment`)
  * `src/environments/simulated_broker.py` (Clase `SimulatedBroker`)
  * `src/environments/environment_config.yaml` (Archivo de configuración)

**Próximos Pasos para la Implementación de este Módulo:**

1.  **Implementar `SimulatedBroker`:** Con sus métodos de cálculo sin estado.
2.  **Implementar `TradingEnvironment`:**
      * **`__init__`:** Cargar configuración, datos de mercado preprocesados (del Módulo 2), instanciar `SimulatedBroker`, definir `observation_space` y `action_space`.
      * **`reset`:** Lógica para reiniciar la cartera, seleccionar un punto de inicio aleatorio en los datos, y devolver la observación inicial.
      * **`step`:** Lógica principal de interacción: interpretar acción, usar `SimulatedBroker` para obtener costos/precios de ejecución, actualizar estado de cartera, calcular P\&L, calcular recompensa, verificar condiciones de terminación, y construir la siguiente observación.
      * **Normalización de Características de Cartera:** Implementar la normalización de las 8 características de cartera antes de que se incluyan en la observación devuelta por `reset` y `step`.
3.  **Registro del Entorno:** Registrar el `TradingEnvironment` con `gymnasium.register` para que pueda ser instanciado con `gymnasium.make(env_id)`.

Este diseño detallado te proporciona un plan claro para construir un entorno de trading robusto y flexible, preparado para el entrenamiento de tu agente de RL y con la vista puesta en una futura transición a operaciones en vivo. 