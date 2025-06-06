# Documentación Detallada: Implementación del Modo Live para BTCBot

## 1. Introducción

Este documento detalla la arquitectura, componentes y consideraciones para la implementación del modo de operación en vivo (`live.py`) del BTCBot. El objetivo es crear un sistema robusto y eficiente que pueda ejecutar estrategias de trading basadas en Aprendizaje por Refuerzo (RL) en tiempo real, interactuando con la API de Binance.

Se priorizará la reutilización de la configuración y lógica existentes del módulo de entrenamiento para asegurar la consistencia entre el backtesting/entrenamiento y la operación en vivo.

## 2. Estructura del Proyecto para el Modo Live

El código específico para el modo *live* se organizará de la siguiente manera:

```
btcbot/
├── live.py                     # Script principal orquestador del modo live
├── src/
│   ├── live/                   # Módulos específicos del modo live
│   │   ├── __init__.py
│   │   ├── binance_api.py      # Clase para interactuar con la API de Binance (REST y WebSockets)
│   │   └── live_trader.py      # Clase con la lógica principal del trading en vivo
│   ├── agente/
│   ├── configuration/
│   ├── data/
│   └── entorno/
└── ... (otros archivos y directorios del proyecto)
```

## 3. Componentes Principales del Modo Live

### 3.1. `live.py` (Orquestador Principal)

* **Responsabilidad:** Script ejecutable que inicia, coordina y gestiona el ciclo de vida del bot en modo *live*.
* **Funciones Clave:**
    * Parsear argumentos de línea de comandos específicos del modo *live* (si los hubiera).
    * Configurar el logging para el modo *live*.
    * Instanciar y configurar `LiveTrader` y `BinanceAPI`.
    * Manejar la carga inicial de datos históricos y modelos/scalers.
    * Ejecutar el bucle principal de operación:
        * Esperar nuevas velas del WebSocket.
        * Orquestar el procesamiento de datos.
        * Solicitar acciones al agente.
        * Gestionar la ejecución de órdenes.
        * Manejar el logging a BigQuery.
    * Implementar un mecanismo de apagado elegante (ej., al recibir `KeyboardInterrupt`):
        * Cerrar conexiones (WebSocket, API).
        * Cerrar posiciones abiertas (configurable, ver sección de Gestión de Riesgo).
        * Asegurar que todos los logs pendientes se escriban.

### 3.2. `src/live/binance_api.py` (Clase `BinanceAPI`)

* **Responsabilidad:** Encapsular toda la comunicación con la API de Binance (REST y WebSockets). Debe ser una clase no ejecutable, instanciada por `live.py`.
* **Funciones Clave:**
    * **Configuración:** Leer credenciales de API (key y secret) y URLs de endpoint (producción/testnet) desde `src/configuration/config.py`.
    * **Conexión WebSocket:**
        * Establecer y mantener una conexión WebSocket para el flujo de datos de klines/candlesticks del par de trading especificado (`symbol`, `interval` de `config.py`).
        * Procesar los mensajes del WebSocket y notificar a `LiveTrader` sobre el cierre de nuevas velas.
        * Implementar lógica de reconexión automática en caso de desconexión.
    * **API REST:**
        * `get_historical_klines(symbol, interval, start_time, end_time, limit)`: Para la descarga inicial de `L+N` velas. Similar a la lógica en `src/data/Adquisicion.py` pero adaptada para una descarga única al inicio.
        * `get_account_balance()`: Para obtener saldos de los activos relevantes.
        * `get_open_positions(symbol)`: Para obtener información sobre posiciones actualmente abiertas para el símbolo operado.
        * `get_order_status(symbol, order_id)`: Para consultar el estado de una orden específica.
        * `get_recent_trades(symbol)`: Para obtener el historial de trades recientes.
        * `place_market_order(symbol, side, quantity)`: Para colocar órdenes a mercado.
        * `cancel_order(symbol, order_id)`: (Si se implementan órdenes límite/stop en el futuro).
    * **Manejo de Errores:**
        * Implementar reintentos con *exponential backoff* para errores transitorios de red o API.
        * Gestionar códigos de error específicos de Binance (ej., fondos insuficientes, tamaño de orden inválido, *rate limits*).
        * Propagar errores de forma clara a `LiveTrader` o `live.py`.

### 3.3. `src/live/live_trader.py` (Clase `LiveTrader`)

* **Responsabilidad:** Contener la lógica principal del ciclo de trading en vivo. No es un script ejecutable por sí mismo.
* **Funciones Clave:**
    * **Inicialización:**
        * Recibir una instancia de `BinanceAPI`.
        * Cargar el agente de RL entrenado (usando `agent.load()` que maneja GCS/local según `config.storage_mode`).
        * Cargar el *scaler* global y derivar/cargar el `price_scaler` (según la decisión de mantener el método de `train.py`).
        * Obtener la ventana inicial de `L+N` velas históricas usando `BinanceAPI`.
        * Inicializar el estado del portafolio (consultando a `BinanceAPI`).
    * **Gestión de Datos:**
        * Mantener un `pd.DataFrame` (o similar) con la ventana deslizante de `L+N` velas.
        * Al recibir una nueva vela de `BinanceAPI`:
            * Añadir la nueva vela al DataFrame.
            * Eliminar la vela más antigua.
            * Disparar el reprocesamiento de datos.
    * **Procesamiento de Datos por Vela:**
        * **Cálculo de Indicadores:** Reutilizar la clase `Indicadores` de `src/data/indicadores.py`. Pasar la ventana de datos actual (`L+N` velas) para calcular los indicadores.
        * **Normalización:** Usar el método `transform` del *scaler* global cargado para normalizar las `L` velas más recientes (con indicadores) que formarán la observación de mercado para el agente.
        * **Construcción de la Observación del Agente:** Combinar los datos de mercado normalizados con las características del portafolio actualizadas (obtenidas de `BinanceAPI`). La estructura debe coincidir con `observation_space_shape` del entrenamiento.
    * **Toma de Decisiones:**
        * Pasar la observación construida al método `agent.select_action(observation, deterministic=True)`.
    * **Ejecución de Órdenes:**
        * Interpretar la acción del agente (ej., valor en [-1,1]).
        * Traducir la acción en una orden de mercado (COMPRAR/VENDER) y determinar la cantidad basada en el `porcentaje_max_inversion_por_trade` del balance actual y la magnitud de la acción, respetando las reglas de Binance (mínimos, precisión).
        * Utilizar `BinanceAPI.place_market_order()`.
    * **Gestión de Estado del Portafolio:**
        * Actualizar el estado interno del portafolio (balance, posiciones) basado en la información de `BinanceAPI` después de confirmaciones de trades o periódicamente.
    * **Gestión de Riesgo:**
        * Implementar el "interruptor de pánico" por pérdida diaria del 10%.
        * Si el interruptor se activa: pausar nuevas operaciones, cerrar posiciones existentes, y registrar el evento.
        * Lógica para cerrar posiciones al iniciar/detener el bot (configurable).
    * **Logging a BigQuery:**
        * Después de cada evento significativo (nueva vela, decisión del agente, orden colocada, orden ejecutada, error), recopilar los datos relevantes y enviarlos para su inserción en BigQuery.

## 4. Flujo de Datos y Lógica Operativa

1.  **Inicio (`live.py`):**
    * Cargar configuración (`config.py`).
    * Instanciar `BinanceAPI` y `LiveTrader`.
    * `LiveTrader` carga el modelo de RL y los *scalers* (de GCS o local).
    * `LiveTrader` solicita a `BinanceAPI` la descarga inicial de `L+N` velas históricas.
    * `LiveTrader` consulta a `BinanceAPI` el estado actual de la cuenta/posiciones.
    * Opcional: Si está configurado, cerrar posiciones existentes.
    * `BinanceAPI` establece la conexión WebSocket para klines.

2.  **Bucle Principal (gestionado por `live.py` o dentro de `LiveTrader`):**
    * `BinanceAPI` recibe un mensaje del WebSocket indicando el cierre de una nueva vela.
    * `BinanceAPI` pasa la información de la nueva vela a `LiveTrader`.
    * `LiveTrader`:
        * Añade la nueva vela a su ventana de datos interna y elimina la más antigua.
        * Calcula los indicadores técnicos para la ventana actualizada (`L+N` velas) usando la lógica de `src/data/indicadores.py`.
        * Prepara la observación para el agente:
            * Toma las `L` velas más recientes con indicadores.
            * Normaliza estos datos de mercado usando el *scaler* global cargado.
            * Obtiene las características actuales del portafolio consultando a `BinanceAPI` (balance, posición actual, etc.) y las normaliza como en el entrenamiento.
            * Combina los datos de mercado y portafolio para formar la observación.
        * Pasa la observación al agente: `action = agent.select_action(observation, deterministic=True)`.
        * Interpreta la `action`.
        * Si la acción implica un trade:
            * Calcula la cantidad de la orden basada en la magnitud de la acción y el `porcentaje_max_inversion_por_trade` del balance actual.
            * Solicita a `BinanceAPI` que coloque una orden a mercado.
        * Actualiza el estado interno del portafolio (puede esperar confirmación de la API o ser optimista).
        * Registra toda la información relevante (observación, acción, detalles de la orden, P&L, etc.) en BigQuery.
        * Verifica las condiciones del "interruptor de pánico".

3.  **Apagado (`live.py`):**
    * Al recibir una señal de interrupción (ej. Ctrl+C):
        * `LiveTrader` (o `live.py`) instruye a `BinanceAPI` para cerrar la conexión WebSocket.
        * Si está configurado, `LiveTrader` instruye a `BinanceAPI` para cerrar todas las posiciones abiertas.
        * Asegura que todos los logs pendientes (especialmente a BigQuery) se completen.
        * Libera otros recursos.

## 5. Gestión de la Configuración

* **Reutilización Máxima:** Se debe depender de `src/configuration/config.yaml` y `src/configuration/config.py` para todos los parámetros compartibles entre entrenamiento y *live*:
    * `symbol`, `interval`
    * `ventana_observacion_size` (L)
    * Configuración de indicadores técnicos (períodos, etc.)
    * `storage_mode` (para modelos/scalers)
    * `project_id` de GCP
    * Parámetros de riesgo como `porcentaje_max_inversion_por_trade`.
    * Modo `testnet`.
* **Parámetros Específicos del Modo Live (en `config.yaml`):**
    * Nombre del *dataset* de BigQuery.
    * Umbral de pérdida diaria para el interruptor de pánico (ej., 0.10 para 10%).
    * Bandera para `close_positions_on_startup`.
    * Bandera para `close_positions_on_shutdown`.

## 6. Logging a BigQuery

* **Objetivo:** Recopilar datos detallados de cada ciclo de decisión y trade para análisis posterior en Looker Studio.
* **Conexión:** Usar la librería `google-cloud-bigquery`.
* **Nombre de Tabla Dinámico:** `TRADE_LOGS_TESTNET` o `TRADE_LOGS_LIVE` dentro de un *dataset* (base de datos) especificado por el usuario en `config.yaml`.
* **Creación de Tabla:** El script `live.py` (o un módulo de utilidad) debe verificar si la tabla existe al inicio y crearla con el esquema definido si no existe.
* **Esquema de la Tabla (Sugerido):**
    * `event_timestamp` (TIMESTAMP, NOT NULL, partición diaria recomendada)
    * `bot_run_id` (STRING, identificador único para esta ejecución del bot)
    * `environment_mode` (STRING, 'TESTNET' o 'LIVE')
    * `symbol` (STRING)
    * `interval` (STRING)
    * `ventana_L_size` (INTEGER)
    * `raw_agent_action` (FLOAT)
    * `interpreted_action` (STRING, ej: 'BUY', 'SELL', 'HOLD_NEUTRAL', 'HOLD_POSITION')
    * `order_id_binance` (STRING, NULLABLE)
    * `order_status_binance` (STRING, NULLABLE)
    * `order_side` (STRING, NULLABLE, 'BUY' o 'SELL')
    * `order_type` (STRING, NULLABLE, 'MARKET')
    * `requested_quantity_asset` (FLOAT, NULLABLE)
    * `executed_quantity_asset` (FLOAT, NULLABLE)
    * `avg_fill_price` (FLOAT, NULLABLE)
    * `commission_paid` (FLOAT, NULLABLE)
    * `commission_asset` (STRING, NULLABLE)
    * `balance_before_action_USDT` (FLOAT, por ejemplo)
    * `equity_before_action_USDT` (FLOAT)
    * `balance_after_trade_USDT` (FLOAT, NULLABLE)
    * `equity_after_trade_USDT` (FLOAT, NULLABLE)
    * `current_position_type_before` (STRING, 'NEUTRAL', 'LONG', 'SHORT')
    * `current_position_size_before` (FLOAT)
    * `current_position_entry_price_before` (FLOAT, NULLABLE)
    * `current_position_pnl_unrealized_roe_before` (FLOAT, NULLABLE)
    * `pnl_realized_trade_abs` (FLOAT, NULLABLE)
    * `pnl_realized_trade_roe` (FLOAT, NULLABLE)
    * `daily_loss_tracker_pct` (FLOAT, para el interruptor de pánico)
    * `error_message` (STRING, NULLABLE)
    * `raw_observation_json` (STRING, JSON de la observación del agente, opcional para depuración profunda)
* **Inserción de Datos:** Usar `client.insert_rows_json()` para enviar lotes de registros. Decidir frecuencia (por evento, por minuto, etc.).

## 7. Gestión de Riesgo

* **Consistencia con Entrenamiento:** La magnitud de la acción del agente debe ser interpretada de la misma forma que en `src/entorno/environment.py` para determinar el tamaño de la posición, utilizando `porcentaje_max_inversion_por_trade`.
* **Interruptor de Pánico:**
    * **Umbral:** Pérdida del 10% del capital inicial del día/sesión (configurable).
    * **Acción:**
        1.  Detener la colocación de nuevas órdenes.
        2.  Cerrar todas las posiciones abiertas existentes para el símbolo operado.
        3.  Registrar el evento críticamente en los logs y en BigQuery.
        4.  El bot puede detenerse o entrar en un modo de solo monitoreo hasta el día siguiente (configurable).
* **Cierre de Posiciones al Iniciar/Detener:**
    * **Al Iniciar:** Parámetro configurable (`--close-on-startup` o en `config.yaml`) para cerrar posiciones existentes para el símbolo al iniciar el bot.
    * **Al Detener (Ctrl+C o Apagado Controlado):** Parámetro configurable para cerrar posiciones existentes antes de que el script finalice.

## 8. Consideraciones Adicionales Finales

* **`price_scaler`:** Como se decidió, se mantendrá el método actual de `train.py` para derivarlo del scaler global. Es crucial asegurar que el conjunto y orden de características del scaler global no cambien entre el entrenamiento y la operación en vivo.
* **Testing Riguroso en Testnet:** Antes de operar con fondos reales, es imperativo probar exhaustivamente en el entorno testnet de Binance durante un período prolongado y bajo diversas condiciones de mercado.
* **Sincronización:** Dado que los WebSockets son asíncronos, asegurar que las actualizaciones de datos y las decisiones de trading se manejen de forma secuencial y sin condiciones de carrera. Python `asyncio` podría ser útil aquí si se elige ese camino, o una gestión cuidadosa de colas y eventos.
* **Monitoreo Externo:** Además del logging a BigQuery, considera tener un sistema de monitoreo simple (quizás usando los logs de la aplicación) para la salud general del bot (estado de conexión, último trade, errores recientes).

Esta documentación debería proporcionar una base sólida para comenzar la implementación. ¡Adelante con el desarrollo!