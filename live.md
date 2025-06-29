# Diseño del Modo de Operación en Vivo (Live Trading Bot)

Este documento detalla la conceptualización, arquitectura y consideraciones operacionales para el agente de trading en su modo de operación en vivo. El objetivo es que el bot opere con capital real en el mercado, utilizando el modelo de Reinforcement Learning entrenado como su "cerebro" de decisión.

## 1. Filosofía Central y Principios Operacionales

*   **Operación en Tiempo Real:** El bot interactuará directamente con los datos de mercado y las APIs de ejecución de órdenes de Binance en tiempo real.
*   **No Simulación:** A diferencia del entorno de entrenamiento, no habrá simulación alguna. Todas las interacciones (adquisición de datos, estado del portfolio, ejecución de órdenes) reflejarán la realidad del mercado.
*   **Modelo como "Caja Negra":** El modelo de RL entrenado y desplegado en Vertex AI actuará como una caja negra de predicción. El bot en vivo le enviará observaciones y recibirá acciones, sin necesidad de conocer la lógica interna del modelo.
*   **Reutilización de Componentes:** Se priorizará la adaptación y reutilización de los módulos existentes del proyecto (adquisición de datos, cálculo de indicadores, normalización, configuración) para mantener la consistencia y eficiencia.
*   **Principios SOLID:** El diseño de nuevos componentes y la adaptación de los existentes se adherirán estrictamente a los principios SOLID, con un énfasis particular en el Principio de Responsabilidad Única (SRP).

## 2. Arquitectura de Componentes

El modo live se estructurará en torno a varios componentes clave, algunos nuevos y otros adaptados de la fase de entrenamiento:

### 2.1. Componentes Nuevos

*   **`LiveTradingManager` (Orquestador Principal)**
    *   **Responsabilidad:** Coordinar el ciclo de vida completo del bot en vivo. Es el "cerebro" que orquesta la interacción entre todos los demás componentes.
    *   **Funcionalidades:**
        *   Inicialización: Carga de configuraciones, conexión a servicios, inicialización de componentes.
        *   Bucle Principal: Ejecución continua del ciclo de trading (obtener datos, construir observación, predecir, ejecutar, registrar).
        *   Manejo de Eventos: Reaccionar a la llegada de nuevas velas, señales de riesgo, etc.
        *   Cierre Elegante (`Graceful Shutdown`): Gestionar el cierre ordenado del bot, incluyendo el cierre de posiciones abiertas y el envío de logs pendientes.
        *   Control de `Kill Switch`: Monitorear la señal de parada de emergencia.

*   **`BinanceLiveDataReader` (Adquisición de Datos en Tiempo Real)**
    *   **Responsabilidad:** Conectarse a los WebSockets de Binance para recibir datos de mercado en tiempo real (velas o ticks).
    *   **Funcionalidades:**
        *   Conexión y Mantenimiento de WebSocket: Establecer y mantener una conexión persistente.
        *   Reconexión Automática: Lógica robusta para manejar desconexiones y reestablecer la conexión.
        *   Adquisición de Histórico Inicial (`Warm-up Data`): Al iniciar, descargar un número mínimo de velas históricas (usando la API REST) para satisfacer los requisitos de los indicadores técnicos.
        *   Ventana Deslizante (`Sliding Window`): Mantener una `deque` (o estructura similar) en memoria con la ventana de datos históricos requerida por el `LiveObservationBuilder`. Cada nueva vela se añade y la más antigua se elimina.
        *   Disparador de Eventos: Emitir un evento o llamar a un callback en el `LiveTradingManager` cada vez que una nueva vela completa y validada esté disponible. Este evento será el principal disparador del ciclo de trading.

*   **`LiveObservationBuilder` (Construcción de la Observación)**
    *   **Responsabilidad:** Preparar la observación para el modelo de RL, aplicando los mismos pasos de preprocesamiento que durante el entrenamiento.
    *   **Funcionalidades:**
        *   Cálculo de Indicadores: Aplicar los indicadores técnicos configurados (`src/data/indicadores.py`) sobre la ventana de datos de mercado actualizada.
        *   Normalización: Utilizar los `scaler` y `price_scaler` (cargados desde el `run_id` específico del modelo) para normalizar todas las características de la observación al rango esperado por el modelo.
        *   Formateo: Asegurar que la observación final tenga la forma y el tipo de datos exactos que espera el endpoint de Vertex AI.

*   **`VertexAIModelClient` (Cliente de Inferencia del Modelo)**
    *   **Responsabilidad:** Encapsular la comunicación con el endpoint de Vertex AI donde está desplegado el modelo de RL.
    *   **Funcionalidades:**
        *   Envío de Peticiones: Enviar la observación formateada al endpoint de Vertex AI.
        *   Recepción de Predicciones: Parsear la respuesta del endpoint para extraer la acción predicha por el modelo.
        *   Manejo de Errores: Gestionar errores de red, timeouts y errores de inferencia del modelo.

*   **`LivePortfolioManager` (Gestor de Portfolio y Ejecución de Órdenes)**
    *   **Responsabilidad:** Mantener el estado real del portfolio (balance, posiciones abiertas, PNL no realizado) y ejecutar órdenes de trading a través de la API de Binance.
    *   **Funcionalidades:**
        *   Sincronización de Estado: Al iniciar, consultar el estado real de la cuenta en Binance. Durante la operación, mantener el estado interno sincronizado con la cuenta real (posiblemente a través de WebSockets de usuario o polling).
        *   Cierre de Posiciones al Inicio: Si se detectan posiciones abiertas al iniciar el bot, cerrarlas de forma controlada para asegurar un estado limpio.
        *   Ejecución de Órdenes: Enviar órdenes a mercado a la API de Binance (compra/venta). La lógica de cálculo de tamaño de posición y aplicación de comisiones se adaptará de la fase de entrenamiento.
        *   Gestión de Posiciones: Abrir, cerrar y actualizar el PNL no realizado de las posiciones.
        *   Cierre de Posiciones al Fallo/Cierre: Si el bot se detiene por un error o un cierre elegante, cerrar todas las posiciones abiertas.

*   **`RiskManager` (Control de Riesgo Independiente)**
    *   **Responsabilidad:** Monitorear el riesgo del portfolio de forma independiente de la lógica de decisión del agente y tomar acciones correctivas si se cruzan umbrales de riesgo.
    *   **Funcionalidades:**
        *   Monitoreo de Métricas: Vigilar el `equity_actual`, `drawdown_actual`, `PNL_diario`, etc.
        *   Umbrales de Riesgo: Definir y aplicar umbrales configurables (ej. `max_drawdown_configurado_cuenta`).
        *   Acción de Emergencia: Si un umbral se cruza:
            *   Instruir al `LivePortfolioManager` para cerrar **todas las posiciones abiertas** de forma inmediata.
            *   Emitir una señal al `LiveTradingManager` para **detener la operativa** (no más predicciones ni nuevas órdenes).
            *   Registrar el evento crítico en el `BigQueryLogger`.

*   **`BigQueryLogger` (Registro de Operaciones)**
    *   **Responsabilidad:** Almacenar de forma persistente todos los eventos y datos relevantes de la operativa en BigQuery para análisis posterior.
    *   **Funcionalidades:**
        *   Conexión a BigQuery: Gestionar la autenticación y conexión.
        *   Definición de Esquemas: Crear y mantener los esquemas de las tablas para diferentes tipos de eventos (acciones del agente, trades ejecutados, estado del portfolio, eventos de riesgo).
        *   Inserción de Datos: Métodos para insertar datos de forma eficiente (streaming o micro-batches).
        *   Tipos de Datos a Registrar:
            *   `acciones_agente`: `timestamp`, `observation_hash`, `action_raw`, `intencion`, `magnitud_efectiva`, `prediccion_modelo_id`.
            *   `trades_ejecutados`: `timestamp_apertura`, `timestamp_cierre`, `tipo_operacion`, `precio_entrada`, `precio_salida`, `tamaño_activo`, `margen_usado`, `pnl_abs`, `roe`, `pasos_duracion`, `trade_id`.
            *   `estado_portfolio`: `timestamp`, `balance_actual`, `equity_actual`, `posicion_tipo`, `pnl_no_realizado_abs`, `pnl_no_realizado_roe`, `max_equity_alcanzado_sesion`.
            *   `eventos_riesgo`: `timestamp`, `tipo_evento`, `descripcion`, `valor_metrica`, `umbral_metrica`.

*   **`TelegramNotifier` (Notificaciones de Eventos)**
    *   **Responsabilidad:** Enviar notificaciones en tiempo real a un bot de Telegram para eventos importantes.
    *   **Funcionalidades:**
        *   Conexión a la API de Telegram.
        *   Envío de Mensajes: Métodos para enviar mensajes de texto con información relevante.
        *   Eventos a Notificar: Inicio/cierre del bot, apertura/cierre de posición (con PNL), alcanzar umbrales de riesgo, errores críticos, desconexiones.

### 2.2. Componentes Reutilizados/Adaptados

*   **`TransformerSACAgent` (Rol en Modo Live)**
    *   **Adaptación:** En el modo live, el agente no "aprende" ni "selecciona" acciones directamente de sus redes locales. Su método `select_action` se adaptará para llamar al `VertexAIModelClient` y obtener la acción del endpoint desplegado.

*   **`config.py` (Gestión de Configuración)**
    *   **Reutilización:** Continuará siendo la fuente central de configuración, incluyendo API keys de Binance (desde Secret Manager), URLs del endpoint de Vertex AI, y parámetros específicos del trading en vivo.
    *   **Adaptación:** La configuración `is_testnet` determinará si el bot opera en modo "paper trading" (Binance Testnet) o con capital real (Binance Production), utilizando las API keys correspondientes.

## 3. Estrategia de Despliegue

*   **Plataforma de Despliegue:** **Compute Engine (VM)**
    *   **Justificación:** Para un proceso de larga duración (24/7) que mantiene conexiones WebSocket persistentes y gestiona un estado continuo, una VM ofrece el control, la estabilidad y la persistencia en memoria necesarios. Esto simplifica la lógica del bot al no tener que lidiar con los reinicios efímeros y la naturaleza sin estado de plataformas como Cloud Run para este caso de uso.
    *   **Tipo de Máquina:** Se optará por una VM de bajo costo (ej. `e2-micro` o `e2-small`) que debería ser suficiente para la carga de trabajo de un solo bot.
*   **Contenedorización:** El bot se desplegará como un **contenedor Docker** en la VM. Esto asegura la portabilidad y la consistencia del entorno de ejecución entre desarrollo y producción. La imagen Docker se construirá y se subirá a Google Container Registry (GCR) o Artifact Registry.
*   **No Endpoint Propio:** El bot en la VM no expondrá ningún endpoint API propio. Su función es consumir el endpoint de Vertex AI para la inferencia del modelo.

## 4. Consideraciones Operacionales Clave

*   **Disparador del Ciclo de Trading:** La llegada de una **nueva vela completa y validada** a través del WebSocket de Binance será el único disparador del ciclo de predicción y ejecución. Esto elimina la necesidad de temporizadores y asegura una sincronización perfecta con el mercado.
*   **Scalers Específicos del `run_id`:** Los `scaler.pkl` y `price_scaler.pkl` utilizados para la normalización de la observación serán cargados desde el `run_id` específico del modelo desplegado en Vertex AI. Esto garantiza que la entrada al modelo en producción sea idéntica en formato a la que vio durante el entrenamiento.
*   **Órdenes a Mercado:** Todas las operaciones de compra/venta se ejecutarán como órdenes a mercado, simplificando la lógica de ejecución y manteniendo la consistencia con el entrenamiento (conceptualmente).
*   **Kill Switch de Emergencia:** Se implementará un mecanismo de "Kill Switch" (ej. un archivo en GCS o un mensaje en Pub/Sub) que, al activarse, forzará el cierre de todas las posiciones y la detención inmediata de la operativa del bot.

## 5. ¿Qué cosas me he dejado? (Consideraciones Adicionales)

Aunque hemos cubierto los aspectos fundamentales, la operación de un bot de trading en vivo implica desafíos adicionales que debemos tener en cuenta para futuras iteraciones o para una implementación más robusta:

*   **Manejo Granular de Errores y Reintentos:**
    *   Implementar lógica de reintentos con backoff exponencial para errores transitorios de la API de Binance (ej. límites de tasa, problemas de red).
    *   Distinguir entre errores transitorios y errores persistentes (ej. credenciales inválidas, órdenes mal formadas) que requieren una acción diferente (ej. detener la operativa y alertar).
*   **Persistencia del Estado para Recuperación ante Desastres:**
    *   Más allá del "graceful shutdown", ¿cómo recupera el bot su estado si la VM falla inesperadamente?
    *   Necesidad de persistir el estado clave del `LivePortfolioManager` (balance, posiciones abiertas, última vela procesada, PNL no realizado) en un almacenamiento duradero externo (ej. Cloud SQL, Firestore, Redis) para poder retomar la operativa desde el último punto conocido.
*   **Sincronización Horaria:**
    *   Asegurar que la VM tenga una hora precisa y sincronizada con los servidores de Binance para evitar problemas con los timestamps de las órdenes y los datos de mercado.
*   **Manejo de Órdenes Complejas:**
    *   Actualmente, asumimos órdenes a mercado simples. En la realidad, pueden ocurrir **fills parciales** o **rechazos** de órdenes.
    *   El `LivePortfolioManager` debería ser capaz de:
        *   Confirmar que una orden se ha ejecutado completamente.
        *   Manejar fills parciales (ej. enviar una nueva orden para el resto, o ajustar el tamaño de la posición).
        *   Detectar y reaccionar a órdenes rechazadas.
        *   Considerar la latencia entre el envío de la orden y la confirmación de la ejecución.
*   **Monitoreo Proactivo y Alertas Detalladas:**
    *   Más allá de las notificaciones de Telegram, implementar un sistema de monitoreo más completo (ej. Cloud Monitoring) para la salud del proceso del bot (uso de CPU/Memoria, si el proceso sigue vivo, errores en los logs del sistema).
    *   Configurar alertas para eventos críticos que no solo se envíen a Telegram, sino que también puedan escalar a otros canales (ej. correo electrónico, PagerDuty) si no se resuelven.
*   **Gestión de Múltiples Instancias:**
    *   Implementar un mecanismo para asegurar que solo una instancia del bot de trading en vivo esté activa en un momento dado para evitar interferencias y doble ejecución de órdenes (ej. un archivo de bloqueo en GCS o una entrada en una base de datos).
*   **Actualizaciones del Modelo en Vivo:**
    *   Definir un proceso para actualizar el modelo que el bot utiliza en producción sin detener la operativa o con una interrupción mínima (ej. "hot-swapping" del modelo, o un proceso de despliegue controlado).
*   **Gestión de la Sesión de Trading:**
    *   Si el bot no operará 24/7 (ej. solo en ciertos días/horas), definir cómo se gestiona el cierre de posiciones al final de una sesión y la reapertura al inicio de la siguiente.
