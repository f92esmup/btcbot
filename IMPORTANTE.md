Tienes toda la razón al señalar esos aspectos. La "duración de las órdenes" en el sentido de "Time in Force" (TIF) o cuánto tiempo permanece una orden *limitada* en el libro antes de ser cancelada, y el "tiempo en posición" como una característica explícita, se manejan de la siguiente manera en el diseño actual:

**1. Duración de las Órdenes (Time in Force - TIF)**

* **No se tiene en cuenta explícitamente porque se usan ÓRDENES A MERCADO:**
    * Tanto para entrar como para cerrar posiciones, el `LiveBinanceAPIManager` está configurado para colocar órdenes de tipo `MARKET`.
    * Las órdenes a mercado están diseñadas para ejecutarse inmediatamente al mejor precio disponible. No tienen una "duración" en el libro de órdenes como las órdenes limitadas (que podrían usar políticas TIF como GTC - Good 'Til Canceled, IOC - Immediate Or Cancel, FOK - Fill Or Kill).
    * Dado que el bot toma decisiones al cierre de cada vela y busca una ejecución inmediata, las órdenes a mercado son la elección natural y simplifican la lógica al no tener que gestionar órdenes abiertas que no se ejecutan.

* **Implicaciones de no usar órdenes limitadas:**
    * **Ventaja:** Simplicidad y alta probabilidad de ejecución.
    * **Desventaja:** Posible *slippage* (deslizamiento), especialmente en mercados volátiles o con poca liquidez. El precio de ejecución puede ser peor que el precio de mercado en el momento de decidir la orden. El `SimulatedBroker` en tu entorno de entrenamiento tiene un parámetro `slippage_atr_multiplier` para simular esto. En vivo, el slippage es una realidad del uso de órdenes a mercado.

**2. "Tiempo en Posición" como Característica para el Modelo**

* **En el Entrenamiento (`TradingEnvironment`):**
    * Sí se tiene en cuenta. La característica `portfolio_features[6]` corresponde a `steps_in_current_position` normalizado. Esta variable cuenta cuántas velas (pasos de tiempo) ha estado abierta la posición actual.
    * Se normaliza dividiendo por `max_steps_in_position` definido en `config.yaml` bajo `environment.portfolio_features_normalization`.

* **En el Modo Live (`portfolio_feature_builder.py`):**
    * **Actualmente no se replica de forma precisa y se establece en `0.0` (placeholder).**
    * **Razón:** Replicar esto exactamente en el modo en vivo requeriría que el `run_live_trader.py` mantenga un estado persistente sobre cuándo se abrió la posición actual (por ejemplo, el timestamp de la vela de apertura) y luego contar cuántas velas han pasado desde entonces.
    * Si el bot se reinicia, este estado se perdería a menos que se guarde y recupere (ej. en un archivo, base de datos o GCS).
    * Para un bot que toma decisiones basadas principalmente en la información de la vela actual y la secuencia de mercado reciente, esta característica de "duración de la posición" podría ser menos crítica o, si es crítica, el modelo podría haber aprendido patrones relacionados con ella que no se traducen bien si la feature es constantemente cero en vivo.

**Implicaciones y Consideraciones:**

1.  **Consistencia entre Entrenamiento y Live para "Tiempo en Posición":**
    * Si la característica `steps_in_current_position` fue importante para el aprendizaje del modelo, tenerla siempre en `0.0` en vivo podría llevar a decisiones subóptimas.
    * **Soluciones posibles para el modo live:**
        * **Implementar el seguimiento del estado:** Modificar `run_live_trader.py` para que, cuando abra una posición, guarde el timestamp (o el contador de velas si el bot funciona sin interrupciones). En cada ciclo, calcularía la "duración" y la pasaría a `build_live_portfolio_features`. Esto añade complejidad por la necesidad de persistencia del estado.
        * **Usar un proxy:** Por ejemplo, si el P&L no realizado de la posición es muy alto o muy bajo, podría ser un indicativo de que la posición ha estado abierta por un tiempo considerable o ha tenido un movimiento significativo. Sin embargo, esto es indirecto.
        * **Re-entrenar el modelo sin esa característica:** Si es demasiado complejo implementarla en vivo de forma fiable y se sospecha que no es crucial, se podría eliminar del `observation_space` y reentrenar.
        * **Aceptar la discrepancia:** Si las pruebas en Testnet muestran un buen rendimiento a pesar de que esta feature sea 0.0, podría ser aceptable (aunque es una fuente potencial de diferencia entre el backtesting/entrenamiento y el rendimiento en vivo).

2.  **Órdenes a Mercado y Estrategia:**
    * El uso de órdenes a mercado implica una estrategia que prioriza la entrada/salida inmediata sobre la obtención de un precio específico. Esto es coherente con un modelo que decide al cierre de una vela.
    * Si tu estrategia requiriese más control sobre el precio de entrada/salida (ej. actuar como *maker* en lugar de *taker*), necesitarías implementar órdenes limitadas y, con ello, la gestión de su duración (TIF), su posible no ejecución, y la lógica para cancelarlas o modificarlas. Esto añadiría una capa significativa de complejidad.

En resumen:
* La "duración de las órdenes" (TIF) no se considera porque se usan órdenes a mercado.
* El "tiempo en posición" (`steps_in_current_position`) sí se usa en el entrenamiento, pero su implementación actual en la guía para el modo en vivo es un placeholder (`0.0`) debido a la complejidad de mantener ese estado de forma persistente. Deberías evaluar la importancia de esta característica para tu modelo y decidir si necesitas una implementación más precisa en el modo en vivo.