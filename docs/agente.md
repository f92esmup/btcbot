Hemos llegado al Módulo 4: Agente de Reinforcement Learning (RL Agent), que es el componente central donde reside la inteligencia de tu bot de trading. Tus especificaciones son muy claras y apuntan a una solución moderna y potente utilizando Soft Actor-Critic (SAC) con una arquitectura basada en Transformers.

Antes de detallar el módulo, abordemos tus dos observaciones:

1.  **Actualización de `N_features`**: Correcto. En el Módulo 2 (Preprocesamiento), definimos `N_features_mercado = 20`. En el Módulo 3 (Entorno de Trading), especificamos que la observación para el agente incluiría estas 20 características de mercado *más* 8 características de cartera/portafolio. Por lo tanto, si estas características de cartera se concatenan a cada paso de la secuencia de mercado para alimentar el Transformer, la dimensión de entrada por paso temporal para el Transformer sería `N_features_concatenadas = 20 (mercado) + 8 (cartera) = 28`. Ajustaremos la especificación del agente para reflejar esto.

2.  **Configuración del Entorno para Datos del Portafolio en la Observación**: Sí, en la definición del Módulo 3, diseñamos el `observation_space` como un `gymnasium.spaces.Dict`. Este diccionario contendrá `'market_features': Box(L, N_features_mercado)` y `'portfolio_features': Box(N_features_cartera)`. El agente (Módulo 4) necesitará una arquitectura de red (específicamente, un extractor de características dentro de la política de Stable Baselines3) que pueda manejar esta entrada estructurada, procesar cada parte adecuadamente (el Transformer para las `market_features` después de combinarlas con `portfolio_features`) y luego alimentar las representaciones aprendidas a las redes del actor y el crítico.

Con estas aclaraciones, procedamos a definir el Módulo 4.

**Módulo 4: Agente de Reinforcement Learning (RL Agent)**

Este módulo se encarga de implementar, entrenar y gestionar el agente de RL que tomará las decisiones de trading.

**1. Responsabilidades y Funcionalidades Exactas:**

  * **Implementación del Algoritmo Soft Actor-Critic (SAC):**
      * Se utilizará el algoritmo SAC, conocido por su eficiencia de muestra y estabilidad en espacios de acción continuos.
      * La implementación se basará en una librería robusta como Stable Baselines3 (SB3).
  * **Arquitectura del Modelo (Política y Redes de Valor con Backbone Transformer):**
      * **Entrada de Observación:** El agente recibirá el `DictSpace` definido en el Módulo 3: `{'market_features': (L=96, N_features_mercado=20), 'portfolio_features': (8,)}`.
      * **Procesamiento de Características de Entrada y Fusión:**
          * Las 8 `portfolio_features` se replicarán/transmitirán y se concatenarán a cada uno de los `L=96` pasos temporales de las `market_features`.
          * Esto crea una secuencia de entrada efectiva para el Transformer de forma `(L=96, N_features_entrada_transformer = 20 + 8 = 28)`.
      * **Transformer Encoder (para procesar la secuencia combinada):**
          * **Capa de Embedding de Entrada:** Una capa lineal para proyectar los `N_features_entrada_transformer = 28` de cada paso temporal a la dimensión del modelo `d_model=128`.
          * **Positional Encoding:** Se añadirá un Positional Encoding Sinusoidal fijo a los embeddings de entrada para proporcionar información sobre el orden secuencial.
          * **Pila de Capas de Encoder Transformer:**
              * Número de Capas de Encoder: **3 capas** (configurable).
              * Número de Cabezas de Atención (por capa): **4 cabezas** (configurable).
              * `d_model` (Dimensión del Modelo): **128** (configurable).
              * Dimensión de la Feed-Forward Network (FFN) interna de cada capa de encoder: `4 * d_model = 512` (configurable).
              * Se puede incluir Dropout para regularización.
          * **Salida del Transformer:** El Transformer procesará la secuencia y su salida (ej., la representación del último paso temporal `(d_model,)` o una agregación como Global Average Pooling sobre la dimensión temporal) se utilizará como la representación del estado aprendida.
      * **Red del Actor (Política):**
          * Toma la representación del estado aprendida del Transformer como entrada.
          * Consiste en un Multi-Layer Perceptron (MLP), ej., con capas ocultas `[256, 256]` (configurable).
          * Produce los parámetros (media y desviación estándar) para una **distribución Gaussiana Escalonada y Reescalada (Squashed Gaussian)**. La acción se muestrea de esta distribución y se pasa por una función `tanh` para acotarla al rango `[-1, 1]`.
      * **Redes del Crítico (Redes Q - dos para Clipped Double-Q Learning, parte de SAC):**
          * Cada una de las dos redes Q toma la representación del estado aprendida del Transformer *y* la acción (ya sea del actor o del replay buffer) como entrada.
          * Consiste en un MLP (ej., con capas ocultas `[256, 256]` cada una, configurable).
          * Produce el valor Q estimado.
  * **Replay Buffer (Memoria de Repetición de Experiencias):**
      * Almacena tuplas de transición: `(observacion, accion, recompensa, siguiente_observacion, done_flags)`.
      * Tamaño configurable: **100,000 transiciones**.
      * Para `DictSpace`, SB3 utiliza un `DictReplayBuffer`.
  * **Aprendizaje del Coeficiente de Entropía (`alpha`):**
      * SAC ajusta automáticamente el coeficiente `alpha` que pondera el término de entropía en la función objetivo, para equilibrar la maximización de la recompensa y la exploración. Se configurará como `ent_coef='auto'` en SB3.
  * **Manejo de Exploración vs. Explotación:**
      * Inherente al algoritmo SAC a través de su política estocástica y la maximización de la entropía.
  * **Capacidad de Guardar y Cargar Modelos Entrenados:**
      * Se utilizarán las funcionalidades `model.save()` y `model.load()` de Stable Baselines3 para persistir y reutilizar los agentes entrenados (incluyendo los pesos de las redes y, opcionalmente, el estado del optimizador y el replay buffer).

**2. Tecnologías, Librerías o Frameworks Específicos:**

  * **Lenguaje de Programación:** Python.
  * **Framework de Deep Learning:** **PyTorch** (Stable Baselines3 se basa en PyTorch).
  * **Librería de Reinforcement Learning:** **Stable Baselines3 (SB3)**.

**3. Entradas y Salidas del Agente (durante la interacción):**

  * **Entradas (del entorno):**
      * `observacion (dict)`: Estructura con `'market_features'` y `'portfolio_features'`.
      * `recompensa (float)`.
      * `terminated (bool)`, `truncated (bool)`.
  * **Salidas (al entorno):**
      * `accion (np.ndarray)`: La `action_signal` continua `[-1, 1]`.

**4. Parámetros Configurables (Hiperparámetros del Agente y Modelo):**

Se almacenarán en un archivo YAML, por ejemplo, `src/agent/agent_config.yaml`.

```yaml
# src/agent/agent_config.yaml

# --- Configuración del Algoritmo SAC (para Stable Baselines3) ---
algorithm: "SAC"
# policy: "MlpPolicy" # Aunque usaremos una política con extractor personalizado, la base puede ser MlpPolicy
learning_rate: 0.0003
buffer_size: 100000
learning_starts: 10000      # Pasos para llenar el buffer antes de entrenar
batch_size: 256
tau: 0.005                  # Coeficiente de actualización suave para redes objetivo
gamma: 0.99                 # Factor de descuento
train_freq: 1               # Frecuencia de entrenamiento (en pasos de entorno, ej., (1, "step"))
gradient_steps: 1           # Pasos de gradiente por actualización
ent_coef: "auto"            # Aprendizaje automático del coeficiente de entropía alpha
# target_entropy: "auto"    # Calculado automáticamente por SB3 si ent_coef='auto'
use_sde: false              # State-Dependent Exploration (SAC ya es estocástico)

# --- Configuración de la Arquitectura de Red Personalizada (para policy_kwargs en SB3) ---
policy_kwargs:
  features_extractor_class: "CustomTransformerFeatureExtractor" # Nombre de tu clase extractor
  features_extractor_kwargs:
    market_features_key: "market_features"
    portfolio_features_key: "portfolio_features"
    # N_features_mercado = 20, N_features_cartera = 8 => N_features_entrada_transformer = 28
    features_in_transformer: 28 # 20 (mercado) + 8 (cartera)
    d_model: 128
    n_heads: 4
    n_encoder_layers: 3
    dim_feedforward: 512 # d_model * 4
    dropout_rate: 0.1
    # features_dim: 128 # Dimensión de salida del extractor, usualmente d_model

  # Arquitectura de las redes MLP del actor (pi) y crítico (qf) DESPUÉS del extractor
  net_arch:
    pi: [256, 256] # Capas ocultas para el actor
    qf: [256, 256] # Capas ocultas para los críticos

# --- Configuración de Entrenamiento ---
total_training_timesteps: 1000000
log_interval_episodes: 10 # Loguear cada N episodios
save_path_prefix: "models/sac_transformer_trading_agent"
save_frequency_steps: 50000

# --- Carga de Modelo (Opcional) ---
# load_trained_model_on_start: false
# model_path_to_load: "models/sac_transformer_trading_agent_XXXX_steps.zip"
```

**5. Estructura de Código Propuesta:**

  * **`src/agent/agent_config.yaml`**: Archivo de configuración.
  * **`src/agent/custom_transformer_extractor.py`**:
      * Definición de la clase `CustomTransformerFeatureExtractor(BaseFeaturesExtractor)` de SB3.
      * Esta clase recibirá el `Dict` de observaciones.
      * Implementará la lógica para:
        1.  Extraer `market_features` y `portfolio_features`.
        2.  Replicar `portfolio_features` y concatenarlas a cada uno de los `L` pasos de `market_features` para formar la entrada `(L, 28)`.
        3.  Pasar esta secuencia a través de la capa de embedding lineal.
        4.  Añadir Positional Encoding.
        5.  Procesar con el stack de `nn.TransformerEncoderLayer`.
        6.  Agregar la salida del Transformer (ej., tomar la salida del último paso `[:, -1, :]` o aplicar Global Average Pooling) para obtener un vector de características de dimensión `d_model`.
        7.  Devolver este vector de características planas.
  * **`src/agent/rl_agent_manager.py`** (o similar):
      * Clase `RLAgentManager` que encapsula la creación, entrenamiento, guardado/carga del agente SAC.
      * Método `setup_agent()`: Carga configuración, crea el entorno (Módulo 3), instancia el modelo SAC de SB3, pasando `policy_kwargs` que especifica el `CustomTransformerFeatureExtractor` y la arquitectura de las redes `pi` y `qf`.
      * Método `train_agent()`: Llama a `model.learn()`.
      * Método `predict_action()`: Llama a `model.predict()`.
  * **`raiz/scripts/train_rl_agent.py`**: Script principal para orquestar el entrenamiento.
  * **`raiz/scripts/evaluate_rl_agent.py`**: Script para evaluar un agente entrenado.

**Consideraciones Clave:**

  * **Implementación del `CustomTransformerFeatureExtractor`:** Este es el componente más crítico y personalizado dentro de SB3. Debe manejar correctamente la observación `Dict`, la fusión de características, la arquitectura Transformer, y devolver la dimensión de características esperada por las redes MLP del actor y crítico. SB3 espera que el `features_extractor` devuelva un tensor de características de una dimensión (`features_dim`, que en este caso sería `d_model`).
  * **Positional Encoding:** Asegurar una correcta implementación y adición.
  * **Flujo de Datos:** El flujo desde la observación `Dict` hasta las redes finales del actor/crítico a través del extractor personalizado debe ser coherente.

Este diseño para el Módulo 4 establece una base sólida para un agente de trading avanzado. La combinación de SAC para la toma de decisiones y un Transformer para la comprensión de las secuencias de mercado y estado de la cartera es una aproximación de vanguardia.