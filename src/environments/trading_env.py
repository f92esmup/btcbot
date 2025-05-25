import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, Union, List
import logging
import io
import datetime # Added import
from google.cloud import storage

from src.utils.config import ConfigManager
from src.environments.simulated_broker import SimulatedBroker
from src.utils.logging_utils import get_madrid_timestamp_str

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TradingEnv')

import numpy as np
# Importar torch de manera condicional para no crear dependencia obligatoria
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class TradingEnvironment(gym.Env):
    """
    Entorno de trading de futuros para Gymnasium.
    
    Este entorno simula el trading de futuros de criptomonedas con apalancamiento
    fijo y tamaño de posición relativo al equity.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config_path: str = 'src/config.yaml', render_mode: Optional[str] = None):
        """
        Inicializa el entorno de trading.
        
        Args:
            config_path: Ruta al archivo de configuración yaml centralizada
            render_mode: Modo de renderización (human, etc.)
        """
        # Carga la configuración centralizada
        self.config_manager = ConfigManager(config_path=config_path)
        self.config = self.config_manager.get_environment_config()
        
        # Configura el render_mode
        self.render_mode = render_mode
        
        # Initialize sequence length from preprocessing config
        preprocessing_config = self.config_manager.get_preprocessing_config()
        self.L = preprocessing_config['sequence_length_L']
        
        # Carga los datos de mercado preprocesados
        self.market_data, self.feature_names = self._load_market_data()
        
        # Inicializa el broker simulado
        self.broker = SimulatedBroker(
            taker_fee_rate=self.config['taker_fee_rate'],
            slippage_atr_multiplier=self.config['slippage_atr_multiplier'],
            min_order_size_btc=self.config['min_order_size_btc']
        )
        
        # Configuración del entorno
        self.initial_equity = self.config['initial_equity']
        self.leverage = self.config['leverage']
        self.position_size_pct_equity = self.config['position_size_pct_equity']
        self.action_threshold = self.config['action_threshold']
        self.equity_drawdown_threshold = self.config['equity_drawdown_threshold_episode_end']
        self.liquidation_safety_factor = self.config['liquidation_safety_factor']
        
        # Estado de la cartera (se inicializa en reset())
        self.current_step_index = 0
        self.initial_equity_episode = 0.0
        self.current_equity = 0.0
        self.balance = 0.0
        self.active_position_side = 0  # -1 (Corto), 0 (Neutral), 1 (Largo)
        self.active_position_size_contracts = 0.0
        self.active_position_entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.margin_used = 0.0
        self.available_margin = 0.0
        self.steps_in_current_position = 0
        self.liquidation_price = 0.0
        self.last_equity = 0.0  # Para el cálculo de recompensa

        # For BigQuery logging
        self.current_episode_step_data = []
        self.latest_episode_summary = None
        self.current_step_in_episode = 0
        self.episode_id_counter = 0 # To assign unique IDs to episodes
        # For aggregating stats per episode for the summary
        self.current_episode_agg_stats = {
            'total_reward': 0.0,
            'pnl_realized': 0.0,
            'num_trades': 0,
            'total_fees': 0.0
        }
        
        # Define el espacio de observación como un Dict
        market_features_dim = len(self.feature_names)
        self.observation_space = spaces.Dict({
            # Características de mercado: secuencia de L pasos con N características c/u
            'market_features': spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.L, market_features_dim), dtype=np.float32
            ),
            # Características de cartera: 8 valores normalizados
            'portfolio_features': spaces.Box(
                low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
            )
        })
        
        # Define el espacio de acción como un Box continuo de una dimensión
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Información adicional para episodios
        self.episode_stats = {
            'trades': 0,
            'profitable_trades': 0,
            'unprofitable_trades': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
            'max_equity': 0.0,
            'min_equity': float('inf')
        }
        
        # Registrar en gym solo si no está ya registrado
        env_id = self.config.get('env_id', 'FuturesTradingEnv-v0')
        if env_id not in gym.envs.registry:
            gym.register(
                id=env_id,
                entry_point='src.environments.trading_env:TradingEnvironment',
                max_episode_steps=None  # Se configura dinámicamente en reset()
            )
        
        logger.info("TradingEnvironment initialized successfully.")
    
    def _load_market_data(self) -> Tuple[np.ndarray, List[str]]:
        """
        Carga los datos de mercado preprocesados desde Google Cloud Storage.
        
        Returns:
            Tuple con (datos_de_mercado, nombres_de_características)
        """
        # Inicializar cliente de GCS
        try:
            # Obtener variables de GCS
            gcp_project_id = self.config_manager.get_env_variable('GCP_PROJECT_ID')
            gcs_bucket_name = self.config_manager.get_env_variable('GCS_BUCKET_NAME')
            
            if not gcp_project_id or not gcs_bucket_name:
                raise ValueError("Variables de entorno GCP_PROJECT_ID o GCS_BUCKET_NAME no configuradas.")
                
            # Inicializar cliente de almacenamiento
            storage_client = storage.Client(project=gcp_project_id)
            bucket = storage_client.bucket(gcs_bucket_name)
        except Exception as e:
            logger.error(f"Error al inicializar cliente de Google Cloud Storage: {e}")
            raise
            
        # Busca el archivo con los datos preprocesados
        data_dir = self.config['processed_data_directory']
        file_identifier = self.config['processed_data_file_identifier']
        
        # Listar blobs en el bucket con el prefijo del directorio
        prefix = f"{data_dir}/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        # Filtrar por el identificador en el nombre del archivo
        matching_files = [blob.name.split('/')[-1] for blob in blobs if file_identifier in blob.name]
        
        if not matching_files:
            raise FileNotFoundError(f"No se encontraron archivos con el identificador {file_identifier} en GCS: {gcs_bucket_name}/{data_dir}")
        
        # Utiliza el primer archivo que coincida (se podría hacer más sofisticado si hay múltiples)
        matching_file = sorted(matching_files, reverse=True)[0]  # Usar el más reciente
        gcs_file_path = f"{data_dir}/{matching_file}"
        logger.info(f"Cargando datos de mercado desde GCS: {gcs_bucket_name}/{gcs_file_path}")
        
        # Descargar archivo a un buffer de memoria
        blob = bucket.blob(gcs_file_path)
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        
        # Carga los datos
        data = np.load(buffer)
        
        # Verificar las claves disponibles en el archivo
        logger.info(f"Claves disponibles en el archivo: {list(data.keys())}")
        
        # Intenta cargar los datos según las claves disponibles
        if 'X_market' in data:
            market_features = data['X_market']
            # Dado que no tenemos feature_names, creamos nombres genéricos basados en la forma
            feature_names = [f"feature_{i}" for i in range(market_features.shape[2])]
            logger.info(f"Datos cargados con la clave 'X_market' de forma {market_features.shape}")
        elif 'market_features' in data:
            market_features = data['market_features']
            feature_names = data['feature_names'].tolist() if 'feature_names' in data else [f"feature_{i}" for i in range(market_features.shape[2])]
            logger.info(f"Datos cargados con la clave 'market_features' de forma {market_features.shape}")
        else:
            raise KeyError(f"No se encontró ninguna clave válida para datos de mercado en el archivo descargado de GCS")
        
        # Cargar datos adicionales si están disponibles (para optimización)
        self.close_prices = None
        self.atr_values = None
        
        # Verificar si tenemos precios de cierre y ATR no normalizados
        if 'close_prices' in data:
            self.close_prices = data['close_prices']
            logger.info(f"Precios de cierre no normalizados cargados: {len(self.close_prices)} valores")
            
        if 'atr_values' in data:
            self.atr_values = data['atr_values']
            logger.info(f"Valores ATR no normalizados cargados: {len(self.atr_values)} valores")
            
        # Si alguno de los dos no está disponible, eliminamos ambos para consistencia
        if self.close_prices is None or self.atr_values is None:
            self.close_prices = None
            self.atr_values = None
            logger.warning("No se encontraron datos de precio y ATR sin normalizar. Usando aproximaciones.")
        
        # Asegurar que market_features sea float32
        if market_features.dtype != np.float32:
            logger.info(f"Convirtiendo market_features de {market_features.dtype} a float32")
            market_features = market_features.astype(np.float32)
            
        # Verificar que self.L coincide con la dimensión secuencial de los datos
        if market_features.shape[1] != self.L:
            logger.warning(f"Advertencia: La longitud de secuencia en los datos ({market_features.shape[1]}) no coincide con self.L ({self.L})")
            logger.warning(f"Ajustando self.L para que coincida con los datos")
            self.L = market_features.shape[1]
            
            # También actualizar el observation_space
            market_features_dim = market_features.shape[2]
            self.observation_space = spaces.Dict({
                'market_features': spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.L, market_features_dim), dtype=np.float32
                ),
                'portfolio_features': spaces.Box(
                    low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
                )
            })
            
        logger.info(f"Datos de mercado cargados: {market_features.shape}, dtype: {market_features.dtype}")
        return market_features, feature_names
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reinicia el entorno al principio de un nuevo episodio.
        
        Args:
            seed: Semilla para la generación de números aleatorios (opcional)
            options: Opciones adicionales (opcional)
            
        Returns:
            Tuple con (observación, info)
        """
        # Reinicia el generador de números aleatorios si se proporciona una semilla
        if seed is not None:
            super().reset(seed=seed)

        # Increment episode counter and reset BQ logging structures
        self.episode_id_counter += 1
        self.current_episode_step_data.clear()
        self.latest_episode_summary = None
        self.current_step_in_episode = 0
        self.current_episode_agg_stats = {
            'total_reward': 0.0,
            'pnl_realized': 0.0,
            'num_trades': 0,
            'total_fees': 0.0
        }
        
        # Reinicia las estadísticas del episodio
        self.episode_stats = {
            'trades': 0,
            'profitable_trades': 0,
            'unprofitable_trades': 0,
            'total_pnl': 0.0,
            'total_fees': 0.0,
            'max_equity': self.initial_equity,
            'min_equity': self.initial_equity
        }
        
        # Selecciona un punto de inicio aleatorio en el conjunto de datos si está configurado
        if self.config['allow_random_episode_start']:
            # Asegura que haya suficientes datos para al menos un episodio razonable
            min_episode_steps = 10  # Número mínimo de pasos para un episodio
            max_start_idx = len(self.market_data) - min_episode_steps
            self.current_step_index = self.np_random.integers(0, max_start_idx) if max_start_idx > 0 else 0
        else:
            # Comienza desde la primera secuencia disponible
            self.current_step_index = 0
        
        # Reinicia el estado de la cartera
        self.initial_equity_episode = self.initial_equity
        self.current_equity = self.initial_equity
        self.last_equity = self.initial_equity
        self.balance = self.initial_equity
        self.active_position_side = 0
        self.active_position_size_contracts = 0.0
        self.active_position_entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.margin_used = 0.0
        self.available_margin = self.initial_equity
        self.steps_in_current_position = 0
        self.liquidation_price = 0.0
        
        # Obtiene la observación inicial
        observation = self._get_observation()
        
        # Info adicional
        info = {}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Ejecuta un paso en el entorno.
        
        Args:
            action: Acción continua del agente en el rango [-1, 1]
            
        Returns:
            Tuple con (observación, recompensa, terminado, truncado, info)
        """
        # Extrae la señal de acción del array (evitando copias innecesarias)
        action_signal = float(action[0])
        
        # Guarda el equity anterior para el cálculo de la recompensa
        self.last_equity = self.current_equity
        
        # Avanza al siguiente paso en los datos de mercado
        self.current_step_index += 1
        self.current_step_in_episode += 1 # Increment step in episode
        
        # Obtiene los datos actuales del mercado (referencia, no copia)
        current_market_data = self.market_data[self.current_step_index]
        
        # Obtiene el precio de cierre usando nuestro método optimizado
        close_price = self._get_current_close_price()
        
        # Obtiene el ATR usando el método optimizado que ya maneja atr_values
        atr_value = self._get_atr_value_optimized(current_market_data, close_price)
            
        # Procesa la acción y actualiza el estado del entorno
        info = self._process_action(action_signal, close_price, atr_value)
        
        # Comprueba si hubo liquidación
        liquidated = self._check_liquidation(current_market_data)
        
        # Actualiza estadísticas de episodio
        self.episode_stats['max_equity'] = max(self.episode_stats['max_equity'], self.current_equity)
        self.episode_stats['min_equity'] = min(self.episode_stats['min_equity'], self.current_equity)
        
        # Calcula la recompensa (retorno logarítmico del equity)
        reward = np.log(self.current_equity / self.last_equity) # Ensure self.last_equity is not zero if equity can be zero
        if np.isinf(reward) or np.isnan(reward): # Handle potential log(0) or log(neg)
            reward = 0.0 

        # --- BigQuery Step Logging ---
        step_info_log = {}
        step_info_log['episode_id'] = self.episode_id_counter
        step_info_log['step_in_episode'] = self.current_step_in_episode
        step_info_log['timestamp_event'] = get_madrid_timestamp_str()
        step_info_log['event_type'] = 'step_info'
        step_info_log['reward_step'] = float(reward)
        step_info_log['action_value_raw'] = float(action_signal)
        step_info_log['current_equity_step'] = float(self.current_equity)
        step_info_log['current_position_side_step'] = int(self.active_position_side)
        step_info_log['current_position_avg_price_step'] = float(self.active_position_entry_price) if self.active_position_side != 0 else 0.0
        step_info_log['current_position_size_step'] = float(self.active_position_size_contracts)
        step_info_log['market_price_at_step'] = float(close_price) # close_price from this step

        # Log market and portfolio features
        # Market features for the current step (that influenced this state)
        current_market_data_for_log = self.market_data[self.current_step_index]
        if current_market_data_for_log.ndim == 2 and current_market_data_for_log.shape[0] > 0 and current_market_data_for_log.shape[1] > 0:
            last_market_feats = current_market_data_for_log[-1] # Last row of the sequence for current step
            step_info_log['obs_market_feat_0_step'] = float(last_market_feats[0]) if len(last_market_feats) > 0 else None
            step_info_log['obs_market_feat_1_step'] = float(last_market_feats[1]) if len(last_market_feats) > 1 else None
            step_info_log['obs_market_feat_2_step'] = float(last_market_feats[2]) if len(last_market_feats) > 2 else None
        else: # Handle cases where market features might not be as expected (e.g. 1D array)
            step_info_log['obs_market_feat_0_step'] = None
            step_info_log['obs_market_feat_1_step'] = None
            step_info_log['obs_market_feat_2_step'] = None
            
        # Portfolio features reflecting the state *after* the current action
        current_portfolio_features_for_log = self._get_normalized_portfolio_features() # This gets current state
        step_info_log['obs_portfolio_pnl_norm_step'] = float(current_portfolio_features_for_log[3])
        step_info_log['obs_portfolio_steps_in_pos_norm_step'] = float(current_portfolio_features_for_log[6])
        
        self.current_episode_step_data.append(step_info_log)
        # --- End BigQuery Step Logging ---

        # Comprueba condiciones de fin de episodio
        terminated = False
        truncated = False
        
        # Condición 1: Drawdown máximo
        current_drawdown = (self.current_equity / self.initial_equity_episode) - 1.0
        if current_drawdown <= self.equity_drawdown_threshold:
            terminated = True
            info['termination_reason'] = 'max_drawdown'
        
        # Condición 2: Liquidación
        if liquidated:
            terminated = True
            info['termination_reason'] = 'liquidation'
        
        # Condición 3: Fin de datos disponibles
        if self.current_step_index >= len(self.market_data) - 1:
            truncated = True
            info['termination_reason'] = 'data_end'
        
        # Obtiene la siguiente observación
        observation = self._get_observation()

        # Update aggregated reward for BQ
        self.current_episode_agg_stats['total_reward'] += reward
        
        # Episode End Logging for BigQuery
        if terminated or truncated:
            episode_summary_log = {}
            episode_summary_log['episode_id'] = self.episode_id_counter
            episode_summary_log['timestamp_event'] = get_madrid_timestamp_str()
            episode_summary_log['event_type'] = 'episode_summary'
            episode_summary_log['total_reward_episode'] = float(self.current_episode_agg_stats['total_reward'])
            episode_summary_log['pnl_realized_episode'] = float(self.current_episode_agg_stats['pnl_realized'])
            episode_summary_log['num_trades_episode'] = int(self.current_episode_agg_stats['num_trades'])
            episode_summary_log['total_fees_episode'] = float(self.current_episode_agg_stats['total_fees'])
            episode_summary_log['episode_duration_steps'] = int(self.current_step_in_episode)
            episode_summary_log['final_equity_episode'] = float(self.current_equity)
            episode_summary_log['termination_reason'] = info.get('termination_reason', None)
            
            self.latest_episode_summary = episode_summary_log
            self.current_episode_step_data.append(episode_summary_log) # Append to step data for callback
        
        # Actualiza info con estadísticas de episodio
        info.update({
            'current_equity': self.current_equity,
            'episode_return': self.current_equity / self.initial_equity_episode - 1.0,
            'current_drawdown': current_drawdown,
            'trades_count': self.episode_stats['trades'],
            'win_rate': self.episode_stats['profitable_trades'] / max(1, self.episode_stats['trades']),
            'position_side': self.active_position_side,
            'position_size': self.active_position_size_contracts,
            'entry_price': self.active_position_entry_price,
            'unrealized_pnl': self.unrealized_pnl,
            'episode_stats': self.episode_stats
        })
        
        # Renderiza si está configurado
        if self.render_mode == 'human':
            self.render()
        
        return observation, reward, terminated, truncated, info
        
        # Procesa la acción y actualiza el estado del entorno
        info = self._process_action(action_signal, close_price, atr_value)
        
        # Comprueba si hubo liquidación
        liquidated = self._check_liquidation(current_market_data)
        
        # Actualiza estadísticas de episodio
        self.episode_stats['max_equity'] = max(self.episode_stats['max_equity'], self.current_equity)
        self.episode_stats['min_equity'] = min(self.episode_stats['min_equity'], self.current_equity)
        
        # Calcula la recompensa (retorno logarítmico del equity)
        reward = np.log(self.current_equity / self.last_equity)
        
        # Comprueba condiciones de fin de episodio
        terminated = False
        truncated = False
        
        # Condición 1: Drawdown máximo
        current_drawdown = (self.current_equity / self.initial_equity_episode) - 1.0
        if current_drawdown <= self.equity_drawdown_threshold:
            terminated = True
            info['termination_reason'] = 'max_drawdown'
        
        # Condición 2: Liquidación
        if liquidated:
            terminated = True
            info['termination_reason'] = 'liquidation'
        
        # Condición 3: Fin de datos disponibles
        if self.current_step_index >= len(self.market_data) - 1:
            truncated = True
            info['termination_reason'] = 'data_end'
        
        # Obtiene la siguiente observación
        observation = self._get_observation()
        
        # Actualiza info con estadísticas de episodio
        info.update({
            'current_equity': self.current_equity,
            'episode_return': self.current_equity / self.initial_equity_episode - 1.0,
            'current_drawdown': current_drawdown,
            'trades_count': self.episode_stats['trades'],
            'win_rate': self.episode_stats['profitable_trades'] / max(1, self.episode_stats['trades']),
            'position_side': self.active_position_side,
            'position_size': self.active_position_size_contracts,
            'entry_price': self.active_position_entry_price,
            'unrealized_pnl': self.unrealized_pnl,
            'market_price': close_price,  # Add current market price
            'episode_stats': self.episode_stats
        })
        
        # Renderiza si está configurado
        if self.render_mode == 'human':
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _process_action(self, action_signal: float, close_price: float, atr_value: float) -> Dict[str, Any]:
        """
        Procesa la acción del agente y actualiza el estado del entorno.
        
        Args:
            action_signal: Señal de acción del agente [-1, 1]
            close_price: Precio de cierre actual
            atr_value: Valor ATR actual
            
        Returns:
            Diccionario con información sobre la acción procesada
        """
        info = {"action_type": "hold"}
        
        # Interpreta la acción según el umbral configurado
        if action_signal > self.action_threshold:  # Intento de Abrir Largo
            if self.active_position_side == 0:  # Neutral
                self._open_position(1, close_price, atr_value)
                info["action_type"] = "open_long"
            elif self.active_position_side == -1:  # Corto
                self._close_position(close_price, atr_value)
                self._open_position(1, close_price, atr_value)
                info["action_type"] = "close_short_open_long"
            # Si ya está en posición larga, mantiene
        
        elif action_signal < -self.action_threshold:  # Intento de Abrir Corto
            if self.active_position_side == 0:  # Neutral
                self._open_position(-1, close_price, atr_value)
                info["action_type"] = "open_short"
            elif self.active_position_side == 1:  # Largo
                self._close_position(close_price, atr_value)
                self._open_position(-1, close_price, atr_value)
                info["action_type"] = "close_long_open_short"
            # Si ya está en posición corta, mantiene
        
        else:  # Intento de Neutralizar
            if self.active_position_side != 0:
                self._close_position(close_price, atr_value)
                info["action_type"] = "close_position"
        
        # Si hay posición activa, incrementa los pasos en esta posición
        if self.active_position_side != 0:
            self.steps_in_current_position += 1
            
            # Actualiza P&L no realizado
            if self.active_position_side == 1:  # Largo
                self.unrealized_pnl = (close_price - self.active_position_entry_price) * self.active_position_size_contracts
            else:  # Corto
                self.unrealized_pnl = (self.active_position_entry_price - close_price) * self.active_position_size_contracts
            
            # Actualiza el equity actual con el P&L no realizado
            self.current_equity = self.balance + self.unrealized_pnl
        else:
            self.steps_in_current_position = 0
            self.unrealized_pnl = 0.0
            # En estado neutral, equity = balance
            self.current_equity = self.balance
        
        # Actualiza el margen disponible
        self.available_margin = self.current_equity - self.margin_used
        
        return info
    
    def _open_position(self, side: int, market_close_price: float, atr_value: float) -> None:
        """
        Abre una nueva posición.
        
        Args:
            side: Lado de la posición (1 para Largo, -1 para Corto)
            market_close_price: Precio de cierre actual
            atr_value: Valor ATR actual
        """
        # Determina el tipo de acción
        desired_action = "OPEN_LONG" if side == 1 else "OPEN_SHORT"
        
        # Calcula los detalles de ejecución con el broker simulado
        execution_details = self.broker.calculate_execution_details(
            desired_action=desired_action,
            market_close_price=market_close_price,
            atr_value=atr_value
        )
        
        execution_price = execution_details["execution_price"]
        
        # Calcula el tamaño de la posición
        position_size_contracts, commission, margin_required = self.broker.calculate_position_size_contracts(
            equity=self.current_equity,
            position_size_pct_equity=self.position_size_pct_equity,
            leverage=self.leverage,
            execution_price=execution_price
        )
        
        # Si el tamaño de la posición es 0, no se puede abrir (ej. si no hay suficiente equity)
        if position_size_contracts <= 0:
            logger.warning(f"No se pudo abrir posición {desired_action}: tamaño insuficiente.")
            return
        
        # Actualiza el estado de la cartera
        self.active_position_side = side
        self.active_position_size_contracts = position_size_contracts
        self.active_position_entry_price = execution_price
        self.margin_used = margin_required
        
        # Calcula el valor nocional
        notional_value = position_size_contracts * execution_price
        
        # Deduce la comisión del balance
        self.balance -= commission
        self.current_equity = self.balance  # El P&L no realizado se actualizará en el próximo paso
        
        # Calcula el precio de liquidación
        self.liquidation_price = self.broker.calculate_liquidation_price(
            position_side=side, 
            entry_price=execution_price, 
            leverage=self.leverage, 
            safety_factor=self.liquidation_safety_factor
        )
        
        # Inicializa los pasos en la posición
        self.steps_in_current_position = 0
        
        # Actualiza estadísticas
        self.episode_stats['trades'] += 1
        self.episode_stats['total_fees'] += commission
        # BQ Agg Stats
        self.current_episode_agg_stats['num_trades'] += 1
        self.current_episode_agg_stats['total_fees'] += commission
        
        logger.info(f"Posición abierta: {desired_action}, Precio: {execution_price}, Tamaño: {position_size_contracts}, Comisión: {commission}")
    
    def _close_position(self, market_close_price: float, atr_value: float) -> None:
        """
        Cierra la posición activa.
        
        Args:
            market_close_price: Precio de cierre actual
            atr_value: Valor ATR actual
        """
        # Si no hay posición activa, no hace nada
        if self.active_position_side == 0:
            return
        
        # Determina el tipo de acción
        desired_action = "CLOSE_LONG" if self.active_position_side == 1 else "CLOSE_SHORT"
        
        # Calcula los detalles de ejecución
        execution_details = self.broker.calculate_execution_details(
            desired_action=desired_action,
            market_close_price=market_close_price,
            atr_value=atr_value,
            position_to_close_entry_price=self.active_position_entry_price,
            position_to_close_size=self.active_position_size_contracts
        )
        
        execution_price = execution_details["execution_price"]
        pnl = execution_details["potential_pnl"]
        commission = execution_details["commission_to_be_paid"]
        
        # Actualiza el balance con el P&L realizado y deduce la comisión
        self.balance += pnl - commission
        
        # Actualiza estadísticas
        if pnl > 0:
            self.episode_stats['profitable_trades'] += 1
        else:
            self.episode_stats['unprofitable_trades'] += 1
        
        self.episode_stats['total_pnl'] += pnl
        self.episode_stats['total_fees'] += commission
        # BQ Agg Stats
        self.current_episode_agg_stats['num_trades'] += 1 # Closing a position is also a trade event for this counter
        self.current_episode_agg_stats['total_fees'] += commission
        self.current_episode_agg_stats['pnl_realized'] += pnl
        
        logger.info(f"Posición cerrada: {desired_action}, Precio: {execution_price}, P&L: {pnl}, Comisión: {commission}")
        
        # Reinicia el estado de la posición
        self.active_position_side = 0
        self.active_position_size_contracts = 0.0
        self.active_position_entry_price = 0.0
        self.margin_used = 0.0
        self.unrealized_pnl = 0.0
        self.steps_in_current_position = 0
        self.liquidation_price = 0.0
        
        # Actualiza el equity (ahora igual al balance, ya que no hay posición)
        self.current_equity = self.balance
    
    def _check_liquidation(self, current_market_data) -> bool:
        """
        Comprueba si la posición activa debe ser liquidada.
        
        Args:
            current_market_data: Datos de mercado actuales
            
        Returns:
            True si la posición fue liquidada, False en caso contrario
        """
        # Comprobación rápida: si no hay posición activa, retornar inmediatamente
        if self.active_position_side == 0:
            return False
            
        # Si no hay precio de liquidación válido, no puede haber liquidación
        if self.liquidation_price <= 0:
            return False
            
        # Comprobación rápida con precio de cierre
        close_price = self._get_current_close_price()
        
        # Para posiciones largas, liquidación cuando precio <= liquidation_price
        if self.active_position_side == 1 and close_price <= self.liquidation_price:
            self._liquidate_position(self.liquidation_price)
            return True
            
        # Para posiciones cortas, liquidación cuando precio >= liquidation_price
        if self.active_position_side == -1 and close_price >= self.liquidation_price:
            self._liquidate_position(self.liquidation_price)
            return True
            
        # Si hemos llegado aquí, no hay liquidación con el precio de cierre.
        # Para optimizar, evitamos cálculos adicionales si la diferencia con el precio 
        # de liquidación es suficientemente grande
        margin_to_liquidation = abs(close_price - self.liquidation_price) / close_price
        if margin_to_liquidation > 0.02:  # 2% de margen es suficientemente seguro
            return False
            
        # Si estamos cerca de la liquidación, hacemos una verificación más precisa
        # Cache para índices de características - calculamos una vez y reutilizamos
        if not hasattr(self, '_price_indices_cache'):
            self._price_indices_cache = {
                'high_feat_idx': self.feature_names.index('log_ret_H_O_norm') if 'log_ret_H_O_norm' in self.feature_names else -1,
                'low_feat_idx': self.feature_names.index('log_ret_L_O_norm') if 'log_ret_L_O_norm' in self.feature_names else -1
            }
        
        # Inicializar high/low con valores por defecto (siendo conservadores)
        high_price = close_price * 1.01
        low_price = close_price * 0.99
        
        # Usar ATR directamente desde datos preprocesados si están disponibles (más rápido)
        if hasattr(self, 'atr_values') and self.atr_values is not None:
            try:
                atr = float(self.atr_values[self.current_step_index])
                # Usar ATR para aproximaciones más precisas de high y low
                high_price = close_price + atr * 0.5
                low_price = close_price - atr * 0.5
            except (IndexError, AttributeError):
                pass
        
        # Si tenemos índices para high/low, usarlos para cálculo más preciso
        high_feat_idx = self._price_indices_cache['high_feat_idx']
        low_feat_idx = self._price_indices_cache['low_feat_idx']
        
        if high_feat_idx >= 0 and low_feat_idx >= 0:
            try:
                # Acceso directo a los datos con comprobaciones para evitar errores
                open_price = close_price
                high_price = open_price * np.exp(current_market_data[-1, high_feat_idx])
                low_price = open_price * np.exp(current_market_data[-1, low_feat_idx])
            except Exception:
                pass
        
        # Comprobación de liquidación final
        if ((self.active_position_side == 1 and low_price <= self.liquidation_price) or
           (self.active_position_side == -1 and high_price >= self.liquidation_price)):
            self._liquidate_position(self.liquidation_price)
            return True
        
        return False
        
    def _get_current_close_price(self) -> float:
        """
        Obtiene el precio de cierre actual del mercado.
        
        Returns:
            Precio de cierre actual
        """
        # Primero intentamos usar close_prices no normalizados si están disponibles
        try:
            if hasattr(self, 'close_prices') and self.close_prices is not None:
                return float(self.close_prices[self.current_step_index])
        except (IndexError, AttributeError) as e:
            # Si hay un error, continuamos con el método anterior
            logger.warning(f"Error al acceder a self.close_prices[{self.current_step_index}]: {e}. Usando fallback.")
            pass
            
        # Si no tenemos close_prices, usamos el método anterior (optimizado)
        if not hasattr(self, '_close_idx_cache'):
            # Cache the index to avoid repeated searches
            self._close_idx_cache = -1
            for possible_name in ['close', 'Close', 'price', 'Price', 'log_ret_C_O_norm', 'log_ret_C_C_prev_norm']:
                if possible_name in self.feature_names:
                    self._close_idx_cache = self.feature_names.index(possible_name)
                    break
        
        close_idx = self._close_idx_cache
        
        # Si no se encuentra, usamos un valor por defecto
        if close_idx == -1:
            return 30000.0  # Valor por defecto para BTC
        
        # Acceso directo al dato usando el índice cacheado
        current_market_data = self.market_data[self.current_step_index]
        
        # Si es un retorno log, aproximamos el precio absoluto
        if 'log_ret' in self.feature_names[close_idx]:
            return 30000.0 * (1 + current_market_data[-1, close_idx])
        else:
            # Si es el precio directo
            return float(current_market_data[-1, close_idx])
    
    def _liquidate_position(self, liquidation_price: float) -> None:
        """
        Ejecuta la liquidación forzosa de la posición actual.
        
        Args:
            liquidation_price: Precio al que se liquida la posición
        """
        # Si no hay posición activa, no hace nada
        if self.active_position_side == 0:
            return
        
        # Calcula el P&L realizado (siempre será negativo en una liquidación)
        if self.active_position_side == 1:  # Largo
            pnl = (liquidation_price - self.active_position_entry_price) * self.active_position_size_contracts
        else:  # Corto
            pnl = (self.active_position_entry_price - liquidation_price) * self.active_position_size_contracts
        
        # Calcula la comisión (taker fee sobre el valor nocional)
        notional_value = self.active_position_size_contracts * liquidation_price
        commission = notional_value * self.broker.taker_fee_rate
        
        # Actualiza el balance
        self.balance += pnl - commission
        
        # Actualiza estadísticas
        self.episode_stats['trades'] += 1
        self.episode_stats['unprofitable_trades'] += 1
        self.episode_stats['total_pnl'] += pnl
        self.episode_stats['total_fees'] += commission
        # BQ Agg Stats
        self.current_episode_agg_stats['num_trades'] += 1 # Liquidation is a trade event
        self.current_episode_agg_stats['total_fees'] += commission
        self.current_episode_agg_stats['pnl_realized'] += pnl
        
        logger.warning(f"Posición liquidada: Precio: {liquidation_price}, P&L: {pnl}, Comisión: {commission}")
        
        # Reinicia el estado de la posición
        self.active_position_side = 0
        self.active_position_size_contracts = 0.0
        self.active_position_entry_price = 0.0
        self.margin_used = 0.0
        self.unrealized_pnl = 0.0
        self.steps_in_current_position = 0
        self.liquidation_price = 0.0
        
        # Actualiza el equity (ahora igual al balance, ya que no hay posición)
        self.current_equity = self.balance
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Construye la observación para el agente.
        
        Returns:
            Dict con características de mercado y cartera
        """
        try:
            # Cada índice en market_data ya representa una secuencia completa de longitud L
            # Seleccionamos la secuencia actual directamente
            
            # Extraemos la secuencia actual de datos de mercado
            market_features = self.market_data[self.current_step_index]
            
            # Aseguramos que sea float32
            if market_features.dtype != np.float32:
                market_features = market_features.astype(np.float32)
            
            # Normaliza las características de cartera (ya en float32)
            portfolio_features = self._get_normalized_portfolio_features()
            
            # Construye la observación como un Dict (reusamos el mismo diccionario si ya existe)
            if not hasattr(self, '_observation_cache'):
                self._observation_cache = {
                    'market_features': market_features,
                    'portfolio_features': portfolio_features
                }
            else:
                # Actualizamos el diccionario existente sin crear uno nuevo
                self._observation_cache['market_features'] = market_features
                self._observation_cache['portfolio_features'] = portfolio_features
            
            return self._observation_cache
        except Exception as e:
            logger.error(f"Error al construir observación: {e}")
            logger.error(f"Índice actual: {self.current_step_index}, L: {self.L}, Forma market_data: {self.market_data.shape}")
            raise
            
    def get_torch_observation(self, observation: Dict[str, np.ndarray], device: str = "cuda") -> Dict[str, torch.Tensor]:
        """
        Convierte una observación numpy a tensores de PyTorch para uso con GPU.
        
        Args:
            observation: Observación en formato numpy
            device: Dispositivo donde colocar los tensores ('cuda', 'mps', 'cpu')
            
        Returns:
            Diccionario con tensores de PyTorch
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch no está disponible. Instálalo para usar esta función.")
            
        # Crear diccionario para tensores
        torch_obs = {}
        
        # Convertir cada array numpy a tensor
        for key, value in observation.items():
            # Asegurar que los datos están en float32 para PyTorch
            if value.dtype != np.float32:
                value = value.astype(np.float32)
                
            # Convertir a tensor y mover al dispositivo especificado
            torch_obs[key] = torch.tensor(value, dtype=torch.float32, device=device)
            
        return torch_obs
    
    def _get_normalized_portfolio_features(self) -> np.ndarray:
        """
        Normaliza las características de la cartera optimizado con buffer preasignado.
        
        Returns:
            Array con 8 características normalizadas
        """
        # Inicializar buffer de características si no existe
        if not hasattr(self, '_portfolio_features_buffer'):
            self._portfolio_features_buffer = np.zeros(8, dtype=np.float32)
            
        # Ref para más claridad en el código
        features = self._portfolio_features_buffer
        
        # 1. Estado Posición: {-1, 0, 1}
        features[0] = self.active_position_side
        
        is_neutral = self.active_position_side == 0
        
        # 2. Tamaño Posición Normalizado
        if is_neutral:
            features[1] = 0.0
        else:
            features[1] = (self.active_position_size_contracts * self.active_position_entry_price) / self.initial_equity_episode
        
        # 3. Precio Entrada Normalizado
        if is_neutral:
            features[2] = 0.0
        else:
            # Calculamos close_price una sola vez y lo reutilizamos
            close_price = self._get_current_close_price()
            features[2] = self.active_position_entry_price / close_price - 1.0
        
        # 4. P&L No Realizado Normalizado
        features[3] = self.unrealized_pnl / max(self.current_equity, 1.0)  # Evita división por cero
        
        # 5. Retorno Log Equity (último paso)
        features[4] = np.log(self.current_equity / max(self.last_equity, 1.0))  # max evita división por cero
        
        # 6. Ratio de Margen Disponible
        features[5] = self.available_margin / max(self.current_equity, 1.0)
        
        # 7. Pasos en Posición Normalizados
        max_steps = self.config['portfolio_features_normalization']['max_steps_in_position']
        features[6] = min(self.steps_in_current_position / max_steps, 1.0)  # Clamp a 1.0
        
        # 8. Apalancamiento Configurado
        features[7] = self.leverage
        
        # Devolver la referencia al buffer ya rellenado (sin crear copias)
        return features
        
        return portfolio_features
    
    def render(self):
        """
        Renderiza el estado actual del entorno.
        """
        if self.render_mode == 'human':
            print(f"\n--- Paso {self.current_step_index} ---")
            print(f"Equity: ${self.current_equity:.2f} (Inicial: ${self.initial_equity_episode:.2f})")
            print(f"Retorno: {(self.current_equity / self.initial_equity_episode - 1.0) * 100:.2f}%")
            
            if self.active_position_side == 1:
                position_text = f"LARGO: {self.active_position_size_contracts:.3f} contratos @ ${self.active_position_entry_price:.2f}"
                print(position_text)
                print(f"Precio de liquidación: ${self.liquidation_price:.2f}")
                print(f"P&L No Realizado: ${self.unrealized_pnl:.2f}")
            elif self.active_position_side == -1:
                position_text = f"CORTO: {self.active_position_size_contracts:.3f} contratos @ ${self.active_position_entry_price:.2f}"
                print(position_text)
                print(f"Precio de liquidación: ${self.liquidation_price:.2f}")
                print(f"P&L No Realizado: ${self.unrealized_pnl:.2f}")
            else:
                print("Posición: NEUTRAL")
            
            print(f"Operaciones: {self.episode_stats['trades']}, Ganadores: {self.episode_stats['profitable_trades']}")
            print(f"P&L Total: ${self.episode_stats['total_pnl']:.2f}, Comisiones: ${self.episode_stats['total_fees']:.2f}")
            print("-------------------")
    
    def close(self):
        """
        Cierra el entorno y libera recursos.
        """
        pass
    
    def _get_atr_value_optimized(self, current_market_data, close_price):
        """
        Método auxiliar para obtener el valor ATR con caché de índices.
        
        Args:
            current_market_data: Datos de mercado actuales
            close_price: Precio de cierre actual
            
        Returns:
            Valor ATR (desnormalizado)
        """
        # Intentar usar ATR pre-calculado primero si está disponible
        try:
            if hasattr(self, 'atr_values') and self.atr_values is not None:
                return float(self.atr_values[self.current_step_index])
        except (IndexError, AttributeError, TypeError):
            # Si falla, continuamos con el método basado en features
            pass
            
        # Cachear el índice ATR para evitar búsquedas repetidas
        if not hasattr(self, '_atr_idx_cache'):
            self._atr_idx_cache = self.feature_names.index('ATR_norm') if 'ATR_norm' in self.feature_names else -1
            
        atr_idx = self._atr_idx_cache
        
        # Si tenemos el índice, extraer el valor
        if atr_idx >= 0:
            # Des-normalizar si ATR_norm = ATR / Close
            atr_value = current_market_data[-1, atr_idx] * close_price
            return atr_value
        else:
            # Valor por defecto
            return close_price * 0.01  # 1% del precio como aproximación

    def get_current_episode_step_data(self) -> list:
        """Returns a copy of the current episode's step data and clears the internal list."""
        data_to_return = list(self.current_episode_step_data) # Return a shallow copy
        self.current_episode_step_data.clear()
        return data_to_return
