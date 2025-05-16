import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, Union, List
import logging

from src.utils.config import ConfigManager
from src.environments.simulated_broker import SimulatedBroker

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TradingEnv')

class TradingEnvironment(gym.Env):
    """
    Entorno de trading de futuros para Gymnasium.
    
    Este entorno simula el trading de futuros de criptomonedas con apalancamiento
    fijo y tamaño de posición relativo al equity.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config_path: str = 'src/environments/environment_config.yaml', render_mode: Optional[str] = None):
        """
        Inicializa el entorno de trading.
        
        Args:
            config_path: Ruta al archivo de configuración yaml
            render_mode: Modo de renderización (human, etc.)
        """
        # Carga la configuración del entorno
        self.config_manager = ConfigManager(config_path=config_path)
        self.config = self._load_env_config()
        
        # Configura el render_mode
        self.render_mode = render_mode
        
        # Carga los datos de mercado preprocesados
        self.market_data, self.feature_names = self._load_market_data()
        self.L = self.config['sequence_length_L']
        
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
        
        logger.info("TradingEnvironment initialized successfully.")
    
    def _load_env_config(self) -> Dict[str, Any]:
        """Carga la configuración del entorno desde el archivo yaml"""
        env_config = {}
        
        # Configuración general del entorno
        env_config['env_id'] = self.config_manager.get_config_value('env_id', 'FuturesTradingEnv-v0')
        env_config['initial_equity'] = self.config_manager.get_config_value('initial_equity', 10000.0)
        env_config['max_episode_steps_use_dataset_length'] = self.config_manager.get_config_value('max_episode_steps_use_dataset_length', True)
        env_config['allow_random_episode_start'] = self.config_manager.get_config_value('allow_random_episode_start', True)
        
        # Configuración de Trading
        env_config['leverage'] = self.config_manager.get_config_value('leverage', 10.0)
        env_config['position_size_pct_equity'] = self.config_manager.get_config_value('position_size_pct_equity', 0.05)
        env_config['taker_fee_rate'] = self.config_manager.get_config_value('taker_fee_rate', 0.0004)
        env_config['slippage_atr_multiplier'] = self.config_manager.get_config_value('slippage_atr_multiplier', 0.1)
        env_config['min_order_size_btc'] = self.config_manager.get_config_value('min_order_size_btc', 0.001)
        
        # Lógica de Acción
        env_config['action_threshold'] = self.config_manager.get_config_value('action_threshold', 0.15)
        
        # Lógica de Finalización y Liquidación
        env_config['equity_drawdown_threshold_episode_end'] = self.config_manager.get_config_value('equity_drawdown_threshold_episode_end', -0.20)
        env_config['liquidation_safety_factor'] = self.config_manager.get_config_value('liquidation_safety_factor', 0.8)
        
        # Características de Observación
        max_steps_in_position = self.config_manager.get_config_value('portfolio_features_normalization.max_steps_in_position', 288)
        env_config['portfolio_features_normalization'] = {
            'max_steps_in_position': max_steps_in_position
        }
        
        # Carga de Datos de Mercado
        env_config['processed_data_directory'] = self.config_manager.get_config_value('processed_data_directory', 'data/processed/')
        env_config['processed_data_file_identifier'] = self.config_manager.get_config_value('processed_data_file_identifier', '_L96_market_features.npz')
        
        # Obtiene la longitud de secuencia del archivo de preprocesamiento
        preprocessing_config_path = 'src/data/preprocessing_config.yaml'
        preprocessing_config = ConfigManager(config_path=preprocessing_config_path)
        env_config['sequence_length_L'] = preprocessing_config.get_config_value('sequence_length_L', 96)
        
        return env_config
    
    def _load_market_data(self) -> Tuple[np.ndarray, List[str]]:
        """
        Carga los datos de mercado preprocesados.
        
        Returns:
            Tuple con (datos_de_mercado, nombres_de_características)
        """
        # Busca el archivo con los datos preprocesados
        data_dir = self.config['processed_data_directory']
        file_identifier = self.config['processed_data_file_identifier']
        
        # Lista todos los archivos en el directorio
        all_files = os.listdir(data_dir)
        
        # Filtra por el identificador
        matching_files = [f for f in all_files if file_identifier in f]
        
        if not matching_files:
            raise FileNotFoundError(f"No se encontraron archivos con el identificador {file_identifier} en {data_dir}")
        
        # Utiliza el primer archivo que coincida (se podría hacer más sofisticado si hay múltiples)
        data_file = os.path.join(data_dir, matching_files[0])
        logger.info(f"Cargando datos de mercado desde: {data_file}")
        
        # Carga los datos
        data = np.load(data_file)
        
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
            raise KeyError(f"No se encontró ninguna clave válida para datos de mercado en {data_file}")
        
        logger.info(f"Datos de mercado cargados: {market_features.shape}")
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
            # Asegura que haya suficientes datos para al menos una secuencia completa
            max_start_idx = len(self.market_data) - self.L * 2  # * 2 para tener espacio para al menos un episodio razonable
            self.current_step_index = self.np_random.integers(0, max_start_idx) if max_start_idx > 0 else 0
        else:
            # Comienza desde el principio (asegura al menos una secuencia de longitud L)
            self.current_step_index = self.L
        
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
        # Extrae la señal de acción del array
        action_signal = float(action[0])
        
        # Guarda el equity anterior para el cálculo de la recompensa
        self.last_equity = self.current_equity
        
        # Avanza al siguiente paso en los datos de mercado
        self.current_step_index += 1
        
        # Obtiene los datos actuales del mercado
        current_market_data = self.market_data[self.current_step_index]
        
        # Obtiene el precio de cierre usando nuestro método auxiliar
        close_price = self._get_current_close_price()
        
        # Intenta obtener el ATR si está disponible
        atr_idx = self.feature_names.index('ATR_norm') if 'ATR_norm' in self.feature_names else -1
        
        # Si no tenemos ATR directamente, usamos un valor por defecto
        if atr_idx >= 0:
            # Des-normalizar si ATR_norm = ATR / Close
            atr_value = current_market_data[-1, atr_idx] * close_price
            logger.debug(f"ATR extraído: {atr_value:.2f}")
        else:
            # Valor por defecto o calcular ATR en tiempo real
            atr_value = close_price * 0.01  # 1% del precio como aproximación
            logger.debug(f"ATR aproximado: {atr_value:.2f} (1% del precio {close_price:.2f})")
        
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
        if self.current_step_index >= len(self.market_data) - self.L:
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
        # Si no hay posición activa, no hay nada que liquidar
        if self.active_position_side == 0:
            return False
        
        # Busca los índices para High y Low en los datos
        # Suponemos que las primeras 4 características pueden ser OHLC o extractos de ellas
        # Si no están disponibles directamente, usamos el precio de cierre como aproximación
        close_price = self._get_current_close_price()
        
        # Aproximamos high y low como ±1% del cierre si no están disponibles directamente
        high_price = close_price * 1.01  # Aproximación
        low_price = close_price * 0.99   # Aproximación
        
        # Intentamos extraer high y low si están disponibles
        try:
            # Los precios probablemente estén codificados como retornos log o normalizados
            # Revirtiendo la normalización aproximadamente
            high_feat_idx = self.feature_names.index('log_ret_H_O_norm') if 'log_ret_H_O_norm' in self.feature_names else -1
            low_feat_idx = self.feature_names.index('log_ret_L_O_norm') if 'log_ret_L_O_norm' in self.feature_names else -1
            
            if high_feat_idx >= 0 and low_feat_idx >= 0:
                # Si tenemos estos valores, podemos aproximar high y low
                open_price = close_price  # Aproximación, usamos close como open
                high_price = open_price * np.exp(current_market_data[-1, high_feat_idx])
                low_price = open_price * np.exp(current_market_data[-1, low_feat_idx])
                logger.debug(f"Precios High/Low calculados: {high_price:.2f}/{low_price:.2f}")
        except Exception as e:
            logger.warning(f"No se pudieron extraer High/Low de los datos: {e}")
            logger.warning("Usando aproximación de High/Low")
        
        # Para posición larga, comprueba si el precio mínimo del periodo bajó del precio de liquidación
        if self.active_position_side == 1 and low_price <= self.liquidation_price:
            # Ejecuta liquidación a precio de liquidación
            self._liquidate_position(self.liquidation_price)
            return True
        
        # Para posición corta, comprueba si el precio máximo del periodo subió del precio de liquidación
        if self.active_position_side == -1 and high_price >= self.liquidation_price:
            # Ejecuta liquidación a precio de liquidación
            self._liquidate_position(self.liquidation_price)
            return True
        
        return False
        
    def _get_current_close_price(self) -> float:
        """
        Obtiene el precio de cierre actual del mercado.
        
        Returns:
            Precio de cierre actual
        """
        # Buscamos si existe una columna específica para Close
        close_idx = -1
        
        # Intenta localizar la columna del precio de cierre en las características disponibles
        for possible_name in ['close', 'Close', 'price', 'Price', 'log_ret_C_O_norm', 'log_ret_C_C_prev_norm']:
            if possible_name in self.feature_names:
                close_idx = self.feature_names.index(possible_name)
                break
        
        # Si no se encuentra, usamos un valor promedio o aproximado
        if close_idx == -1:
            # Usamos el último precio conocido o un valor arbitrario para simulación
            return 30000.0  # Valor por defecto para BTC si no hay datos
        
        # Si es un retorno log, necesitamos convertirlo a precio absoluto
        # Esto es una aproximación, ya que necesitaríamos el precio anterior
        if 'log_ret' in self.feature_names[close_idx]:
            # Asumiendo una base de precio aproximada para Bitcoin
            return 30000.0 * (1 + self.market_data[self.current_step_index, -1, close_idx])
        else:
            # Si es el precio directo
            return float(self.market_data[self.current_step_index, -1, close_idx])
    
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
            # Extrae la secuencia de datos de mercado de longitud L
            market_features = self.market_data[self.current_step_index - self.L + 1:self.current_step_index + 1]
            
            # Normaliza las características de cartera
            portfolio_features = self._get_normalized_portfolio_features()
            
            # Construye la observación como un Dict
            observation = {
                'market_features': market_features.astype(np.float32),
                'portfolio_features': portfolio_features.astype(np.float32)
            }
            
            return observation
        except Exception as e:
            logger.error(f"Error al construir observación: {e}")
            logger.error(f"Índice actual: {self.current_step_index}, L: {self.L}, Forma market_data: {self.market_data.shape}")
            raise
    
    def _get_normalized_portfolio_features(self) -> np.ndarray:
        """
        Normaliza las características de la cartera.
        
        Returns:
            Array con 8 características normalizadas
        """
        # 1. Estado Posición: {-1, 0, 1}
        position_state = self.active_position_side
        
        # 2. Tamaño Posición Normalizado
        if self.active_position_side == 0:
            normalized_position_size = 0.0
        else:
            normalized_position_size = (self.active_position_size_contracts * self.active_position_entry_price) / self.initial_equity_episode
        
        # 3. Precio Entrada Normalizado
        if self.active_position_side == 0:
            normalized_entry_price = 0.0
        else:
            # Obtiene el precio de cierre actual usando nuestro método auxiliar
            close_price = self._get_current_close_price()
            normalized_entry_price = self.active_position_entry_price / close_price - 1.0
        
        # 4. P&L No Realizado Normalizado
        normalized_unrealized_pnl = self.unrealized_pnl / max(self.current_equity, 1.0)  # Evita división por cero
        
        # 5. Retorno Log Equity (último paso)
        if self.last_equity > 0:
            log_equity_return = np.log(self.current_equity / self.last_equity)
        else:
            log_equity_return = 0.0
        
        # 6. Ratio de Margen Disponible
        margin_ratio = self.available_margin / max(self.current_equity, 1.0)  # Evita división por cero
        
        # 7. Pasos en Posición Normalizados
        max_steps = self.config['portfolio_features_normalization']['max_steps_in_position']
        normalized_steps = self.steps_in_current_position / max_steps
        
        # 8. Apalancamiento Configurado
        leverage = self.leverage
        
        # Combina las características en un array
        portfolio_features = np.array([
            position_state,
            normalized_position_size,
            normalized_entry_price,
            normalized_unrealized_pnl,
            log_equity_return,
            margin_ratio,
            normalized_steps,
            leverage
        ], dtype=np.float32)
        
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
