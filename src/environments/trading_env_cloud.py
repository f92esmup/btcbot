import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, Union, List
import logging
import tempfile
import fsspec

from src.environments.simulated_broker import SimulatedBroker

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TradingEnvCloud')

# Importar torch de manera condicional para no crear dependencia obligatoria
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class TradingEnvironmentCloud(gym.Env):
    """
    Entorno de trading de futuros para Gymnasium, optimizado para GCP.
    
    Este entorno simula el trading de futuros de criptomonedas con apalancamiento
    fijo y tamaño de posición relativo al equity, cargando datos directamente
    de Google Cloud Storage.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 sequence_length_L: int = 96,  # Longitud de secuencia para el Transformer
                 initial_equity: float = 10000.0,  # Saldo inicial
                 leverage: int = 1,  # Apalancamiento
                 position_size_percentage: float = 0.2,  # Porcentaje de equity para posiciones
                 stop_loss_percentage: Optional[float] = None,  # Stop loss, si se usa
                 take_profit_percentage: Optional[float] = None,  # Take profit, si se usa
                 trading_fees: float = 0.0004,  # Comisiones de trading (0.04% por defecto)
                 slippage: float = 0.0001,  # Deslizamiento (0.01%)
                 data_gcs_path: Optional[str] = None,  # Ruta a los datos en GCS
                 use_reward_scaling: bool = True,  # Escalar recompensas
                 render_mode: Optional[str] = None):
        """
        Inicializa el entorno de trading con parámetros directos (sin YAML).
        
        Args:
            sequence_length_L: Longitud de secuencia para el Transformer
            initial_equity: Saldo inicial para trading
            leverage: Apalancamiento a usar
            position_size_percentage: Porcentaje de equity para abrir posiciones
            stop_loss_percentage: Porcentaje de stop loss (None = desactivado)
            take_profit_percentage: Porcentaje de take profit (None = desactivado)
            trading_fees: Comisiones por trade (porcentaje)
            slippage: Deslizamiento por trade (porcentaje)
            data_gcs_path: Ruta a los datos en GCS (formato NPZ)
            use_reward_scaling: Si es True, escala las recompensas
            render_mode: Modo de renderización
        """
        # Guardar parámetros
        self.L = sequence_length_L
        self.initial_equity = initial_equity
        self.leverage = leverage
        self.position_size_percentage = position_size_percentage
        self.stop_loss_percentage = stop_loss_percentage
        self.take_profit_percentage = take_profit_percentage
        self.trading_fees = trading_fees
        self.slippage = slippage
        self.data_gcs_path = data_gcs_path
        self.use_reward_scaling = use_reward_scaling
        self.render_mode = render_mode
        
        # Inicializar configuración adicional
        self.funding_rate = 0.0001  # Tasa de financiación promedio diaria
        self.funding_period = 8  # Periodos por día para financiación (8h)
        self.max_position_duration = None  # Sin límite de duración por defecto
        self.time_feature_count = 5  # Características de tiempo (hora, día, etc.)
        
        # Carga los datos de mercado preprocesados
        self._load_market_data(data_gcs_path)
        
        # Estructura de observación: mercado + cartera
        market_feature_dim = self.X_market.shape[2]  # Dim. de características de mercado
        
        # Para las características de la cartera (8): 
        # [equity_normalized, pnl_actualized, current_price_entry_ratio, 
        #  position_size_normalized, long_filled, short_filled, 
        #  duration_normalized, funding_accrued_normalized]
        portfolio_feature_dim = 8  
        
        # Definir el espacio de observación como un Dict con dos tensores
        self.observation_space = spaces.Dict({
            'market_features': spaces.Box(
                low=-10, high=10, 
                shape=(self.L, market_feature_dim), 
                dtype=np.float32
            ),
            'portfolio_features': spaces.Box(
                low=-10, high=10, 
                shape=(self.L, portfolio_feature_dim), 
                dtype=np.float32
            )
        })
        
        # Espacio de acción: [posición]
        # -1 = 100% short, 0 = cerrado, 1 = 100% long
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Inicializar otras variables
        self.current_step = 0
        self.episode_step = 0
        self.total_steps = self.X_market.shape[0] - 1  # -1 para poder avanzar al menos una vez
        self.episode_number = 0
        self.broker = None
        
        # Estadísticas para el episodio actual
        self.episode_stats = {
            'returns': [],
            'equity_curve': [],
            'trades': [],
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'last_equity_value': self.initial_equity,
            'last_equity_timestamp': None
        }
        
        # Resetea el entorno (inicializa todo)
        self.reset()
        
        logger.info(f"Entorno de trading inicializado: {self.total_steps} pasos disponibles, "
                   f"secuencia L={self.L}, mercado {market_feature_dim}D, cartera {portfolio_feature_dim}D")
    
    def _load_market_data(self, data_path: str):
        """
        Carga los datos de mercado desde un archivo NPZ en GCS.
        
        Args:
            data_path: Ruta completa al archivo NPZ en GCS
        """
        if data_path is None:
            raise ValueError("No se proporcionó ruta a los datos de mercado (data_gcs_path)")
        
        logger.info(f"Cargando datos de mercado desde: {data_path}")
        
        try:
            # Usar fsspec para cargar el archivo NPZ desde GCS
            with fsspec.open(data_path, 'rb') as f:
                with tempfile.NamedTemporaryFile() as temp:
                    # Copiar el contenido a un archivo temporal
                    temp.write(f.read())
                    temp.flush()
                    
                    # Cargar desde el archivo temporal
                    data = np.load(temp.name)
                    
                    # Cargar arrays del NPZ
                    self.X_market = data['X_market']
                    self.timestamps = data['timestamps']
                    try:
                        self.feature_names = data['feature_names']
                    except KeyError:
                        logger.warning("No se encontraron nombres de características en el archivo NPZ")
                        self.feature_names = [f"feature_{i}" for i in range(self.X_market.shape[2])]
            
            # Validar las dimensiones
            if len(self.X_market.shape) != 3:
                raise ValueError(f"Formato de datos incorrecto. Se esperan 3 dimensiones (num_samples, L, features), "
                                f"pero se obtuvo {self.X_market.shape}")
            
            # Asegurar que los datos estén en float32
            if self.X_market.dtype != np.float32:
                self.X_market = self.X_market.astype(np.float32)
            
            logger.info(f"Datos cargados: {self.X_market.shape[0]} secuencias de longitud {self.X_market.shape[1]} "
                       f"con {self.X_market.shape[2]} características")
            
        except Exception as e:
            logger.error(f"Error cargando datos desde GCS: {e}")
            raise
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reinicia el entorno al principio o a un punto aleatorio si options['random_start']=True.
        
        Args:
            seed: Semilla para generación de números aleatorios
            options: Opciones para el reset. Soporta 'random_start' (bool).
        
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        # Opciones de reset (valores por defecto si no se proporcionan)
        if options is None:
            options = {}
        
        random_start = options.get('random_start', False)
        
        # Escoger punto de inicio (aleatorio o desde el principio)
        if random_start:
            # Limitar el inicio aleatorio para asegurar que no nos quedamos sin datos
            max_start = max(0, self.total_steps - 1000)  # Al menos 1000 pasos, si hay suficientes
            self.current_step = self.np_random.integers(0, max_start)
        else:
            self.current_step = 0
        
        # Reiniciar contador de pasos del episodio
        self.episode_step = 0
        self.episode_number += 1
        
        # Inicializar broker simulado
        self.broker = SimulatedBroker(
            initial_equity=self.initial_equity,
            leverage=self.leverage,
            position_size_pct=self.position_size_percentage,
            stop_loss_pct=self.stop_loss_percentage,
            take_profit_pct=self.take_profit_percentage,
            trading_fee_pct=self.trading_fees,
            slippage_pct=self.slippage,
            funding_rate=self.funding_rate,
            funding_period=self.funding_period
        )
        
        # Obtener la secuencia de precios actual
        market_obs = self.X_market[self.current_step]
        
        # Construir historial de características de cartera (todo ceros al inicio)
        portfolio_features = np.zeros((self.L, 8), dtype=np.float32)
        
        # Normalizar el equity inicial (primera dimensión)
        portfolio_features[:, 0] = 0.0  # equity_normalized (cambio porcentual desde inicio, inicia en 0)
        
        observation = {
            'market_features': market_obs,
            'portfolio_features': portfolio_features
        }
        
        # Reiniciar estadísticas del episodio
        self._reset_episode_stats()
        
        # Información adicional
        info = {
            'equity': self.broker.equity,
            'step': self.episode_step,
            'timestamp': self.timestamps[self.current_step][-1]  # Último timestamp de la secuencia
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Ejecuta un paso en el entorno de trading.
        
        Args:
            action: Array numpy con la acción (-1 a 1) para ajustar posición
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Asegurar que el tipo de datos sea el correcto
        if not isinstance(action, np.ndarray):
            if isinstance(action, (list, tuple)):
                action = np.array(action, dtype=np.float32)
            else:
                action = np.array([action], dtype=np.float32)
        
        # Clamp la acción al rango válido
        action_value = np.clip(action[0], -1.0, 1.0)
        
        # Obtener el precio actual de cierre (desde la secuencia actual, último timestep)
        current_price = self._get_current_price()
        
        # Ejecutar la acción en el broker
        position_delta, execution_price = self.broker.update_position(
            action_value, current_price, self.timestamps[self.current_step][-1]
        )
        
        # Calcular recompensa (cambio porcentual en equity)
        reward = self._calculate_reward()
        
        # Avanzar al siguiente paso
        self.current_step += 1
        self.episode_step += 1
        
        # Verificar si el episodio ha terminado
        terminated = False
        truncated = False
        termination_reason = None
        
        # Terminar si el equity cae por debajo del umbral crítico (90% de pérdida)
        if self.broker.equity <= self.initial_equity * 0.1:
            terminated = True
            termination_reason = 'critical_loss'
            logger.info(f"Episodio {self.episode_number} terminado por pérdida crítica: "
                       f"equity = {self.broker.equity:.2f}")
        
        # Verificar si hubo liquidación
        if self._check_liquidation():
            terminated = True
            termination_reason = 'liquidation'
            logger.info(f"Episodio {self.episode_number} terminado por liquidación")
        
        # Truncar si llegamos al final de los datos
        if self.current_step >= self.total_steps:
            truncated = True
            termination_reason = 'data_end'
            logger.debug(f"Episodio {self.episode_number} truncado: fin de datos")
        
        # Actualizar las estadísticas del episodio
        self._update_episode_stats(reward, terminated or truncated)
        
        # Preparar la nueva observación
        observation, info = self._get_observation()
        
        # Añadir información adicional útil para análisis
        info.update({
            'equity': self.broker.equity,
            'action_taken': action_value,
            'position': self.broker.current_position,
            'price': current_price,
            'step': self.episode_step,
            'timestamp': self.timestamps[self.current_step-1][-1] if not truncated else None,
            'equity_change_pct': (self.broker.equity / self.episode_stats['last_equity_value'] - 1) * 100,
            'position_delta': position_delta,
            'execution_price': execution_price,
            'termination_reason': termination_reason,
            'drawdown_pct': ((self.episode_stats['max_equity'] - self.broker.equity) / self.episode_stats['max_equity']) * 100 if self.episode_stats['max_equity'] > 0 else 0
        })
        
        # Actualizar el valor de equity para la próxima comparación
        self.episode_stats['last_equity_value'] = self.broker.equity
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Construye la observación actual con las características de mercado y cartera.
        
        Returns:
            Tuple de (observación, info)
        """
        if self.current_step >= self.total_steps:
            # Si llegamos al final, reproducir la última observación
            market_obs = self.X_market[self.total_steps - 1].copy()
        else:
            market_obs = self.X_market[self.current_step].copy()
        
        # Construir las características de la cartera
        portfolio_features = self._build_portfolio_features()
        
        observation = {
            'market_features': market_obs,
            'portfolio_features': portfolio_features
        }
        
        # Info básica
        info = {
            'current_step': self.current_step,
            'equity': self.broker.equity
        }
        
        return observation, info
    
    def _build_portfolio_features(self) -> np.ndarray:
        """
        Construye el tensor de características de la cartera.
        
        Returns:
            Array de características de cartera (L, 8)
        """
        # Inicializar tensor de características de cartera
        portfolio_features = np.zeros((self.L, 8), dtype=np.float32)
        
        # 1. Equity normalizado (cambio porcentual desde inicio)
        equity_change = (self.broker.equity / self.initial_equity) - 1.0
        portfolio_features[:, 0] = equity_change
        
        # 2. PnL actualizado (si hay posición abierta)
        unrealized_pnl_pct = self.broker.unrealized_pnl_pct
        portfolio_features[:, 1] = unrealized_pnl_pct
        
        # 3. Ratio precio actual / precio de entrada
        price_entry_ratio = 0.0
        if self.broker.entry_price > 0:
            current_price = self._get_current_price()
            price_entry_ratio = (current_price / self.broker.entry_price) - 1.0
            # Invertir el signo para posiciones cortas
            if self.broker.current_position < 0:
                price_entry_ratio *= -1
        portfolio_features[:, 2] = price_entry_ratio
        
        # 4. Tamaño de posición normalizado (-1 a 1)
        portfolio_features[:, 3] = self.broker.current_position
        
        # 5-6. Indicadores de posición (long=1/0, short=1/0)
        long_filled = 1.0 if self.broker.current_position > 0 else 0.0
        short_filled = 1.0 if self.broker.current_position < 0 else 0.0
        portfolio_features[:, 4] = long_filled
        portfolio_features[:, 5] = short_filled
        
        # 7. Duración de la posición normalizada (0-1)
        duration_normalized = 0.0
        if self.broker.position_duration > 0:
            # Normalizar según una duración máxima (por ejemplo, 100 pasos)
            max_duration = 100 if self.max_position_duration is None else self.max_position_duration
            duration_normalized = min(1.0, self.broker.position_duration / max_duration)
        portfolio_features[:, 6] = duration_normalized
        
        # 8. Financiación acumulada normalizada
        funding_normalized = self.broker.funding_payments_accrued / (self.initial_equity * 0.01)
        portfolio_features[:, 7] = funding_normalized
        
        return portfolio_features
    
    def _get_current_price(self) -> float:
        """
        Obtiene el precio actual de cierre de la secuencia.
        
        Returns:
            Precio de cierre actual
        """
        # Implementación simple: usa el cierre del último valor de la secuencia actual
        try:
            # Asumimos que estamos en un mercado de futuros donde el precio de interés
            # es típicamente el cierre normalizado, pero debemos desnormalizarlo
            # En este caso, usamos un precio de referencia base
            base_price = 30000.0  # Precio base artificial BTCUSDT
            
            # Extraer el precio normalizado - esto dependerá de las características exactas
            # en los datos. Asumimos que el retorno log_ret_C_C_prev_norm es la 4ª columna (índice 3)
            # (esto es solo un ejemplo, deberías ajustarlo a tus datos reales)
            log_ret_norm_idx = 3  # Índice de log_ret_C_C_prev_norm
            
            # Para simplicidad, devolvemos un precio artificial constante
            # En una implementación real, usarías los datos normalizados para generar
            # un precio realista desnormalizado
            return base_price
            
        except Exception as e:
            logger.error(f"Error al obtener precio actual: {e}")
            return 30000.0  # Un valor predeterminado en caso de error
    
    def _check_liquidation(self) -> bool:
        """
        Verifica si la posición actual sería liquidada debido a falta de margen.
        
        Returns:
            True si la posición sería liquidada, False en caso contrario
        """
        if self.broker.current_position == 0:
            # No hay posición activa, no puede haber liquidación
            return False
            
        current_price = self._get_current_price()
        
        # Calcular el precio de liquidación basado en el margen disponible
        # Esta es una implementación simplificada del mecanismo de liquidación
        if self.broker.current_position > 0:  # Posición larga
            # Si el precio cae lo suficiente para que las pérdidas > margen disponible
            liquidation_price = self.broker.entry_price * (1 - 1/(self.broker.leverage))
            return current_price <= liquidation_price
        else:  # Posición corta
            # Si el precio sube lo suficiente para que las pérdidas > margen disponible
            liquidation_price = self.broker.entry_price * (1 + 1/(self.broker.leverage))
            return current_price >= liquidation_price
    
    def _calculate_reward(self) -> float:
        """
        Calcula la recompensa basada en el cambio de equity.
        
        Returns:
            Valor de recompensa
        """
        # Recompensa simple: cambio porcentual en el equity
        equity_change_pct = (self.broker.equity / self.episode_stats['last_equity_value']) - 1.0
        
        # Opcional: escalar la recompensa para valores más manejables
        if self.use_reward_scaling:
            # Escalar para que +/-1% cambio sea aproximadamente +/-1 recompensa
            reward = equity_change_pct * 100.0
        else:
            reward = equity_change_pct
        
        return reward
    
    def _reset_episode_stats(self):
        """Reinicia las estadísticas del episodio."""
        self.episode_stats = {
            'returns': [],
            'equity_curve': [self.initial_equity],
            'trades': [],
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'last_equity_value': self.initial_equity,
            'last_equity_timestamp': None
        }
    
    def _update_episode_stats(self, reward: float, is_terminal: bool):
        """
        Actualiza las estadísticas del episodio con la información más reciente.
        
        Args:
            reward: Recompensa del último paso
            is_terminal: Si el episodio ha terminado
        """
        # Actualizar retornos
        if self.use_reward_scaling:
            # Convertir de recompensa escalada a retorno real
            returns = reward / 100.0
        else:
            returns = reward
        
        self.episode_stats['returns'].append(returns)
        self.episode_stats['equity_curve'].append(self.broker.equity)
        
        # Actualizar estadísticas si el episodio ha terminado
        if is_terminal:
            # Estadísticas de trades
            self.episode_stats['total_trades'] = self.broker.trade_count
            
            if self.broker.trade_count > 0:
                # Win rate
                self.episode_stats['win_rate'] = self.broker.winning_trades / self.broker.trade_count
                
                # Profit factor
                if self.broker.total_losses != 0:
                    self.episode_stats['profit_factor'] = abs(self.broker.total_profits / self.broker.total_losses)
                else:
                    self.episode_stats['profit_factor'] = float('inf') if self.broker.total_profits > 0 else 0.0
            
            # Sharpe Ratio (anualizado, asumiendo que un paso es 1h)
            if len(self.episode_stats['returns']) > 1:
                returns_array = np.array(self.episode_stats['returns'])
                avg_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                
                if std_return != 0:
                    # Anualizar (√8760 para pasos horarios)
                    self.episode_stats['sharpe_ratio'] = (avg_return / std_return) * np.sqrt(8760)
                else:
                    self.episode_stats['sharpe_ratio'] = 0.0
            
            # Maximum Drawdown
            equity_curve = np.array(self.episode_stats['equity_curve'])
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            self.episode_stats['max_drawdown'] = np.max(drawdown)
            
            logger.info(f"Episodio {self.episode_number} estadísticas: "
                       f"Equity final={self.broker.equity:.2f}, "
                       f"Cambio={((self.broker.equity/self.initial_equity)-1)*100:.2f}%, "
                       f"Trades={self.broker.trade_count}, "
                       f"Win Rate={self.episode_stats['win_rate']*100:.1f}%, "
                       f"Sharpe={self.episode_stats['sharpe_ratio']:.2f}, "
                       f"Max DD={self.episode_stats['max_drawdown']*100:.2f}%")
    
    def render(self):
        """Renderiza el estado actual del entorno."""
        if self.render_mode == "human":
            # Implementación básica: imprime información básica
            print(f"Step: {self.episode_step}, Equity: ${self.broker.equity:.2f}, "
                  f"Position: {self.broker.current_position:.2f}, "
                  f"Unrealized PnL: {self.broker.unrealized_pnl:.2f}")
    
    def close(self):
        """Cierra los recursos del entorno."""
        pass
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Obtiene las estadísticas del episodio actual.
        
        Returns:
            Diccionario con estadísticas
        """
        return self.episode_stats