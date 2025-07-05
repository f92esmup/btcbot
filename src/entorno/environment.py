"""
Entorno de Trading de Futuros para Reinforcement Learning.

Este módulo implementa un entorno de trading de futuros que hereda de gymnasium.Env,
diseñado para simular trading de Bitcoin con apalancamiento, comisiones y slippage.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, TYPE_CHECKING, Union
import logging
from sklearn.preprocessing import MinMaxScaler

from .portfolio import Portfolio
from .base_portfolio import BasePortfolio, TipoOperacion
from .reward_strategy import BaseRewardStrategy, EquityChangeRewardStrategy, SortinoRewardStrategy
from .portfolio_state import get_normalized_portfolio_features
# Importar tipo de configuración solo para type hints
if TYPE_CHECKING:
    from ..configuration import EnvironmentConfig
    from ..utils.observation_builder import ObservationBuilder


logger = logging.getLogger(__name__)


class FuturesTradingEnv(gym.Env):
    """
    Entorno de trading de futuros para un único símbolo (Bitcoin).
    
    Su responsabilidad es orquestar la simulación, delegando la lógica de trading
    al Portfolio y la de recompensas a una RewardStrategy.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        price_scaler: MinMaxScaler,
        env_config: Union["EnvironmentConfig", Dict[str, Any]],
        portfolio: BasePortfolio, # Inyección de dependencia
        observation_builder: "ObservationBuilder", # Inyección de dependencia para construcción de observaciones
        reward_strategy: Optional[BaseRewardStrategy] = None
    ):
        """
        Inicializa el entorno de trading.
        
        Args:
            data_df: DataFrame con datos OHLCV + indicadores normalizados [0,1]
            price_scaler: Scaler ajustado para normalizar precios
            env_config: Configuración del entorno (objeto Pydantic o diccionario)
            portfolio: Una instancia que cumple con la interfaz BasePortfolio.
            observation_builder: Constructor de observaciones centralizado.
            reward_strategy: Estrategia para el cálculo de recompensas.
        """
        super().__init__()
        
        self.price_scaler = price_scaler
        self.config_entorno = env_config
        self.data_array = data_df.to_numpy(dtype=np.float32)
        self.column_names = data_df.columns.tolist()
        self.observation_builder = observation_builder  # Inyección de dependencia
        
        if 'Close' not in self.column_names:
            raise ValueError("Columna 'Close' no encontrada en los datos")
        
        close_index = self.column_names.index('Close')
        close_data_normalized = self.data_array[:, close_index].reshape(-1, 1)
        self.original_prices = self.price_scaler.inverse_transform(close_data_normalized).ravel()
        
        # Acceso compatible con Pydantic y diccionarios
        ventana_observacion_size = self._get_config_value('ventana_observacion_size')
        
        if len(self.data_array) < ventana_observacion_size:
            raise ValueError(
                f"Dataset debe tener al menos {ventana_observacion_size} filas"
            )
        
        self.portfolio = portfolio
        
        # Dynamic reward strategy selection based on configuration
        strategy_name = self._get_config_value('reward_strategy', 'EquityChange')
        if reward_strategy is not None:
            self.reward_strategy = reward_strategy
        elif strategy_name == 'Sortino':
            self.reward_strategy = SortinoRewardStrategy(env_config)
        else:
            self.reward_strategy = EquityChangeRewardStrategy(env_config)
            
        self._setup_spaces()
        
        self.paso_actual = 0
        self.pasos_totales_episodio = 0

        logger.info(f"Entorno inicializado con {len(self.data_array)} filas de datos")

    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Helper para acceder a valores de configuración independientemente del tipo."""
        if hasattr(self.config_entorno, key):
            return getattr(self.config_entorno, key)
        elif isinstance(self.config_entorno, dict):
            return self.config_entorno.get(key, default)
        else:
            return default

    def _setup_spaces(self):
        """Configura los espacios de acción y observación."""
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Acceso compatible con Pydantic y diccionarios
        ventana_size = self._get_config_value('ventana_observacion_size')
        
        num_features_mercado = len(self.column_names)
        
        # Para portfolio_features, usar valor por defecto si no está configurado
        num_features_portfolio = 4
            
        total_features = ventana_size * num_features_mercado + num_features_portfolio
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(total_features,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reinicia el entorno para un nuevo episodio."""
        super().reset(seed=seed)
        
        self._select_start_point(options)
        self.portfolio.reset()
        self.pasos_totales_episodio = 0
        
        observation = self._get_current_observation()
        info = self._get_step_info(0.0, "RESET", False, 0.0)
        
        logger.info(f"Episodio reiniciado en paso {self.paso_actual}")
        return observation, info

    def _select_start_point(self, options: Optional[Dict[str, Any]]):
        """Selecciona el punto de inicio para el episodio."""
        ventana_size = self._get_config_value('ventana_observacion_size')
        max_start = len(self.data_array) - ventana_size - 1
        
        if options and 'start_index' in options:
            self.paso_actual = max(ventana_size, min(options['start_index'], max_start))
        else:
            self.paso_actual = self.np_random.integers(ventana_size, max_start + 1)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Ejecuta un paso en el entorno."""
        action_raw = float(np.clip(action[0], -1.0, 1.0))
        equity_anterior = self.portfolio.equity

        intencion, magnitud_efectiva = self._interpret_action(action_raw)
        
        trade_ejecutado, pnl_realizado = self.portfolio.execute_order(
            intencion, magnitud_efectiva, self._get_current_price()
        )
        
        self.portfolio.update_state(self._get_current_price())
        self.portfolio.historial_equity.append(self.portfolio.equity)
        self.portfolio.advance_step()

        self.paso_actual += 1
        self.pasos_totales_episodio += 1

        reward = self.reward_strategy.calculate_reward(self.portfolio, trade_ejecutado, pnl_realizado, equity_anterior)
        
        terminated, truncated = self._check_episode_termination()
        observation = self._get_current_observation()
        info = self._get_step_info(action_raw, intencion, trade_ejecutado, pnl_realizado)

        return observation, reward, terminated, truncated, info

    def _interpret_action(self, action_raw: float) -> Tuple[str, float]:
        """Interpreta la acción cruda del agente."""
        zona_muerta = self._get_config_value('zona_muerta_mantener')
        if -zona_muerta <= action_raw <= zona_muerta:
            return "MANTENER", 0.0
        elif action_raw > zona_muerta:
            return "COMPRAR", (action_raw - zona_muerta) / (1.0 - zona_muerta)
        else:
            return "VENDER", (abs(action_raw) - zona_muerta) / (1.0 - zona_muerta)

    def _check_episode_termination(self) -> Tuple[bool, bool]:
        """Verifica las condiciones de finalización del episodio."""
        capital_inicial = self._get_config_value('capital_inicial')
        max_drawdown = self._get_config_value('max_drawdown_configurado_cuenta')
        drawdown_threshold = capital_inicial * (1 - max_drawdown)
        terminated = (self.portfolio.equity <= drawdown_threshold) or self.portfolio.is_max_consecutive_losses_reached

        end_of_data = self.paso_actual >= len(self.data_array) - 1
        usar_max_pasos = self._get_config_value('usar_max_pasos_episodio')
        max_pasos = self._get_config_value('max_pasos_episodio')
        max_steps_reached = usar_max_pasos and self.pasos_totales_episodio >= max_pasos
        truncated = end_of_data or max_steps_reached

        return terminated, truncated

    def _get_current_observation(self) -> np.ndarray:
        """Construye la observación actual delegando al ObservationBuilder."""
        # 1. Obtener la ventana de datos de mercado
        ventana_size = self._get_config_value('ventana_observacion_size')
        start_idx = max(0, self.paso_actual - ventana_size + 1)
        end_idx = self.paso_actual + 1
        market_data_window = self.data_array[start_idx:end_idx]

        # 2. Aplicar padding si es necesario
        if market_data_window.shape[0] < ventana_size:
            padding = np.repeat(self.data_array[0:1], ventana_size - market_data_window.shape[0], axis=0)
            market_data_window = np.vstack([padding, market_data_window])

        # 3. Convertir la ventana de datos a DataFrame para el ObservationBuilder
        market_df = pd.DataFrame(market_data_window, columns=self.column_names)

        # 4. Obtener el estado actual del portafolio
        portfolio_state = self.portfolio.get_current_state()

        # 5. Delegar la construcción de la observación al ObservationBuilder
        return self.observation_builder.build(market_df, portfolio_state)

    def _get_current_price(self) -> float:
        """Obtiene el precio Close actual (desnormalizado)."""
        return self.original_prices[self.paso_actual]

    def _get_step_info(self, action_raw: float, intencion: str, trade_ejecutado: bool, pnl_realizado: float) -> Dict[str, Any]:
        """Prepara el diccionario de información para el paso actual."""
        posicion = self.portfolio.posicion_actual
        return {
            'step': self.paso_actual,
            'action_raw': action_raw,
            'intencion': intencion,
            'trade_ejecutado': trade_ejecutado,
            'pnl_realizado': pnl_realizado,
            'balance': self.portfolio.balance,
            'equity': self.portfolio.equity,
            'max_equity_episodio': self.portfolio.max_equity_alcanzado_episodio,
            'posicion_tipo': posicion['tipo'].name,
            'posicion_pnl_roe': posicion['pnl_no_realizado_roe'],
            'posicion_pasos': posicion['pasos_en_posicion'],
            'precio_actual': self._get_current_price(),
            'num_trades_episodio': len(self.portfolio.historial_trades),
            'pasos_totales_episodio': self.pasos_totales_episodio,
            'consecutive_losses_reached': self.portfolio.is_max_consecutive_losses_reached
        }

    def render(self, mode: str = "human") -> Optional[str]:
        """Renderiza el estado actual del entorno."""
        if mode == "human":
            info = self.portfolio.render_state(self._get_current_price(), self.paso_actual, self.pasos_totales_episodio)
            print(info)
            return info
        return None

    def close(self):
        """Cierra el entorno y libera recursos."""
        logger.info("Entorno cerrado")

    def get_equity_series(self) -> pd.Series:
        """Obtiene la serie temporal del equity."""
        return pd.Series(self.portfolio.historial_equity)
