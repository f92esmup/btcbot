"""
Entorno de Trading de Futuros para Reinforcement Learning.

Este módulo implementa un entorno de trading de futuros que hereda de gymnasium.Env,
diseñado para simular trading de Bitcoin con apalancamiento, comisiones y slippage.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import logging
from sklearn.preprocessing import MinMaxScaler

from .portfolio import Portfolio
from .base_portfolio import BasePortfolio, TipoOperacion
from .reward_strategy import BaseRewardStrategy, EquityChangeRewardStrategy

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
        env_config: Dict[str, Any],
        portfolio: BasePortfolio, # Inyección de dependencia
        reward_strategy: Optional[BaseRewardStrategy] = None
    ):
        """
        Inicializa el entorno de trading.
        
        Args:
            data_df: DataFrame con datos OHLCV + indicadores normalizados [0,1]
            price_scaler: Scaler ajustado para normalizar precios
            env_config: Diccionario con la configuración específica para el entorno.
            portfolio: Una instancia que cumple con la interfaz BasePortfolio.
            reward_strategy: Estrategia para el cálculo de recompensas.
        """
        super().__init__()
        
        self.price_scaler = price_scaler
        self.config_entorno = env_config
        self.data_array = data_df.to_numpy(dtype=np.float32)
        self.column_names = data_df.columns.tolist()
        
        if 'Close' not in self.column_names:
            raise ValueError("Columna 'Close' no encontrada en los datos")
        
        close_index = self.column_names.index('Close')
        close_data_normalized = self.data_array[:, close_index].reshape(-1, 1)
        self.original_prices = self.price_scaler.inverse_transform(close_data_normalized).ravel()
        
        if len(self.data_array) < self.config_entorno['ventana_observacion_size']:
            raise ValueError(
                f"Dataset debe tener al menos {self.config_entorno['ventana_observacion_size']} filas"
            )
        
        self.portfolio = portfolio
        self.reward_strategy = reward_strategy or EquityChangeRewardStrategy(env_config)
        self._setup_spaces()
        
        self.paso_actual = 0
        self.pasos_totales_episodio = 0
        self.historial_equity = []

        logger.info(f"Entorno inicializado con {len(self.data_array)} filas de datos")

    def _setup_spaces(self):
        """Configura los espacios de acción y observación."""
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        ventana_size = self.config_entorno['ventana_observacion_size']
        num_features_mercado = len(self.column_names)
        num_features_portfolio = self.config_entorno.get('architecture', {}).get('portfolio_features', 4)
        total_features = ventana_size * num_features_mercado + num_features_portfolio
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(total_features,), dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reinicia el entorno para un nuevo episodio."""
        super().reset(seed=seed)
        
        self._select_start_point(options)
        self.portfolio.reset()
        self.pasos_totales_episodio = 0
        self.historial_equity = [self.portfolio.equity]
        
        observation = self._get_current_observation()
        info = self._get_step_info(0.0, "RESET", False, 0.0)
        
        logger.info(f"Episodio reiniciado en paso {self.paso_actual}")
        return observation, info

    def _select_start_point(self, options: Optional[Dict[str, Any]]):
        """Selecciona el punto de inicio para el episodio."""
        ventana_size = self.config_entorno['ventana_observacion_size']
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
        self.portfolio.advance_step()

        self.paso_actual += 1
        self.pasos_totales_episodio += 1

        reward = self.reward_strategy.calculate_reward(self.portfolio, trade_ejecutado, pnl_realizado, equity_anterior)
        
        terminated, truncated = self._check_episode_termination()
        observation = self._get_current_observation()
        info = self._get_step_info(action_raw, intencion, trade_ejecutado, pnl_realizado)

        self.historial_equity.append(self.portfolio.equity)
        
        return observation, reward, terminated, truncated, info

    def _interpret_action(self, action_raw: float) -> Tuple[str, float]:
        """Interpreta la acción cruda del agente."""
        zona_muerta = self.config_entorno['zona_muerta_mantener']
        if -zona_muerta <= action_raw <= zona_muerta:
            return "MANTENER", 0.0
        elif action_raw > zona_muerta:
            return "COMPRAR", (action_raw - zona_muerta) / (1.0 - zona_muerta)
        else:
            return "VENDER", (abs(action_raw) - zona_muerta) / (1.0 - zona_muerta)

    def _check_episode_termination(self) -> Tuple[bool, bool]:
        """Verifica las condiciones de finalización del episodio."""
        drawdown_threshold = self.config_entorno['capital_inicial'] * (1 - self.config_entorno['max_drawdown_configurado_cuenta'])
        terminated = self.portfolio.equity <= drawdown_threshold

        end_of_data = self.paso_actual >= len(self.data_array) - 1
        max_steps_reached = self.config_entorno['usar_max_pasos_episodio'] and self.pasos_totales_episodio >= self.config_entorno['max_pasos_episodio']
        truncated = end_of_data or max_steps_reached

        return terminated, truncated

    def _get_current_observation(self) -> np.ndarray:
        """Construye la observación actual."""
        ventana_size = self.config_entorno['ventana_observacion_size']
        start_idx = max(0, self.paso_actual - ventana_size + 1)
        end_idx = self.paso_actual + 1
        market_data = self.data_array[start_idx:end_idx]

        if market_data.shape[0] < ventana_size:
            padding = np.repeat(self.data_array[0:1], ventana_size - market_data.shape[0], axis=0)
            market_data = np.vstack([padding, market_data])

        portfolio_features = self._get_normalized_portfolio_features()
        return np.concatenate([market_data.ravel(), portfolio_features]).astype(np.float32)

    def _get_normalized_portfolio_features(self) -> np.ndarray:
        """Obtiene las características del portafolio normalizadas."""
        posicion_actual = self.portfolio.posicion_actual
        tipo_posicion = posicion_actual['tipo']
        
        if tipo_posicion == TipoOperacion.LARGO:
            tipo_posicion_norm = 1.0
        elif tipo_posicion == TipoOperacion.NEUTRAL:
            tipo_posicion_norm = 0.5
        else:  # CORTO
            tipo_posicion_norm = 0.0

        min_roe = self.config_entorno['min_clip_pnl_roe']
        max_roe = self.config_entorno['max_clip_pnl_roe']
        pnl_roe_clipped = np.clip(posicion_actual['pnl_no_realizado_roe'], min_roe, max_roe)
        pnl_roe_norm = (pnl_roe_clipped - min_roe) / (max_roe - min_roe) if max_roe > min_roe else 0.5

        pasos_norm = min(1.0, posicion_actual['pasos_en_posicion'] / self.config_entorno['max_pasos_en_posicion'])

        if tipo_posicion != TipoOperacion.NEUTRAL:
            precio_entrada_scaled = self.price_scaler.transform([[posicion_actual['precio_entrada']]])[0][0]
            precio_entrada_norm = np.clip(precio_entrada_scaled, 0.0, 1.0)
        else:
            precio_entrada_norm = 0.5

        return np.array([tipo_posicion_norm, pnl_roe_norm, pasos_norm, precio_entrada_norm], dtype=np.float32)

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
            'pasos_totales_episodio': self.pasos_totales_episodio
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
        return pd.Series(self.historial_equity)
