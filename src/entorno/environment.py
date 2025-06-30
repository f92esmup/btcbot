"""
Entorno de Trading de Futuros para Reinforcement Learning.

Este módulo implementa un entorno de trading de futuros que hereda de gymnasium.Env,
diseñado para simular trading de Bitcoin con apalancamiento, comisiones y slippage.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional, Union
from enum import Enum
import logging
from sklearn.preprocessing import MinMaxScaler

# Configurar logging
logger = logging.getLogger(__name__)


class TipoOperacion(Enum):
    """Tipos de operación/posición."""
    NEUTRAL = 0
    LARGO = 1
    CORTO = -1


class FuturesTradingEnv(gym.Env):
    """
    Entorno de trading de futuros para un único símbolo (Bitcoin).
    
    Simula trading de futuros con:
    - Apalancamiento configurable
    - Comisiones y slippage realistas
    - Una sola operación a la vez
    - Recompensas basadas en rendimiento y gestión de riesgo
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        price_scaler: MinMaxScaler,
        env_config: Dict[str, Any]
    ):
        """
        Inicializa el entorno de trading.
        
        Args:
            data_df: DataFrame con datos OHLCV + indicadores normalizados [0,1]
            price_scaler: Scaler ajustado para normalizar precios
            env_config: Diccionario con la configuración específica para el entorno.
        """
        super().__init__()
        
        # Guardar datos y configuración
        self.data_df = data_df.copy()
        self.price_scaler = price_scaler
        self.config_entorno = env_config
        
        # Convertir DataFrame a array de NumPy para acceso más rápido
        self.data_array = self.data_df.to_numpy(dtype=np.float32)
        
        # Guardar nombres de columnas para compatibilidad
        self.column_names = self.data_df.columns.tolist()
        
        # Pre-calcular precios originales (desnormalizados) para optimizar _get_current_price
        if 'Close' not in self.column_names:
            raise ValueError("Columna 'Close' no encontrada en los datos")
        
        close_index = self.column_names.index('Close')
        close_data_normalized = self.data_array[:, close_index].reshape(-1, 1)
        self.original_prices = self.price_scaler.inverse_transform(close_data_normalized).ravel()
        
        # Validar datos mínimos
        if len(self.data_array) < self.config_entorno['ventana_observacion_size']:
            raise ValueError(
                f"Dataset debe tener al menos {self.config_entorno['ventana_observacion_size']} filas"
            )
        
        # Liberar memoria del DataFrame después de la conversión
        del self.data_df
        
        # Configurar espacios de acción y observación
        self._setup_spaces()
        
        # Inicializar estado interno
        self._initialize_state()
        
        logger.info(f"Entorno inicializado con {len(self.data_array)} filas de datos")
    
    def _setup_spaces(self):
        """Configura los espacios de acción y observación."""
        # Espacio de acción: continuo [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Espacio de observación: ventana de mercado + características del portafolio
        ventana_size = self.config_entorno['ventana_observacion_size']
        num_features_mercado = len(self.column_names)
        num_features_portfolio = 4  # tipo_posicion, pnl_roe, pasos_posicion, precio_entrada
        
        total_features = ventana_size * num_features_mercado + num_features_portfolio
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_features,),
            dtype=np.float32
        )
        
        logger.info(f"Espacio observación: {total_features} features total")
        logger.info(f"  - Mercado: {ventana_size} x {num_features_mercado} = {ventana_size * num_features_mercado}")
        logger.info(f"  - Portfolio: {num_features_portfolio}")
    
    def _initialize_state(self):
        """Inicializa el estado interno del entorno."""
        # Estado del portafolio
        self.balance_actual = self.config_entorno['capital_inicial']
        self.equity_actual = self.config_entorno['capital_inicial']
        self.max_equity_alcanzado_episodio = self.config_entorno['capital_inicial']
        
        # Estado de la posición
        self.posicion_actual = {
            'tipo': TipoOperacion.NEUTRAL,
            'precio_entrada': 0.0,
            'tamaño_activo': 0.0,  # Cantidad del activo (BTC)
            'valor_nocional': 0.0,  # Valor total de la posición
            'margen_usado': 0.0,    # Margen/colateral usado
            'pnl_no_realizado_abs': 0.0,  # P&L absoluto no realizado
            'pnl_no_realizado_roe': 0.0,  # P&L ROE (Return on Equity/Margin)
            'pasos_en_posicion': 0
        }
        
        # Estado del episodio
        self.paso_actual = 0
        self.historial_trades = []
        self.historial_equity = []
        self.pasos_totales_episodio = 0
        
        # Para recompensas
        self.equity_anterior = self.config_entorno['capital_inicial']
        self.retornos_realizados_episodio = []
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reinicia el entorno para un nuevo episodio.
        
        Args:
            seed: Semilla para reproducibilidad
            options: Opciones adicionales
            
        Returns:
            Tupla con (observación_inicial, info)
        """
        super().reset(seed=seed)
        
        # Seleccionar punto de inicio
        self._select_start_point(options)
        
        # Reinicializar estado
        self._initialize_state()
        
        # Construir observación inicial
        observation = self._get_current_observation()
        
        info = {
            'step': self.paso_actual,
            'balance': self.balance_actual,
            'equity': self.equity_actual,
            'posicion_tipo': self.posicion_actual['tipo'].name,
            'precio_actual': self._get_current_price()
        }
        
        logger.info(f"Episodio reiniciado en paso {self.paso_actual}")
        
        return observation, info
    
    def _select_start_point(self, options: Optional[Dict[str, Any]]):
        """
        Selecciona el punto de inicio para el episodio.
        
        Args:
            options: Puede contener 'start_index' para especificar inicio
        """
        ventana_size = self.config_entorno['ventana_observacion_size']
        max_start = len(self.data_array) - ventana_size - 1
        
        if options and 'start_index' in options:
            self.paso_actual = max(ventana_size, min(options['start_index'], max_start))
        else:
            # Inicio aleatorio, asegurando suficientes datos para la ventana
            self.paso_actual = self.np_random.integers(ventana_size, max_start + 1)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Ejecuta un paso en el entorno.
        
        Args:
            action: Acción del agente (array de tamaño 1 con valor en [-1, 1])
            
        Returns:
            Tupla con (observación, recompensa, terminated, truncated, info)
        """
        if len(action) != 1:
            raise ValueError(f"Acción debe ser array de tamaño 1, recibido: {len(action)}")
        
        action_raw = float(action[0])
        action_raw = np.clip(action_raw, -1.0, 1.0)  # Asegurar rango válido
        
        # Guardar estado anterior para cálculo de recompensas
        equity_anterior = self.equity_actual
        
        # 1. Interpretar acción
        intencion, magnitud_efectiva = self._interpret_action(action_raw)
        
        # 2. Ejecutar lógica de trading
        trade_ejecutado, pnl_realizado = self._execute_trade(intencion, magnitud_efectiva)
        
        # 3. Actualizar estado del portafolio
        self._update_equity_and_pnl()
        
        # 4. Avanzar al siguiente paso
        self.paso_actual += 1
        self.pasos_totales_episodio += 1
        
        # Actualizar pasos en posición
        if self.posicion_actual['tipo'] != TipoOperacion.NEUTRAL:
            self.posicion_actual['pasos_en_posicion'] += 1
        
        # 5. Calcular recompensa
        reward = self._calculate_reward(trade_ejecutado, pnl_realizado, equity_anterior)
        
        # 6. Verificar condiciones de finalización
        terminated, truncated = self._check_episode_termination()
        
        # 7. Construir nueva observación
        observation = self._get_current_observation()
        
        # 8. Preparar info
        info = self._get_step_info(action_raw, intencion, trade_ejecutado, pnl_realizado)
        
        # Actualizar historiales
        self.historial_equity.append(self.equity_actual)
        self.equity_anterior = self.equity_actual
        
        return observation, reward, terminated, truncated, info
    
    def _interpret_action(self, action_raw: float) -> Tuple[str, float]:
        """
        Interpreta la acción cruda del agente.
        
        Args:
            action_raw: Valor de acción en [-1, 1]
            
        Returns:
            Tupla con (intención, magnitud_efectiva)
        """
        zona_muerta = self.config_entorno['zona_muerta_mantener']
        
        if -zona_muerta <= action_raw <= zona_muerta:
            return "MANTENER", 0.0
        elif action_raw > zona_muerta:
            magnitud = (action_raw - zona_muerta) / (1.0 - zona_muerta)
            return "COMPRAR", magnitud
        else:  # action_raw < -zona_muerta
            magnitud = (abs(action_raw) - zona_muerta) / (1.0 - zona_muerta)
            return "VENDER", magnitud
    
    def _execute_trade(self, intencion: str, magnitud_efectiva: float) -> Tuple[bool, float]:
        """
        Ejecuta la lógica de trading basada en la intención.
        
        Args:
            intencion: "COMPRAR", "VENDER", o "MANTENER"
            magnitud_efectiva: Magnitud normalizada [0, 1]
            
        Returns:
            Tupla con (trade_ejecutado, pnl_realizado)
        """
        posicion_actual_tipo = self.posicion_actual['tipo']
        precio_mercado = self._get_current_price()
        
        # Caso 1: MANTENER o misma dirección que posición actual
        if intencion == "MANTENER":
            return False, 0.0
        
        # Determinar si es operación opuesta
        es_operacion_opuesta = (
            (intencion == "VENDER" and posicion_actual_tipo == TipoOperacion.LARGO) or
            (intencion == "COMPRAR" and posicion_actual_tipo == TipoOperacion.CORTO)
        )
        
        # Caso 2: Operación opuesta -> Cerrar posición actual
        if es_operacion_opuesta:
            pnl_realizado = self._close_position(precio_mercado)
            return True, pnl_realizado
        
        # Caso 3: Abrir nueva posición (agente está NEUTRAL)
        if posicion_actual_tipo == TipoOperacion.NEUTRAL and magnitud_efectiva > 0:
            tipo_operacion = TipoOperacion.LARGO if intencion == "COMPRAR" else TipoOperacion.CORTO
            self._open_position(tipo_operacion, precio_mercado, magnitud_efectiva)
            return True, 0.0
        
        # Caso 4: Misma dirección que posición actual -> No hacer nada
        return False, 0.0
    
    def _open_position(self, tipo_operacion: TipoOperacion, precio_mercado: float, magnitud: float):
        """
        Abre una nueva posición.
        
        Args:
            tipo_operacion: LARGO o CORTO
            precio_mercado: Precio actual del mercado
            magnitud: Magnitud normalizada de la operación [0, 1]
        """
        # Calcular tamaño de la posición
        margen_a_usar = self.balance_actual * self.config_entorno['porcentaje_max_inversion_por_trade'] * magnitud
        valor_nocional = margen_a_usar * self.config_entorno['apalancamiento']
        tamaño_activo = valor_nocional / precio_mercado
        
        # Aplicar costos (comisiones y slippage)
        precio_ejecucion, coste_total = self._apply_costs(valor_nocional, precio_mercado, tipo_operacion)
        
        # Actualizar balance (restar comisión y margen usado)
        self.balance_actual -= (coste_total + margen_a_usar)
        
        # Registrar posición
        self.posicion_actual = {
            'tipo': tipo_operacion,
            'precio_entrada': precio_ejecucion,
            'tamaño_activo': tamaño_activo,
            'valor_nocional': valor_nocional,
            'margen_usado': margen_a_usar,
            'pnl_no_realizado_abs': 0.0,
            'pnl_no_realizado_roe': 0.0,
            'pasos_en_posicion': 0
        }
        
        logger.debug(f"Posición {tipo_operacion.name} abierta: {tamaño_activo:.6f} BTC @ {precio_ejecucion:.2f}")
    
    def _close_position(self, precio_mercado: float) -> float:
        """
        Cierra la posición actual.
        
        Args:
            precio_mercado: Precio actual del mercado
            
        Returns:
            PNL realizado absoluto
        """
        if self.posicion_actual['tipo'] == TipoOperacion.NEUTRAL:
            return 0.0
        
        # Aplicar costos de cierre
        precio_ejecucion, coste_cierre = self._apply_costs(
            self.posicion_actual['valor_nocional'],
            precio_mercado,
            self.posicion_actual['tipo']
        )
        
        # Calcular PNL realizado
        if self.posicion_actual['tipo'] == TipoOperacion.LARGO:
            pnl_bruto = (precio_ejecucion - self.posicion_actual['precio_entrada']) * self.posicion_actual['tamaño_activo']
        else:  # CORTO
            pnl_bruto = (self.posicion_actual['precio_entrada'] - precio_ejecucion) * self.posicion_actual['tamaño_activo']
        
        pnl_neto = pnl_bruto - coste_cierre
        
        # Actualizar balance (devolver margen + PNL neto)
        self.balance_actual += (self.posicion_actual['margen_usado'] + pnl_neto)
        
        # Guardar para historial y recompensas
        roe_operacion = pnl_neto / self.posicion_actual['margen_usado']
        self.retornos_realizados_episodio.append(roe_operacion)
        
        # Registrar trade en historial
        self.historial_trades.append({
            'tipo': self.posicion_actual['tipo'].name,
            'precio_entrada': self.posicion_actual['precio_entrada'],
            'precio_salida': precio_ejecucion,
            'tamaño_activo': self.posicion_actual['tamaño_activo'],
            'margen_usado': self.posicion_actual['margen_usado'],
            'pnl_abs': pnl_neto,
            'roe': roe_operacion,
            'pasos_duracion': self.posicion_actual['pasos_en_posicion'],
            'paso_cierre': self.paso_actual
        })
        
        logger.debug(f"Posición cerrada: PNL = {pnl_neto:.2f}, ROE = {roe_operacion:.4f}")
        
        # Resetear posición
        self.posicion_actual = {
            'tipo': TipoOperacion.NEUTRAL,
            'precio_entrada': 0.0,
            'tamaño_activo': 0.0,
            'valor_nocional': 0.0,
            'margen_usado': 0.0,
            'pnl_no_realizado_abs': 0.0,
            'pnl_no_realizado_roe': 0.0,
            'pasos_en_posicion': 0
        }
        
        return pnl_neto
    
    def _apply_costs(self, valor_nocional: float, precio_mercado: float, tipo_operacion: TipoOperacion) -> Tuple[float, float]:
        """
        Aplica comisiones y slippage.
        
        Args:
            valor_nocional: Valor nocional de la operación
            precio_mercado: Precio de mercado actual
            tipo_operacion: LARGO o CORTO
            
        Returns:
            Tupla con (precio_ejecucion_final, coste_total_absoluto)
        """
        # Comisión (sobre valor nocional)
        comision_abs = valor_nocional * self.config_entorno['comision_taker_porcentaje']
        
        # Slippage (afecta al precio de ejecución)
        slippage_factor = self.config_entorno['slippage_porcentaje']
        
        if tipo_operacion == TipoOperacion.LARGO:
            # Comprar: precio sube por slippage
            precio_ejecucion = precio_mercado * (1 + slippage_factor)
        else:
            # Vender: precio baja por slippage
            precio_ejecucion = precio_mercado * (1 - slippage_factor)
        
        coste_total = comision_abs
        
        return precio_ejecucion, coste_total
    
    def _update_equity_and_pnl(self):
        """Actualiza el equity y PNL no realizado de la posición actual."""
        if self.posicion_actual['tipo'] == TipoOperacion.NEUTRAL:
            self.equity_actual = self.balance_actual
            return
        
        precio_actual = self._get_current_price()
        
        # Calcular PNL no realizado
        if self.posicion_actual['tipo'] == TipoOperacion.LARGO:
            pnl_no_realizado = (precio_actual - self.posicion_actual['precio_entrada']) * self.posicion_actual['tamaño_activo']
        else:  # CORTO
            pnl_no_realizado = (self.posicion_actual['precio_entrada'] - precio_actual) * self.posicion_actual['tamaño_activo']
        
        # ROE (Return on Equity/Margin)
        pnl_roe = pnl_no_realizado / self.posicion_actual['margen_usado'] if self.posicion_actual['margen_usado'] > 0 else 0.0
        
        # Actualizar posición
        self.posicion_actual['pnl_no_realizado_abs'] = pnl_no_realizado
        self.posicion_actual['pnl_no_realizado_roe'] = pnl_roe
        
        # Actualizar equity
        self.equity_actual = self.balance_actual + pnl_no_realizado
        
        # Actualizar máximo equity del episodio
        self.max_equity_alcanzado_episodio = max(self.max_equity_alcanzado_episodio, self.equity_actual)
    
    def _calculate_reward(self, trade_ejecutado: bool, pnl_realizado: float, equity_anterior: float) -> float:
        """
        Calcula la recompensa híbrida.
        
        Args:
            trade_ejecutado: Si se ejecutó un trade en este paso
            pnl_realizado: PNL realizado si se cerró una posición
            equity_anterior: Equity del paso anterior
            
        Returns:
            Recompensa total
        """
        # TODO: Aquí se integrará con el "cerebro" del agente cuando esté definido
        # Por ahora implementamos la lógica de recompensa básica
        
        # Componente por paso: cambio en equity
        if equity_anterior > 0:
            recompensa_paso = (self.equity_actual - equity_anterior) / equity_anterior
        else:
            recompensa_paso = 0.0
        
        recompensa_paso *= self.config_entorno['peso_recompensa_paso']
        
        # Componente por cierre de operación
        recompensa_cierre = 0.0
        if trade_ejecutado and pnl_realizado != 0.0:
            # Encontrar la operación que se acaba de cerrar
            if len(self.historial_trades) > 0:
                ultimo_trade = self.historial_trades[-1]
                roe_operacion = ultimo_trade['roe']
                
                if self.config_entorno['usar_log1p_en_pnl']:
                    recompensa_cierre = np.sign(roe_operacion) * np.log1p(abs(roe_operacion))
                else:
                    recompensa_cierre = roe_operacion
                
                recompensa_cierre *= self.config_entorno['peso_recompensa_cierre']
        
        # Recompensa total
        reward_total = recompensa_paso + recompensa_cierre
        
        return float(reward_total)
    
    def _check_episode_termination(self) -> Tuple[bool, bool]:
        """
        Verifica las condiciones de finalización del episodio.
        
        Returns:
            Tupla con (terminated, truncated)
        """
        terminated = False
        truncated = False
        
        # Terminated: drawdown excesivo
        drawdown_threshold = self.config_entorno['capital_inicial'] * (1 - self.config_entorno['max_drawdown_configurado_cuenta'])
        if self.equity_actual <= drawdown_threshold:
            terminated = True
            logger.info(f"Episodio terminado por drawdown: equity={self.equity_actual:.2f} <= threshold={drawdown_threshold:.2f}")
        
        # Truncated: fin de datos
        if self.paso_actual >= len(self.data_array) - 1:
            truncated = True
            logger.info("Episodio truncado: fin de datos alcanzado")
        
        # Truncated: máximo pasos por episodio (si está habilitado)
        if (self.config_entorno['usar_max_pasos_episodio'] and 
            self.pasos_totales_episodio >= self.config_entorno['max_pasos_episodio']):
            truncated = True
            logger.info(f"Episodio truncado: máximo pasos alcanzado ({self.pasos_totales_episodio})")
        
        return terminated, truncated
    
    def _get_current_observation(self) -> np.ndarray:
        """
        Construye la observación actual.
        
        Returns:
            Vector de observación normalizado
        """
        # 1. Ventana de datos de mercado
        ventana_size = self.config_entorno['ventana_observacion_size']
        start_idx = max(0, self.paso_actual - ventana_size + 1)
        end_idx = self.paso_actual + 1
        
        market_data = self.data_array[start_idx:end_idx]
        
        # Si no hay suficientes datos históricos, rellenar con la primera fila disponible
        if market_data.shape[0] < ventana_size:
            padding_needed = ventana_size - market_data.shape[0]
            primera_fila = self.data_array[0:1]
            padding = np.repeat(primera_fila, padding_needed, axis=0)
            market_data = np.vstack([padding, market_data])
        
        ventana_flat = market_data.ravel()
        
        # 2. Características del portafolio normalizadas
        portfolio_features = self._get_normalized_portfolio_features()
        
        # 3. Combinar todo
        observation = np.concatenate([ventana_flat, portfolio_features])
        
        return observation.astype(np.float32)
    
    def _get_normalized_portfolio_features(self) -> np.ndarray:
        """
        Obtiene las características del portafolio normalizadas.
        
        Returns:
            Array con 4 características normalizadas a [0, 1]
        """
        # 1. Tipo de posición normalizado
        if self.posicion_actual['tipo'] == TipoOperacion.LARGO:
            tipo_posicion_norm = 1.0
        elif self.posicion_actual['tipo'] == TipoOperacion.NEUTRAL:
            tipo_posicion_norm = 0.5
        else:  # CORTO
            tipo_posicion_norm = 0.0
        
        # 2. PNL ROE normalizado y clipeado
        pnl_roe = self.posicion_actual['pnl_no_realizado_roe']
        pnl_roe_clipped = np.clip(
            pnl_roe,
            self.config_entorno['min_clip_pnl_roe'],
            self.config_entorno['max_clip_pnl_roe']
        )
        
        # Normalizar a [0, 1]
        min_roe = self.config_entorno['min_clip_pnl_roe']
        max_roe = self.config_entorno['max_clip_pnl_roe']
        if max_roe != min_roe:
            pnl_roe_norm = (pnl_roe_clipped - min_roe) / (max_roe - min_roe)
        else:
            pnl_roe_norm = 0.5
        
        # 3. Pasos en posición normalizado
        pasos_norm = min(1.0, self.posicion_actual['pasos_en_posicion'] / self.config_entorno['max_pasos_en_posicion'])
        
        # 4. Precio de entrada normalizado
        if self.posicion_actual['tipo'] != TipoOperacion.NEUTRAL:
            # Usar el price_scaler para normalizar el precio de entrada
            precio_entrada_scaled = self.price_scaler.transform([[self.posicion_actual['precio_entrada']]])[0][0]
            precio_entrada_norm = np.clip(precio_entrada_scaled, 0.0, 1.0)
        else:
            precio_entrada_norm = 0.5  # Valor neutral
        
        return np.array([tipo_posicion_norm, pnl_roe_norm, pasos_norm, precio_entrada_norm], dtype=np.float32)
    
    def _get_current_price(self) -> float:
        """
        Obtiene el precio actual del mercado.
        
        Returns:
            Precio Close actual (desnormalizado)
        """
        return self.original_prices[self.paso_actual]
    
    def _get_step_info(self, action_raw: float, intencion: str, trade_ejecutado: bool, pnl_realizado: float) -> Dict[str, Any]:
        """
        Prepara el diccionario de información para el paso actual.
        
        Args:
            action_raw: Acción cruda del agente
            intencion: Intención interpretada
            trade_ejecutado: Si se ejecutó un trade
            pnl_realizado: PNL realizado si se cerró posición
            
        Returns:
            Diccionario con información del estado actual
        """
        return {
            'step': self.paso_actual,
            'action_raw': action_raw,
            'intencion': intencion,
            'trade_ejecutado': trade_ejecutado,
            'pnl_realizado': pnl_realizado,
            'balance': self.balance_actual,
            'equity': self.equity_actual,
            'max_equity_episodio': self.max_equity_alcanzado_episodio,
            'posicion_tipo': self.posicion_actual['tipo'].name,
            'posicion_pnl_roe': self.posicion_actual['pnl_no_realizado_roe'],
            'posicion_pasos': self.posicion_actual['pasos_en_posicion'],
            'precio_actual': self._get_current_price(),
            'num_trades_episodio': len(self.historial_trades),
            'pasos_totales_episodio': self.pasos_totales_episodio
        }
    
    def render(self, mode: str = "human") -> Optional[str]:
        """
        Renderiza el estado actual del entorno.
        
        Args:
            mode: Modo de renderizado ("human" o "rgb_array")
            
        Returns:
            String con información o None
        """
        if mode == "human":
            precio_actual = self._get_current_price()
            drawdown = (self.max_equity_alcanzado_episodio - self.equity_actual) / self.max_equity_alcanzado_episodio * 100
            
            info = f"""
=== TRADING ENVIRONMENT STATE ===
Paso: {self.paso_actual} | Episodio: {self.pasos_totales_episodio}
Precio BTC: ${precio_actual:.2f}
Balance: ${self.balance_actual:.2f}
Equity: ${self.equity_actual:.2f}
Max Equity: ${self.max_equity_alcanzado_episodio:.2f}
Drawdown: {drawdown:.2f}%

Posición: {self.posicion_actual['tipo'].name}
PNL No Realizado: ${self.posicion_actual['pnl_no_realizado_abs']:.2f} ({self.posicion_actual['pnl_no_realizado_roe']:.2%} ROE)
Pasos en Posición: {self.posicion_actual['pasos_en_posicion']}

Trades Realizados: {len(self.historial_trades)}
Retornos del Episodio: {len(self.retornos_realizados_episodio)}
=================================
"""
            print(info)
            return info
        
        return None
    
    def close(self):
        """Cierra el entorno y libera recursos."""
        # Por ahora no hay recursos específicos que liberar
        logger.info("Entorno cerrado")
    
    def get_equity_series(self) -> pd.Series:
        """
        Obtiene la serie temporal del equity.
        
        Returns:
            Serie con la evolución del equity
        """
        return pd.Series(self.historial_equity)
