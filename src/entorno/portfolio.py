"""
Módulo que encapsula la lógica y el estado del portfolio de trading para simulación.
"""

import logging
import pandas as pd
from typing import Dict, Any, Tuple, List
from .base_portfolio import BasePortfolio, TipoOperacion

logger = logging.getLogger(__name__)

class Portfolio(BasePortfolio):
    """
    Gestiona el estado financiero y las operaciones de un portfolio de trading en modo simulación.

    Esta clase es responsable de:
    - Mantener el estado del balance, equity y la posición actual en memoria.
    - Ejecutar la lógica para abrir y cerrar posiciones.
    - Calcular costos de transacción (comisiones y slippage).
    - Actualizar el P&L no realizado.
    - Mantener un historial de trades ejecutados.
    """
    def __init__(self, env_config: Dict[str, Any]):
        """
        Inicializa el portfolio de simulación.

        Args:
            env_config (Dict[str, Any]): Diccionario con la configuración del entorno,
                                         incluyendo capital_inicial, apalancamiento, etc.
        """
        self.config = env_config
        self._balance_actual = 0.0
        self._equity_actual = 0.0
        self._posicion_actual = {}
        self._historial_trades = []
        self.historial_equity = []
        self.max_equity_alcanzado_episodio = 0.0
        self.retornos_realizados_episodio = []
        self.max_consecutive_losses = self.config.max_consecutive_losses
        self._consecutive_losses = 0
        self.reset()

    def reset(self):
        """Reinicia el estado del portfolio a sus valores iniciales."""
        self._balance_actual = self.config.capital_inicial
        self._equity_actual = self.config.capital_inicial
        self.historial_equity = [self._equity_actual]
        self.max_equity_alcanzado_episodio = self.config.capital_inicial
        self._consecutive_losses = 0
        self._posicion_actual = {
            'tipo': TipoOperacion.NEUTRAL,
            'precio_entrada': 0.0,
            'tamaño_activo': 0.0,
            'valor_nocional': 0.0,
            'margen_usado': 0.0,
            'pnl_no_realizado_abs': 0.0,
            'pnl_no_realizado_roe': 0.0,
            'pasos_en_posicion': 0
        }
        self._historial_trades = []
        self.retornos_realizados_episodio = []

    def execute_order(self, intencion: str, magnitud: float, precio: float, paso_cierre: int = None) -> Tuple[bool, float]:
        posicion_actual_tipo = self._posicion_actual['tipo']

        if intencion == "MANTENER":
            return False, 0.0

        es_operacion_opuesta = (
            (intencion == "VENDER" and posicion_actual_tipo == TipoOperacion.LARGO) or
            (intencion == "COMPRAR" and posicion_actual_tipo == TipoOperacion.CORTO)
        )

        if es_operacion_opuesta:
            pnl_realizado = self._close_position(precio, paso_cierre)
            return True, pnl_realizado

        if posicion_actual_tipo == TipoOperacion.NEUTRAL and magnitud > 0:
            tipo_operacion = TipoOperacion.LARGO if intencion == "COMPRAR" else TipoOperacion.CORTO
            self._open_position(tipo_operacion, precio, magnitud)
            return True, 0.0

        return False, 0.0

    def _open_position(self, tipo_operacion: TipoOperacion, precio_mercado: float, magnitud: float):
        margen_a_usar = self._balance_actual * self.config.porcentaje_max_inversion_por_trade * magnitud
        valor_nocional = margen_a_usar * self.config.apalancamiento
        tamaño_activo = valor_nocional / precio_mercado

        precio_ejecucion, coste_total = self._apply_costs(valor_nocional, precio_mercado, tipo_operacion)

        self._balance_actual -= (coste_total + margen_a_usar)

        self._posicion_actual = {
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

    def _close_position(self, precio_mercado: float, paso_cierre: int = None) -> float:
        if self._posicion_actual['tipo'] == TipoOperacion.NEUTRAL:
            return 0.0

        precio_ejecucion, coste_cierre = self._apply_costs(
            self._posicion_actual['valor_nocional'],
            precio_mercado,
            self._posicion_actual['tipo']
        )

        if self._posicion_actual['tipo'] == TipoOperacion.LARGO:
            pnl_bruto = (precio_ejecucion - self._posicion_actual['precio_entrada']) * self._posicion_actual['tamaño_activo']
        else:  # CORTO
            pnl_bruto = (self._posicion_actual['precio_entrada'] - precio_ejecucion) * self._posicion_actual['tamaño_activo']

        pnl_neto = pnl_bruto - coste_cierre

        if pnl_neto > 0:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        self._balance_actual += (self._posicion_actual['margen_usado'] + pnl_neto)

        roe_operacion = pnl_neto / self._posicion_actual['margen_usado'] if self._posicion_actual['margen_usado'] > 0 else 0.0
        self.retornos_realizados_episodio.append(roe_operacion)

        # Crear diccionario del trade para el historial
        trade_dict = {
            'tipo': self._posicion_actual['tipo'].name,
            'precio_entrada': self._posicion_actual['precio_entrada'],
            'precio_salida': precio_ejecucion,
            'tamaño_activo': self._posicion_actual['tamaño_activo'],
            'margen_usado': self._posicion_actual['margen_usado'],
            'pnl_abs': pnl_neto,
            'roe': roe_operacion,
            'pasos_duracion': self._posicion_actual['pasos_en_posicion'],
            'cierre_por': 'AGENT'  # Valor por defecto, se sobrescribe si es por stop-loss
        }
        
        # Añadir paso_cierre si está disponible
        if paso_cierre is not None:
            trade_dict['paso_cierre'] = paso_cierre
            
        self._historial_trades.append(trade_dict)

        logger.debug(f"Posición cerrada: PNL = {pnl_neto:.2f}, ROE = {roe_operacion:.4f}")

        self._posicion_actual = {
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
        comision_abs = valor_nocional * self.config.comision_taker_porcentaje
        slippage_factor = self.config.slippage_porcentaje

        if tipo_operacion == TipoOperacion.LARGO:
            precio_ejecucion = precio_mercado * (1 + slippage_factor)
        else:
            precio_ejecucion = precio_mercado * (1 - slippage_factor)

        return precio_ejecucion, comision_abs

    def update_state(self, precio_actual: float):
        if self._posicion_actual['tipo'] == TipoOperacion.NEUTRAL:
            self._equity_actual = self._balance_actual
            return

        if self._posicion_actual['tipo'] == TipoOperacion.LARGO:
            pnl_no_realizado = (precio_actual - self._posicion_actual['precio_entrada']) * self._posicion_actual['tamaño_activo']
        else:  # CORTO
            pnl_no_realizado = (self._posicion_actual['precio_entrada'] - precio_actual) * self._posicion_actual['tamaño_activo']

        pnl_roe = pnl_no_realizado / self._posicion_actual['margen_usado'] if self._posicion_actual['margen_usado'] > 0 else 0.0

        self._posicion_actual['pnl_no_realizado_abs'] = pnl_no_realizado
        self._posicion_actual['pnl_no_realizado_roe'] = pnl_roe

        self._equity_actual = self._balance_actual + pnl_no_realizado
        self.max_equity_alcanzado_episodio = max(self.max_equity_alcanzado_episodio, self._equity_actual)

    def advance_step(self):
        if self._posicion_actual['tipo'] != TipoOperacion.NEUTRAL:
            self._posicion_actual['pasos_en_posicion'] += 1

    def get_current_state(self) -> Dict[str, Any]:
        return self.posicion_actual

    @property
    def equity(self) -> float:
        return self._equity_actual

    @property
    def balance(self) -> float:
        return self._balance_actual

    @property
    def posicion_actual(self) -> Dict[str, Any]:
        return self._posicion_actual

    @property
    def historial_trades(self) -> List[Dict[str, Any]]:
        return self._historial_trades

    @property
    def is_max_consecutive_losses_reached(self) -> bool:
        """Verifica si se ha alcanzado el umbral de pérdidas consecutivas."""
        return self._consecutive_losses >= self.max_consecutive_losses

    def get_equity_series(self) -> pd.Series:
        """
        Devuelve la serie temporal del equity como pandas Series.
        
        Returns:
            pd.Series: Serie temporal del historial de equity
        """
        return pd.Series(self.historial_equity)

    def check_and_execute_stop_loss(self, low_price: float, high_price: float, paso_cierre: int) -> bool:
        """
        Verifica y ejecuta el stop-loss si se alcanza el nivel configurado.

        Args:
            low_price (float): Precio mínimo de la vela actual
            high_price (float): Precio máximo de la vela actual
            paso_cierre (int): Paso del episodio en el que se ejecuta el cierre

        Returns:
            bool: True si se ejecutó el stop-loss, False en caso contrario
        """
        if self._posicion_actual['tipo'] == TipoOperacion.NEUTRAL:
            return False
        
        precio_entrada = self._posicion_actual['precio_entrada']
        stop_loss_pct = self.config.stop_loss_pct

        if self._posicion_actual['tipo'] == TipoOperacion.LARGO:
            stop_loss_price = precio_entrada * (1 - stop_loss_pct)
            if low_price <= stop_loss_price:
                logger.debug(f"Stop-loss ejecutado en LARGO: {stop_loss_price:.2f}")
                pnl = self._close_position(stop_loss_price, paso_cierre)
                # Modificar el último trade del historial para indicar que fue por stop-loss
                if self._historial_trades:
                    self._historial_trades[-1]['cierre_por'] = 'STOP_LOSS'
                return True
        else:  # CORTO
            stop_loss_price = precio_entrada * (1 + stop_loss_pct)
            if high_price >= stop_loss_price:
                logger.debug(f"Stop-loss ejecutado en CORTO: {stop_loss_price:.2f}")
                pnl = self._close_position(stop_loss_price, paso_cierre)
                # Modificar el último trade del historial para indicar que fue por stop-loss
                if self._historial_trades:
                    self._historial_trades[-1]['cierre_por'] = 'STOP_LOSS'
                return True
        
        return False
