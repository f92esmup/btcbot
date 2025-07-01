"""
Módulo que encapsula la lógica y el estado del portfolio de trading.
"""

import logging
from typing import Dict, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class TipoOperacion(Enum):
    """Tipos de operación/posición."""
    NEUTRAL = 0
    LARGO = 1
    CORTO = -1

class Portfolio:
    """
    Gestiona el estado financiero y las operaciones de un portfolio de trading.

    Esta clase es responsable de:
    - Mantener el estado del balance, equity y la posición actual.
    - Ejecutar la lógica para abrir y cerrar posiciones.
    - Calcular costos de transacción (comisiones y slippage).
    - Actualizar el P&L no realizado.
    - Mantener un historial de trades ejecutados.
    """
    def __init__(self, env_config: Dict[str, Any]):
        """
        Inicializa el portfolio.

        Args:
            env_config (Dict[str, Any]): Diccionario con la configuración del entorno,
                                         incluyendo capital_inicial, apalancamiento, etc.
        """
        self.config = env_config
        self.reset()

    def reset(self):
        """Reinicia el estado del portfolio a sus valores iniciales."""
        self.balance_actual = self.config['capital_inicial']
        self.equity_actual = self.config['capital_inicial']
        self.max_equity_alcanzado_episodio = self.config['capital_inicial']
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
        self.historial_trades = []
        self.retornos_realizados_episodio = []

    def execute_trade(self, intencion: str, magnitud_efectiva: float, precio_mercado: float) -> Tuple[bool, float]:
        """
        Ejecuta la lógica de trading basada en la intención del agente.

        Args:
            intencion (str): "COMPRAR", "VENDER", o "MANTENER".
            magnitud_efectiva (float): Magnitud normalizada de la operación [0, 1].
            precio_mercado (float): Precio actual del mercado.

        Returns:
            Tuple[bool, float]: Una tupla con (trade_ejecutado, pnl_realizado).
        """
        posicion_actual_tipo = self.posicion_actual['tipo']

        if intencion == "MANTENER":
            return False, 0.0

        es_operacion_opuesta = (
            (intencion == "VENDER" and posicion_actual_tipo == TipoOperacion.LARGO) or
            (intencion == "COMPRAR" and posicion_actual_tipo == TipoOperacion.CORTO)
        )

        if es_operacion_opuesta:
            pnl_realizado = self._close_position(precio_mercado)
            return True, pnl_realizado

        if posicion_actual_tipo == TipoOperacion.NEUTRAL and magnitud_efectiva > 0:
            tipo_operacion = TipoOperacion.LARGO if intencion == "COMPRAR" else TipoOperacion.CORTO
            self._open_position(tipo_operacion, precio_mercado, magnitud_efectiva)
            return True, 0.0

        return False, 0.0

    def _open_position(self, tipo_operacion: TipoOperacion, precio_mercado: float, magnitud: float):
        """Abre una nueva posición."""
        margen_a_usar = self.balance_actual * self.config['porcentaje_max_inversion_por_trade'] * magnitud
        valor_nocional = margen_a_usar * self.config['apalancamiento']
        tamaño_activo = valor_nocional / precio_mercado

        precio_ejecucion, coste_total = self._apply_costs(valor_nocional, precio_mercado, tipo_operacion)

        self.balance_actual -= (coste_total + margen_a_usar)

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
        """Cierra la posición actual y retorna el PNL realizado."""
        if self.posicion_actual['tipo'] == TipoOperacion.NEUTRAL:
            return 0.0

        precio_ejecucion, coste_cierre = self._apply_costs(
            self.posicion_actual['valor_nocional'],
            precio_mercado,
            self.posicion_actual['tipo']
        )

        if self.posicion_actual['tipo'] == TipoOperacion.LARGO:
            pnl_bruto = (precio_ejecucion - self.posicion_actual['precio_entrada']) * self.posicion_actual['tamaño_activo']
        else:  # CORTO
            pnl_bruto = (self.posicion_actual['precio_entrada'] - precio_ejecucion) * self.posicion_actual['tamaño_activo']

        pnl_neto = pnl_bruto - coste_cierre

        self.balance_actual += (self.posicion_actual['margen_usado'] + pnl_neto)

        roe_operacion = pnl_neto / self.posicion_actual['margen_usado'] if self.posicion_actual['margen_usado'] > 0 else 0.0
        self.retornos_realizados_episodio.append(roe_operacion)

        self.historial_trades.append({
            'tipo': self.posicion_actual['tipo'].name,
            'precio_entrada': self.posicion_actual['precio_entrada'],
            'precio_salida': precio_ejecucion,
            'tamaño_activo': self.posicion_actual['tamaño_activo'],
            'margen_usado': self.posicion_actual['margen_usado'],
            'pnl_abs': pnl_neto,
            'roe': roe_operacion,
            'pasos_duracion': self.posicion_actual['pasos_en_posicion'],
        })

        logger.debug(f"Posición cerrada: PNL = {pnl_neto:.2f}, ROE = {roe_operacion:.4f}")

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
        """Aplica comisiones y slippage."""
        comision_abs = valor_nocional * self.config['comision_taker_porcentaje']
        slippage_factor = self.config['slippage_porcentaje']

        if tipo_operacion == TipoOperacion.LARGO:
            precio_ejecucion = precio_mercado * (1 + slippage_factor)
        else:
            precio_ejecucion = precio_mercado * (1 - slippage_factor)

        return precio_ejecucion, comision_abs

    def update_equity_and_pnl(self, precio_actual: float):
        """Actualiza el equity y PNL no realizado de la posición actual."""
        if self.posicion_actual['tipo'] == TipoOperacion.NEUTRAL:
            self.equity_actual = self.balance_actual
            return

        if self.posicion_actual['tipo'] == TipoOperacion.LARGO:
            pnl_no_realizado = (precio_actual - self.posicion_actual['precio_entrada']) * self.posicion_actual['tamaño_activo']
        else:  # CORTO
            pnl_no_realizado = (self.posicion_actual['precio_entrada'] - precio_actual) * self.posicion_actual['tamaño_activo']

        pnl_roe = pnl_no_realizado / self.posicion_actual['margen_usado'] if self.posicion_actual['margen_usado'] > 0 else 0.0

        self.posicion_actual['pnl_no_realizado_abs'] = pnl_no_realizado
        self.posicion_actual['pnl_no_realizado_roe'] = pnl_roe

        self.equity_actual = self.balance_actual + pnl_no_realizado
        self.max_equity_alcanzado_episodio = max(self.max_equity_alcanzado_episodio, self.equity_actual)

    def advance_step(self):
        """Avanza el contador de pasos en la posición si está abierta."""
        if self.posicion_actual['tipo'] != TipoOperacion.NEUTRAL:
            self.posicion_actual['pasos_en_posicion'] += 1
