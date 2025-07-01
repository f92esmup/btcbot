from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Any
from enum import Enum

class TipoOperacion(Enum):
    """Tipos de operación/posición."""
    NEUTRAL = 0
    LARGO = 1
    CORTO = -1

class BasePortfolio(ABC):
    """
    Interfaz abstracta que define el contrato para cualquier gestor de portfolio.

    Establece las operaciones y propiedades fundamentales que un portfolio debe
    soportar, independientemente de si es para simulación (en memoria) o para
    operativa real (conectado a una API).
    """

    @abstractmethod
    def reset(self):
        """
        Reinicia el estado del portfolio a sus valores iniciales.
        """
        pass

    @abstractmethod
    def execute_order(self, intencion: str, magnitud: float, precio: float) -> Tuple[bool, float]:
        """
        Ejecuta una decisión de trading (abrir, cerrar, o modificar una posición).

        Args:
            intencion (str): La intención de la operación ('COMPRAR', 'VENDER', 'MANTENER').
            magnitud (float): La magnitud normalizada de la operación [0, 1].
            precio (float): El precio de mercado actual para la ejecución.

        Returns:
            Tuple[bool, float]: Una tupla indicando (si se ejecutó un trade, pnl realizado en el cierre).
        """
        pass

    @abstractmethod
    def update_state(self, precio_actual: float):
        """
        Actualiza el estado interno del portfolio basado en el precio de mercado más reciente.
        Principalmente, calcula el P&L no realizado de la posición abierta.
        """
        pass

    @abstractmethod
    def get_current_state(self) -> Dict[str, Any]:
        """
        Devuelve un diccionario que representa el estado actual del portfolio.
        Este estado es utilizado para construir parte de la observación para el agente.

        Returns:
            Dict[str, Any]: Un diccionario con el estado del portfolio.
        """
        pass
    
    @abstractmethod
    def advance_step(self):
        """
        Avanza el contador de pasos en la posición si está abierta.
        """
        pass

    @property
    @abstractmethod
    def equity(self) -> float:
        """
        Propiedad que devuelve el equity actual del portfolio (balance + P&L no realizado).
        """
        pass

    @property
    @abstractmethod
    def balance(self) -> float:
        """
        Propiedad que devuelve el balance actual de la cuenta (dinero no comprometido en márgenes).
        """
        pass
    
    @property
    @abstractmethod
    def posicion_actual(self) -> Dict[str, Any]:
        """
        Propiedad que devuelve un diccionario con la información de la posición abierta.
        """
        pass

    @property
    @abstractmethod
    def historial_trades(self) -> List[Dict[str, Any]]:
        """
        Propiedad que devuelve una lista con el historial de todos los trades cerrados.
        """
        pass
