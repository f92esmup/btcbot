import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

class SimulatedBroker:
    """
    Clase para simular un broker de futuros.
    
    Responsable de calcular:
     - Precios de ejecución (con slippage)
     - Comisiones
     - Tamaños de posición
     - Margen requerido
    
    Es una clase sin estado, lo que facilita su reemplazo por una implementación real
    """
    
    def __init__(self, taker_fee_rate: float, slippage_atr_multiplier: float, min_order_size_btc: float = 0.001):
        """
        Inicializa el broker simulado.
        
        Args:
            taker_fee_rate: Comisión taker (ej. 0.0004 para 0.04%)
            slippage_atr_multiplier: Multiplicador de ATR para el slippage
            min_order_size_btc: Tamaño mínimo de orden en BTC
        """
        self.taker_fee_rate = taker_fee_rate
        self.slippage_atr_multiplier = slippage_atr_multiplier
        self.min_order_size_btc = min_order_size_btc
    
    def calculate_execution_details(
        self, 
        desired_action: str, 
        market_close_price: float, 
        atr_value: float, 
        position_to_close_entry_price: Optional[float] = None, 
        position_to_close_size: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calcula los detalles de ejecución para una acción deseada.
        
        Args:
            desired_action: "OPEN_LONG", "OPEN_SHORT", "CLOSE_LONG", "CLOSE_SHORT"
            market_close_price: Precio de cierre actual del mercado
            atr_value: Valor del ATR para calcular el slippage
            position_to_close_entry_price: Precio de entrada de la posición a cerrar (solo para cierres)
            position_to_close_size: Tamaño de la posición a cerrar en contratos/BTC (solo para cierres)
            
        Returns:
            Dict con execution_price, slippage, commission_to_be_paid, potential_pnl (si es cierre)
        """
        # Calcula el slippage basado en el ATR
        slippage_amount = self.slippage_atr_multiplier * atr_value
        
        # Determina el precio de ejecución basado en el tipo de acción y el slippage
        if desired_action == "OPEN_LONG" or desired_action == "CLOSE_SHORT":
            # Para comprar, el precio es mayor (worst case)
            execution_price = market_close_price + slippage_amount
        else:  # "OPEN_SHORT" o "CLOSE_LONG"
            # Para vender, el precio es menor (worst case)
            execution_price = market_close_price - slippage_amount
        
        # Inicializa el resultado
        result = {
            "execution_price": execution_price,
            "slippage": slippage_amount,
            "slippage_pct": slippage_amount / market_close_price
        }
        
        # Si es un cierre, calcula el P&L potencial
        if "CLOSE" in desired_action:
            if not position_to_close_entry_price or not position_to_close_size:
                raise ValueError("Para cerrar posiciones, se requiere precio de entrada y tamaño")
            
            if desired_action == "CLOSE_LONG":
                # Para posición larga, P&L = (precio_salida - precio_entrada) * tamaño
                potential_pnl = (execution_price - position_to_close_entry_price) * position_to_close_size
            else:  # "CLOSE_SHORT"
                # Para posición corta, P&L = (precio_entrada - precio_salida) * tamaño
                potential_pnl = (position_to_close_entry_price - execution_price) * position_to_close_size
            
            result["potential_pnl"] = potential_pnl
            
            # Calcula comisión basada en el valor nocional de cierre
            value_notional = position_to_close_size * execution_price
            commission = value_notional * self.taker_fee_rate
            result["commission_to_be_paid"] = commission
            result["notional_value"] = value_notional
        
        # Para aperturas, la comisión se calcula cuando se determina el tamaño
        
        return result
    
    def calculate_position_size_contracts(
        self, 
        equity: float, 
        position_size_pct_equity: float, 
        leverage: float, 
        execution_price: float
    ) -> Tuple[float, float, float]:
        """
        Calcula el tamaño de la posición en contratos/BTC.
        
        Args:
            equity: Equity actual
            position_size_pct_equity: Porcentaje del equity a utilizar (ej. 0.05 para 5%)
            leverage: Apalancamiento configurado
            execution_price: Precio de ejecución estimado
            
        Returns:
            Tuple con (tamaño_en_contratos, commission_to_be_paid, margin_required)
        """
        # Calcula el valor nocional deseado (% del equity)
        desired_notional_value = equity * position_size_pct_equity * leverage
        
        # Calcula el tamaño en contratos/BTC
        position_size_contracts = desired_notional_value / execution_price
        
        # Redondea a la precisión adecuada (0.001 BTC)
        position_size_contracts = np.floor(position_size_contracts * 1000) / 1000
        
        # Verifica el tamaño mínimo
        if position_size_contracts < self.min_order_size_btc:
            # No alcanza el mínimo, ajusta al mínimo
            if equity * position_size_pct_equity * leverage >= self.min_order_size_btc * execution_price:
                position_size_contracts = self.min_order_size_btc
            else:
                # No hay suficiente equity para el mínimo
                position_size_contracts = 0.0
        
        # Recalcula el valor nocional real
        actual_notional_value = position_size_contracts * execution_price
        
        # Calcula la comisión
        commission = actual_notional_value * self.taker_fee_rate
        
        # Calcula el margen requerido
        margin_required = self.calculate_margin_required(position_size_contracts, execution_price, leverage)
        
        return position_size_contracts, commission, margin_required
    
    def calculate_margin_required(
        self, 
        position_size_contracts: float, 
        execution_price: float, 
        leverage: float
    ) -> float:
        """
        Calcula el margen requerido para una posición.
        
        Args:
            position_size_contracts: Tamaño de la posición en contratos/BTC
            execution_price: Precio de ejecución
            leverage: Apalancamiento configurado
            
        Returns:
            Margen requerido
        """
        # En futuros, el margen es el valor nocional / leverage
        notional_value = position_size_contracts * execution_price
        margin_required = notional_value / leverage
        
        return margin_required

    def calculate_liquidation_price(
        self,
        position_side: int,  # 1 para largo, -1 para corto
        entry_price: float,
        leverage: float,
        safety_factor: float = 0.8
    ) -> float:
        """
        Calcula el precio de liquidación para una posición.
        
        Args:
            position_side: Lado de la posición (1 para largo, -1 para corto)
            entry_price: Precio de entrada
            leverage: Apalancamiento configurado
            safety_factor: Factor de seguridad para la liquidación (< 1.0)
            
        Returns:
            Precio de liquidación
        """
        # Cálculo simplificado: liquidación ocurre al moverse (1/leverage * safety_factor) en contra
        # Para posiciones largas: entry_price * (1 - move_pct)
        # Para posiciones cortas: entry_price * (1 + move_pct)
        
        move_pct = (1.0 / leverage) * safety_factor
        
        if position_side == 1:  # Largo
            liquidation_price = entry_price * (1.0 - move_pct)
        else:  # Corto
            liquidation_price = entry_price * (1.0 + move_pct)
            
        return liquidation_price
