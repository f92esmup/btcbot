"""
Simulated broker module for Bitcoin trading bot.

This module contains the SimulatedBroker class, which is responsible for simulating
the execution of trading orders and calculating execution details like commissions,
slippage, position sizing, margin requirements, and liquidation prices.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class SimulatedBroker:
    """
    A class for simulating the execution of trading orders on Binance Futures.
    
    This class simulates the execution of trading orders, calculates commissions,
    slippage, position sizing, margin requirements, and liquidation prices based
    on the specifications of Binance Futures for BTC/USDT perpetual contracts.
    
    Attributes:
        logger (logging.Logger): Class logger.
        commission_rate (float): Trading commission rate (as a decimal).
        max_leverage (int): Maximum allowed leverage.
        min_order_size_usd (float): Minimum order size in USD.
        slippage_model (str): Model to use for simulating slippage.
        slippage_factor (float): Factor for slippage calculation.
        maintenance_margin_rate (float): Maintenance margin requirement rate.
        initial_margin_rate (float): Initial margin requirement rate.
    """
    
    def __init__(
        self,
        commission_rate: float = 0.0004,  # 0.04% for both maker and taker
        max_leverage: int = 20,
        min_order_size_usd: float = 5.0,  # Minimum order size in USD
        slippage_model: str = 'atr_based',
        slippage_factor: float = 0.05,  # 5% of ATR
        maintenance_margin_rate: float = 0.004,  # 0.4% for BTC futures with 20x leverage
        initial_margin_rate: float = 0.05  # 5% initial margin (1/max_leverage)
    ):
        """
        Initialize the SimulatedBroker with trading parameters.
        
        Args:
            commission_rate (float, optional): Trading commission rate (as a decimal).
                Defaults to 0.0004 (0.04%).
            max_leverage (int, optional): Maximum allowed leverage. Defaults to 20.
            min_order_size_usd (float, optional): Minimum order size in USD.
                Defaults to 5.0 USD.
            slippage_model (str, optional): Model to use for simulating slippage.
                Options: 'none', 'fixed_bps', 'atr_based'. Defaults to 'atr_based'.
            slippage_factor (float, optional): Factor for slippage calculation.
                For 'fixed_bps', this is in basis points (e.g., 5 = 0.05%).
                For 'atr_based', this is a fraction of ATR (e.g., 0.05 = 5% of ATR).
                Defaults to 0.05.
            maintenance_margin_rate (float, optional): Maintenance margin requirement rate.
                Defaults to 0.004 (0.4%).
            initial_margin_rate (float, optional): Initial margin requirement rate.
                Defaults to 0.05 (5% = 1/max_leverage).
        """
        self.logger = logging.getLogger(__name__)
        self.commission_rate = commission_rate
        self.max_leverage = max_leverage
        self.min_order_size_usd = min_order_size_usd
        self.slippage_model = slippage_model
        self.slippage_factor = slippage_factor
        self.maintenance_margin_rate = maintenance_margin_rate
        self.initial_margin_rate = initial_margin_rate
        
        # Validate parameters
        if not 0 <= commission_rate < 0.01:
            self.logger.warning(f"Unusual commission rate: {commission_rate}. Normally between 0 and 0.01 (1%).")
        
        if max_leverage not in [1, 2, 3, 5, 10, 20, 50, 75, 100, 125]:
            self.logger.warning(f"Unusual max leverage: {max_leverage}. Common values on Binance: 1, 3, 5, 10, 20, 50, 75, 100, 125.")
        
        if not 0 < initial_margin_rate <= 1:
            self.logger.warning(f"Unusual initial margin rate: {initial_margin_rate}. Should be between 0 and 1.")
            
        if not 0 < maintenance_margin_rate < initial_margin_rate:
            self.logger.warning(f"Maintenance margin rate ({maintenance_margin_rate}) should be greater than 0 and less than initial margin rate ({initial_margin_rate}).")
        
        self.logger.info("SimulatedBroker initialized with parameters.")
    
    def calculate_execution_details(
        self,
        order_type: str,
        market_price: float,
        order_size_usd: float,
        atr: float = None,
        current_position: float = 0.0,
        current_position_entry_price: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate execution details for a simulated order.
        
        Args:
            order_type (str): Type of order ('buy', 'sell', or 'close').
            market_price (float): Current market price.
            order_size_usd (float): Size of the order in USD.
            atr (float, optional): Average True Range for slippage calculation.
                Required if slippage_model is 'atr_based'. Defaults to None.
            current_position (float, optional): Current position size in BTC.
                Positive for long, negative for short. Defaults to 0.0.
            current_position_entry_price (float, optional): Entry price of the current position.
                Defaults to 0.0.
                
        Returns:
            Dict[str, Any]: Dictionary containing execution details:
                - 'executed' (bool): Whether the order was executed.
                - 'execution_price' (float): Price at which the order was executed.
                - 'slippage_usd' (float): Slippage cost in USD.
                - 'commission_usd' (float): Commission cost in USD.
                - 'size_btc' (float): Size of the executed order in BTC.
                - 'size_usd' (float): Size of the executed order in USD.
                - 'reason' (str, optional): Reason if the order was not executed.
                
        Raises:
            ValueError: If order_type is not 'buy', 'sell', or 'close'.
        """
        # Validate inputs
        if order_type not in ['buy', 'sell', 'close']:
            raise ValueError(f"Invalid order type: {order_type}. Must be 'buy', 'sell', or 'close'.")
        
        if market_price <= 0:
            raise ValueError(f"Invalid market price: {market_price}. Must be positive.")
        
        if order_size_usd < 0:
            raise ValueError(f"Invalid order size: {order_size_usd}. Must be non-negative.")
        
        if self.slippage_model == 'atr_based' and atr is None:
            raise ValueError("ATR is required for ATR-based slippage calculation.")
        
        # If order type is 'close', determine buy/sell based on current position
        if order_type == 'close':
            if current_position > 0:
                order_type = 'sell'
                order_size_usd = abs(current_position) * market_price
            elif current_position < 0:
                order_type = 'buy'
                order_size_usd = abs(current_position) * market_price
            else:
                # No position to close
                return {
                    'executed': False,
                    'reason': 'No position to close',
                    'execution_price': market_price,
                    'slippage_usd': 0.0,
                    'commission_usd': 0.0,
                    'size_btc': 0.0,
                    'size_usd': 0.0
                }
        
        # Check minimum order size
        if order_size_usd < self.min_order_size_usd:
            return {
                'executed': False,
                'reason': f'Order size ({order_size_usd} USD) below minimum ({self.min_order_size_usd} USD)',
                'execution_price': market_price,
                'slippage_usd': 0.0,
                'commission_usd': 0.0,
                'size_btc': 0.0,
                'size_usd': 0.0
            }
        
        # Calculate execution price with slippage
        execution_price, slippage_usd = self._calculate_slippage(
            order_type=order_type,
            market_price=market_price,
            order_size_usd=order_size_usd,
            atr=atr
        )
        
        # Calculate commission
        commission_usd = order_size_usd * self.commission_rate
        
        # Calculate order size in BTC
        size_btc = order_size_usd / execution_price
        if order_type == 'sell':
            size_btc = -size_btc  # Negative for sell orders
        
        return {
            'executed': True,
            'execution_price': execution_price,
            'slippage_usd': slippage_usd,
            'commission_usd': commission_usd,
            'size_btc': size_btc,
            'size_usd': order_size_usd,
            'order_type': order_type
        }
    
    def _calculate_slippage(
        self,
        order_type: str,
        market_price: float,
        order_size_usd: float,
        atr: float = None
    ) -> Tuple[float, float]:
        """
        Calculate execution price with slippage and the cost of slippage.
        
        Args:
            order_type (str): Type of order ('buy' or 'sell').
            market_price (float): Current market price.
            order_size_usd (float): Size of the order in USD.
            atr (float, optional): Average True Range for slippage calculation.
                Required if slippage_model is 'atr_based'. Defaults to None.
                
        Returns:
            Tuple[float, float]: A tuple containing:
                - Execution price with slippage.
                - Slippage cost in USD.
        """
        if self.slippage_model == 'none':
            # No slippage
            return market_price, 0.0
            
        elif self.slippage_model == 'fixed_bps':
            # Fixed basis points slippage
            slippage_pct = self.slippage_factor / 10000  # Convert bps to decimal
            
            if order_type == 'buy':
                # Buy orders execute at a higher price
                execution_price = market_price * (1 + slippage_pct)
            else:  # sell
                # Sell orders execute at a lower price
                execution_price = market_price * (1 - slippage_pct)
            
            # Calculate slippage cost
            slippage_usd = order_size_usd * slippage_pct
            
            return execution_price, slippage_usd
            
        elif self.slippage_model == 'atr_based':
            # ATR-based slippage
            if atr is None or atr <= 0:
                return market_price, 0.0
            
            # Slippage as a percentage of ATR
            slippage_amount = atr * self.slippage_factor
            
            if order_type == 'buy':
                # Buy orders execute at a higher price
                execution_price = market_price + slippage_amount
            else:  # sell
                # Sell orders execute at a lower price
                execution_price = market_price - slippage_amount
            
            # Calculate slippage cost in USD
            order_size_btc = order_size_usd / market_price
            slippage_usd = order_size_btc * slippage_amount
            
            return execution_price, slippage_usd
        
        else:
            # Unsupported slippage model, use market price
            self.logger.warning(f"Unsupported slippage model: {self.slippage_model}. Using market price.")
            return market_price, 0.0
    
    def calculate_liquidation_price(
        self,
        position_size_btc: float,
        entry_price: float,
        account_balance_usd: float,
        additional_collateral_usd: float = 0.0
    ) -> float:
        """
        Calculate the liquidation price for a position.
        
        The liquidation price is the price at which the position will be forcibly closed
        by the exchange due to insufficient margin. This is calculated based on the
        maintenance margin requirement.
        
        Args:
            position_size_btc (float): Position size in BTC. Positive for long, negative for short.
            entry_price (float): Entry price of the position.
            account_balance_usd (float): Account balance in USD.
            additional_collateral_usd (float, optional): Additional collateral allocated to this position.
                Defaults to 0.0.
                
        Returns:
            float: Liquidation price. Returns None if there is no position or the position is fully collateralized.
        """
        if position_size_btc == 0:
            return None  # No position, no liquidation price
        
        # Calculate position value
        position_value_usd = abs(position_size_btc) * entry_price
        
        # Calculate total available collateral
        total_collateral = account_balance_usd + additional_collateral_usd
        
        # If the position is fully collateralized, there's no liquidation risk
        if position_value_usd <= total_collateral:
            return None
        
        # Calculate the liquidation price
        if position_size_btc > 0:  # Long position
            # For long positions, liquidation happens when:
            # position_value * (1 - (price_change / entry_price)) = maintenance_margin
            # Solve for the price at which this happens
            liquidation_price = entry_price * (1 - ((total_collateral - position_value_usd * self.maintenance_margin_rate) / position_value_usd))
        else:  # Short position
            # For short positions, liquidation happens when:
            # position_value * (1 + (price_change / entry_price)) = maintenance_margin
            # Solve for the price at which this happens
            liquidation_price = entry_price * (1 + ((total_collateral - position_value_usd * self.maintenance_margin_rate) / position_value_usd))
        
        return max(0.01, liquidation_price)  # Ensure liquidation price is positive
    
    def calculate_max_position_size(
        self,
        account_balance_usd: float,
        current_price: float,
        desired_leverage: float = None
    ) -> float:
        """
        Calculate the maximum position size that can be taken based on account balance and leverage.
        
        Args:
            account_balance_usd (float): Account balance in USD.
            current_price (float): Current market price.
            desired_leverage (float, optional): Desired leverage to use.
                If None, uses the maximum allowed leverage. Defaults to None.
                
        Returns:
            float: Maximum position size in BTC.
        """
        # Use maximum leverage if desired leverage is not specified or exceeds maximum
        effective_leverage = min(desired_leverage or self.max_leverage, self.max_leverage)
        
        # Calculate maximum position value
        max_position_value_usd = account_balance_usd * effective_leverage
        
        # Convert to BTC
        max_position_size_btc = max_position_value_usd / current_price
        
        return max_position_size_btc
    
    def calculate_required_margin(
        self,
        position_size_btc: float,
        entry_price: float,
        leverage: float = None
    ) -> Dict[str, float]:
        """
        Calculate the initial and maintenance margin required for a position.
        
        Args:
            position_size_btc (float): Position size in BTC. Positive for long, negative for short.
            entry_price (float): Entry price of the position.
            leverage (float, optional): Leverage used for the position.
                If None, uses the maximum allowed leverage. Defaults to None.
                
        Returns:
            Dict[str, float]: Dictionary containing margin requirements:
                - 'initial_margin_usd': Initial margin required in USD.
                - 'maintenance_margin_usd': Maintenance margin required in USD.
                - 'effective_leverage': Effective leverage used for the calculation.
        """
        # Use maximum leverage if desired leverage is not specified or exceeds maximum
        effective_leverage = min(leverage or self.max_leverage, self.max_leverage)
        
        # Calculate position value
        position_value_usd = abs(position_size_btc) * entry_price
        
        # Calculate margin requirements
        initial_margin_usd = position_value_usd / effective_leverage
        maintenance_margin_usd = position_value_usd * self.maintenance_margin_rate
        
        return {
            'initial_margin_usd': initial_margin_usd,
            'maintenance_margin_usd': maintenance_margin_usd,
            'effective_leverage': effective_leverage
        }
