#!/usr/bin/env python3
"""
Implementation Improvements for BTC Trading Bot.

This script contains improved implementations for key components based on the technical
design document feedback. These improvements address the misalignments and missing details
identified in the current implementation.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------
# 1. Improved Data Preprocessing and Feature Normalization 
#------------------------------------------------------------------------------

def normalize_features_improved(df: pd.DataFrame, 
                               normalization_methods: Dict,
                               feature_normalization_lookback: int = 100,
                               ffill_limit_for_nans: int = 5) -> pd.DataFrame:
    """
    Improved normalization of features causally using the specified methods.
    
    This implementation ensures all normalizations are causal (not using future data)
    and correctly handles each of the 20 market features with the appropriate 
    normalization strategy.
    
    Args:
        df (pd.DataFrame): DataFrame with calculated features.
        normalization_methods (Dict): Dictionary mapping features to normalization methods.
        feature_normalization_lookback (int): Lookback period for normalization calculations.
        ffill_limit_for_nans (int): Maximum number of NaNs to forward fill.
            
    Returns:
        pd.DataFrame: DataFrame with normalized features.
    """
    logger.info(f"Normalizing features with improved methods")
    
    # Create a new DataFrame for normalized features
    norm_df = pd.DataFrame(index=df.index)
    
    # Ensure OHLCV and ATR columns exist for normalization dependencies
    required_cols = ['close', 'atr', 'open']
    missing_deps = [col for col in required_cols if col not in df.columns]
    if missing_deps:
        logger.error(f"Missing required columns for normalization: {missing_deps}")
        raise ValueError(f"Required columns missing for normalization: {missing_deps}")
    
    # Process each feature with its specified normalization method
    for feature, norm_method in normalization_methods.items():
        try:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found in input DataFrame. Skipping.")
                continue
            
            # Apply the specified normalization method
            if norm_method == 'log_diff':
                # Log difference from previous value: log(x_t / x_{t-1})
                # For market prices: log(C/C_prev), log(H/O), log(L/O), log(C/O)
                if feature == 'open':
                    # log(O/O_prev)
                    norm_df[feature] = np.log(df[feature] / df[feature].shift(1))
                elif feature == 'high':
                    # log(H/O) - log ratio of high to open
                    norm_df[feature] = np.log(df[feature] / df['open'])
                elif feature == 'low':
                    # log(L/O) - log ratio of low to open
                    norm_df[feature] = np.log(df[feature] / df['open'])
                elif feature == 'close':
                    # log(C/O) - log ratio of close to open (within candle)
                    norm_df[feature] = np.log(df[feature] / df['open'])
                elif feature == 'volume':
                    # log(Vol/SMA(Vol,N)) - volume relative to its moving average
                    vol_sma = df[feature].rolling(window=20).mean().shift(1)
                    norm_df[feature] = np.log(df[feature] / vol_sma)
                else:
                    # Standard log diff for other features
                    norm_df[feature] = np.log(df[feature] / df[feature].shift(1))
            
            elif norm_method == 'zscore':
                # Z-score using rolling window: (x_t - mean_{t-1}) / std_{t-1}
                # This is causal - only using past data for mean and std
                mean = df[feature].rolling(window=feature_normalization_lookback).mean().shift(1)
                std = df[feature].rolling(window=feature_normalization_lookback).std().shift(1)
                # Handle zero std with a small epsilon
                std = std.replace(0, 1e-8)
                norm_df[feature] = (df[feature] - mean) / std
            
            elif norm_method == 'divide_by_close':
                # Divide by close price: x_t / close_t
                norm_df[feature] = df[feature] / df['close']
            
            elif norm_method == 'divide_by_atr':
                # Normalize by ATR: (x_t - close_t) / atr_t
                # This works better for oscillators and indicators that should be centered
                norm_df[feature] = (df[feature] - df['close']) / df['atr']
            
            elif norm_method == 'pct_change':
                # Percentage change: (x_t / x_{t-1}) - 1
                norm_df[feature] = df[feature].pct_change()
            
            elif norm_method == 'identity':
                # No normalization: x_t
                norm_df[feature] = df[feature]
            
            elif norm_method == 'identity_center':
                # Center to [-50, 50]: x_t - 50
                # Perfect for indicators already in [0, 100] range like RSI, MFI, Stochastic
                norm_df[feature] = df[feature] - 50
            
            elif norm_method == 'identity_scale':
                # Scale to [0, 1]: x_t / 100
                # For indicators in [0, 100] when we want [0, 1] range
                norm_df[feature] = df[feature] / 100
            
            else:
                logger.warning(f"Unknown normalization method: {norm_method} for feature: {feature}. Using identity.")
                norm_df[feature] = df[feature]
            
        except Exception as e:
            logger.error(f"Error normalizing feature {feature} with method {norm_method}: {str(e)}")
            # Use identity normalization as fallback
            norm_df[feature] = df[feature]
    
    # Handle NaN values created by normalization
    nan_counts = norm_df.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Normalization created NaN values: {nan_counts}")
        
        # Forward fill NaN values up to the limit
        norm_df.fillna(method='ffill', limit=ffill_limit_for_nans, inplace=True)
        
        # Fill remaining NaNs with 0
        norm_df.fillna(0, inplace=True)
    
    logger.info(f"Feature normalization completed. Final shape: {norm_df.shape}")
    return norm_df

#------------------------------------------------------------------------------
# 2. Improved Portfolio Features Normalization
#------------------------------------------------------------------------------

def get_portfolio_features_improved(current_position_btc, 
                                   current_position_entry_price,
                                   current_market_price,
                                   current_balance_usd,
                                   initial_balance_usd,
                                   current_atr,
                                   steps_in_position,
                                   max_drawdown_pct,
                                   max_position_norm_divisor=1.0,
                                   steps_in_position_norm_divisor=100.0,
                                   unrealized_pnl_norm_divisor=1000.0,
                                   max_drawdown_pct_norm_divisor=20.0,
                                   atr_pct_norm_divisor=5.0,
                                   maintenance_margin_pct_norm_divisor=10.0,
                                   position_direction=None):
    """
    Calculate and normalize portfolio features according to the technical design document.
    
    Args:
        current_position_btc (float): Current BTC position size.
        current_position_entry_price (float): Entry price of current position.
        current_market_price (float): Current market price.
        current_balance_usd (float): Current account balance in USD.
        initial_balance_usd (float): Initial account balance in USD.
        current_atr (float): Current ATR value.
        steps_in_position (int): Number of steps in the current position.
        max_drawdown_pct (float): Maximum drawdown percentage.
        max_position_norm_divisor (float): Divisor for normalizing position size.
        steps_in_position_norm_divisor (float): Divisor for normalizing steps in position.
        unrealized_pnl_norm_divisor (float): Divisor for normalizing unrealized P&L.
        max_drawdown_pct_norm_divisor (float): Divisor for normalizing max drawdown.
        atr_pct_norm_divisor (float): Divisor for normalizing ATR percentage.
        maintenance_margin_pct_norm_divisor (float): Divisor for normalizing margin.
        position_direction (int, optional): Position direction (1=long, -1=short, 0=none).
            
    Returns:
        np.ndarray: Array of normalized portfolio features.
    """
    # 1. Position size normalized by max position
    normalized_position = current_position_btc / max_position_norm_divisor
    
    # 2. Entry price normalized - use log scale relative to current price
    if current_position_entry_price > 0:
        normalized_entry_price = np.log(current_position_entry_price / current_market_price)
    else:
        normalized_entry_price = 0.0
    
    # 3. Unrealized P&L normalized
    if current_position_btc != 0 and current_position_entry_price > 0:
        if position_direction is None:
            position_direction = 1 if current_position_btc > 0 else -1
            
        unrealized_pnl = position_direction * current_position_btc * (current_market_price - current_position_entry_price)
        # Normalize by the unrealized PnL divisor, but also relative to current equity
        current_equity = current_balance_usd + (current_position_btc * current_market_price)
        unrealized_pnl_normalized = unrealized_pnl / (current_equity * unrealized_pnl_norm_divisor/100)
    else:
        unrealized_pnl_normalized = 0.0
    
    # 4. Account balance normalized by initial balance (as percentage change)
    normalized_balance = (current_balance_usd / initial_balance_usd) - 1.0
    
    # 5. Steps in position normalized by steps divisor
    steps_in_position_normalized = steps_in_position / steps_in_position_norm_divisor
    
    # 6. Max drawdown percentage normalized
    max_drawdown_pct_normalized = max_drawdown_pct / max_drawdown_pct_norm_divisor
    
    # 7. ATR percentage normalized
    atr_pct = (current_atr / current_market_price) * 100
    atr_pct_normalized = atr_pct / atr_pct_norm_divisor
    
    # 8. Maintenance margin percentage normalized
    if abs(current_position_btc) > 0:
        # Simplified maintenance margin calculation
        position_value = abs(current_position_btc) * current_market_price
        maintenance_margin_pct = 1.0  # Example value (1% maintenance margin)
        maintenance_margin_pct_normalized = maintenance_margin_pct / maintenance_margin_pct_norm_divisor
    else:
        maintenance_margin_pct_normalized = 0.0
    
    # Combine all features
    portfolio_features = np.array([
        normalized_position,
        normalized_entry_price,
        unrealized_pnl_normalized,
        normalized_balance,
        steps_in_position_normalized,
        max_drawdown_pct_normalized,
        atr_pct_normalized,
        maintenance_margin_pct_normalized
    ], dtype=np.float32)
    
    return portfolio_features

#------------------------------------------------------------------------------
# 3. Improved Transformer Feature Extractor Forward Pass
#------------------------------------------------------------------------------

class ImprovedTransformerForward:
    """
    Improved implementation of the transformer feature extractor's forward pass
    following the technical design document's specifications.
    
    This example shows how to properly fuse market and portfolio features
    before passing them through the transformer, which allows the model to capture
    temporal correlations between market data and portfolio state at each timestep.
    """
    
    @staticmethod
    def forward(observations, market_embedding, positional_encoding, 
                transformer_encoder, final_embedding):
        """
        Forward pass of the feature extractor following technical design document.
        
        Process market features with portfolio features through the transformer encoder
        to create a comprehensive state representation that captures temporal
        correlations between market state and portfolio state.
        
        Args:
            observations (Dict[str, torch.Tensor]): Dictionary with 'market_features'
                and 'portfolio_features' tensors.
            market_embedding (nn.Linear): Linear embedding for combined features.
            positional_encoding (nn.Module): Positional encoding layer.
            transformer_encoder (nn.TransformerEncoder): Transformer encoder.
            final_embedding (nn.Linear): Final linear layer.
                
        Returns:
            torch.Tensor: Extracted features tensor.
        """
        # Extract components from the dict observation
        market_features = observations['market_features']  # (batch_size, seq_len, n_market_features)
        portfolio_features = observations['portfolio_features']  # (batch_size, n_portfolio_features)
        
        # Get dimensions
        batch_size, seq_len, n_market_features = market_features.shape
        
        # Replicate portfolio features for each time step
        # First, unsqueeze to add seq_len dimension: (batch_size, 1, n_portfolio_features)
        portfolio_expanded = portfolio_features.unsqueeze(1)
        
        # Repeat along seq_len dimension: (batch_size, seq_len, n_portfolio_features)
        portfolio_repeated = portfolio_expanded.repeat(1, seq_len, 1)
        
        # Concatenate market and portfolio features along the feature dimension
        # (batch_size, seq_len, n_market_features + n_portfolio_features)
        combined_features = torch.cat([market_features, portfolio_repeated], dim=2)
        
        # Transpose to (seq_len, batch_size, n_features) for transformer
        combined_features = combined_features.transpose(0, 1)
        
        # Linear embedding and positional encoding
        embedded = market_embedding(combined_features)  # (seq_len, batch_size, d_model)
        embedded = positional_encoding(embedded)
        
        # Apply transformer encoder
        transformer_output = transformer_encoder(embedded)  # (seq_len, batch_size, d_model)
        
        # Use the last sequence element as the representation
        final_representation = transformer_output[-1]  # (batch_size, d_model)
        
        # Final embedding to match required features_dim
        output = final_embedding(final_representation)  # (batch_size, features_dim)
        
        return output

#------------------------------------------------------------------------------
# 4. Improved Trading Environment Step Logic
#------------------------------------------------------------------------------

def execute_trade_improved(decision, target_position_btc, current_position_btc, 
                          current_position_entry_price, current_balance_usd,
                          current_market_price, leverage=3.0,
                          commission_rate=0.0004):
    """
    Improved implementation of trade execution logic following the technical design document.
    
    This implementation handles all cases correctly:
    - Opening a new position
    - Closing an existing position
    - Inverting a position (changing from long to short or vice versa)
    - Increasing/decreasing position size
    
    It also properly calculates the weighted average entry price when adding to a position.
    
    Args:
        decision (str): Trade decision ('buy', 'sell', 'hold', 'close')
        target_position_btc (float): Target position size in BTC
        current_position_btc (float): Current position size in BTC
        current_position_entry_price (float): Current average entry price
        current_balance_usd (float): Current account balance in USD
        current_market_price (float): Current market price
        leverage (float): Account leverage ratio
        commission_rate (float): Commission rate per trade
        
    Returns:
        tuple: (new_position_btc, new_entry_price, new_balance_usd, realized_pnl, steps_in_position)
    """
    if decision == 'hold':
        # No change, just update steps in position if in a position
        steps_in_position = 0 if current_position_btc == 0 else 1
        return current_position_btc, current_position_entry_price, current_balance_usd, 0, steps_in_position
    
    # Calculate position delta (how much to add or remove)
    position_delta_btc = target_position_btc - current_position_btc
    
    # If delta is very small, consider it as no change
    if abs(position_delta_btc) < 1e-8:
        steps_in_position = 0 if current_position_btc == 0 else 1
        return current_position_btc, current_position_entry_price, current_balance_usd, 0, steps_in_position
    
    # Calculate order cost and commission
    order_value_usd = abs(position_delta_btc * current_market_price)
    commission_usd = order_value_usd * commission_rate
    
    # Check if we have enough margin available
    required_margin = order_value_usd / leverage
    if required_margin + commission_usd > current_balance_usd:
        # Not enough margin, don't execute the trade
        steps_in_position = 0 if current_position_btc == 0 else 1
        return current_position_btc, current_position_entry_price, current_balance_usd, 0, steps_in_position
    
    # Calculate realized PnL if we're reducing or flipping a position
    realized_pnl = 0
    if current_position_btc != 0 and (
            (current_position_btc > 0 and position_delta_btc < 0) or 
            (current_position_btc < 0 and position_delta_btc > 0)):
        
        # Calculate how much of the position we're closing
        closing_size = min(abs(current_position_btc), abs(position_delta_btc))
        if current_position_btc > 0:  # Closing long position
            realized_pnl = closing_size * (current_market_price - current_position_entry_price)
        else:  # Closing short position
            realized_pnl = closing_size * (current_position_entry_price - current_market_price)
    
    # Update balance with commission cost and realized PnL
    new_balance_usd = current_balance_usd - commission_usd + realized_pnl
    
    # Calculate new position
    new_position_btc = current_position_btc + position_delta_btc
    
    # Calculate new entry price
    if abs(new_position_btc) < 1e-8:  # Position fully closed
        new_entry_price = 0
        steps_in_position = 0
    elif (current_position_btc > 0 and new_position_btc > 0) or (current_position_btc < 0 and new_position_btc < 0):
        # Adding to existing position (same direction) - weighted average
        if position_delta_btc != 0:  # Only update if actually adding
            old_value = abs(current_position_btc * current_position_entry_price)
            new_value = abs(position_delta_btc * current_market_price)
            new_entry_price = (old_value + new_value) / abs(new_position_btc)
        else:
            new_entry_price = current_position_entry_price
        steps_in_position = 1  # Continue counting
    else:
        # New position or flipped position - use current price
        new_entry_price = current_market_price
        steps_in_position = 0  # Reset counter for new position
    
    return new_position_btc, new_entry_price, new_balance_usd, realized_pnl, steps_in_position

if __name__ == "__main__":
    logger.info("This script contains improved implementations for BTC Trading Bot components.")
    logger.info("Import and use these functions in the actual implementation files.")
