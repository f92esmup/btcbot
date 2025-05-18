"""
Feature engineering module for Bitcoin trading bot.

This module contains the FeatureEngineer class, which is responsible for calculating
technical indicators and other derived features from OHLCV market data using pandas-ta.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from typing import List, Dict, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class FeatureEngineer:
    """
    A class for engineering features from OHLCV market data using pandas-ta.
    
    This class is responsible for calculating technical indicators and other
    derived features from OHLCV (Open, High, Low, Close, Volume) market data.
    It uses the pandas-ta library for efficient calculation of indicators.
    
    Attributes:
        logger (logging.Logger): Class logger.
        default_feature_params (Dict): Default parameters for feature calculation.
    """
    
    def __init__(self, feature_params: Optional[Dict] = None):
        """
        Initialize the FeatureEngineer with custom parameters.
        
        Args:
            feature_params (Dict, optional): Parameters for feature calculation.
                If not provided, default parameters will be used.
        """
        self.logger = logging.getLogger(__name__)
        
        # Define default parameters for technical indicators
        self.default_feature_params = {
            # Moving Averages
            'sma_fast_period': 20,
            'sma_slow_period': 50,
            'ema_fast_period': 12,
            'ema_slow_period': 26,
            
            # RSI
            'rsi_period': 14,
            
            # Bollinger Bands
            'bbands_period': 20,
            'bbands_std_dev': 2,
            
            # MACD
            'macd_fast_period': 12,
            'macd_slow_period': 26,
            'macd_signal_period': 9,
            
            # ADX
            'adx_period': 14,
            
            # ATR
            'atr_period': 14,
            
            # Stochastic
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            
            # OBV (On-Balance Volume)
            # No parameters needed
            
            # Ichimoku Cloud
            'ichimoku_tenkan_period': 9,
            'ichimoku_kijun_period': 26,
            'ichimoku_senkou_b_period': 52,
        }
        
        # Update default parameters with provided parameters
        if feature_params:
            self.default_feature_params.update(feature_params)
        
        self.logger.info("FeatureEngineer initialized with parameters.")
    
    def engineer_features(
        self, 
        df: pd.DataFrame, 
        include_all: bool = False
    ) -> pd.DataFrame:
        """
        Calculate technical indicators and derived features from OHLCV data.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data. Must contain columns:
                'open', 'high', 'low', 'close', 'volume'.
            include_all (bool, optional): If True, return all calculated features.
                If False, return only the 20 features specified in the design document.
                Defaults to False.
        
        Returns:
            pd.DataFrame: Original DataFrame with additional feature columns.
        
        Raises:
            ValueError: If input DataFrame does not contain required columns.
        """
        # Validate input DataFrame
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Input DataFrame missing required columns: {missing_cols}")
        
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Log the data range
        self.logger.info(f"Engineering features for data from {result_df.index.min()} to {result_df.index.max()}")
        
        # 1. Process OHLCV base features
        self._calculate_ohlcv_derived_features(result_df)
        
        # 2. Calculate Moving Averages
        self._calculate_moving_averages(result_df)
        
        # 3. Calculate Momentum Indicators
        self._calculate_momentum_indicators(result_df)
        
        # 4. Calculate Volatility Indicators
        self._calculate_volatility_indicators(result_df)
        
        # 5. Calculate Volume Indicators
        self._calculate_volume_indicators(result_df)
        
        # 6. Calculate Trend Indicators
        self._calculate_trend_indicators(result_df)
        
        # If include_all is False, only return the specified features
        if not include_all:
            # Select the 20 specified features as per the design document
            selected_features = self._select_final_features(result_df)
            return selected_features
        
        return result_df
    
    def _calculate_ohlcv_derived_features(self, df: pd.DataFrame) -> None:
        """
        Calculate basic features derived directly from OHLCV data.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
        """
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-Low Range
        df['hl_range'] = df['high'] - df['low']
        
        # High-Close Range
        df['hc_range'] = df['high'] - df['close'].shift(1)
        
        # Low-Close Range
        df['lc_range'] = df['low'] - df['close'].shift(1)
        
        # Body size (absolute and relative)
        df['body_size'] = np.abs(df['close'] - df['open'])
        df['body_size_rel'] = df['body_size'] / (df['high'] - df['low'])
        
        # Upper and lower shadows
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> None:
        """
        Calculate various moving average indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
        """
        # Simple Moving Averages
        df['sma_fast'] = ta.sma(df['close'], length=self.default_feature_params['sma_fast_period'])
        df['sma_slow'] = ta.sma(df['close'], length=self.default_feature_params['sma_slow_period'])
        
        # Exponential Moving Averages
        df['ema_fast'] = ta.ema(df['close'], length=self.default_feature_params['ema_fast_period'])
        df['ema_slow'] = ta.ema(df['close'], length=self.default_feature_params['ema_slow_period'])
        
        # Moving Average Crossover
        df['sma_cross'] = df['sma_fast'] - df['sma_slow']
        df['ema_cross'] = df['ema_fast'] - df['ema_slow']
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> None:
        """
        Calculate momentum indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
        """
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=self.default_feature_params['rsi_period'])
        
        # Stochastic Oscillator
        stoch = ta.stoch(
            df['high'], 
            df['low'], 
            df['close'], 
            k=self.default_feature_params['stoch_k_period'], 
            d=self.default_feature_params['stoch_d_period'], 
            smooth_k=3
        )
        # pandas-ta returns a DataFrame, we need to add the columns to our main df
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
        
        # MACD
        macd = ta.macd(
            df['close'], 
            fast=self.default_feature_params['macd_fast_period'], 
            slow=self.default_feature_params['macd_slow_period'], 
            signal=self.default_feature_params['macd_signal_period']
        )
        # Add MACD columns to main df
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_histogram'] = macd['MACDh_12_26_9']
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> None:
        """
        Calculate volatility indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
        """
        # Bollinger Bands
        bbands = ta.bbands(
            df['close'], 
            length=self.default_feature_params['bbands_period'], 
            std=self.default_feature_params['bbands_std_dev']
        )
        # Add BBands columns to main df
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_middle'] = bbands['BBM_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # ATR (Average True Range)
        df['atr'] = ta.atr(
            df['high'], 
            df['low'], 
            df['close'], 
            length=self.default_feature_params['atr_period']
        )
        
        # ATR Percentage (ATR relative to close price)
        df['atr_pct'] = df['atr'] / df['close'] * 100
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> None:
        """
        Calculate volume-based indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
        """
        # On-Balance Volume
        df['obv'] = ta.obv(df['close'], df['volume'])
        
        # Volume SMA
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        
        # Volume relative to its moving average
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Money Flow Index
        df['mfi'] = ta.mfi(
            df['high'], 
            df['low'], 
            df['close'], 
            df['volume'], 
            length=14
        )
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> None:
        """
        Calculate trend indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data.
        """
        # ADX (Average Directional Index)
        adx = ta.adx(
            df['high'], 
            df['low'], 
            df['close'], 
            length=self.default_feature_params['adx_period']
        )
        # Add ADX columns to main df
        df['adx'] = adx['ADX_14']
        df['di_plus'] = adx['DMP_14']
        df['di_minus'] = adx['DMN_14']
        
        try:
            # Try Ichimoku Cloud with new API
            ichimoku = ta.ichimoku(
                df['high'],
                df['low'],
                df['close'],
                tenkan=self.default_feature_params['ichimoku_tenkan_period'],
                kijun=self.default_feature_params['ichimoku_kijun_period'],
                senkou=self.default_feature_params['ichimoku_senkou_b_period']
            )
            
            # Check if ichimoku is a tuple (newer pandas_ta versions) or DataFrame (older versions)
            if isinstance(ichimoku, tuple):
                ichi_df = ichimoku[0]  # First element contains the main indicators
                # Extract columns using numerical indices if needed
                if 'ITS_9' in ichi_df.columns:
                    df['tenkan_sen'] = ichi_df['ITS_9']
                    df['kijun_sen'] = ichi_df['IKS_26']
                    df['senkou_span_a'] = ichi_df['ISA_9_26']
                    df['senkou_span_b'] = ichi_df['ISB_26_52']
                else:
                    # Use the first few columns as a fallback
                    df['tenkan_sen'] = ichi_df.iloc[:, 0]
                    df['kijun_sen'] = ichi_df.iloc[:, 1]
                    df['senkou_span_a'] = ichi_df.iloc[:, 2]
                    df['senkou_span_b'] = ichi_df.iloc[:, 3]
            else:
                # Older pandas_ta versions return a DataFrame
                df['tenkan_sen'] = ichimoku['ITS_9']
                df['kijun_sen'] = ichimoku['IKS_26']
                df['senkou_span_a'] = ichimoku['ISA_9_26']
                df['senkou_span_b'] = ichimoku['ISB_26_52']
                
        except Exception as e:
            # If Ichimoku fails, use simple moving averages as fallback
            print(f"Error calculating Ichimoku: {e}")
            df['tenkan_sen'] = ta.sma(df['close'], length=9)
            df['kijun_sen'] = ta.sma(df['close'], length=26)
            df['senkou_span_a'] = (df['tenkan_sen'] + df['kijun_sen']) / 2
            df['senkou_span_b'] = ta.sma(df['close'], length=52)
    
    def _select_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select the 20 final features as specified in the design document.
        
        Args:
            df (pd.DataFrame): DataFrame with all calculated features.
            
        Returns:
            pd.DataFrame: DataFrame with only the selected features.
        """
        # List of the 20 features to keep according to the technical design document
        selected_cols = [
            # OHLCV Base (5)
            'open', 'high', 'low', 'close', 'volume',
            
            # Price-derived features (3)
            'log_return',          # Log return
            'hl_range',            # High-Low range
            'body_size_rel',       # Relative body size
            
            # Technical indicators (12)
            'atr',                 # Average True Range
            'rsi',                 # Relative Strength Index
            'macd',                # MACD line
            'macd_signal',         # MACD signal line
            'macd_histogram',      # MACD Histogram
            'bb_width',            # Bollinger Band Width
            'sma_cross',           # SMA Crossover (fast - slow)
            'stoch_k',             # Stochastic %K
            'adx',                 # Average Directional Index
            'volume_ratio',        # Volume / Volume SMA
            'mfi',                 # Money Flow Index
            'obv',                 # On-Balance Volume
        ]
        
        # Ensure all columns exist
        missing_cols = [col for col in selected_cols if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Some selected features are missing: {missing_cols}")
            selected_cols = [col for col in selected_cols if col in df.columns]
        
        # Return only the selected columns
        return df[selected_cols]