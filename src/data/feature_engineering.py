import pandas as pd
import numpy as np
import logging
import math
import pandas_ta as ta

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, indicators_config: dict, ohlcv_config: dict):
        self.ic = indicators_config
        self.oc = ohlcv_config
        
    def add_ohlcv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula características basadas en OHLCV:
        - log_ret_C_O: log(Close/Open)
        - log_ret_H_O: log(High/Open)
        - log_ret_L_O: log(Low/Open)
        - log_ret_C_C_prev: log(Close/Close_prev)
        - log_ret_Vol_SMAVol: log(Volume/SMA(Volume))
        """
        logger.debug("Calculando características OHLCV procesadas.")
        
        df_out = df.copy()
        
        # Logaritmo de retornos dentro de la vela
        df_out['log_ret_C_O'] = np.log(df_out['Close'] / df_out['Open'])
        df_out['log_ret_H_O'] = np.log(df_out['High'] / df_out['Open'])
        df_out['log_ret_L_O'] = np.log(df_out['Low'] / df_out['Open'])
        
        # Logaritmo de retorno de cierre a cierre
        df_out['log_ret_C_C_prev'] = np.log(df_out['Close'] / df_out['Close'].shift(1))
        
        # SMA del volumen
        volume_sma_period = self.oc.get('volume_sma_period', 20)
        df_out['Volume_SMA'] = df_out['Volume'].rolling(window=volume_sma_period).mean()
        # Evitar división por cero
        df_out['Volume_SMA'] = df_out['Volume_SMA'].replace(0, 1e-9)
        # Logaritmo de la relación volumen / SMA volumen
        df_out['log_ret_Vol_SMAVol'] = np.log(df_out['Volume'] / df_out['Volume_SMA'])
        
        return df_out
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores técnicos basados en el DataFrame OHLCV usando pandas_ta
        """
        logger.debug("Calculando indicadores técnicos con pandas_ta.")
        df_out = df.copy()
        self._add_pandas_ta_indicators(df_out)
        return df_out
    
    def _add_talib_indicators(self, df_out: pd.DataFrame) -> None:
        """
        Implementa indicadores técnicos usando TA-Lib
        """
        import talib
        
        # SMAs
        df_out['SMA_short'] = talib.SMA(df_out['Close'], timeperiod=self.ic['sma_short_period'])
        df_out['SMA_long'] = talib.SMA(df_out['Close'], timeperiod=self.ic['sma_long_period'])
        
        # EMAs
        df_out['EMA_short'] = talib.EMA(df_out['Close'], timeperiod=self.ic['ema_short_period'])
        df_out['EMA_long'] = talib.EMA(df_out['Close'], timeperiod=self.ic['ema_long_period'])
        
        # RSI
        df_out['RSI'] = talib.RSI(df_out['Close'], timeperiod=self.ic['rsi_period'])
        
        # ATR
        df_out['ATR'] = talib.ATR(df_out['High'], df_out['Low'], df_out['Close'], 
                                timeperiod=self.ic['atr_period'])
        
        # MACD
        macd, macdsignal, macdhist = talib.MACD(
            df_out['Close'],
            fastperiod=self.ic['macd_fast_period'],
            slowperiod=self.ic['macd_slow_period'],
            signalperiod=self.ic['macd_signal_period']
        )
        df_out['MACD_line'] = macd
        df_out['MACD_signal'] = macdsignal
        df_out['MACD_hist'] = macdhist
        
        # Bandas de Bollinger
        upper, middle, lower = talib.BBANDS(
            df_out['Close'],
            timeperiod=self.ic['bollinger_period'],
            nbdevup=self.ic['bollinger_std_dev'],
            nbdevdn=self.ic['bollinger_std_dev'],
            matype=0  # 0 para SMA
        )
        df_out['BB_upper'] = upper
        df_out['BB_middle'] = middle  # Es la SMA(periodo_bollinger)
        df_out['BB_lower'] = lower
        df_out['BB_width'] = upper - lower  # Ancho absoluto
        
        # CCI
        df_out['CCI'] = talib.CCI(
            df_out['High'], 
            df_out['Low'], 
            df_out['Close'], 
            timeperiod=self.ic['cci_period']
        )
        
        # Stochastic Oscillator (%K lento, %D)
        # TA-Lib STOCH: fastk_period, slowk_period (slowing), slowd_period (smoothing de slowk)
        slowk, slowd = talib.STOCH(
            df_out['High'], 
            df_out['Low'], 
            df_out['Close'],
            fastk_period=self.ic['stochastic_k_period'],
            slowk_period=self.ic['stochastic_slowing_period'],
            slowk_matype=0,  # SMA para cálculo de %K lento
            slowd_period=self.ic['stochastic_d_period'],
            slowd_matype=0  # SMA para suavizar %K lento (que es %D)
        )
        df_out['STOCH_slowk'] = slowk
        df_out['STOCH_slowd'] = slowd
            
    def _add_pandas_ta_indicators(self, df_out: pd.DataFrame) -> None:
        """
        Implementa indicadores técnicos usando pandas_ta
        """
        import pandas_ta as ta
        
        # SMAs
        df_out['SMA_short'] = ta.sma(df_out['Close'], length=self.ic['sma_short_period'])
        df_out['SMA_long'] = ta.sma(df_out['Close'], length=self.ic['sma_long_period'])
        
        # EMAs
        df_out['EMA_short'] = ta.ema(df_out['Close'], length=self.ic['ema_short_period'])
        df_out['EMA_long'] = ta.ema(df_out['Close'], length=self.ic['ema_long_period'])
        
        # RSI
        df_out['RSI'] = ta.rsi(df_out['Close'], length=self.ic['rsi_period'])
        
        # ATR
        atr = ta.atr(
            df_out['High'], 
            df_out['Low'], 
            df_out['Close'], 
            length=self.ic['atr_period']
        )
        df_out['ATR'] = atr
        
        # MACD
        macd_result = ta.macd(
            df_out['Close'],
            fast=self.ic['macd_fast_period'],
            slow=self.ic['macd_slow_period'],
            signal=self.ic['macd_signal_period']
        )
        df_out['MACD_line'] = macd_result['MACD_' + str(self.ic['macd_fast_period']) + '_' + 
                                        str(self.ic['macd_slow_period']) + '_' + 
                                        str(self.ic['macd_signal_period'])]
        df_out['MACD_signal'] = macd_result['MACDs_' + str(self.ic['macd_fast_period']) + '_' + 
                                            str(self.ic['macd_slow_period']) + '_' + 
                                            str(self.ic['macd_signal_period'])]
        df_out['MACD_hist'] = macd_result['MACDh_' + str(self.ic['macd_fast_period']) + '_' + 
                                        str(self.ic['macd_slow_period']) + '_' + 
                                        str(self.ic['macd_signal_period'])]
        
        # Bandas de Bollinger
        bb_result = ta.bbands(
            df_out['Close'],
            length=self.ic['bollinger_period'],
            std=self.ic['bollinger_std_dev']
        )
        df_out['BB_upper'] = bb_result['BBU_' + str(self.ic['bollinger_period']) + '_' + 
                                    str(self.ic['bollinger_std_dev']) + '.0']
        df_out['BB_middle'] = bb_result['BBM_' + str(self.ic['bollinger_period']) + '_' + 
                                    str(self.ic['bollinger_std_dev']) + '.0']
        df_out['BB_lower'] = bb_result['BBL_' + str(self.ic['bollinger_period']) + '_' + 
                                    str(self.ic['bollinger_std_dev']) + '.0']
        df_out['BB_width'] = df_out['BB_upper'] - df_out['BB_lower']  # Ancho absoluto
        
        # CCI
        df_out['CCI'] = ta.cci(
            df_out['High'], 
            df_out['Low'], 
            df_out['Close'], 
            length=self.ic['cci_period']
        )
        
        # Stochastic Oscillator (%K lento, %D)
        stoch_result = ta.stoch(
            df_out['High'], 
            df_out['Low'], 
            df_out['Close'],
            k=self.ic['stochastic_k_period'],
            d=self.ic['stochastic_d_period'],
            smooth_k=self.ic['stochastic_slowing_period']
        )
        df_out['STOCH_slowk'] = stoch_result['STOCHk_' + str(self.ic['stochastic_k_period']) + '_' + 
                                            str(self.ic['stochastic_d_period']) + '_' + 
                                            str(self.ic['stochastic_slowing_period'])]
        df_out['STOCH_slowd'] = stoch_result['STOCHd_' + str(self.ic['stochastic_k_period']) + '_' + 
                                            str(self.ic['stochastic_d_period']) + '_' + 
                                            str(self.ic['stochastic_slowing_period'])]
