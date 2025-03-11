"""
Technical analysis indicators for trading.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class TradingIndicators:
    """Container for trading indicators."""
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    sma_20: float
    sma_50: float
    upper_band: float
    lower_band: float
    atr: float

def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: List of prices
        period: RSI period
        
    Returns:
        List of RSI values
    """
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].mean()
    down = -seed[seed < 0].mean()
    
    if down != 0:
        rs = up/down
    else:
        rs = 1
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100./(1.+rs)

    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(period-1) + upval)/period
        down = (down*(period-1) + downval)/period
        
        if down != 0:
            rs = up/down
        else:
            rs = 1
        rsi[i] = 100. - 100./(1.+rs)
    
    return rsi.tolist()

def calculate_macd(prices: List[float], 
                  fast_period: int = 12, 
                  slow_period: int = 26, 
                  signal_period: int = 9) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    prices = np.array(prices)
    exp1 = prices.ewm(span=fast_period, adjust=False).mean()
    exp2 = prices.ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return macd.tolist(), signal.tolist(), hist.tolist()

def calculate_bollinger_bands(prices: List[float], 
                            period: int = 20, 
                            num_std: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculate Bollinger Bands.
    
    Returns:
        Tuple of (Middle band, Upper band, Lower band)
    """
    prices = np.array(prices)
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    return middle_band.tolist(), upper_band.tolist(), lower_band.tolist()

def calculate_atr(high: List[float], 
                 low: List[float], 
                 close: List[float], 
                 period: int = 14) -> List[float]:
    """
    Calculate Average True Range.
    """
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    prev_close = np.roll(close, 1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr = np.zeros_like(tr)
    atr[period-1] = tr[:period].mean()
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr.tolist()

def get_all_indicators(prices: List[float], 
                      high: List[float] = None,
                      low: List[float] = None) -> List[TradingIndicators]:
    """
    Calculate all trading indicators.
    
    Args:
        prices: List of closing prices
        high: List of high prices (optional)
        low: List of low prices (optional)
        
    Returns:
        List of TradingIndicators objects
    """
    if high is None:
        high = prices
    if low is None:
        low = prices
        
    # Calculate all indicators
    rsi = calculate_rsi(prices)
    macd, macd_signal, macd_hist = calculate_macd(prices)
    sma_20, upper_band, lower_band = calculate_bollinger_bands(prices)
    sma_50 = np.array(prices).rolling(window=50).mean().tolist()
    atr = calculate_atr(high, low, prices)
    
    # Combine into indicator objects
    indicators = []
    for i in range(len(prices)):
        ind = TradingIndicators(
            rsi=rsi[i],
            macd=macd[i],
            macd_signal=macd_signal[i],
            macd_hist=macd_hist[i],
            sma_20=sma_20[i],
            sma_50=sma_50[i] if i >= 49 else prices[i],
            upper_band=upper_band[i],
            lower_band=lower_band[i],
            atr=atr[i]
        )
        indicators.append(ind)
    
    return indicators 