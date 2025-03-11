"""
Bitcoin price data fetcher.
"""

import ccxt
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import time

class BitcoinDataFetcher:
    """Fetch Bitcoin price data from exchanges."""
    
    def __init__(self, exchange_id: str = 'binance'):
        """
        Initialize the data fetcher.
        
        Args:
            exchange_id: ID of the exchange to use (default: binance)
        """
        self.exchange = getattr(ccxt, exchange_id)()
        
    def fetch_historical_data(self, 
                            timeframe: str = '1h',
                            limit: int = 1000,
                            since: datetime = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            timeframe: Time interval ('1m', '5m', '1h', '1d', etc.)
            limit: Number of candles to fetch
            since: Start time (if None, fetches most recent data)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Convert since to timestamp if provided
        since_ts = int(since.timestamp() * 1000) if since else None
        
        # Fetch OHLCV data
        ohlcv = self.exchange.fetch_ohlcv(
            symbol='BTC/USDT',
            timeframe=timeframe,
            limit=limit,
            since=since_ts
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    
    def fetch_recent_trades(self, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch recent trades.
        
        Args:
            limit: Number of trades to fetch
            
        Returns:
            DataFrame with trade data
        """
        trades = self.exchange.fetch_trades(
            symbol='BTC/USDT',
            limit=limit
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    
    def get_training_data(self, 
                         days: int = 30,
                         timeframe: str = '1h') -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Get historical data formatted for training.
        
        Args:
            days: Number of days of historical data to fetch
            timeframe: Time interval for the data
            
        Returns:
            Tuple of (prices, volumes, highs, lows)
        """
        since = datetime.now() - timedelta(days=days)
        df = self.fetch_historical_data(
            timeframe=timeframe,
            since=since
        )
        
        return (
            df['close'].tolist(),
            df['volume'].tolist(),
            df['high'].tolist(),
            df['low'].tolist()
        ) 