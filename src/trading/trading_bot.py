"""
Bitcoin trading bot using GPT predictions and technical analysis.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime

from ..models import GPT
from ..config import ModelConfig, TrainingConfig
from ..data.price_tokenizer import PriceTokenizer
from .indicators import get_all_indicators, TradingIndicators
from .data_fetcher import BitcoinDataFetcher

@dataclass
class TradingSignal:
    """Trading signal with confidence and supporting data."""
    action: str  # 'buy', 'sell', or 'hold'
    confidence: float  # 0 to 1
    price: float
    timestamp: datetime
    gpt_prediction: float
    indicators: TradingIndicators
    reason: str

class BitcoinTradingBot:
    """Trading bot that combines GPT predictions with technical analysis."""
    
    def __init__(
        self,
        model: GPT,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        price_tokenizer: PriceTokenizer,
        sequence_length: int = 168,  # 1 week of hourly data
        prediction_horizon: int = 24,  # Predict 24 hours ahead
    ):
        """
        Initialize the trading bot.
        
        Args:
            model: Trained GPT model
            model_config: Model configuration
            training_config: Training configuration
            price_tokenizer: Price tokenizer instance
            sequence_length: Number of time steps to use for prediction
            prediction_horizon: Number of time steps to predict ahead
        """
        self.model = model
        self.model_config = model_config
        self.training_config = training_config
        self.tokenizer = price_tokenizer
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        self.data_fetcher = BitcoinDataFetcher()
        self.device = next(model.parameters()).device
        
    def get_gpt_prediction(self, prices: List[float], volumes: List[float]) -> float:
        """Get price prediction from the GPT model."""
        # Prepare input sequence
        sequence = prices[-self.sequence_length:]
        volumes = volumes[-self.sequence_length:]
        
        # Get technical indicators
        indicators = get_all_indicators(sequence)
        indicator_values = [[
            ind.rsi/100,  # Normalize RSI
            ind.macd,
            ind.macd_signal,
            ind.macd_hist,
            (ind.sma_20 - sequence[-1])/sequence[-1],  # Normalize to percentage
            (ind.sma_50 - sequence[-1])/sequence[-1],
            (ind.upper_band - sequence[-1])/sequence[-1],
            (ind.lower_band - sequence[-1])/sequence[-1],
            ind.atr/sequence[-1]  # Normalize ATR
        ] for ind in indicators]
        
        # Encode sequence with features
        encoded = self.tokenizer.encode_with_features(
            prices=sequence,
            volumes=volumes,
            indicators=indicator_values
        )
        
        # Add batch dimension and move to device
        encoded = encoded.unsqueeze(0).to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            logits, _ = self.model(encoded)
            next_token = torch.argmax(logits[0, -1, :]).item()
        
        # Convert token to price
        predicted_price = self.tokenizer.token_to_price(next_token)
        return predicted_price
    
    def analyze_technicals(self, 
                         current_price: float,
                         indicators: TradingIndicators) -> Tuple[str, float, str]:
        """
        Analyze technical indicators for trading signals.
        
        Returns:
            Tuple of (action, confidence, reason)
        """
        reasons = []
        buy_signals = 0
        sell_signals = 0
        
        # RSI signals
        if indicators.rsi < 30:
            buy_signals += 1
            reasons.append("RSI oversold")
        elif indicators.rsi > 70:
            sell_signals += 1
            reasons.append("RSI overbought")
        
        # MACD signals
        if indicators.macd > indicators.macd_signal:
            buy_signals += 1
            reasons.append("MACD bullish crossover")
        elif indicators.macd < indicators.macd_signal:
            sell_signals += 1
            reasons.append("MACD bearish crossover")
        
        # Bollinger Bands signals
        if current_price < indicators.lower_band:
            buy_signals += 1
            reasons.append("Price below lower Bollinger Band")
        elif current_price > indicators.upper_band:
            sell_signals += 1
            reasons.append("Price above upper Bollinger Band")
        
        # Moving Average signals
        if indicators.sma_20 > indicators.sma_50:
            buy_signals += 1
            reasons.append("20 SMA above 50 SMA")
        elif indicators.sma_20 < indicators.sma_50:
            sell_signals += 1
            reasons.append("20 SMA below 50 SMA")
        
        # Determine action and confidence
        total_signals = max(buy_signals + sell_signals, 1)
        if buy_signals > sell_signals:
            action = "buy"
            confidence = buy_signals / total_signals
        elif sell_signals > buy_signals:
            action = "sell"
            confidence = sell_signals / total_signals
        else:
            action = "hold"
            confidence = 0.5
        
        reason = " | ".join(reasons) if reasons else "No strong signals"
        return action, confidence, reason
    
    def get_trading_signal(self) -> TradingSignal:
        """
        Generate a trading signal based on current market conditions.
        
        Returns:
            TradingSignal object with action and supporting data
        """
        # Fetch recent data
        prices, volumes, highs, lows = self.data_fetcher.get_training_data(
            days=10,  # Get more data than needed to calculate indicators
            timeframe='1h'
        )
        
        current_price = prices[-1]
        current_time = datetime.now()
        
        # Get GPT prediction
        predicted_price = self.get_gpt_prediction(prices, volumes)
        
        # Get technical indicators
        indicators = get_all_indicators(prices, highs, lows)[-1]
        
        # Analyze technicals
        tech_action, tech_confidence, tech_reason = self.analyze_technicals(
            current_price,
            indicators
        )
        
        # Combine GPT prediction with technical analysis
        price_change = (predicted_price - current_price) / current_price
        
        if abs(price_change) < 0.01:  # Less than 1% change
            action = "hold"
            confidence = 0.5
            reason = f"GPT predicts minimal price change ({price_change:.2%})"
        else:
            # If GPT and technicals agree, increase confidence
            gpt_action = "buy" if price_change > 0 else "sell"
            if gpt_action == tech_action:
                action = gpt_action
                confidence = min(0.95, tech_confidence + 0.2)
                reason = f"GPT and technicals agree: {tech_reason}"
            else:
                # If they disagree, go with the stronger signal
                if abs(price_change) > 0.05:  # GPT predicts >5% change
                    action = gpt_action
                    confidence = 0.6
                    reason = f"Strong GPT signal: predicts {price_change:.2%} change"
                else:
                    action = tech_action
                    confidence = tech_confidence
                    reason = f"Following technical signals: {tech_reason}"
        
        return TradingSignal(
            action=action,
            confidence=confidence,
            price=current_price,
            timestamp=current_time,
            gpt_prediction=predicted_price,
            indicators=indicators,
            reason=reason
        ) 