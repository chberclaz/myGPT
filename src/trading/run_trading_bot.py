"""
Script to run the Bitcoin trading bot with a trained model.
"""

import torch
import logging
from pathlib import Path
import json
from datetime import datetime
import time
import argparse

from ..models import GPT
from ..config import ModelConfig, TrainingConfig
from ..data.price_tokenizer import PriceTokenizer
from .trading_bot import BitcoinTradingBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger(__name__)

def load_model(checkpoint_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> tuple[GPT, ModelConfig, TrainingConfig]:
    """Load model and configs from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_config = checkpoint['model_config']
    training_config = checkpoint['training_config']
    
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model, model_config, training_config

def save_trading_signal(signal, output_dir: Path):
    """Save trading signal to JSON file."""
    signal_dict = {
        'action': signal.action,
        'confidence': signal.confidence,
        'price': signal.price,
        'timestamp': signal.timestamp.isoformat(),
        'gpt_prediction': signal.gpt_prediction,
        'indicators': {
            'rsi': signal.indicators.rsi,
            'macd': signal.indicators.macd,
            'macd_signal': signal.indicators.macd_signal,
            'macd_hist': signal.indicators.macd_hist,
            'sma_20': signal.indicators.sma_20,
            'sma_50': signal.indicators.sma_50,
            'upper_band': signal.indicators.upper_band,
            'lower_band': signal.indicators.lower_band,
            'atr': signal.indicators.atr
        },
        'reason': signal.reason
    }
    
    filename = output_dir / f"signal_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(signal_dict, f, indent=2)
    
    logger.info(f"Saved trading signal to {filename}")

def run_trading_bot(
    checkpoint_path: str,
    interval: int = 3600,  # 1 hour
    output_dir: str = 'trading_signals',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Run the trading bot continuously.
    
    Args:
        checkpoint_path: Path to model checkpoint
        interval: Time between predictions in seconds
        output_dir: Directory to save trading signals
        device: Device to run model on
    """
    # Load model and configs
    model, model_config, training_config = load_model(checkpoint_path, device)
    
    # Initialize price tokenizer
    price_tokenizer = PriceTokenizer(
        min_price=1000,
        max_price=100000,
        n_bins=model_config.vocab_size
    )
    
    # Initialize trading bot
    bot = BitcoinTradingBot(
        model=model,
        model_config=model_config,
        training_config=training_config,
        price_tokenizer=price_tokenizer
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"Starting trading bot with {interval}s interval")
    
    try:
        while True:
            try:
                # Get trading signal
                signal = bot.get_trading_signal()
                
                # Log signal
                logger.info(
                    f"Signal: {signal.action.upper()} with {signal.confidence:.2%} confidence | "
                    f"Current: ${signal.price:.2f} | "
                    f"Predicted: ${signal.gpt_prediction:.2f} | "
                    f"Reason: {signal.reason}"
                )
                
                # Save signal
                save_trading_signal(signal, output_path)
                
            except Exception as e:
                logger.error(f"Error getting trading signal: {str(e)}", exc_info=True)
            
            # Wait for next interval
            time.sleep(interval)
            
    except KeyboardInterrupt:
        logger.info("Stopping trading bot")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bitcoin trading bot")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Time between predictions in seconds (default: 3600)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="trading_signals",
        help="Directory to save trading signals (default: trading_signals)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help="Device to run model on (default: cuda if available, else cpu)"
    )
    
    args = parser.parse_args()
    
    run_trading_bot(
        checkpoint_path=args.checkpoint,
        interval=args.interval,
        output_dir=args.output_dir,
        device=args.device
    ) 