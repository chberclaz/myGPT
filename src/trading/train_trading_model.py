"""
Script to train the GPT model on historical Bitcoin price data.
"""

import torch
from pathlib import Path
import logging
from datetime import datetime

from ..models import GPT
from ..config import ModelConfig, TrainingConfig
from ..data.price_tokenizer import PriceTokenizer
from .data_fetcher import BitcoinDataFetcher
from .indicators import get_all_indicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_training_data(
    data_fetcher: BitcoinDataFetcher,
    price_tokenizer: PriceTokenizer,
    sequence_length: int,
    prediction_horizon: int,
    days: int = 365  # 1 year of data by default
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare training data from historical Bitcoin prices.
    
    Returns:
        Tuple of (x, y) tensors for training
    """
    logger.info(f"Fetching {days} days of historical data...")
    prices, volumes, highs, lows = data_fetcher.get_training_data(
        days=days,
        timeframe='1h'
    )
    
    # Calculate technical indicators
    logger.info("Calculating technical indicators...")
    indicators = get_all_indicators(prices, highs, lows)
    
    # Prepare sequences
    sequences = []
    targets = []
    
    for i in range(len(prices) - sequence_length - prediction_horizon):
        # Get price sequence and corresponding volume data
        price_seq = prices[i:i + sequence_length]
        volume_seq = volumes[i:i + sequence_length]
        
        # Get indicators for the sequence
        ind_seq = indicators[i:i + sequence_length]
        indicator_values = [[
            ind.rsi/100,
            ind.macd,
            ind.macd_signal,
            ind.macd_hist,
            (ind.sma_20 - price_seq[-1])/price_seq[-1],
            (ind.sma_50 - price_seq[-1])/price_seq[-1],
            (ind.upper_band - price_seq[-1])/price_seq[-1],
            (ind.lower_band - price_seq[-1])/price_seq[-1],
            ind.atr/price_seq[-1]
        ] for ind in ind_seq]
        
        # Encode sequence with features
        encoded = price_tokenizer.encode_with_features(
            prices=price_seq,
            volumes=volume_seq,
            indicators=indicator_values
        )
        
        # Get target prices
        target_prices = prices[i + sequence_length:i + sequence_length + prediction_horizon]
        target = price_tokenizer.encode(target_prices)
        
        sequences.append(encoded)
        targets.append(target)
    
    # Convert to tensors
    x = torch.stack(sequences)
    y = torch.stack(targets)
    
    return x, y

def train_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    price_tokenizer: PriceTokenizer,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir: str = 'checkpoints'
) -> GPT:
    """Train a new GPT model on Bitcoin price data."""
    logger.info(f"Training new model on {device}")
    
    # Initialize model
    model = GPT(model_config).to(device)
    
    # Initialize data fetcher
    data_fetcher = BitcoinDataFetcher()
    
    # Prepare training data
    x, y = prepare_training_data(
        data_fetcher=data_fetcher,
        price_tokenizer=price_tokenizer,
        sequence_length=training_config.sequence_length,
        prediction_horizon=training_config.prediction_horizon
    )
    
    # Split into train/val
    split_idx = int(0.9 * len(x))
    train_x, train_y = x[:split_idx].to(device), y[:split_idx].to(device)
    val_x, val_y = x[split_idx:].to(device), y[split_idx:].to(device)
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(
        weight_decay=training_config.weight_decay,
        learning_rate=training_config.learning_rate,
        betas=(training_config.beta1, training_config.beta2),
    )
    
    # Training loop
    best_val_loss = float('inf')
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    
    logger.info("Starting training...")
    for epoch in range(training_config.max_epochs):
        # Training
        model.train()
        total_loss = 0
        for i in range(0, len(train_x), training_config.batch_size):
            batch_x = train_x[i:i + training_config.batch_size]
            batch_y = train_y[i:i + training_config.batch_size]
            
            logits, loss = model(batch_x, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / (len(train_x) / training_config.batch_size)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_x), training_config.batch_size):
                batch_x = val_x[i:i + training_config.batch_size]
                batch_y = val_y[i:i + training_config.batch_size]
                
                _, loss = model(batch_x, batch_y)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / (len(val_x) / training_config.batch_size)
        
        logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        # Save checkpoint if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'model_config': model_config,
                'training_config': training_config,
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'timestamp': datetime.now().isoformat()
            }
            checkpoint_file = checkpoint_path / f"model_epoch{epoch}_val{avg_val_loss:.4f}.pt"
            torch.save(checkpoint, checkpoint_file)
            logger.info(f"Saved checkpoint to {checkpoint_file}")
    
    return model

if __name__ == "__main__":
    # Example usage
    model_config = ModelConfig(
        n_layer=8,
        n_head=8,
        n_embd=512,
        vocab_size=10000,  # Adjust based on price tokenizer
        block_size=168,    # 1 week of hourly data
        dropout=0.1
    )
    
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=3e-4,
        max_epochs=50,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        grad_clip=1.0,
        sequence_length=168,
        prediction_horizon=24
    )
    
    # Initialize price tokenizer
    price_tokenizer = PriceTokenizer(
        min_price=1000,    # Adjust based on historical range
        max_price=100000,  # Adjust based on historical range
        n_bins=10000       # Should match vocab_size
    )
    
    # Train model
    model = train_model(model_config, training_config, price_tokenizer) 