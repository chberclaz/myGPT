"""
File I/O utilities.
"""

import os
import torch
from typing import Optional, Dict, Any
from ..config.training_config import TrainingConfig
from ..config.model_config import ModelConfig
from ..models.gpt import GPT

def load_checkpoint(config: TrainingConfig) -> Optional[Dict[str, Any]]:
    """Load model checkpoint if it exists."""
    checkpoint_path = os.path.join(config.out_dir, 'ckpt.pt')
    if os.path.exists(checkpoint_path):
        print(f"loading checkpoint from {checkpoint_path}")
        return torch.load(checkpoint_path, map_location=config.device)
    return None

def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    iter_num: int,
    best_val_loss: float
):
    """
    Save model checkpoint.
    
    Args:
        model: The GPT model to save
        optimizer: The optimizer used for training
        model_config: Model configuration
        training_config: Training configuration
        iter_num: Current iteration number
        best_val_loss: Best validation loss so far
    """
    os.makedirs(training_config.out_dir, exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_config': model_config.__dict__,
        'training_config': training_config.__dict__,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, os.path.join(training_config.out_dir, 'ckpt.pt'))

def get_latest_checkpoint(config: TrainingConfig) -> Optional[str]:
    """Get the path to the latest checkpoint."""
    checkpoint_dir = config.out_dir
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        return None
    
    return os.path.join(checkpoint_dir, max(checkpoints)) 