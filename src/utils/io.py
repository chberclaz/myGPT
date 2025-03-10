"""
File I/O utilities.
"""

import os
import torch
from typing import Optional, Dict, Any
from ..config.training_config import TrainingConfig
from ..models.gpt import GPT

def load_checkpoint(config: TrainingConfig) -> Optional[Dict[str, Any]]:
    """Load model checkpoint if it exists."""
    checkpoint_path = os.path.join(config.out_dir, 'ckpt.pt')
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, map_location=config.device)
    return None

def save_checkpoint(model: GPT, optimizer: torch.optim.Optimizer, config: TrainingConfig):
    """Save model checkpoint."""
    os.makedirs(config.out_dir, exist_ok=True)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': config.__dict__,
    }
    torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))

def get_latest_checkpoint(config: TrainingConfig) -> Optional[str]:
    """Get the path to the latest checkpoint."""
    checkpoint_dir = config.out_dir
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        return None
    
    return os.path.join(checkpoint_dir, max(checkpoints)) 