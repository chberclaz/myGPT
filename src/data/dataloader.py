"""
Data loading utilities.
"""

import os
import numpy as np
import torch
from typing import Tuple, Optional
from ..config.training_config import TrainingConfig

def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data for training."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def estimate_loss(model: torch.nn.Module, eval_data: torch.Tensor, eval_iters: int, 
                 batch_size: int, block_size: int, device: torch.device) -> float:
    """Estimate the loss on the evaluation data."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(eval_data, batch_size, block_size, device)
        with torch.no_grad():
            _, loss = model(x, y)
            losses[k] = loss.item()
    model.train()
    return losses.mean()

def prepare_data(config: TrainingConfig) -> torch.Tensor:
    """Prepare and load the dataset."""
    data_path = os.path.join('data', config.dataset, 'train.bin')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run prepare.py first.")
        exit(1)
    
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    data = torch.from_numpy(data).type(torch.long)
    return data 