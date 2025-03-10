"""
Data loading utilities.
"""

import os
import numpy as np
import torch
from typing import Tuple, Optional
from ..config.training_config import TrainingConfig

def prepare_data(config: TrainingConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare and load the dataset.
    
    Args:
        config: Training configuration containing data paths and device
        
    Returns:
        Tuple of (train_data, val_data) tensors
    """
    data_path = os.path.join('data', config.dataset, 'train.bin')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run prepare.py first.")
        exit(1)
    
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    data = torch.from_numpy(data).type(torch.long)
    
    # Move data to device
    if hasattr(config, 'device'):
        data = data.to(config.device)
    
    # Split into train and validation sets
    n = int(0.9 * len(data))  # 90% for training
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data 