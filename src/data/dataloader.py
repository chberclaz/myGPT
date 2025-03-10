"""
Data loading utilities.
"""

import os
import numpy as np
import torch
from typing import Tuple, Optional
from ..config.training_config import TrainingConfig

def prepare_data(config: TrainingConfig) -> torch.Tensor:
    """Prepare and load the dataset."""
    data_path = os.path.join('data', config.dataset, 'train.bin')
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run prepare.py first.")
        exit(1)
    
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    data = torch.from_numpy(data).type(torch.long)
    return data 