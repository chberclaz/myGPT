"""
Training configuration for GPT-2.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class TrainingConfig:
    """Configuration for training the GPT-2 model."""
    
    # Training parameters
    batch_size: int = 64
    block_size: int = 256
    learning_rate: float = 1e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    
    # Optimization parameters
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    
    # Learning rate schedule
    warmup_iters: int = 2000
    lr_decay_iters: int = 5000
    min_lr: float = 1e-5
    
    # Distributed training
    distributed: bool = False
    zero_stage: int = 0
    process_rank: int = 0
    num_processes: int = 1
    
    # Device settings
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False
    
    # Data parameters
    dataset: str = "shakespeare"
    train_data_path: str = "data/shakespeare/input.txt"
    val_data_path: Optional[str] = None
    
    # Generation parameters
    temperature: float = 0.8
    top_k: int = 200
    max_new_tokens: int = 500
    num_samples: int = 10
    
    # Output settings
    out_dir: str = "out"
    save_interval: int = 1000 