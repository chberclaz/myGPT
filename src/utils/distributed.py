"""
Distributed training utilities.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional
from ..config.training_config import TrainingConfig

def setup_distributed(config: TrainingConfig):
    """Set up distributed training."""
    if not config.distributed:
        return
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    # Set device based on rank
    config.device = f"cuda:{config.process_rank}"
    
    # Set up distributed model
    torch.cuda.set_device(config.device)

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Get optimizer with optional distributed settings."""
    # Start with all parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    
    # Filter out parameters that don't require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    # Create optimizer groups
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    # Create optimizer
    if config.distributed and config.zero_stage > 0:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(
            optim_groups,
            optimizer_class=torch.optim.AdamW,
            lr=config.learning_rate,
            betas=config.betas,
            eps=1e-8
        )
    else:
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=config.betas,
            eps=1e-8
        )
    
    return optimizer 