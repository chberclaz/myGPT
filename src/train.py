"""
Main training script for the GPT model.
"""

import os
import time
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from data.dataloader import get_batch, estimate_loss, prepare_data
from data.tokenizer import get_tokenizer
from models import GPT
from utils.io import load_checkpoint, save_checkpoint
from utils.distributed import setup_distributed, cleanup_distributed, get_optimizer

def train(
    model_config: ModelConfig = None,
    training_config: TrainingConfig = None,
    train_data: Dataset = None,
    val_data: Dataset = None,
    model: GPT = None,
    optimizer: torch.optim.Optimizer = None,
    device: str = None,
    rank: int = 0,
    world_size: int = 1,
    ddp: bool = False,
    wandb_run = None
):
    """
    Train the GPT model.
    
    Args:
        model_config: Model configuration. If None, loads from config/model_config.py
        training_config: Training configuration. If None, loads from config/training_config.py
        train_data: Training dataset. If None, loads from config/data_config.py
        val_data: Validation dataset. If None, loads from config/data_config.py
        model: Pre-initialized model. If None, creates new model
        optimizer: Pre-initialized optimizer. If None, creates new optimizer
        device: Device to train on. If None, uses CUDA if available
        rank: Process rank for distributed training
        world_size: Total number of processes for distributed training
        ddp: Whether to use distributed training
        wandb_run: Weights & Biases run object for logging
    """
    # Load configurations if not provided
    if model_config is None:
        model_config = ModelConfig()
    if training_config is None:
        training_config = TrainingConfig()
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set up distributed training if enabled
    if ddp:
        setup_distributed(rank, world_size)
    
    # Prepare data if not provided
    if train_data is None or val_data is None:
        train_data, val_data = prepare_data(training_config)
    
    # Initialize model if not provided
    if model is None:
        model = GPT(model_config)
        model = model.to(device)
        if ddp:
            model = DDP(model, device_ids=[rank])
    
    # Create optimizer if not provided
    if optimizer is None:
        optimizer = get_optimizer(model, training_config)
    
    # Load checkpoint if exists
    best_val_loss = float('inf')
    start_iter = 0
    if os.path.exists(os.path.join(training_config.out_dir, 'ckpt.pt')):
        checkpoint = load_checkpoint(training_config)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    
    # Training loop
    model.train()
    X, Y = get_batch(train_data, training_config.batch_size, training_config.block_size, device)
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    
    while True:
        # Determine and set the learning rate for this iteration
        lr = get_lr(training_config, start_iter + local_iter_num) if training_config.decay_lr else training_config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate the loss on train/val sets and write checkpoints
        if local_iter_num % training_config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, training_config, device)
            print(f"step {local_iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_run:
                wandb_run.log({
                    'train/loss': losses['train'],
                    'val/loss': losses['val'],
                    'lr': lr,
                    'mfu': running_mfu*100, # convert to percentage
                })
            if losses['val'] < best_val_loss or training_config.always_save_checkpoint:
                best_val_loss = losses['val']
                if local_iter_num > 0:
                    save_checkpoint(model, optimizer, training_config)
        
        # Forward backward update, with optional gradient accumulation to simulate larger batch size
        for micro_step in range(training_config.gradient_accumulation_steps):
            # Forward pass
            logits, loss = model(X, Y)
            loss = loss / training_config.gradient_accumulation_steps # scale the loss to account for gradient accumulation
            loss.backward()
            if training_config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
            optimizer.step()
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if local_iter_num % training_config.log_interval == 0:
            lossf = loss.item() * training_config.gradient_accumulation_steps
            print(f"iter {local_iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if local_iter_num % training_config.log_interval == 0:
            # Get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * training_config.gradient_accumulation_steps
            print(f"iter {local_iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
            # Check if we should stop training
            if training_config.max_iters is not None and local_iter_num >= training_config.max_iters:
                break
    
    # Clean up distributed training
    if ddp:
        cleanup_distributed()
    
    return model, optimizer, best_val_loss

if __name__ == '__main__':
    # Example usage
    model_config = ModelConfig()
    training_config = TrainingConfig()
    model, optimizer, best_val_loss = train(model_config, training_config)
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}") 