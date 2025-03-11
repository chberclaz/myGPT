"""
Main training script for the GPT model.
"""

import os
import time
import torch
import torch.nn as nn
from typing import Tuple
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import math

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.data.dataloader import DataManager
from src.data.tokenizer import get_tokenizer
from src.models import GPT
from src.utils.io import load_checkpoint, save_checkpoint
from src.utils.distributed import setup_distributed, cleanup_distributed

def get_lr(config: TrainingConfig, it: int) -> float:
    """
    Calculate learning rate based on iteration number.
    
    Args:
        config: Training configuration
        it: Current iteration number
        
    Returns:
        Learning rate for the current iteration
    """
    # Linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # Cosine learning rate decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

class Trainer:
    """Class for training GPT models."""
    
    def __init__(
        self,
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
        Initialize the trainer.
        
        Args:
            model_config: Model configuration. If None, loads from config/model_config.py
            training_config: Training configuration. If None, loads from config/training_config.py
            train_data: Training dataset. If None, loads from data directory
            val_data: Validation dataset. If None, loads from data directory
            model: Pre-trained GPT model. If None, creates new model
            optimizer: Optimizer. If None, creates new optimizer
            device: Device to train on. If None, uses CUDA if available
            rank: Process rank for distributed training
            world_size: Total number of processes for distributed training
            ddp: Whether to use DistributedDataParallel
            wandb_run: Optional wandb run for logging
        """
        # Load configurations if not provided
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_config.device = self.device  # Ensure device is set in config
        self.rank = rank
        self.world_size = world_size
        self.ddp = ddp
        self.wandb_run = wandb_run
        
        # Prepare data if not provided
        if train_data is None or val_data is None:
            data_manager = DataManager(self.training_config)
            try:
                # Try to load existing data
                self.train_data, self.val_data = data_manager.load_data()
            except FileNotFoundError:
                # If no existing data, prepare new data
                self.train_data, self.val_data = data_manager.prepare_data()
                data_manager.save_data()
        else:
            # Move provided data to device if needed
            self.train_data = train_data.to(self.device) if isinstance(train_data, torch.Tensor) else train_data
            self.val_data = val_data.to(self.device) if isinstance(val_data, torch.Tensor) else val_data
        
        # Initialize model if not provided
        if model is None:
            self.model = GPT(self.model_config)
            self.model = self.model.to(self.device)
            if ddp:
                self.model = DDP(self.model, device_ids=[rank])
        else:
            self.model = model.to(self.device)
            if ddp:
                self.model = DDP(self.model, device_ids=[rank])
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = GPT.create_optimizer(self.model, self.training_config)
        else:
            self.optimizer = optimizer
        
        # Load checkpoint if exists
        self.best_val_loss = float('inf')
        self.start_iter = 0
        """   if os.path.exists(os.path.join(self.training_config.out_dir, 'ckpt.pt')):
            print(f"from Trainer")
            checkpoint = load_checkpoint(self.training_config)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_iter = checkpoint['iter_num']
            self.best_val_loss = checkpoint['best_val_loss']
        """
    def get_batch(self, data: Dataset, batch_size: int, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of data from the dataset.
        
        Args:
            data: Dataset to get batch from
            batch_size: Size of the batch
            block_size: Size of each sequence in the batch
            
        Returns:
            Tuple of (input tensor, target tensor)
        """
        # Move data to device if it's not already there
        if isinstance(data, torch.Tensor) and data.device != self.device:
            data = data.to(self.device)
            
        # Generate a small batch of data of inputs x and targets y
        ix = torch.randint(len(data) - block_size - 1, (batch_size,), device=self.device)  # -1 to account for target shift
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+1+block_size] for i in ix])  # Shifted by 1 but same length as x
        
        return x, y  # No need to move to device since data is already there
    
    def estimate_loss(self, model: GPT, train_data: Dataset, val_data: Dataset) -> dict:
        """
        Estimate the loss on train and validation sets.
        
        Args:
            model: Model to evaluate
            train_data: Training dataset
            val_data: Validation dataset
            
        Returns:
            Dictionary containing train and validation losses
        """
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.training_config.eval_iters, device=self.device)
            for k in range(self.training_config.eval_iters):
                X, Y = self.get_batch(
                    train_data if split == 'train' else val_data,
                    self.training_config.batch_size,
                    self.training_config.block_size
                )
                with torch.no_grad():
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    def train(self) -> Tuple[GPT, torch.optim.Optimizer, float]:
        """
        Train the model.
        
        Returns:
            Tuple of (model, optimizer, best_val_loss)
        """
        # Ensure model is on the correct device
        self.model = self.model.to(self.device)
        self.model.train()
        
        # Get initial batch
        X, Y = self.get_batch(self.train_data, self.training_config.batch_size, self.training_config.block_size)
        t0 = time.time()
        local_iter_num = 0
        running_mfu = -1.0
        
        # Calculate total iterations
        total_iters = self.training_config.max_iters if self.training_config.max_iters is not None else float('inf')
        print(f"total_iters: {total_iters}")
        for local_iter_num in range(total_iters):
            # Determine and set the learning rate for this iteration
            #lr = get_lr(self.training_config, self.start_iter + local_iter_num) if self.training_config.lr_decay_iters else self.training_config.learning_rate
            #for param_group in self.optimizer.param_groups:
            #    param_group['lr'] = lr
            
            # Evaluate the loss on train/val sets and write checkpoints
            if local_iter_num % self.training_config.eval_interval == 0:
                losses = self.estimate_loss(self.model, self.train_data, self.val_data)
                print(f"iter {local_iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if self.wandb_run:
                    self.wandb_run.log({
                        'train/loss': losses['train'],
                        'val/loss': losses['val'],
                        #'lr': lr,
                        'mfu': running_mfu*100, # convert to percentage
                    })
                if losses['val'] < self.best_val_loss or self.training_config.always_save_checkpoint:
                    self.best_val_loss = losses['val']
                    if local_iter_num > 0:
                        self.model.save(
                            optimizer=self.optimizer,
                            training_config=self.training_config,
                            iter_num=self.start_iter + local_iter_num,
                            best_val_loss=self.best_val_loss
                        )
            
            # Forward backward update, with optional gradient accumulation to simulate larger batch size
            for micro_step in range(self.training_config.gradient_accumulation_steps):
                # Forward pass
                logits, loss = self.model(X, Y)
                loss = loss / self.training_config.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                loss.backward()
                if self.training_config.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()  # Reset gradients after each step
                
                # Get next batch
                X, Y = self.get_batch(self.train_data, self.training_config.batch_size, self.training_config.block_size)
            
            # Timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if local_iter_num % self.training_config.log_interval == 0:
                lossf = loss.item() * self.training_config.gradient_accumulation_steps
                print(f"iter {local_iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        
        # Clean up distributed training
        if self.ddp:
            cleanup_distributed()
        
        return self.model, self.optimizer, self.best_val_loss

def main():
    """Example usage of the Trainer class."""
    # Create configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Create trainer
    trainer = Trainer(
        model_config=model_config,
        training_config=training_config
    )
    
    # Train the model
    model, optimizer, best_val_loss = trainer.train()
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main() 