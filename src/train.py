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

def train():
    """Main training function."""
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Set up distributed training if enabled
    if training_config.distributed:
        setup_distributed(training_config)
    
    # Set random seed for reproducibility
    torch.manual_seed(training_config.seed)
    torch.cuda.manual_seed(training_config.seed)
    
    # Prepare data
    train_data, val_data = prepare_data(training_config)
    
    # Get tokenizer
    encode, decode = get_tokenizer(training_config.dataset)
    
    # Create model
    model = GPT(model_config)
    model.to(training_config.device)
    
    # Wrap model in DDP if using distributed training
    if training_config.distributed:
        model = DDP(model, device_ids=[training_config.process_rank])
    
    # Create optimizer
    optimizer = get_optimizer(model, training_config)
    
    # Load checkpoint if exists
    if os.path.exists(os.path.join(training_config.out_dir, 'ckpt.pt')):
        checkpoint = load_checkpoint(training_config.out_dir)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        training_config.iter_num = checkpoint['iter_num']
        training_config.best_val_loss = checkpoint['best_val_loss']
    
    # Training loop
    X, Y = get_batch(train_data, training_config)
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    
    while True:
        # Determine and set the learning rate for this iteration
        lr = training_config.get_lr(training_config.iter_num) if training_config.decay_lr else training_config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate the loss on train/val sets and write checkpoints
        if training_config.iter_num % training_config.eval_interval == 0 and training_config.process_rank == 0:
            losses = estimate_loss(model, train_data, val_data, training_config)
            print(f"step {training_config.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < training_config.best_val_loss or training_config.always_save_checkpoint:
                training_config.best_val_loss = losses['val']
                if training_config.iter_num > 0:
                    save_checkpoint(model, optimizer, training_config)
        if training_config.iter_num == 0 and training_config.eval_only:
            break
        
        # Forward backward update, with optional gradient clipping, logging, etc
        for micro_step in range(training_config.gradient_accumulation_steps):
            if training_config.distributed:
                # In DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == training_config.gradient_accumulation_steps - 1)
            
            with torch.cuda.amp.autocast(enabled=training_config.fp16):
                logits, loss = model(X, Y)
                loss = loss / training_config.gradient_accumulation_steps
            if training_config.fp16:
                training_config.scaler.scale(loss).backward()
                if training_config.grad_clip != 0.0:
                    training_config.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
                training_config.scaler.step(optimizer)
                training_config.scaler.update()
            else:
                loss.backward()
                if training_config.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if training_config.iter_num % training_config.log_interval == 0 and training_config.process_rank == 0:
            lossf = loss.item() * training_config.gradient_accumulation_steps
            print(f"iter {training_config.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        training_config.iter_num += 1
        local_iter_num += 1
        
        # Check if we should stop training
        if training_config.iter_num > training_config.max_iters:
            break
    
    # Clean up distributed training
    if training_config.distributed:
        cleanup_distributed()

if __name__ == '__main__':
    train() 