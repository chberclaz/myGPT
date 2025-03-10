"""
Script demonstrating how to save and load a trained GPT model.
"""

import os
import torch
from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from models import GPT
from utils.io import save_checkpoint, load_checkpoint
from utils.distributed import get_optimizer

def save_model_example():
    """Example of saving a trained model."""
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Create model and optimizer
    model = GPT(model_config)
    model.to(training_config.device)
    optimizer = get_optimizer(model, training_config)
    
    # Example: Train the model (you would replace this with your actual training loop)
    print("Training model...")
    # ... your training code here ...
    
    # Save the model
    print(f"Saving model to {training_config.out_dir}")
    save_checkpoint(model, optimizer, training_config)
    print("Model saved successfully!")

def load_model_example():
    """Example of loading a saved model."""
    # Load configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Create model and optimizer
    model = GPT(model_config)
    model.to(training_config.device)
    optimizer = get_optimizer(model, training_config)
    
    # Load the checkpoint
    checkpoint_path = os.path.join(training_config.out_dir, 'ckpt.pt')
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = load_checkpoint(training_config)
        
        # Restore model and optimizer state
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Restore training state if needed
        training_config.iter_num = checkpoint.get('iter_num', 0)
        training_config.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print("Model loaded successfully!")
        print(f"Resuming from iteration {training_config.iter_num}")
        print(f"Best validation loss: {training_config.best_val_loss:.4f}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")

def main():
    """Main function demonstrating save and load functionality."""
    print("=== Model Save and Load Example ===")
    
    # Example 1: Save a model
    print("\n1. Saving a model:")
    save_model_example()
    
    # Example 2: Load a model
    print("\n2. Loading a model:")
    load_model_example()

if __name__ == '__main__':
    main() 