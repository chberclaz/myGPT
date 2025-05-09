"""
Example script demonstrating how to work with a custom dataset:
1. Prepare and load custom data
2. Train the model on the custom dataset
3. Save the trained model
4. Load the model later and generate text
"""

import os
import torch
import numpy as np
from typing import Tuple, Optional
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.models import GPT
from src.train import Trainer
from src.generate import Generator
from src.data.dataloader import DataManager

def train_on_custom_data(model_config: ModelConfig, training_config: TrainingConfig):
    """Example of training the model on a custom dataset."""
    
    # Initialize data manager
    data_manager = DataManager(training_config)
    
    # Prepare or load custom data
    try:
        print("Attempting to load existing data...")
        train_data, val_data = data_manager.load_data()
    except FileNotFoundError:
        print("No existing data found. Preparing new data...")
        train_data, val_data = data_manager.prepare_data(os.path.join(training_config.data_dir, training_config.input_file))
        # Save the prepared data
        data_manager.save_data()
    
    print(f"Dataset loaded. Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Create trainer with custom data
    trainer = Trainer(
        model_config=model_config,
        training_config=training_config,
        train_data=train_data,
        val_data=val_data
    )
    
    # Train the model
    print("Starting training...")
    model, optimizer, best_val_loss = trainer.train()
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Save the model
    print("Saving model...")
    model.save(optimizer, training_config)
    print(f"model parameters trained & saved: {model.parameters()}")
    print(f"Model saved to {training_config.out_dir}/ckpt.pt")
    
    return model, model_config, training_config

def main():
    """Run the complete example workflow."""
    # Create configurations (must match training configuration)
    model_config = ModelConfig(
        vocab_size=50257,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dataset='custom'
    )
    
    training_config = TrainingConfig(
        batch_size=16,
        learning_rate=3e-4,
        max_iters=8000,
        eval_interval=1000,
        eval_iters=800,
        warmup_iters=100,
        lr_decay_iters=5000,
        min_lr=3e-5,
        device='cuda' if torch.cuda.is_available() else 'cpu',  # Set device in config
        dataset='custom',
        data_dir='data/custom/',
        input_file='input_dante.txt'
    )

    # Check if model checkpoint exists
    checkpoint_path = os.path.join(training_config.out_dir, 'ckpt.pt')
    if os.path.exists(checkpoint_path):
        print("=== Loading Existing Model ===")
        # Load the model from checkpoint
        model, optimizer, model_config, training_config = GPT.load(model_config, training_config)
        print(f"Model loaded from {checkpoint_path}")
        
        # Load the training data
        data_manager = DataManager(training_config)
        train_data, val_data = data_manager.load_data()
        training_config.max_iters = 2000
        
        trainer = Trainer(
            model=model,
            model_config=model_config,
            training_config=training_config,
            train_data=train_data,
            val_data=val_data
        )
        
        # Train the model
        print("=== Starting training... ===")
        #model, optimizer, best_val_loss = trainer.train()
        #print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        print("=== Saving model... ===")
        #model.save(optimizer, training_config)
        print(f"model parameters trained & saved: {model.parameters()}")
        print(f"Model saved to {training_config.out_dir}/ckpt.pt")
    else:
        print("=== Training New Model ===")
        # Train and save a new model
        model, model_config, training_config = train_on_custom_data(model_config, training_config)
    
    print(f"model parameters trained before generating: {model.parameters()}")
    # Create generator with the model (whether loaded or newly trained)
    print("\n=== Creating Generator with Model ===")
    generator = Generator(
        model=model,
        model_config=model_config,
        training_config=training_config
    )
    
    # Generate text with different prompts and settings
    print("\n=== Generating Text ===")
    
    # Example 1: Basic generation
    prompt1 = "Ich bin"
    print(f"\nGenerating text from prompt: {prompt1}")
    generated_text1 = generator.generate(
        prompt=prompt1,
        max_tokens=200,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.0,
        stop_tokens=['.']
    )
    print(f"Generated text: {generated_text1}")
    
    # Example 2: More creative generation
    prompt2 = "So wie von dir erkannt"
    print(f"\nGenerating text with different settings from prompt: {prompt2}")
    generated_text2 = generator.generate(
        prompt=prompt2,
        max_tokens=300,
        temperature=0.9,  # Higher temperature for more creativity
        top_k=50,         # Consider more tokens
        top_p=0.95,       # Higher cumulative probability threshold
        repetition_penalty=1.2,  # Stronger repetition penalty
        stop_tokens=['.']  # Also stop at newlines
    )
    print(f"Generated text: {generated_text2}")

if __name__ == '__main__':
    main()