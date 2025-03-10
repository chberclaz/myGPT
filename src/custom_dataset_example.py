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
from typing import Tuple
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.models import GPT
from src.train import Trainer
from src.generate import Generator
from src.data.tokenizer import get_tokenizer

def prepare_custom_data(text_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare custom data from a text file.
    
    Args:
        text_file: Path to the text file
        
    Returns:
        Tuple of (train_data, val_data) tensors
    """
    # Read the text file
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Get the tokenizer
    encode, decode = get_tokenizer('custom')
    
    # Encode the text
    data = torch.tensor(encode(text), dtype=torch.long)
    
    # Split into train and validation sets
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data

def train_on_custom_data():
    """Example of training the model on a custom dataset."""
    # Create configurations
    model_config = ModelConfig(
        vocab_size=50257,  # Adjust based on your tokenizer
        block_size=256,    # Adjust based on your needs
        n_layer=6,         # Smaller model for faster training
        n_head=6,
        n_embd=384
    )
    
    training_config = TrainingConfig(
        batch_size=32,
        learning_rate=3e-4,
        max_iters=5000,    # Adjust based on your needs
        eval_interval=500,
        eval_iters=200,
        warmup_iters=100,
        lr_decay_iters=5000,
        min_lr=3e-5,
        out_dir='out/custom_model'
    )
    
    # Prepare custom data
    print("Loading custom dataset...")
    train_data, val_data = prepare_custom_data(
        text_file='data/custom/input_dante.txt'
    )
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
    print(f"Model saved to {training_config.out_dir}/ckpt.pt")
    
    return model, model_config, training_config

def main():
    """Run the complete example workflow."""
    # Create configurations (must match training configuration)
    model_config = ModelConfig(
        vocab_size=50304,
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384
    )
    
    training_config = TrainingConfig(
        batch_size=8,
        learning_rate=3e-4,
        max_iters=5000,
        eval_interval=500,
        eval_iters=200,
        warmup_iters=100,
        lr_decay_iters=5000,
        min_lr=3e-5,
        out_dir='out/custom_model'
    )
    
    # Check if model checkpoint exists
    checkpoint_path = os.path.join(training_config.out_dir, 'ckpt.pt')
    if os.path.exists(checkpoint_path):
        print("=== Loading Existing Model ===")
        # Load the model from checkpoint
        model, optimizer = GPT.load(model_config, training_config)
        print(f"Model loaded from {checkpoint_path}")
    else:
        print("=== Training New Model ===")
        # Train and save a new model
        model, model_config, training_config = train_on_custom_data()
    
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
        stop_tokens=['.', '!', '?']
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
        stop_tokens=['.', '!', '?', '\n']  # Also stop at newlines
    )
    print(f"Generated text: {generated_text2}")

if __name__ == '__main__':
    main()