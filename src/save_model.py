"""
Example script demonstrating how to use the Trainer and Generator classes.
"""

import os
import torch
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.models import GPT
from src.train import Trainer
from src.generate import Generator

def train_and_save_example():
    """Example of training a model and saving it using the Trainer class."""
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
    
    # Save the model using the built-in method
    model.save(optimizer, training_config)

def load_and_generate_example():
    """Example of loading a saved model and using it for generation with the Generator class."""
    # Create configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Check if model checkpoint exists
    checkpoint_path = os.path.join(training_config.out_dir, 'ckpt.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Error: No model checkpoint found at {checkpoint_path}")
        print("Please run train_and_save_example() first to train and save a model.")
        return
    
    # Load the pretrained model
    model, optimizer = GPT.load(model_config, training_config)
    print(f"Successfully loaded model from {checkpoint_path}")
    
    # Create generator with the loaded model
    generator = Generator(
        model=model,
        model_config=model_config,
        training_config=training_config
    )
    
    # Example of using the loaded model for generation
    prompt = "Once upon a time"
    print(f"\nGenerating text from prompt: {prompt}")
    generated_text = generator.generate(
        prompt=prompt,
        max_tokens=100,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.0,
        stop_tokens=['.', '!', '?']
    )
    
    print(f"Generated text: {generated_text}")

def main():
    """Run the examples."""
    print("=== Training and Saving Model Example ===")
    train_and_save_example()
    
    print("\n=== Loading Model and Generation Example ===")
    load_and_generate_example()

if __name__ == '__main__':
    main() 