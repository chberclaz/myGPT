"""
Example script demonstrating how to save and load models, and use the training and generation functions.
"""

import torch
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.models import GPT
from src.utils.io import save_checkpoint, load_checkpoint
from src.train import train
from src.generate import generate_text

def save_model_example():
    """Example of saving a model after training."""
    # Create configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Train the model
    model, optimizer, best_val_loss = train(
        model_config=model_config,
        training_config=training_config,
        # Optional parameters:
        # train_data=your_train_data,
        # val_data=your_val_data,
        # device='cuda',
        # ddp=False,
        # wandb_run=your_wandb_run
    )
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    # Save the model
    save_checkpoint(model, optimizer, training_config)
    print(f"Model saved to {training_config.out_dir}/ckpt.pt")

def load_model_example():
    """Example of loading a saved model and using it for generation."""
    # Create configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Load the checkpoint
    checkpoint = load_checkpoint(training_config)
    
    # Create model and optimizer
    model = GPT(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    
    # Restore model and optimizer state
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    print(f"Model loaded from {training_config.out_dir}/ckpt.pt")
    
    # Example of using the loaded model for generation
    prompt = "Once upon a time"
    generated_text = generate_text(
        prompt=prompt,
        model=model,
        model_config=model_config,
        training_config=training_config,
        max_tokens=100,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.0,
        stop_tokens=['.', '!', '?']
    )
    
    print(f"\nPrompt: {prompt}")
    print(f"Generated text: {generated_text}")

def main():
    """Run the examples."""
    print("=== Training and Saving Model Example ===")
    save_model_example()
    
    print("\n=== Loading Model and Generation Example ===")
    load_model_example()

if __name__ == '__main__':
    main() 