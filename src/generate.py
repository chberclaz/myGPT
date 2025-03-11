"""
Script for generating text using a trained GPT model.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.data.tokenizer import get_tokenizer
from src.models import GPT
from src.utils.io import load_checkpoint

class Generator:
    """Class for generating text using trained GPT models."""
    
    def __init__(
        self,
        model: GPT = None,
        model_config: ModelConfig = None,
        training_config: TrainingConfig = None,
        device: str = None
    ):
        """
        Initialize the generator.
        
        Args:
            model: Pre-trained GPT model. If None, loads from checkpoint
            model_config: Model configuration. If None, loads from config/model_config.py
            training_config: Training configuration. If None, loads from config/training_config.py
            device: Device to run generation on. If None, uses model's device or CUDA if available
        """
        # Load configurations if not provided
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        
        # Set device
        if device is not None:
            self.device = device
        elif model is not None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model if not provided
        if model is None:
            self.model, _ = GPT.load(self.model_config, self.training_config)
            self.model = self.model.to(self.device)
        else:
            self.model = model.to(self.device)
        
        print(f"Generator initialized with device: {self.device}")
        
        # Get tokenizer
        self.encode, self.decode = get_tokenizer(self.training_config)
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.01,
        top_k: int = 100,
        top_p: float = 0.5,
        repetition_penalty: float = 2.0,
        stop_tokens: List[str] = None
    ) -> str:
        """
        Generate text using the model.
        
        Args:
            prompt: Input text to generate from
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Number of top tokens to consider for sampling
            top_p: Cumulative probability threshold for sampling
            repetition_penalty: Penalty for repeating tokens
            stop_tokens: List of tokens to stop generation at
            
        Returns:
            Generated text as string
        """
        return self.model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
            device=self.device
        )

def main():
    """Example usage of the Generator class."""
    # Create configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Create generator
    generator = Generator(
        model_config=model_config,
        training_config=training_config
    )
    
    # Example prompt
    prompt = "Once upon a time"
    
    # Generate text with default settings
    generated_text = generator.generate(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")
    
    # Example with custom settings
    generated_text = generator.generate(
        prompt=prompt,
        max_tokens=200,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        stop_tokens=['.', '!', '?']
    )
    print(f"\nGenerated text with custom settings: {generated_text}")

if __name__ == '__main__':
    main() 