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
            device: Device to run generation on. If None, uses CUDA if available
        """
        # Load configurations if not provided
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model if not provided
        if model is None:
            self.model, _ = GPT.load(self.model_config, self.training_config)
            self.model = self.model.to(self.device)
        else:
            self.model = model
        
        # Get tokenizer
        self.encode, self.decode = get_tokenizer(self.model_config.dataset)
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
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
        # Encode the prompt
        context = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Generate tokens
        generated_tokens = []
        for _ in range(max_tokens):
            # Get model predictions
            with torch.no_grad():
                logits, _ = self.model(context)
                logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token in generated_tokens:
                    logits[:, token] /= repetition_penalty
            
            # Sample from the filtered distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for stop tokens
            if stop_tokens and self.decode(next_token.item()) in stop_tokens:
                break
            
            # Append the generated token
            generated_tokens.append(next_token.item())
            context = torch.cat([context, next_token], dim=1)
        
        # Decode the generated tokens
        generated_text = self.decode(generated_tokens)
        
        return generated_text

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