"""
Script for generating text using a trained GPT model.
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Optional

from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from data.tokenizer import get_tokenizer
from models import GPT
from utils.io import load_checkpoint

def generate_text(
    prompt: str,
    model: GPT = None,
    model_config: ModelConfig = None,
    training_config: TrainingConfig = None,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.0,
    stop_tokens: List[str] = None,
    device: str = None
) -> str:
    """
    Generate text using the GPT model.
    
    Args:
        prompt: Input text to generate from
        model: Pre-trained GPT model. If None, loads from checkpoint
        model_config: Model configuration. If None, loads from config/model_config.py
        training_config: Training configuration. If None, loads from config/training_config.py
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Number of top tokens to consider for sampling
        top_p: Cumulative probability threshold for sampling
        repetition_penalty: Penalty for repeating tokens
        stop_tokens: List of tokens to stop generation at
        device: Device to run generation on. If None, uses CUDA if available
    
    Returns:
        Generated text as string
    """
    # Load configurations if not provided
    if model_config is None:
        model_config = ModelConfig()
    if training_config is None:
        training_config = TrainingConfig()
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model if not provided
    if model is None:
        checkpoint = load_checkpoint(training_config)
        model = GPT(model_config)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
    
    # Get tokenizer
    encode, decode = get_tokenizer(training_config.dataset)
    
    # Encode the prompt
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate tokens
    generated_tokens = []
    for _ in range(max_tokens):
        # Get model predictions
        with torch.no_grad():
            logits, _ = model(context)
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
        if stop_tokens and decode(next_token.item()) in stop_tokens:
            break
        
        # Append the generated token
        generated_tokens.append(next_token.item())
        context = torch.cat([context, next_token], dim=1)
    
    # Decode the generated tokens
    generated_text = decode(generated_tokens)
    
    return generated_text

def main():
    """Example usage of the generate_text function."""
    # Example prompt
    prompt = "Once upon a time"
    
    # Generate text with default settings
    generated_text = generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")
    
    # Example with custom settings
    generated_text = generate_text(
        prompt,
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