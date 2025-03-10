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
    model: Optional[GPT] = None,
    max_tokens: int = 500,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    stop_tokens: Optional[List[str]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """Generate text from a prompt using the trained model."""
    # Load configurations if model not provided
    if model is None:
        model_config = ModelConfig()
        training_config = TrainingConfig()
        
        # Create model
        model = GPT(model_config)
        model.to(device)
        
        # Load checkpoint
        if os.path.exists(os.path.join(training_config.out_dir, 'ckpt.pt')):
            checkpoint = load_checkpoint(training_config.out_dir)
            model.load_state_dict(checkpoint['model'])
        else:
            raise ValueError("No checkpoint found. Please train the model first.")
    
    # Get tokenizer
    encode, decode = get_tokenizer(training_config.dataset)
    
    # Encode prompt
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate tokens
    for _ in range(max_tokens):
        # Get predictions
        with torch.no_grad():
            logits, _ = model(context)
        
        # Get next token probabilities
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Apply repetition penalty
        for i in range(len(context[0])):
            logits[0, context[0, i]] /= repetition_penalty
        
        # Sample next token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to context
        context = torch.cat((context, next_token), dim=1)
        
        # Check for stop tokens
        if stop_tokens is not None:
            generated_text = decode(context[0].tolist())
            if any(stop in generated_text for stop in stop_tokens):
                break
    
    # Decode and return generated text
    return decode(context[0].tolist())

def main():
    """Main function for text generation."""
    # Example usage
    prompt = "Once upon a time"
    generated_text = generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")

if __name__ == '__main__':
    main() 