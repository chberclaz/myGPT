"""
Model configuration for GPT-2.
"""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for the GPT-2 model."""
    
    # Model architecture parameters
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dataset: str = 'custom'
    # Model variants
    @classmethod
    def gpt2(cls):
        """GPT-2 base model (124M parameters)."""
        return cls(n_layer=12, n_head=12, n_embd=768)
    
    @classmethod
    def gpt2_medium(cls):
        """GPT-2 medium model (350M parameters)."""
        return cls(n_layer=24, n_head=16, n_embd=1024)
    
    @classmethod
    def gpt2_large(cls):
        """GPT-2 large model (774M parameters)."""
        return cls(n_layer=36, n_head=20, n_embd=1280)
    
    @classmethod
    def gpt2_xl(cls):
        """GPT-2 XL model (1558M parameters)."""
        return cls(n_layer=48, n_head=25, n_embd=1600) 