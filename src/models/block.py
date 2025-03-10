"""
Transformer block implementation.
"""

import torch.nn as nn
from ..config.model_config import ModelConfig
from .attention import CausalSelfAttention
from .mlp import MLP

class Block(nn.Module):
    """Transformer block implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        """Forward pass of the transformer block."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x 