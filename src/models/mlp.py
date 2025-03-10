"""
MLP layer implementation.
"""

import torch.nn as nn
from ..config.model_config import ModelConfig
from .gelu import NewGELU

class MLP(nn.Module):
    """MLP layer implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
    
    def forward(self, x):
        """Forward pass of the MLP layer."""
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x 