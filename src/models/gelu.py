"""
GELU activation implementation.
"""

import math
import torch
import torch.nn as nn

class NewGELU(nn.Module):
    """GELU activation function used by OpenAI in GPT-2."""
    
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0)))) 