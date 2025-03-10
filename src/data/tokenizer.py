"""
Tokenization utilities.
"""

import os
import pickle
import tiktoken
from typing import Tuple, Callable, Optional
from ..config.training_config import TrainingConfig

def get_tokenizer(config: TrainingConfig) -> Tuple[Callable[[str], list[int]], Callable[[list[int]], str]]:
    """Get the appropriate tokenizer based on the dataset."""
    if config.dataset == 'shakespeare':
        # Load the meta.pkl file for Shakespeare dataset
        meta_path = os.path.join('data', config.dataset, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            stoi, itos = meta['stoi'], meta['itos']
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])
            return encode, decode
    else:
        # Default to GPT-2 tokenizer
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        return encode, decode 