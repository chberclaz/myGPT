"""
Main GPT model implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.models.block import Block
from src.models.mlp import MLP
from src.models.attention import CausalSelfAttention
from src.models.gelu import NewGELU

class GPT(nn.Module):
    """GPT model with configurable architecture."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        self.dataset = config.dataset
        
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    @staticmethod
    def create_optimizer(model: 'GPT', config: TrainingConfig) -> torch.optim.Optimizer:
        """Create optimizer with proper weight decay settings."""
        # Start with all parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        
        # Filter out parameters that don't require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Create optimizer groups with weight decay only for 2D parameters
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Create optimizer
        if config.distributed and config.zero_stage > 0:
            from torch.distributed.optim import ZeroRedundancyOptimizer
            optimizer = ZeroRedundancyOptimizer(
                optim_groups,
                optimizer_class=torch.optim.AdamW,
                lr=config.learning_rate,
                betas=config.betas,
                eps=1e-8
            )
        else:
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=config.learning_rate,
                betas=config.betas,
                eps=1e-8
            )
        
        return optimizer
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            idx: Input tensor of shape (batch_size, sequence_length)
            targets: Optional target tensor of shape (batch_size, sequence_length)
            
        Returns:
            Tuple of (logits, loss) where loss is None if targets is None
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Get token and position embeddings
        token_embeddings = self.token_embedding(idx)
        position = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        position_embeddings = self.position_embedding(position)
        
        # Combine embeddings
        x = token_embeddings + position_embeddings
        
        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def save(self, optimizer: torch.optim.Optimizer, training_config: TrainingConfig, iter_num: int = 0, best_val_loss: float = float('inf')) -> None:
        """
        Save the model checkpoint.
        
        Args:
            optimizer: The optimizer used for training
            training_config: Training configuration containing output directory
            iter_num: Current iteration number
            best_val_loss: Best validation loss so far
        """
        from src.utils.io import save_checkpoint
        save_checkpoint(self, optimizer, self.config, training_config, iter_num, best_val_loss)
        
    @classmethod
    def load(cls, model_config: ModelConfig = None, training_config: TrainingConfig = None) -> Tuple['GPT', torch.optim.Optimizer, ModelConfig, TrainingConfig]:
        """
        Load a saved model checkpoint.
        
        Args:
            model_config: Optional model configuration. If None, loads from checkpoint
            training_config: Optional training configuration. If None, loads from checkpoint
            
        Returns:
            Tuple of (model, optimizer, model_config, training_config)
        """
        from src.utils.io import load_checkpoint
        
        # Load the checkpoint
        checkpoint = load_checkpoint(training_config)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {training_config.out_dir}/ckpt.pt")
        
        # Load configurations from checkpoint if not provided
        if model_config is None:
            model_config = ModelConfig(**checkpoint['model_config'])
        if training_config is None:
            training_config = TrainingConfig(**checkpoint['training_config'])
        
        model_config.dataset = training_config.dataset
        # Create model and optimizer
        model = cls(model_config)
        optimizer = cls.create_optimizer(model, training_config)
        
        # Restore model and optimizer state
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        #print(f"Model loaded from {training_config.out_dir}/ckpt.pt")
        return model, optimizer, model_config, training_config
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.8,
                top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.0,
                stop_tokens: List[str] = None, device: str = None) -> str:
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
            device: Device to run generation on. If None, uses model's device
            
        Returns:
            Generated text as string
        """
        from src.data.tokenizer import get_tokenizer
        
        # Use model's device if none specified
        if device is None:
            device = next(self.parameters()).device
        
        # Get tokenizer
        encode, decode = get_tokenizer(self.dataset)
        
        # Encode the prompt and move to device
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate tokens
        generated_tokens = []
        for _ in range(max_tokens):
            # Get model predictions
            with torch.no_grad():
                logits, _ = self(context)
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

            
            # Append the generated token
            generated_tokens.append(next_token.item())
            context = torch.cat([context, next_token], dim=1)

            if stop_tokens and decode([next_token.item()]) in stop_tokens:
                break

        # Decode the generated tokens
        generated_text = decode(generated_tokens)
        
        return generated_text
    
    @classmethod
    def from_pretrained(cls, model_type: str) -> 'GPT':
        """Load pretrained GPT-2 model weights from huggingface."""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained gpt: {model_type}")
        
        # Get model configuration based on type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        
        # Create configuration
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = ModelConfig(**config_args)
        
        # Initialize model
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
        
        # Load HuggingFace model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        
        # Copy weights
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model 