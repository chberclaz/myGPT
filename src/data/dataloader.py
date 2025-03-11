"""
Data loading utilities.
"""

import os
import numpy as np
import torch
from typing import Tuple, Optional, Callable
from ..config.training_config import TrainingConfig
from .tokenizer import get_tokenizer

class DataManager:
    """Manages dataset preparation, saving, and loading for both binary and text data."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the data manager.
        
        Args:
            config: Training configuration containing dataset settings and device
        """
        self.config = config
        self.train_data = None
        self.val_data = None
        
    def prepare_data(self, text_file: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare and load the dataset from either binary or text file.
        
        Args:
            text_file: Optional path to text file. If None, loads from binary file
            
        Returns:
            Tuple of (train_data, val_data) tensors
        """
        if text_file is not None:
            return self._prepare_from_text(text_file)
        else:
            return self._prepare_from_binary()
    
    def _prepare_from_binary(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load data from binary file."""
        data_path = os.path.join('data', self.config.data_dir, 'train.bin')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Binary data file not found at {data_path}")
        
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
        data = torch.from_numpy(data).type(torch.long)
        
        # Move data to device if specified
        if hasattr(self.config, 'device'):
            data = data.to(self.config.device)
        
        # Split into train and validation sets
        n = int(0.9 * len(data))  # 90% for training
        self.train_data = data[:n]
        self.val_data = data[n:]
        
        return self.train_data, self.val_data
    
    def _prepare_from_text(self, text_file: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and tokenize data from text file."""
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"Text file not found at {text_file}")
            
        # Read the text file
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Get the tokenizer
        encode, decode = get_tokenizer(self.config)
        
        # Encode the text
        data = torch.tensor(encode(text), dtype=torch.long)
        
        # Move data to device if specified
        if hasattr(self.config, 'device'):
            data = data.to(self.config.device)
        
        # Split into train and validation sets
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
        return self.train_data, self.val_data
    
    def save_data(self) -> None:
        """Save the prepared data to disk."""
        if self.train_data is None or self.val_data is None:
            raise ValueError("No data to save. Call prepare_data first.")
        
        # Create save directory if it doesn't exist
        save_dir = self.config.data_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Save train and validation data
        torch.save(self.train_data, os.path.join(save_dir, 'train_data.pt'))
        torch.save(self.val_data, os.path.join(save_dir, 'val_data.pt'))
        print(f"Data saved to {save_dir}")
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load previously saved data from disk.
        
        Returns:
            Tuple of (train_data, val_data) tensors
        """
        data_dir = self.config.data_dir
        train_path = os.path.join(data_dir, 'train_data.pt')
        val_path = os.path.join(data_dir, 'val_data.pt')
        
        if not (os.path.exists(train_path) and os.path.exists(val_path)):
            print(f"Data files not found in {data_dir}")
            self._prepare_from_text(os.path.join(data_dir, self.config.input_file))
            self.save_data()

        self.train_data = torch.load(train_path)
        self.val_data = torch.load(val_path)
        
        # Move data to device if specified
        if hasattr(self.config, 'device'):
            self.train_data = self.train_data.to(self.config.device)
            self.val_data = self.val_data.to(self.config.device)
            
        print(f"Training & Validation data loaded from {data_dir}")
        return self.train_data, self.val_data
    
    def get_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the current train and validation data.
        
        Returns:
            Tuple of (train_data, val_data) tensors
        """
        if self.train_data is None or self.val_data is None:
            raise ValueError("No data available. Call prepare_data or load_data first.")
        return self.train_data, self.val_data 