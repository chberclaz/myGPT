# TradeGPT

A PyTorch implementation of GPT for text generation, based on the [nanoGPT](https://github.com/karpathy/nanoGPT) project by Andrej Karpathy.

## Features

- Clean, modular implementation of GPT architecture
- Support for distributed training using PyTorch DDP
- Mixed precision training with FP16
- Gradient accumulation for large batch sizes
- Learning rate scheduling with cosine decay
- Built-in model saving and loading
- Text generation with various sampling strategies (temperature, top-k, top-p)
- Support for multiple datasets (Shakespeare, custom datasets)
- Comprehensive logging and experiment tracking with Weights & Biases

## Project Structure

```
TradeGPT/
├── src/
│   ├── config/           # Configuration files
│   │   ├── model_config.py    # Model architecture configuration
│   │   └── training_config.py # Training hyperparameters
│   ├── data/            # Data handling
│   │   ├── dataloader.py      # Data loading utilities
│   │   └── tokenizer.py       # Text tokenization
│   ├── models/          # Model implementation
│   │   └── gpt.py            # GPT model architecture
│   ├── utils/           # Utility functions
│   │   ├── distributed.py    # Distributed training setup
│   │   └── io.py            # Checkpoint saving/loading
│   ├── train.py         # Main training script
│   ├── generate.py      # Text generation script
│   └── save_model.py    # Example usage scripts
├── data/               # Dataset storage
├── out/               # Model checkpoints and outputs
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/TradeGPT.git
cd TradeGPT
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

The project provides a `Trainer` class that handles the entire training process:

```python
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.train import Trainer

# Create configurations
model_config = ModelConfig()
training_config = TrainingConfig()

# Create trainer
trainer = Trainer(
    model_config=model_config,
    training_config=training_config
)

# Train the model
model, optimizer, best_val_loss = trainer.train()
```

### Generating Text

Use the `Generator` class to generate text from a trained model:

```python
from src.generate import Generator

# Create generator with a trained model
generator = Generator(
    model=model,  # Your trained model
    model_config=model_config,
    training_config=training_config
)

# Generate text
generated_text = generator.generate(
    prompt="Once upon a time",
    max_tokens=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.0,
    stop_tokens=['.', '!', '?']
)
```

### Saving and Loading Models

The GPT model includes built-in methods for saving and loading:

```python
# Save a model
model.save(optimizer, training_config)

# Load a model
model, optimizer = GPT.load(model_config, training_config)
```

See `src/save_model.py` for complete examples of training, saving, loading, and generating text.

## Configuration

### Model Configuration

Configure the model architecture in `src/config/model_config.py`:

- Vocabulary size
- Block size (context length)
- Number of layers
- Number of heads
- Embedding dimension
- Dropout rate

### Training Configuration

Set training parameters in `src/config/training_config.py`:

- Batch size
- Learning rate
- Number of iterations
- Evaluation interval
- Gradient clipping
- Mixed precision training
- Distributed training settings

## Distributed Training

To enable distributed training:

1. Set `ddp=True` in `TrainingConfig`
2. Run the training script with:

```bash
torchrun --nproc_per_node=N src/train.py
```

where N is the number of GPUs to use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
