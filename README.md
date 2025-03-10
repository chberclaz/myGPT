# TradeGPT

A PyTorch implementation of GPT for prediction, based on the [nanoGPT](https://github.com/karpathy/nanoGPT) project by Andrej Karpathy.

## Features

- Clean, modular implementation of GPT architecture
- Support for distributed training using PyTorch DDP
- Mixed precision training with FP16
- Gradient accumulation for large batch sizes
- Learning rate scheduling with cosine decay
- Checkpointing and model saving
- Text generation with various sampling strategies (temperature, top-k, top-p)
- Support for multiple datasets (Shakespeare, custom datasets)

## Project Structure

```
nanoGPT/
├── src/
│   ├── config/           # Configuration files
│   ├── data/            # Data loading and preprocessing
│   ├── models/          # Model architecture
│   └── utils/           # Utility functions
├── data/                # Dataset storage
├── checkpoints/         # Model checkpoints
└── requirements.txt     # Project dependencies
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/nanoGPT.git
cd nanoGPT
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

You can train the model in two ways:

1. Using the command line:

```bash
python src/train.py
```

2. Using the modular training function in your code:

```python
from src.train import train
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig

# Create configurations
model_config = ModelConfig()
training_config = TrainingConfig()

# Train the model
model, optimizer, best_val_loss = train(
    model_config=model_config,
    training_config=training_config,
    # Optional parameters:
    # train_data=your_train_data,
    # val_data=your_val_data,
    # device='cuda',
    # ddp=False,
    # wandb_run=your_wandb_run
)

print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
```

### Text Generation

You can generate text in two ways:

1. Using the command line:

```bash
python src/generate.py
```

2. Using the modular generation function in your code:

```python
from src.generate import generate_text
from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig

# Generate text with default settings
text = generate_text("Once upon a time")

# Or with custom settings
text = generate_text(
    prompt="Once upon a time",
    max_tokens=200,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
    stop_tokens=['.', '!', '?']
)

# Or with your own model
text = generate_text(
    prompt="Once upon a time",
    model=your_trained_model,
    device='cuda'
)
```

### Saving and Loading Models

The project provides utilities for saving and loading models:

```python
from src.utils.io import save_checkpoint, load_checkpoint
from src.config.training_config import TrainingConfig

# Save a model
save_checkpoint(model, optimizer, training_config)

# Load a model
checkpoint = load_checkpoint(training_config)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
```

For a complete example of saving and loading models, see `src/save_model.py`.

## Configuration

### Model Configuration

Model parameters can be configured in `src/config/model_config.py`:

- Vocabulary size
- Embedding dimension
- Number of layers
- Number of attention heads
- Block size (context length)

### Training Configuration

Training parameters can be configured in `src/config/training_config.py`:

- Batch size
- Learning rate
- Number of iterations
- Evaluation interval
- Checkpoint directory
- Distributed training settings

## Distributed Training

To enable distributed training:

1. Set `distributed=True` in your training configuration
2. Run the training script with the appropriate number of processes:

```bash
torchrun --nproc_per_node=N src/train.py
```

where N is the number of GPUs you want to use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
