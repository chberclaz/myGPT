# NanoGPT-Trader

A GPT implementation optimized for trading applications, based on the nanoGPT architecture.

## Installation

You can install the package directly from the repository:

```bash
pip install git+https://github.com/chberclaz/trader.git
```

Or for development:

```bash
git clone https://github.com/chberclaz/trader.git
cd nanogpt-trader
pip install -e .
```

## Quick Start

```python
from src import GPT, Trainer, Generator, DataManager
from src import ModelConfig, TrainingConfig

# Create configurations
model_config = ModelConfig(
    vocab_size=50257,
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dataset='custom'
)

training_config = TrainingConfig(
    batch_size=16,
    learning_rate=3e-4,
    max_iters=8000,
    eval_interval=800,
    eval_iters=800,
    warmup_iters=100,
    lr_decay_iters=5000,
    min_lr=3e-5,
    out_dir='out/custom_model',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    dataset='custom'
)

# Initialize data manager and prepare data
data_manager = DataManager(training_config)
train_data, val_data = data_manager.prepare_data(text_file='path/to/your/data.txt')

# Create trainer and train model
trainer = Trainer(
    model_config=model_config,
    training_config=training_config,
    train_data=train_data,
    val_data=val_data
)
model, optimizer, best_val_loss = trainer.train()

# Generate text
generator = Generator(model=model, model_config=model_config, training_config=training_config)
generated_text = generator.generate(
    prompt="Your prompt here",
    max_tokens=100,
    temperature=0.8
)
print(generated_text)
```

## Project Structure

```
nanogpt-trader/
├── src/                    # Main package directory
│   ├── __init__.py        # Package initialization
│   ├── models/            # Model implementations
│   ├── config/            # Configuration classes
│   ├── data/              # Data loading and processing
│   ├── utils/             # Utility functions
│   └── generate.py        # Text generation
├── tests/                 # Test files
├── docs/                  # Documentation
├── data/                  # Data directory (gitignored)
├── custom_dataset_example.py  # Example usage script
├── requirements.txt       # Package dependencies
├── setup.py              # Package setup file
└── README.md             # This file
```

## Features

- Efficient GPT implementation
- Custom dataset support
- Configurable model architecture
- Training with gradient accumulation
- Text generation with various parameters
- Checkpoint saving and loading
- Device (CPU/GPU) support

## Example Scripts

Check `custom_dataset_example.py` for a complete example of:

1. Preparing and loading custom data
2. Training the model
3. Saving and loading checkpoints
4. Generating text

## License

This project is licensed under the MIT License - see the LICENSE file for details.
