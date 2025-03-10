# nanoGPT

A PyTorch implementation of GPT for text generation, based on the [nanoGPT](https://github.com/karpathy/nanoGPT) project by Andrej Karpathy.

## Features

- Clean, modular implementation of GPT architecture
- Support for distributed training using PyTorch DDP
- Mixed precision training with FP16
- Gradient accumulation for training with large batch sizes
- Learning rate scheduling with cosine decay
- Checkpointing and model saving
- Text generation with various sampling strategies (temperature, top-k, top-p)
- Support for multiple datasets (Shakespeare, custom datasets)

## Project Structure

```
nanoGPT/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── model_config.py
│   │   └── training_config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   └── tokenizer.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── attention.py
│   │   ├── block.py
│   │   ├── gelu.py
│   │   ├── gpt.py
│   │   └── mlp.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── distributed.py
│   │   └── io.py
│   ├── train.py
│   └── generate.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/nanoGPT.git
cd nanoGPT
```

2. Create a virtual environment (optional but recommended):

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

To train the model:

```bash
python src/train.py
```

The training script will:

1. Load the configuration
2. Set up distributed training if enabled
3. Prepare the dataset
4. Create and initialize the model
5. Train the model with the specified parameters
6. Save checkpoints periodically

### Text Generation

To generate text using a trained model:

```bash
python src/generate.py
```

You can also use the generation function in your code:

```python
from src.generate import generate_text

prompt = "Once upon a time"
generated_text = generate_text(
    prompt,
    max_tokens=500,
    temperature=0.8,
    top_k=40,
    top_p=0.9
)
print(generated_text)
```

## Configuration

The model and training parameters can be configured in:

- `src/config/model_config.py`: Model architecture parameters
- `src/config/training_config.py`: Training hyperparameters

## Distributed Training

To enable distributed training:

1. Set `distributed=True` in `training_config.py`
2. Run the training script with PyTorch's distributed launch:

```bash
python -m torch.distributed.launch --nproc_per_node=N src/train.py
```

where N is the number of GPUs to use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
