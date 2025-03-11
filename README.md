# GPT-Powered Bitcoin Trading Bot

A sophisticated Bitcoin trading bot that combines GPT-based price predictions with technical analysis to generate trading signals.

## Features

- **GPT Model Integration**: Uses a transformer-based model trained on historical Bitcoin price data
- **Technical Analysis**: Incorporates multiple technical indicators:
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Simple Moving Averages (20 and 50 periods)
  - Average True Range (ATR)
- **Real-time Data**: Fetches live Bitcoin price data from major exchanges
- **Hybrid Decision Making**: Combines AI predictions with traditional technical analysis
- **Detailed Logging**: Comprehensive logging of trading signals and model predictions
- **Configurable Parameters**: Easily adjustable trading and model parameters

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd nanoGPT
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Required dependencies:

- PyTorch
- ccxt (for exchange data)
- pandas
- numpy
- tiktoken

## Project Structure

```
src/
├── models/          # GPT model implementation
├── config/          # Configuration classes
├── data/
│   └── price_tokenizer.py  # Price data tokenization
└── trading/
    ├── data_fetcher.py     # Bitcoin price data fetching
    ├── indicators.py       # Technical analysis indicators
    ├── trading_bot.py     # Main trading bot implementation
    ├── train_trading_model.py  # Model training script
    └── run_trading_bot.py  # Bot execution script
```

## Usage

### 1. Training the Model

Train the GPT model on historical Bitcoin price data:

```bash
python -m src.trading.train_trading_model
```

Configuration options in `train_trading_model.py`:

- Model architecture (layers, heads, embedding size)
- Training parameters (batch size, learning rate, epochs)
- Price tokenization range and resolution

### 2. Running the Trading Bot

Run the bot with a trained model:

```bash
python -m src.trading.run_trading_bot --checkpoint path/to/model.pt
```

Command-line options:

- `--checkpoint`: Path to trained model checkpoint (required)
- `--interval`: Time between predictions in seconds (default: 3600)
- `--output-dir`: Directory for saving trading signals (default: trading_signals)
- `--device`: Device to run model on (default: cuda if available, else cpu)

### Trading Signal Output

The bot generates JSON files containing trading signals with:

- Action (buy/sell/hold)
- Confidence level
- Current price
- Predicted price
- Technical indicators
- Reasoning for the decision

Example signal:

```json
{
  "action": "buy",
  "confidence": 0.85,
  "price": 45000.0,
  "timestamp": "2024-03-21T14:30:00",
  "gpt_prediction": 46500.0,
  "indicators": {
    "rsi": 35.2,
    "macd": 120.5,
    "macd_signal": 100.2,
    "macd_hist": 20.3,
    "sma_20": 44800.0,
    "sma_50": 43500.0,
    "upper_band": 46000.0,
    "lower_band": 43000.0,
    "atr": 800.0
  },
  "reason": "GPT and technicals agree: RSI oversold | MACD bullish crossover"
}
```

## Model Architecture

The system uses a GPT (Generative Pre-trained Transformer) model adapted for price prediction:

- Input: Sequence of tokenized prices with technical indicators
- Output: Predicted future price movements
- Features:
  - Price and volume data
  - Normalized technical indicators
  - Temporal patterns

## Trading Strategy

The bot combines two approaches:

1. **GPT Predictions**: Price movement forecasting using the trained model
2. **Technical Analysis**: Traditional indicator-based signals

Decision making:

- Strong agreement between GPT and technicals → Higher confidence
- Significant GPT prediction (>5% change) → Favor GPT signal
- Minimal predicted change (<1%) → Hold position
- Otherwise → Follow stronger signal with moderate confidence

## Customization

### Price Tokenizer

Adjust in `src/data/price_tokenizer.py`:

- Price range (`min_price`, `max_price`)
- Number of price bins (`n_bins`)

### Model Configuration

Modify in `src/trading/train_trading_model.py`:

- Model architecture parameters
- Training hyperparameters
- Sequence length and prediction horizon

### Trading Parameters

Update in `src/trading/trading_bot.py`:

- Technical indicator thresholds
- Confidence calculation
- Signal combination logic

## Logging

The bot maintains two types of logs:

1. Console/file logging of operations and errors
2. Detailed JSON files of trading signals

Logs are saved in:

- `trading_bot.log`: Operation logs
- `trading_signals/`: Individual signal JSON files

## Warning

This is an experimental trading bot. Use at your own risk. Always:

- Test thoroughly with small amounts
- Monitor the bot's performance
- Understand the risks involved
- Consider market conditions

## License

[Your License Here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
