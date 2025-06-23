#A minimal transformer-based classifier built in PyTorch for intraday stock price movement prediction.

## 🧠 Model Purpose
Predict if the stock price will go **Up**, **Down**, or remain **Neutral** over the next N minutes using past OHLCV data.

## 🗂 Project Structure
```
├── dataset_builder.py         # Downloads intraday data and prepares features/labels
├── stock_transformer_model.py # Transformer model and training logic
├── main.py                    # Main script to run training and evaluation
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview and instructions
```

## 🚀 Quick Start
1. Clone the repo:
```bash
git clone <your_repo_url>
cd stock-transformer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run training:
```bash
python main.py
```

## ⚙️ Configuration
You can modify the following in `main.py`:
- `tickers = ["SPY", "AAPL", ...]`
- `input_window` and `label_window`
- Training epochs and thresholds

## 🧪 Example Task
> "Will the price go up in the next 10 minutes based on the last 10 minutes?"

Labels:
- `1 = Up`, `-1 = Down`, `0 = Neutral`

## 📈 Expand Ideas
- Add more features: RSI, EMA, MACD
- Replace `yfinance` with more historical APIs
- Convert to regression or multi-output prediction
- Add backtesting logic for live signal testing

## 🧑‍💻 Author
Built with ❤️ using PyTorch and transformers.

---

> Disclaimer: This is a learning tool, not financial advice.
