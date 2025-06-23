from stock_transformer_model import StockTransformer, Trainer
from dataset_builder import StockDatasetBuilder

# Define the tickers to include in training
tickers = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]

# Build dataset
builder = StockDatasetBuilder(symbols=tickers)
X, y = builder.build_dataset()
print(f"Dataset shape: {X.shape}, Labels: {set(y)}")

# Define model
model = StockTransformer(
    input_dim=5,   # OHLCV
    seq_len=10,
    num_classes=3  # Down, Neutral, Up
)

# Train and evaluate
trainer = Trainer(model, X, y)
trainer.train(epochs=10)
trainer.evaluate()
