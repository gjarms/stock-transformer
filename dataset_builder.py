# dataset_builder.py

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

class StockDatasetBuilder:
    def __init__(self, symbols, input_window=10, label_window=10, threshold=0.002, lookback_days=7):
        self.symbols = symbols
        self.input_window = input_window
        self.label_window = label_window
        self.threshold = threshold
        self.lookback_days = lookback_days
        self.scaler = StandardScaler()

    def _download_data(self, symbol):
        end = datetime.now()
        start = end - timedelta(days=self.lookback_days)
        df = yf.download(symbol, interval='1m', start=start, end=end, progress=False)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().reset_index()
        df['Symbol'] = symbol
        return df

    def _generate_labels(self, df):
        future_close = df['Close'].shift(-self.label_window)
        current_close = df['Close']
        future_return = (future_close - current_close) / current_close

        def classify(r):
            if r > self.threshold:
                return 1
            elif r < -self.threshold:
                return -1
            return 0

        df['Label'] = future_return.apply(classify)
        return df

    def build_dataset(self):
        all_X, all_y = [], []

        for symbol in self.symbols:
            try:
                df = self._download_data(symbol)
                df = self._generate_labels(df)

                features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values
                features = self.scaler.fit_transform(features)  # normalize
                df[['Open', 'High', 'Low', 'Close', 'Volume']] = features

                for i in range(len(df) - self.input_window - self.label_window):
                    X_window = df.iloc[i:i+self.input_window][['Open', 'High', 'Low', 'Close', 'Volume']].values
                    y_label = df.iloc[i + self.input_window - 1]['Label']
                    all_X.append(X_window)
                    all_y.append(y_label)

            except Exception as e:
                print(f"Error downloading {symbol}: {e}")

        X = np.array(all_X)
        y = np.array(all_y)
        return X, y
