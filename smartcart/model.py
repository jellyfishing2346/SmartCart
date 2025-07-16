import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class SmartCart:
    def __init__(self, ticker, look_back=60):
        self.ticker = ticker
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        self.trainX, self.trainY = None, None

    def model_path(self):
        # Save models in a models/ directory
        import os
        os.makedirs('models', exist_ok=True)
        return f"models/{self.ticker}_lstm.keras"

    def fetch_data(self, period="5y"):
        df = yf.download(self.ticker, period=period)
        # Check for empty DataFrame or missing 'Close' column
        if df is None or df.empty or 'Close' not in df:
            return None
        close = df['Close'].dropna()
        if close.empty:
            return None
        self.data = close.values.reshape(-1, 1)
        self.data = self.scaler.fit_transform(self.data)
        return df

    def prepare_data(self):
        X, y = [], []
        for i in range(self.look_back, len(self.data)):
            X.append(self.data[i-self.look_back:i, 0])
            y.append(self.data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        self.trainX, self.trainY = X, y

    def build_model(self):
        from tensorflow.keras.models import load_model
        import os
        model_file = self.model_path()
        if os.path.exists(model_file):
            self.model = load_model(model_file)
        else:
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(self.trainX.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            self.model = model

    def train(self, epochs=20, batch_size=32):
        self.model.fit(self.trainX, self.trainY, epochs=epochs, batch_size=batch_size, verbose=1)
        # Save model after training
        self.model.save(self.model_path())

    def predict(self, days=7):
        last_seq = self.data[-self.look_back:]
        preds = []
        for _ in range(days):
            X = np.reshape(last_seq, (1, self.look_back, 1))
            pred = self.model.predict(X, verbose=0)
            preds.append(pred[0, 0])
            last_seq = np.append(last_seq, pred)[-self.look_back:]
        preds = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        return preds
