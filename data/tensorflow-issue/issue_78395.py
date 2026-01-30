from tensorflow import keras
from tensorflow.keras import layers

import pandas as pd
import os
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error  # For evaluation after predictions
import tensorflow as tf
import numpy as np
import pickle

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'C:/Users/path')

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

def load_data(symbol):
    """
    Load processed data for a given stock symbol.
    """
    folder = os.path.join(os.path.dirname(__file__), 'C:/Users/path')
    file_path = os.path.join(folder, f'{symbol}.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Processed data for {symbol} not found.")
    
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Preprocess the data for LSTM. Scale it and reshape it for sequential training.
    """
    # Select features and target
    features = ['SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'RSI', 'Volatility', 'ROC', 'MACD', 'Signal_Line']
    target = 'close'
    
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])
    
    # Creating sequences
    sequence_length = 60  # Use past 60 days to predict the next day
    X, y = [], []
    
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(df[target].values[i])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape X to be (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], len(features)))
    
    return X, y, scaler

def build_lstm_model(input_shape):
    """
    Build an LSTM model for stock price prediction.
    """
    model = tf.keras.Sequential()
    
    # LSTM layers
    model.add(tf.keras.layers.LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.2))  # Add dropout to prevent overfitting
    model.add(tf.keras.layers.LSTM(units=100, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    
    # Dense layers
    model.add(tf.keras.layers.Dense(units=50))
    model.add(tf.keras.layers.Dense(units=1))  # Output layer predicting closing price
    
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanSquaredError()])
    
    return model

def train_model(X, y):
    """
    Train the LSTM model.
    """
    # Build the model
    model = build_lstm_model((X.shape[1], X.shape[2]))
    
    # Train the model
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    
    return model

def save_model(model, symbol, scaler):
    """
    Save the trained LSTM model and the scaler for later use.
    """
    folder = os.path.join(os.path.dirname(__file__), 'C:/Users/path')
    os.makedirs(folder, exist_ok=True)
    
    model_file_path = os.path.join(folder, f'{symbol}_lstm_model.h5')
    scaler_file_path = os.path.join(folder, f'{symbol}_scaler.pkl')
    
    # Save the model
    model.save(model_file_path)
    print(f"Model saved as {symbol}_lstm_model.h5")
    
    # Save the scaler
    with open(scaler_file_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved as {symbol}_scaler.pkl")

def train_and_save_models():
    """
    Load data for each stock, preprocess it, train the LSTM model, and save the trained model.
    """
    stock_symbols = config['stocks']
    
    for symbol in stock_symbols:
        print(f"Training LSTM model for {symbol}...")
        
        # Load and preprocess the data
        df = load_data(symbol)
        X, y, scaler = preprocess_data(df)
        
        # Train the model
        model = train_model(X, y)
        
        # Save the model and scaler
        save_model(model, symbol, scaler)

if __name__ == "__main__":
    train_and_save_models()