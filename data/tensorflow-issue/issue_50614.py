# tf.random.uniform((16, 10, 7), dtype=tf.float32)  # Assuming batch=16, timesteps=10, features=7 from data shape hints

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, units=64):
        super().__init__()
        # GRU branch
        self.gru1 = layers.GRU(units, return_sequences=True)
        self.dropout1 = layers.Dropout(0.2)
        self.gru2 = layers.GRU(units)
        self.dropout2 = layers.Dropout(0.2)
        self.dense_gru = layers.Dense(1)
        
        # BiLSTM branch
        self.bilstm1 = layers.Bidirectional(layers.LSTM(units, return_sequences=True))
        self.bilstm2 = layers.Bidirectional(layers.LSTM(units))
        self.dense_bilstm = layers.Dense(1)

    def call(self, inputs, training=False):
        # GRU pathway
        x_gru = self.gru1(inputs)
        x_gru = self.dropout1(x_gru, training=training)
        x_gru = self.gru2(x_gru)
        x_gru = self.dropout2(x_gru, training=training)
        out_gru = self.dense_gru(x_gru)
        
        # BiLSTM pathway
        x_bilstm = self.bilstm1(inputs)
        x_bilstm = self.bilstm2(x_bilstm)
        out_bilstm = self.dense_bilstm(x_bilstm)
        
        # Return both outputs for possible comparison or multi-task usage
        return out_gru, out_bilstm

def my_model_function():
    # Create an instance of MyModel with default 64 units as inferred from the provided code
    model = MyModel(units=64)
    # Compile with adam optimizer and mse loss as original code
    model.compile(optimizer='adam', loss='mse')
    return model

def GetInput():
    # From the original issue:
    # Input shape: (batch_size, timesteps=X_train.shape[1], features=X_train.shape[2])
    # From example data table, 7 features and ~1000 rows, we assume timesteps=10 (typical short window)
    # Batch size chosen as 16 (matching original batch_size in fit)
    batch_size = 16
    timesteps = 10
    features = 7  # Columns a,b,c,d,e,TARGET VALUE,g = 7 features total
    # Generate random float32 tensor of this shape
    return tf.random.uniform((batch_size, timesteps, features), dtype=tf.float32)

