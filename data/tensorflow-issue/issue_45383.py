# tf.random.uniform((B, 140, 1), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Recreate the LSTM model as per issue's original Sequential model
        # Input shape: (seq_length=140, features=1)
        self.lstm1 = LSTM(256, return_sequences=True)
        self.dropout1 = Dropout(0.25)
        self.lstm2 = LSTM(128)
        self.dropout2 = Dropout(0.25)
        # Output units equal to number of vocabulary characters (y.shape[1])
        # Assume vocab size from original code: passing as constructor arg is possible but not from the issue,
        # so we hardcode a placeholder to allow model creation.
        # We will infer y.shape[1] from used input in GetInput() below.
        self.dense = Dense(65, activation='softmax')  # 65 because original example had 65 unique chars

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.dropout1(x, training=training)
        x = self.lstm2(x)
        x = self.dropout2(x, training=training)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    # Note: In practice, number of classes (output units) depends on dataset vocab size,
    # here assumed 65 based on example chars count from the issue.
    return MyModel()

def GetInput():
    # Return a random tensor input compatible with MyModel
    # As per original code: shape = (n_patterns, seq_length=140, features=1)
    # We do not have n_patterns reliable at inference, so create a batch of 1
    B = 1
    H = 140
    W = 1  # features per timestep (reshaped as in original code)
    
    # Original code normalized input indices by vocab size float(n_vocab).
    # Here vocab size is 65, so generate ints 0-64, normalize to [0,1)
    vocab_size = 65
    
    # Generate random input integer indices as floats in range [0,1)
    input_array = np.random.randint(0, vocab_size, size=(B, H, W)).astype(np.float32)
    input_array /= float(vocab_size)
    
    # Convert to TensorFlow tensor
    input_tensor = tf.convert_to_tensor(input_array)
    return input_tensor

