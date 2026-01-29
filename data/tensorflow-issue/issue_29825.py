# tf.random.uniform((B, 1024, 7), dtype=tf.float32) ← Inferred input shape from the LSTM input layer shape

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using 128 units as inferred from discussion (neurons=128)
        self.lstm1 = layers.LSTM(128, return_sequences=True)
        self.lstm2 = layers.LSTM(128)
        self.dense_relu = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.2)
        self.dense_out = layers.Dense(4, activation='softmax')  # 4 output classes, softmax activation

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.dense_relu(x)
        x = self.dropout(x, training=training)
        return self.dense_out(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a batch of random inputs matching model input shape (batch_size, 1024, 7)
    # Batch size is chosen as 128 to align with typical training batch size in issue discussion
    batch_size = 128
    window_size = 1024
    inputs_n = 7
    # Create random float32 input tensor
    return tf.random.uniform((batch_size, window_size, inputs_n), dtype=tf.float32)

