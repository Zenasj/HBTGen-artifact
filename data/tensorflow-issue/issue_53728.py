# tf.random.uniform((B, 1, 4), dtype=tf.float32)  ← Input shape inferred from trainX reshape: (batch, 1, features)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Assuming look_back=1 from code
        self.look_back = 1
        # Two LSTM layers with 256 units each as per original code
        self.lstm1 = tf.keras.layers.LSTM(256, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(256)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(self.look_back)

    def call(self, inputs, training=False):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        if training:
            x = self.dropout(x, training=training)
        output = self.dense(x)
        return output

def my_model_function():
    return MyModel()

def GetInput():
    # Input feature size is 4 based on ValueError and reshape step (None,1,4)
    # Batch size can be arbitrary, here 5 for example
    B = 5
    input_tensor = tf.random.uniform((B, 1, 4), dtype=tf.float32)
    return input_tensor

