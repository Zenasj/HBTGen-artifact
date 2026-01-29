# tf.random.uniform((B=5, H=10), dtype=tf.int32) ← Input shape inferred from SEQUENCES=5 and TIME_STEPS=10

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Embedding layer with vocabulary size 100 and embedding dim 4
        self.embedding = tf.keras.layers.Embedding(input_dim=100, output_dim=4)
        # LSTM layer with 32 units
        self.lstm = tf.keras.layers.LSTM(32)
        # Dense layer with 1 output unit
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.lstm(x)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel, no pretrained weights to load for this example
    return MyModel()

def GetInput():
    # Return a random integer tensor of shape (5, 10) consistent with
    # SEQUENCES=5, TIME_STEPS=10, vocab size=100, dtype int32 as expected by Embedding
    return tf.random.uniform((5, 10), minval=0, maxval=100, dtype=tf.int32)

