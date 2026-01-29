# tf.random.uniform((B, 3, 3), dtype=tf.float32)  ← Input shape inferred from shape=(None, 3, 3) with fixed sequence length 3 and feature size 3

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # LSTM with 4 units, returns last output (default)
        self.lstm = tf.keras.layers.LSTM(4, name="interm_1")
        # Dense output layer with 1 unit
        self.dense = tf.keras.layers.Dense(1, name="dense_1")

    def call(self, inputs):
        # inputs shape: (batch, 3, 3)
        x = self.lstm(inputs)   # shape: (batch, 4)
        output = self.dense(x)  # shape: (batch, 1)
        return output

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # For a stand-alone model, we can build and compile to mimic original script behavior,
    # though not required strictly for TFLite conversion or inference
    model.compile(optimizer='sgd', loss='mean_squared_error')
    return model

def GetInput():
    # Return a random tensor with shape (batch_size, 3, 3)
    # For generality, choose batch_size = 10 as per original example
    batch_size = 10
    input_tensor = tf.random.uniform(shape=(batch_size, 3, 3), dtype=tf.float32)
    return input_tensor

