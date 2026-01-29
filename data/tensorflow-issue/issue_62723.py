# tf.random.uniform((B, 37), dtype=tf.float32) ← input shape inferred from keras.Input(shape=(37,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the model architecture as described:
        # Input shape (37,), Dense 32 with relu, Dense 5 with softmax
        self.dense1 = tf.keras.layers.Dense(32, activation="relu")
        self.dense2 = tf.keras.layers.Dense(5, activation="softmax")

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        return self.dense2(x)

def my_model_function():
    # Create and return an instance of MyModel
    model = MyModel()
    return model

def GetInput():
    # Return a random input tensor consistent with the input shape (B, 37).
    # Use batch size 1 for simplicity.
    batch_size = 1
    input_shape = (batch_size, 37)
    return tf.random.uniform(input_shape, dtype=tf.float32)

