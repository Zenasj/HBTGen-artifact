# tf.random.uniform((16,), dtype=tf.float32) ← inferred input shape from batch_size=16 scalar input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Dummy layer to print actual shape during forward pass to illustrate issue
        self.print_layer = tf.keras.layers.Lambda(
            lambda x: tf.print("Input shape in call:", tf.shape(x)) or x
        )
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.print_layer(inputs)
        return self.relu(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return input tensor shape (batch_size,) representing scalar input per batch element
    batch_size = 16
    # uniform random to simulate scalar inputs per batch
    return tf.random.uniform((batch_size,), dtype=tf.float32)

