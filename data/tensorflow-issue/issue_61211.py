# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred from the example: Input(shape=(1,))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Lambda layer that doubles the input
        self.lambda_layer = tf.keras.layers.Lambda(lambda x: x * 2)
        # Dense layer with 1 output unit
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.lambda_layer(inputs)
        output = self.dense(x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching shape (batch_size, 1)
    # Using batch size 4 as example batch size
    batch_size = 4
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)

