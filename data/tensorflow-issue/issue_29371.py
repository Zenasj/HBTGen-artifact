# tf.random.uniform((B, 20)) ← Based on discussion, input shape is (None, 20) as batch size is variable (B)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple Dense layer as in example (units=512)
        self.dense = tf.keras.layers.Dense(units=512)

    def call(self, inputs):
        # The issue in the original post was caused by using tensors not connected to Input layers.
        # Here we assume inputs come from proper Keras Input tensors.
        # The example combined two random uniform tensors, but that's not allowed as disconnected from input graph.
        # So here we directly apply the Dense to the input.

        # inputs is expected shape (B, 20)
        return self.dense(inputs)

def my_model_function():
    # Return a compiled instance of MyModel (compilation is optional, weights are randomly initialized)
    return MyModel()

def GetInput():
    # Create a random uniform float tensor of shape (batch_size=8, 20) to match Dense input
    return tf.random.uniform((8, 20), dtype=tf.float32)

