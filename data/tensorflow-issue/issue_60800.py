# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape is (batch_size, 1) from the original example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer model as per the minimal example in the issue
        self.dense = tf.keras.layers.Dense(1, input_shape=(1,))

    def call(self, inputs, training=False):
        # Forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    # No pretrained weights required, example shows random initialization for a regression task
    return MyModel()

def GetInput():
    # Generate random input tensor matching the input expected by the model:
    # Shape: (batch_size=10, 1), dtype float32 to match usual TF float type
    # This matches roughly the training data shape used in the minimal example
    return tf.random.uniform((10, 1), dtype=tf.float32)

