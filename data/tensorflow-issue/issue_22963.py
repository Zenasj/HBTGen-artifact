# tf.random.uniform((None, 10), dtype=tf.float32) ← Input shape inferred from original example build shape (None, 10)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a Dense layer with 4 units as per the original example
        self.dense = tf.keras.layers.Dense(4)
        
    def call(self, inputs):
        # Forward pass simply applies the dense layer to inputs
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel
    # Note: No pre-built weights, layers will be built on first call or explicit build
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The original example uses input shape (None, 10), so single batch input shape is (1, 10)
    # Use float32 as typical dtype for tf.keras models
    return tf.random.uniform((1, 10), dtype=tf.float32)

# Notes based on the issue content:
# - The error "layer isn't built" happens because the dense layer was never built before summary is called.
# - Building the model with `model.build((None, 10))` fixes it, or doing a forward pass once with input tensors.
# - In TensorFlow 2.x, eager mode is default and calling model on an input builds the layers.
#   So calling model(GetInput()) builds the layers and allows summary to work correctly.
# - This minimal subclass example reproduces the original problem and the standard solution.
#
# This code is compatible with TF 2.20.0 and supports XLA compilation via:
# @tf.function(jit_compile=True)
# def compiled(x):
#     return model(x)

