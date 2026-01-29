# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← inferred input shape: batch size unknown, height & width unknown, 1 channel

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv2D layer with 5 filters, kernel size 3x3 (from the example)
        self.conv = tf.keras.layers.Conv2D(5, 3)
    
    def call(self, inputs, training=None):
        # The issue is about using tf.print inside a model in TF2.x
        # The solution is to use a Lambda layer to perform print as a side effect.
        # Here, we replicate that behavior by printing the inputs inside the model call,
        # returning the inputs unchanged, then applying Conv2D.
        
        # Use tf.print to print the tensor during the forward pass.
        # tf.print is fully compatible in graph or eager mode.
        tf.print("Input tensor:", inputs)
        
        # Pass input through Conv2D
        out = self.conv(inputs)
        return out

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected shape.
    # Input shape is (batch_size, height, width, channels)
    # From the issue example: batch size variable, height=40, width=40, channels=1
    # We'll produce a batch size of 2 to allow batching like in the original example.
    batch_size = 2
    height = 40
    width = 40
    channels = 1
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

