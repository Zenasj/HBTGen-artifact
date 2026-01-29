# tf.random.uniform((10, 3072), dtype=tf.float32) ← Input shape inferred from (10, 32*32*3) after flattening as in the issue example.

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # According to the issue example, the model is a simple Dense layer with 10 outputs
        self.dense = tf.keras.layers.Dense(10)
    
    def call(self, inputs, training=False):
        # A straightforward forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel; weights are initialized randomly (default)
    return MyModel()

def GetInput():
    # Generate a batch of 10 samples, each input flattened CIFAR-10 image
    # CIFAR-10 images are 32x32 RGB -> 32*32*3 = 3072 features
    # This replicates the example from the issue where input was normalized and reshaped
    B, H, W, C = 10, 32, 32, 3
    # Random float32 tensor in [0, 1) of shape (10, 3072)
    x = tf.random.uniform((B, H * W * C), dtype=tf.float32)
    return x

