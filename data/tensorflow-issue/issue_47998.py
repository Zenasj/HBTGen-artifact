# tf.random.uniform((B, 128, 128, 3), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, n_out=8):
        super().__init__()
        # Using an internal Conv2D layer configured via the input attribute n_out
        # This mimics the original TestLayer with @attr.s decoration logic abstracted away
        # The Conv2D uses a 3x3 kernel with ReLU activation
        self.conv_layer = layers.Conv2D(n_out, 3, activation='relu')
        # Additional downstream Dense layer (like in the Sequential example) to complete the model
        self.dense = layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.conv_layer(inputs)
        # Flatten spatial dims to feed to Dense layer
        x = layers.Flatten()(x)
        return self.dense(x)

def my_model_function():
    # Instantiate MyModel with n_out=8 as per example usage
    return MyModel(n_out=8)

def GetInput():
    # Create a random float32 tensor with batch size 1, height & width 128, 3 channels,
    # which matches the expected input shape as seen in the issue test code
    return tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)

