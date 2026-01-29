# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape inferred as (batch, 240, 480, 3) from original example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__(name="StylePredictionModelDummy_Fixed")
        # Fixed: feature_extractor is an instance attribute (not a static class attribute)
        self.feature_extractor = tf.keras.layers.Conv2D(
            filters=1, kernel_size=9, strides=5, padding='same', name="dummy_conv"
        )

    def call(self, inputs, training=None, mask=None):
        # Forward pass applies Conv2D layer
        x = self.feature_extractor(inputs)
        return x

def my_model_function():
    # Returns a new instance of the fixed model
    return MyModel()

def GetInput():
    # Return a random float32 tensor with shape (1, 240, 480, 3)
    # Based on example: (1, 960//4, 1920//4, 3) == (1, 240, 480, 3)
    return tf.random.uniform(shape=(1, 240, 480, 3), dtype=tf.float32)

