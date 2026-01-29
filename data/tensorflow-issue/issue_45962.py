# tf.random.uniform((B, 360, 640, C), dtype=...)  ← input shape inferred from DataGenerator dim=(360,640) mostly typical for image sizes

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Minimal example model which would accept inputs of shape (B, 360, 640, C)
        # Since original issue only shows generator and training setup, we build a simple conv net here.
        # Assume 3 channel input (e.g. RGB images) as a placeholder.
        self.conv1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')
        self.pool = tf.keras.layers.MaxPooling2D((2,2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of the model
    return MyModel()

def GetInput():
    # Return a random tensor consistent with what the original DataGenerator would produce.
    # The original generator uses dim=(360,640) and batchSize=32 by default
    # We don't know channels exactly; let's assume 3 as typical RGB.
    batch_size = 32
    height = 360
    width = 640
    channels = 3
    # For simplicity, just produce float32 in [0, 1)
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

