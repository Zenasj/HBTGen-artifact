# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Batch dimension B is variable

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Same layers as the Sequential model described in the issue
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits output
        
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Return an instance of MyModel and compile it similarly to the issue
    model = MyModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

def GetInput():
    # Return a batch of random data matching input shape (batch_size, 28, 28)
    # Batch size chosen arbitrarily as 8 per the issue
    batch_size = 8
    # MNIST images are grayscale with pixel values typically in [0,255]
    # but model expects normalized float. Here using uniform [0,1]
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

