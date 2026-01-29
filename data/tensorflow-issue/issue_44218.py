# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST dataset in the issue

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the model architecture described in the issue:
        # Input shape: (28, 28, 1)
        self.conv = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.pool = layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.5)
        self.dense = layers.Dense(10, activation="softmax")
        
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        return self.dense(x)

def my_model_function():
    # Instantiate and return the model. The caller can compile it as needed.
    return MyModel()

def GetInput():
    # Return a random input tensor matching expected shape (batch_size, 28, 28, 1)
    # Use batch size 128 as per the example batch size in the issue for more representative input
    batch_size = 128
    input_shape = (28, 28, 1)
    return tf.random.uniform((batch_size, *input_shape), dtype=tf.float32)

