# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)
import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a small CNN similar to the example to match input shape (28,28,1)
        self.conv = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28,28,1))
        self.pool = tf.keras.layers.MaxPool2D((2, 2))
        self.bn = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.bn(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        out = self.dense2(x)
        return out

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # This model is ready to be compiled with loss and optimizer externally.
    return model

def GetInput():
    # Return a random tensor input matching the expected input shape: (batch_size, 28, 28, 1)
    # Assume a batch size of 8 for demonstration
    batch_size = 8
    # Random float32 tensor normalized between 0 and 1, matching example preprocessing
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32, minval=0, maxval=1)

