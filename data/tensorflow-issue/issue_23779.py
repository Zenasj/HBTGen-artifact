# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Input shape is (batch_size, 28, 28) grayscale images from MNIST

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Flatten input 28x28 → 784
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        # Dense layer with 512 units and ReLU activation
        self.dense1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        # Dropout with rate 0.2
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        # Dense layer with 128 units and ReLU activation
        self.dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        # Dropout with rate 0.2
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        # Final output layer with 10 units and softmax activation for classification
        self.out = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.out(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random float32 tensor shaped (batch_size, 28, 28)
    # Batch size is arbitrarily chosen as 32 for demonstration.
    # Values normalized roughly as MNIST pixel values scaled in [0,1].
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

