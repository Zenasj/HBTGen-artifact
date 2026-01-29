# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from MNIST dataset (default channels_last format)

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super().__init__()
        # The model architecture corresponds to the MNIST example in the issue:
        # 2 Conv2D layers with ReLU, MaxPooling, Dropout,
        # Flatten, Dense with ReLU, Dropout, and final Dense softmax output.
        self.conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')
        self.conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')
        self.pool = layers.MaxPooling2D(pool_size=(2,2))
        self.dropout1 = layers.Dropout(0.25)
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return a fresh MyModel instance, no pretrained weights as none were provided.
    return MyModel()

def GetInput():
    # Generate a random float32 tensor matching the MNIST input used: shape (batch_size, 28, 28, 1)
    # Choose batch size = 32 as a reasonable default for testing
    batch_size = 32  
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

