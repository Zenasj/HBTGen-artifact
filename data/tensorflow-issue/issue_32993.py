# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← The input matches MNIST dataset shape (batch, height, width, channels)

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Convolutional layer with 32 filters, 3x3 kernel, ReLU activation
        self.conv1 = Conv2D(32, 3, activation='relu')
        # Flatten layer to convert 2D feature maps to 1D feature vectors
        self.flatten = Flatten()
        # Dense (fully connected) layer with 128 units and ReLU activation
        self.d1 = Dense(128, activation='relu')
        # Output Dense layer with 10 units for classification logits, softmax applied in call
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        # Forward pass
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a random float32 tensor shaped like a batch of MNIST images with 1 channel
    # Assume batch size 32 to match training batch size
    return tf.random.uniform((32, 28, 28, 1), dtype=tf.float32)

