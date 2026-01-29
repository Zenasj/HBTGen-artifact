# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32)  ← Input shape inferred from MNIST dataset grayscale images

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the CNN model architecture matching the mnist_convnet example
        self.conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.5)
        self.dense = layers.Dense(10, activation="softmax")  # 10 classes for MNIST

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        return self.dense(x)

def my_model_function():
    # Returns an instance of MyModel, no pretrained weights specified
    return MyModel()

def GetInput():
    # Return a batch of inputs with shape (batch_size, 28, 28, 1)
    # Use batch_size=128 consistent with example
    batch_size = 128
    # Random uniform inputs simulating MNIST images normalized [0, 1]
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

