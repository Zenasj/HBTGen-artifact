# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← input shape corresponds to MNIST grayscale images of 28x28

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # MNIST CNN model replicated from the reported issue example
        self.reshape = tf.keras.layers.Reshape((28, 28, 1))
        self.conv = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits output layer

    def call(self, inputs, training=False):
        x = self.reshape(inputs)        # Reshape from (28, 28) to (28, 28, 1)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Instantiate and compile the model similarly to the issue example
    model = MyModel()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=['accuracy'])
    return model

def GetInput():
    # Return a random batch of grayscale MNIST-like images with shape (batch_size, 28, 28)
    # Using batch size of 64 (common per_worker_batch_size from issue)
    batch_size = 64
    # Generate floats in [0, 1] range as in the dataset preprocessing
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

