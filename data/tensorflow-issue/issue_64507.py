# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← inferred input shape for MNIST grayscale images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the Sequential model in the issue: two Conv2D layers, MaxPooling, Flatten, Dense, Dense
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Create a random input tensor with shape (batch_size, 28, 28, 1)
    # MNIST images are 28x28 grayscale, batch size chosen as 128 per the issue example
    batch_size = 128
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

