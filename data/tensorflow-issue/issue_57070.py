# tf.random.uniform((B, 28, 28, 1), dtype=tf.float64) ← inferred input shape based on MNIST dataset flattened to 784 features, input dtype float64

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # MNIST images are 28x28 grayscale images, flattened into 784 features
        self.dense1 = tf.keras.layers.Dense(15, activation='sigmoid', input_shape=(28*28,))
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        # inputs expected shape: (batch_size, 784) and dtype float64 as per issue example
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create a random input tensor that matches MNIST flattened input shape:
    # (batch_size, 784) with dtype float64 (float64 used in original example)
    # Assume a small batch size for testing
    batch_size = 16
    # Generate random float64 tensor in range [0,1) similar to normalized MNIST input
    return tf.random.uniform((batch_size, 28*28), dtype=tf.float64)

