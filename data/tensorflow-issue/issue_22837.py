# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← Assuming input like MNIST grayscale images, 28x28

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # To avoid NotImplementedError on .save(), we define input shape here.
        # This model mirrors the sequential example: Flatten -> Dense(128, ReLU) -> Dense(128, ReLU) -> Dense(10, softmax)
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))  # Input shape required for saving
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

def my_model_function():
    # Return an instance of MyModel, ready for use and saving.
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected input shape: batch 32, 28x28 grayscale images
    # dtype float32 as typical for image inputs
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

