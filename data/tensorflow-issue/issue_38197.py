# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← inferred input shape for MNIST grayscale images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the architecture from the issue: 
        # Conv2D(32, 3, relu) -> MaxPool2D -> Conv2D(64, 3, relu) -> MaxPool2D 
        # -> Flatten -> Dense(64, relu) -> Dense(10)
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D()
        self.conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits output

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits


def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    return model


def GetInput():
    # Return a random tensor input that matches the model's expected input.
    # MNIST input shape: batch size unknown (use batch 64) x 28 x 28 x 1 grayscale.
    B = 64
    H, W, C = 28, 28, 1
    # Uniform random tensor in [0,1) with float32 dtype simulates normalized images
    return tf.random.uniform((B, H, W, C), dtype=tf.float32)

