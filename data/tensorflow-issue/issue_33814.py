# tf.random.uniform((32, 28, 28, 1), dtype=tf.float32) ← inferred input shape from MNIST data and batch size.

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Equivalent to the tf.keras.Sequential model provided:
        # Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        # MaxPooling2D()
        # Flatten()
        # Dense(10, activation='softmax')
        self.conv = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Returns an instance of MyModel, no pretrained or custom loaded weights mentioned.
    return MyModel()

def GetInput():
    # Return a random tensor input matching the input expected by MyModel
    # Batch size 32, height 28, width 28, channels 1, float32 as in preprocess of MNIST images.
    return tf.random.uniform((32, 28, 28, 1), dtype=tf.float32)

