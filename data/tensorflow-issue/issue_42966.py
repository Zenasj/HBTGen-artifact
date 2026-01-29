# tf.random.uniform((1, 250, 250, 3), dtype=tf.float32) ← inferred input shape from tutorial example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This replicates the initial_model from the referenced tutorial:
        # Conv2D layers with specified filters, kernel sizes, strides and activations
        self.conv1 = tf.keras.layers.Conv2D(32, 5, strides=2, activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(32, 3, activation="relu")
        self.conv3 = tf.keras.layers.Conv2D(32, 3, activation="relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

def my_model_function():
    # Return an instance of MyModel initialized with the Conv2D layers and weights uninitialized (default)
    return MyModel()

def GetInput():
    # Return a random tensor matching the input shape expected by MyModel
    # Batch size 1, height 250, width 250, channels 3, dtype float32 as typical image input
    return tf.random.uniform((1, 250, 250, 3), dtype=tf.float32)

