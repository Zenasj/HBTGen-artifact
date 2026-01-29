# tf.random.uniform((1, 5, 5, 3), dtype=tf.float32) ← inferred input shape from example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers as per the example discussed in the issue
        self.conv2d = tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(2, 2)
        )
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
            units=4,
            activation='softmax'
        )
    
    def call(self, x, training=False):
        # Forward pass through Conv2D -> Flatten -> Dense(Softmax)
        x = self.conv2d(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Produce a random tensor input matching the input shape expected: (1, 5, 5, 3)
    # Batch size 1 is used to align with the example in the issue
    input_shape = (1, 5, 5, 3)
    return tf.random.uniform(input_shape, dtype=tf.float32)

