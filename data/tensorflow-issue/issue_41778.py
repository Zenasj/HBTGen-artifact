# tf.random.uniform((1, 10, 5, 1), dtype=tf.float32) ← input shape inferred from: Input(shape=[10, 5, 1])

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv2D with 32 filters, kernel size 3x3, default stride 1 and padding 'valid'
        self.conv = tf.keras.layers.Conv2D(32, (3, 3))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Instantiate and return the model.
    # No special weight loading mentioned, so default initialization.
    return MyModel()

def GetInput():
    # Return a random tensor matching the input expected by MyModel.
    # Shape: batch=1, height=10, width=5, channels=1 (from Input(shape=[10,5,1]))
    # Using dtype=tf.float32 as typical for Conv2D.
    return tf.random.uniform((1, 10, 5, 1), dtype=tf.float32)

