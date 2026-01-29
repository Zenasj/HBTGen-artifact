# tf.random.uniform((B, 10), dtype=tf.float32) ← inferred input shape based on example using tf.zeros([3, 10])

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # A single dense layer with 5 output units as in the example
        self.layer = tf.keras.layers.Dense(5)

    def call(self, x):
        # Forward pass simply applies the dense layer
        return self.layer(x)

def my_model_function():
    # Instantiate and return MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the expected (batch_size, 10)
    # Using batch size 3 to mirror the example
    return tf.random.uniform((3, 10), dtype=tf.float32)

