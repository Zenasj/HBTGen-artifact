# tf.random.uniform((100, 1), dtype=tf.float32) ← inferred input shape and dtype from the issue example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Reproduce the simple model from the issue:
        # Dense(2, input_dim=1, use_bias=True) + LeakyReLU + Dense(1, use_bias=True)
        self.dense1 = tf.keras.layers.Dense(2, use_bias=True)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.dense2 = tf.keras.layers.Dense(1, use_bias=True)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.leaky_relu(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of the MyModel
    return MyModel()

def GetInput():
    # Generate random uniform input matching expected input shape (100, 1)
    # with float32 dtype as typical default for TF layers
    return tf.random.uniform((100, 1), minval=-1.0, maxval=1.0, dtype=tf.float32)

