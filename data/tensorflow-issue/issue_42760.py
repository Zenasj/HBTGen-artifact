# tf.random.uniform((B, 32, 32, 3), dtype=tf.float32) ← inferred input shape from CIFAR-10 dataset

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # PReLU layer with a proper initializer instead of 'Identity' because Identity is only for 2D matrices.
        # Here we use 'zeros' initializer as a safe default for alpha (learnable slope).
        self.prelu = tf.keras.layers.PReLU(alpha_initializer='zeros')

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10)  # CIFAR-10 has 10 classes

    def call(self, inputs):
        x = self.prelu(inputs)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Instantiate and return MyModel
    return MyModel()

def GetInput():
    # Return a random tensor with shape matching CIFAR-10 images (batch size 1, 32x32 RGB images)
    # Values between 0 and 1 as data is normalized in original example.
    # Batch size is set to 1 to allow flexibility.
    input_shape = (1, 32, 32, 3)
    return tf.random.uniform(input_shape, minval=0, maxval=1, dtype=tf.float32)

