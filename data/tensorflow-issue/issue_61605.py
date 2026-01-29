# tf.random.uniform((3, 74, 74, 256), dtype=tf.float32) ← Input shape mentioned in issue reproduction snippet

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Based on the issue, MaxPool2D expects pool_size and strides as tuples of integers.
        # The example had unreasonable large pool_size values (1e+38, 16777216),
        # which causes failures. A reasonable pool_size like (2, 2) is typically used.
        self.pool_size = (2, 2)
        self.strides = (2, 2)
        self.padding = "same"
        self.maxpool = tf.keras.layers.MaxPool2D(
            pool_size=self.pool_size,
            strides=self.strides,
            padding=self.padding
        )

    def call(self, inputs):
        # inputs is expected to be a 4D tensor: [batch, height, width, channels]
        return self.maxpool(inputs)

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel: (3, 74, 74, 256)
    return tf.random.uniform((3, 74, 74, 256), dtype=tf.float32)

