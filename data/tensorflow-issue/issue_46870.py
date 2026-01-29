# tf.random.uniform((1, 256, 64, 3), dtype=tf.float32) ← inferred input shape from input layer: (256, 64, 3), batch size 1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the issue setup:
        # Input shape: (256,64,3)
        # Dense layer without bias applied channel-wise (last dim=3)
        # The Dense layer is applied on each spatial location independently,
        # so it works as a 1x1 Conv2D with output channels=1 and use_bias=False.
        # This matches use_bias=False Dense layer applied on last dimension.
        self.dense_no_bias = tf.keras.layers.Dense(
            units=1, use_bias=False, name="dense_no_bias"
        )
        self.global_max_pool = tf.keras.layers.GlobalMaxPool2D()

    def call(self, inputs):
        # inputs expected shape: (batch, 256, 64, 3)
        # Dense layer applies on last dim
        x = self.dense_no_bias(inputs)  # shape: (batch, 256, 64, 1)
        x = self.global_max_pool(x)     # shape: (batch, 1)
        return x


def my_model_function():
    # Returns model instance matching original issue's architecture
    return MyModel()


def GetInput():
    # Return random input tensor with the expected shape and dtype
    # batch size = 1 is typical and consistent with TFLite conversion example
    return tf.random.uniform(shape=(1, 256, 64, 3), dtype=tf.float32)

