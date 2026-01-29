# tf.random.uniform((1, 28, 28, 16), dtype=tf.float32) ← inferred input shape from the example Conv2D input_shape

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A Conv2D layer with dilation_rate=(2, 2), filters=16, kernel_size=7, bias initialized to ones
        self.conv = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=7,
            dilation_rate=(2, 2),
            use_bias=True,
            bias_initializer='ones',
            input_shape=(28, 28, 16)
        )

    def call(self, inputs):
        return self.conv(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input with shape (1, 28, 28, 16), dtype float32,
    # matching the Conv2D input shape expected by MyModel
    return tf.random.uniform((1, 28, 28, 16), minval=-1, maxval=1, dtype=tf.float32)

