# tf.random.normal((1, 128, 1, 1), dtype=tf.float32) ← Input shape (batch=1, time=128, slices=1, channels=1)

import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
    def __init__(self, filters, dilation):
        super(MyLayer, self).__init__()
        # Three sequential Conv2D layers with kernel (3,1), padding 'same'
        # Second conv uses dilation_rate for the time dimension

        self.c0 = tf.keras.layers.Conv2D(filters=filters,
                                         kernel_size=(3, 1),
                                         padding='same',
                                         activation=None)

        self.c1 = tf.keras.layers.Conv2D(filters=filters,
                                         kernel_size=(3, 1),
                                         dilation_rate=(dilation, 1),
                                         padding='same',
                                         activation=None)

        self.c2 = tf.keras.layers.Conv2D(filters=filters,
                                         kernel_size=(3, 1),
                                         padding='same',
                                         activation=None)

    def call(self, x):
        # Input shape: (B, time, slices, channels)
        x = self.c0(x)
        x = self.c1(x)
        x = self.c2(x)
        return x


class MyModel(tf.keras.Model):
    def __init__(self, filters=32, dilation=1):
        super(MyModel, self).__init__()
        # Initialize the MyLayer with given filters and dilation
        # This wraps the original model of 3 conv2d layers with possible dilation
        self.mylayer = MyLayer(filters=filters, dilation=dilation)

    def call(self, x):
        # Forward pass through the MyLayer
        return self.mylayer(x)

def my_model_function():
    # Returns an instance of MyModel with default parameters as in the original code
    # Using filters=32 and dilation=1 by default (which yields almost equivalent TFLite/TF results)
    return MyModel(filters=32, dilation=1)

def GetInput():
    # Return a random tensor matching input shape used in the original code
    # Shape (batch=1, time=128, slices=1, channels=1), dtype float32
    return tf.random.normal((1, 128, 1, 1), dtype=tf.float32)

