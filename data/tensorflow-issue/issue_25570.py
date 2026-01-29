# tf.random.uniform((64, 3, 128, 128), dtype=tf.float16) ← inferred input shape from example usage with channels_first format

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a Conv2D layer supporting channels_first data format
        # Although TensorFlow's built-in Conv2D officially supports channels_last on GPU,
        # here we demonstrate channels_first on CPU or MKL-enabled backend.
        self.conv = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=8,
            data_format='channels_first',
            padding='valid',
            activation=None)  # no activation to keep it simple
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1)
    
    def call(self, inputs):
        # inputs shape: (batch, channels, height, width)
        x = self.conv(inputs)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching the input shape expected by MyModel
    # The example used batch=64, channels=3, height=128, width=128 with dtype float16
    return tf.random.uniform((64, 3, 128, 128), dtype=tf.float16)

