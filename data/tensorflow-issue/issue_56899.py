# tf.random.normal((1, 640, 480, 3), dtype=tf.float32) ← Input shape inferred from issue representative_data_gen

import tensorflow as tf
import tensorflow_addons as tfa

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a simple keras model matching the issue example:
        # Input shape (640, 480, 3), Conv2D(64, 3x3), then GELU activation from tf-addons
        self.conv = tf.keras.layers.Conv2D(64, kernel_size=3)
        self.gelu = tfa.layers.GELU(False, name='gelu')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.gelu(x)
        return x

def my_model_function():
    # Return an instance of MyModel, mimicking the tf.keras.Model in the issue
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the model input shape expected: (1, 640, 480, 3), float32
    # Using batch size 1 as in representative dataset generator
    return tf.random.normal(shape=(1, 640, 480, 3), dtype=tf.float32)

