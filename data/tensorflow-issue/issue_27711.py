# tf.random.uniform((2, 300, 300, 3), dtype=tf.float32) ← inferred input shape and dtype from issue example
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Conv2D layer with dilation_rate=2 as highlighted in the issue
        self.conv = tf.keras.layers.Conv2D(filters=3, kernel_size=3, dilation_rate=2)

    def call(self, inputs):
        return self.conv(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by the model
    # Batch size = 2, height = 300, width = 300, channels = 3 (RGB-like)
    return tf.random.uniform((2, 300, 300, 3), dtype=tf.float32)

