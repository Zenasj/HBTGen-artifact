# tf.random.uniform((1, 256, 256, 3), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple Conv2D layer as per the original model in the issue
        self.conv = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid')

    def call(self, inputs):
        return self.conv(inputs)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input shape (batch=1, 256x256 RGB image)
    # Using float32 dtype as shown in the inference code on Raspberry Pi
    return tf.random.uniform((1, 256, 256, 3), dtype=tf.float32)

