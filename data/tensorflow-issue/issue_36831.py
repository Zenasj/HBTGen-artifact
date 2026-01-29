# tf.random.uniform((B, H, W, C), dtype=tf.float32) 
# Input is a tuple of 3 tensors (input_1, input_2, input_3) all with shape (batch_size, height, width, channels)
# Assuming inputs are image-like tensors for Conv2D usage with 4 channels for example (B, 64, 64, 3)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, name="MyModel", **kwargs):
        super(MyModel, self).__init__(name=name, **kwargs)
        # A simple convolutional layer as in the example
        self.conv = tf.keras.layers.Conv2D(
            filters=32, kernel_size=7, strides=1, padding="same"
        )

    def call(self, inputs, training=True, **kwargs):
        # Inputs is expected to be a tuple/list of 3 tensors
        input_1, input_2, input_3 = inputs
        
        # Simple forward pass using only input_1 as per original example
        out = self.conv(input_1)
        return out, None  # Returning tuple to mimic (predictions, warping_output)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a tuple of 3 inputs, each a random tensor matching expected input shape
    # Assuming batch size = 4, image size 64x64, channels=3
    B, H, W, C = 4, 64, 64, 3
    input_1 = tf.random.uniform((B, H, W, C), dtype=tf.float32)
    input_2 = tf.random.uniform((B, H, W, C), dtype=tf.float32)
    input_3 = tf.random.uniform((B, H, W, C), dtype=tf.float32)
    return (input_1, input_2, input_3)

