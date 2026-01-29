# tf.random.uniform((B, H, W, 1), dtype=tf.float32) - 
# Input shape here denotes arbitrary batch size (B), height (H), width (W), and 1 channel

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # A simple convolution layer with 1 filter, kernel size 3x3, 'same' padding to preserve spatial dimensions
        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=3,
            padding='same',
            activation=None,
        )
    
    # To ensure the SavedModel signature allows arbitrary spatial dimensions,
    # annotate call with a tf.function and input_signature specifying None for height and width.
    @tf.function(input_signature=[tf.TensorSpec([None, None, None, 1], tf.float32)])
    def call(self, inputs):
        # Forward pass directly applies the conv layer
        return self.conv(inputs)

def my_model_function():
    # Returns an instance of MyModel with the initialized conv layer.
    return MyModel()

def GetInput():
    # Returns a random input tensor compatible with MyModel input_signature:
    # Batch size arbitrarily set to 1, height and width to 32, channels to 1
    return tf.random.uniform((1, 32, 32, 1), dtype=tf.float32)

