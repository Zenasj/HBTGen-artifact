# tf.random.uniform((None, None, 3), dtype=tf.float32)  ← Input shape is a 3-channel image with variable height and width

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model computes variance over the last three input dimensions,
        # replicating tf.math.reduce_variance behavior as in the issue.
        # We will compute variance as E[x²] - (E[x])² along all spatial and channel dims.
    
    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs shape: (H, W, 3) or (None, None, 3)
        # Compute the mean over height, width, and channels:
        mean = tf.reduce_mean(inputs)
        mean_sq = tf.reduce_mean(tf.square(inputs))
        variance = mean_sq - tf.square(mean)
        return variance

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Following the input signature from the original TF Module:
    # Shape: (height, width, channels) with channels = 3, height and width variable.
    # We'll provide a random float32 tensor with shape (256, 256, 3) as a reasonable default.
    # Values on [0, 255), since original input images were of pixel intensity range.
    input_tensor = tf.random.uniform(shape=(256, 256, 3), minval=0, maxval=255, dtype=tf.float32)
    return input_tensor

