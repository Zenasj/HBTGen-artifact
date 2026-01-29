# tf.random.uniform((1, 3, 128, 128), dtype=tf.float32) ← Input tensor shape inferred from PyTorch example and TF graph

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Since the main problematic op is equivalent to PyTorch's F.interpolate with scale_factor=2 and mode='nearest',
        # we implement it here in TF as a simple nearest neighbor upsampling by a factor of 2.
        self.upsample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest')

    def call(self, inputs):
        # inputs shape expected: (batch, height, width, channels)
        # The PyTorch tensor was (N,C,H,W)= (1,3,128,128), typical TF format is NHWC
        # Need to transpose input from NCHW to NHWC before upsampling, and transpose back
        # because the input that was used is in NCHW format.
        # For this reason, we add transpose steps.

        # Transpose NCHW -> NHWC
        x = tf.transpose(inputs, perm=[0, 2, 3, 1])
        
        # Apply nearest neighbor upsampling by scale factor 2
        x = self.upsample(x)
        
        # Transpose back NHWC -> NCHW to maintain consistency
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor compatible with MyModel's expected input:
    # shape (1, 3, 128, 128), dtype float32, similar to input used in the PyTorch example
    return tf.random.uniform((1, 3, 128, 128), dtype=tf.float32)

