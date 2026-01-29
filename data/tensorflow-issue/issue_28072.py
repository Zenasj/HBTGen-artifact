# tf.random.normal((B, H, W, C), dtype=tf.float32) ← the input shape is [batch_size, IMAGE_SIZE, IMAGE_SIZE, channels]

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, kernel_size, channels):
        super(MyModel, self).__init__()
        # Normal convolution
        self.normal_conv = tf.keras.layers.Conv2D(channels, kernel_size, padding="same")
        # Rank-separable convolutions: two 1D convs (kernel_size x 1) then (1 x kernel_size)
        self.rank_conv_1 = tf.keras.layers.Conv2D(channels, (kernel_size, 1), padding="same")
        self.rank_conv_2 = tf.keras.layers.Conv2D(channels, (1, kernel_size), padding="same")
        # Depth-wise separable conv: depthwise conv + pointwise conv(1x1)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size, padding="same")
        self.pointwise_conv = tf.keras.layers.Conv2D(channels, 1, padding="same")
    
    def call(self, x):
        # Compute outputs of all three convolutions
        normal_out = self.normal_conv(x)
        rank_out = self.rank_conv_1(x)
        rank_out = self.rank_conv_2(rank_out)
        depth_out = self.depthwise_conv(x)
        depth_out = self.pointwise_conv(depth_out)

        # As per the issue's context, one might want to compare or benchmark these.
        # Here, we concatenate outputs along channel dim for a combined demonstration.
        # The user can interpret or apply timing/comparison externally.
        combined = tf.concat([normal_out, rank_out, depth_out], axis=-1)
        return combined

def my_model_function():
    # For demonstration, assume kernel_size=3 and channels=64 as those were typical benchmarking params.
    # Users can change them as needed to instantiate model with other params.
    return MyModel(kernel_size=3, channels=64)

def GetInput():
    # Following the reported usage, IMAGE_SIZE=320, batch_size chosen based on 2048 total channels*batch_size
    # For channels=64 (default), batch_size=2048//64=32 as an example
    IMAGE_SIZE = 320
    channels = 64
    batch_size = 2048 // channels  # 32
    # Input tensor shaped [batch_size, IMAGE_SIZE, IMAGE_SIZE, channels]
    return tf.random.normal(shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, channels), dtype=tf.float32)

