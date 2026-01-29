# tf.random.uniform((B, T, C), dtype=tf.float32) ← Input shape inferred as (batch, time_steps, channels) for SeparableConv1D

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using causal padding with SeparableConv1D is supported in TF >= 2.4 nightly.
        # Re-implement causal padding manually to avoid the original bug in older TF versions:
        # That is, pad the input tensor manually on the time dimension before convolution.
        
        self.kernel_size = 2
        self.filters = 3
        self.dilation_rate = 1
        
        # Depthwise separable conv1d is equivalent to tf.keras.layers.SeparableConv1D
        # with causal padding.
        # We'll implement causal padding manually in call().
        
        # Depthwise convolution layer with "valid" padding (since padding done manually)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(self.kernel_size, 1),
            dilation_rate=(self.dilation_rate, 1),
            padding='valid',
            depth_multiplier=1,
            use_bias=False
        )
        
        # Pointwise convolution to mix channels
        self.pointwise_conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            padding='valid',
            use_bias=True
        )
    
    def call(self, inputs):
        # inputs is shape (batch_size, time_steps, channels)
        # SeparableConv1D treats inputs in shape (batch, time, channels).
        # DepthwiseConv2D expects 4D input: (batch, height, width, channels).
        # We treat time as "height" and width=1 to simulate 1D conv.
        
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        channels = tf.shape(inputs)[2]
        
        # Add a width dimension of 1 to convert (B, T, C) -> (B, T, 1, C)
        x = tf.expand_dims(inputs, axis=2)
        
        # Compute causal padding size on the time dimension:
        pad_size = self.dilation_rate * (self.kernel_size - 1)
        
        # Pad only on the "left" (start) side of time dimension to preserve causality
        # Pad shape for tf.pad is [[batch], [height], [width], [channels]]
        padding = [[0, 0], [pad_size, 0], [0, 0], [0, 0]]
        x_padded = tf.pad(x, padding)
        
        # Apply depthwise conv with valid padding on padded input
        x_dw = self.depthwise_conv(x_padded)
        
        # Apply pointwise conv
        x_pw = self.pointwise_conv(x_dw)
        
        # Result shape still (B, T, 1, filters), squeeze width dim
        return tf.squeeze(x_pw, axis=2)


def my_model_function():
    # Return an instance of MyModel, weights initialized randomly
    return MyModel()


def GetInput():
    # Generate input tensor shape compatible with MyModel: (batch, time_steps, channels)
    # Use random uniform float32 values, example shape (2, 10, 3)
    return tf.random.uniform((2, 10, 3), dtype=tf.float32)

