# tf.random.uniform((B, H, W, 24), dtype=tf.float32) ← Input shape inferred from original model code, channel=24 constant

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, final_depth=24):
        super().__init__()
        self.final_depth = final_depth
        
        # Because the original model uses reshape with dynamic shape,
        # and Conv2D with kernel size (1,3) followed by ReLU with negative_slope=0.2
        self.conv2d = layers.Conv2D(1, kernel_size=(1, 3), activation=None,
                                    padding='same', kernel_initializer='glorot_uniform')
        self.leaky_relu = layers.ReLU(negative_slope=0.2)
    
    def call(self, inputs):
        # inputs: shape (batch, height, width, channels=24)
        shape = tf.shape(inputs)
        
        # Reshape inputs: [batch, height*width, channels, 1]
        # This simulates flattening spatial dims into one dim (height*width),
        # then adding a singleton channel dimension on the right for Conv2D.
        x = tf.reshape(inputs, [shape[0], shape[1] * shape[2], shape[3], 1])
        
        # Apply Conv2D + LeakyReLU
        x = self.conv2d(x)
        x = self.leaky_relu(x)
        
        # Reshape back to original 4D shape: [batch, height, width, channels]
        # Note: Because conv2d output channels=1, the reshape expects channels dimension=original channels.
        # But conv output last dim is 1 channel, so output channels become 1.
        # To maintain output shape compatible with input, 
        # We reshape back to [batch, height, width, channels=original final_depth=24]
        # Assumption here: output channels = 1, but reshape targets original input channels=24.
        # This is ambiguous in original snippet, 
        # so we keep output channels = 1 per conv and reshape output to match first 3 dims,
        # but channel dimension changed to 1 (cannot match 24).
        # To avoid discrepancy, we infer output channels=1 after reshape.
        # So output shape = (batch, height, width, 1)
        
        out = tf.reshape(x, [shape[0], shape[1], shape[2], 1])
        return out


def my_model_function():
    # Return an instance of MyModel with default final_depth=24
    return MyModel()


def GetInput():
    # Return random input tensor matching expected input shape
    # Since input channel is fixed at 24 from original code
    # Spatial dimensions are dynamic (None), 
    # so we provide a specific example shape for testing, e.g. batch=2, height=8, width=8
    batch_size = 2
    height = 8
    width = 8
    channels = 24
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

