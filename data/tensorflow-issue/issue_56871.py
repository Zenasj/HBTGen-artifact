# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape is (None, None, 3) with 3 color channels, batch size dynamic

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Add, Lambda, Input
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal
import tensorflow.keras.backend as K


class MyModel(tf.keras.Model):
    def __init__(self, scale=3, in_channels=3, num_fea=28, m=4, out_channels=3):
        super(MyModel, self).__init__()
        self.scale = scale
        
        # Bilinear upsampling branch
        self.upsample = UpSampling2D(size=(scale, scale), interpolation='bilinear')
        
        # Feature extraction convolutions with ReLU activations
        self.conv_in = Conv2D(num_fea, 3, padding='same', activation='relu', 
                              kernel_initializer=glorot_normal(), bias_initializer='zeros')
        
        self.conv_blocks = []
        for _ in range(m):
            self.conv_blocks.append(
                Conv2D(num_fea, 3, padding='same', activation='relu', 
                       kernel_initializer=glorot_normal(), bias_initializer='zeros')
            )
        
        # Output convolution that expands channels for subpixel depth_to_space
        self.conv_out = Conv2D(out_channels * (scale ** 2), 3, padding='same', 
                               kernel_initializer=glorot_normal(), bias_initializer='zeros')
        
        # Depth to space layer for pixel shuffle upsampling
        self.depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, scale))
        
        # Final addition and conv layers
        self.add = Add()
        self.conv_final = Conv2D(out_channels, 3, padding='same',
                                 kernel_initializer=glorot_normal(), bias_initializer='zeros')
        
        # Clipping output between 0 and 255, as images are typically in this range
        self.clip_func = Lambda(lambda x: K.clip(x, 0., 255.))
        
    def call(self, inputs, training=False):
        # Bilinear upsample the input directly
        upsampled_inp = self.upsample(inputs)
        
        # Feature extraction path
        x = self.conv_in(inputs)
        for conv in self.conv_blocks:
            x = conv(x)
        x = self.conv_out(x)
        
        # Pixel shuffle upsample by rearranging depth to space
        out = self.depth_to_space(x)
        
        # Add bilinear upsample and conv path results
        x = self.add([upsampled_inp, out])
        
        # Final convolution and clip output to valid image range
        x = self.conv_final(x)
        out = self.clip_func(x)
        
        return out


def my_model_function():
    # Return an instance of MyModel with default parameters as per original code
    return MyModel()


def GetInput():
    # Return a random input tensor with shape (1, 32, 32, 3) and dtype float32 representing an RGB image batch
    # Input shape is flexible (None, None, 3) - here fixed to 32x32 for testing
    return tf.random.uniform((1, 32, 32, 3), minval=0, maxval=255, dtype=tf.float32)

