# tf.random.uniform((B, None, 512, 64), dtype=tf.float32) ← inferred input shape from Input(shape=(None, 512, 64))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__(name='')
        self.depth = 5
        self.in_channels = 64
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        
        # Create padding and conv layers with increasing dilation rates 2^i
        for i in range(self.depth):
            dil = 2**i
            # Compute padding length according to formula: twidth + (dil-1)*(twidth-1) - 1
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i+1), tf.keras.layers.ZeroPadding2D(((pad_length, 0), (1, 1))))
            setattr(self, 'conv{}'.format(i+1), 
                    tf.keras.layers.Conv2D(filters=self.in_channels, kernel_size=self.kernel_size, 
                                           dilation_rate=(dil, 1), padding='valid'))
            setattr(self, 'norm{}'.format(i+1), tf.keras.layers.LayerNormalization(axis=[1,2,3]))
            setattr(self, 'prelu{}'.format(i+1), tf.keras.layers.PReLU(shared_axes=[1, 2]))

    def call(self, input_tensor):
        skip = input_tensor
        for i in range(self.depth):
            # Fetch padding, conv, norm, prelu layers for this depth
            pad_layer = getattr(self, 'pad{}'.format(i+1))
            conv_layer = getattr(self, 'conv{}'.format(i+1))
            norm_layer = getattr(self, 'norm{}'.format(i+1))
            prelu_layer = getattr(self, 'prelu{}'.format(i+1))
            
            x = pad_layer(skip)
            x = conv_layer(x)
            x = norm_layer(x)
            x = prelu_layer(x)
            # Concatenate along channel axis (axis=3)
            skip = tf.concat([x, skip], axis=3)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create input tensor matching the expected input shape: (batch_size, None, 512, 64)
    # Using None for time dimension, batch=1 here, channels last format
    batch_size = 1
    time_dim = 16  # arbitrary sequence length to simulate "None"
    height = 512
    channels = 64
    # Random float32 tensor with shape (batch_size, time_dim, height, channels)
    return tf.random.uniform((batch_size, time_dim, height, channels), dtype=tf.float32)

