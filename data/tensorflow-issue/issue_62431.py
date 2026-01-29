# tf.random.uniform((3, 5, 1, 1, 1), dtype=tf.float64)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Parameters derived and inferred from the issue details:
        filters = 2
        kernel_size = [3, 3, 3]
        strides = [2, 2, 2]
        padding = "valid"
        output_padding = None
        data_format = "channels_last"
        dilation_rate = [1, 1, 1]
        activation = "relu"
        use_bias = False
        # kernel_initializer, bias_initializer etc. are None by default; 
        # we keep them as None to match issue setup
        
        self.conv3d_transpose = tf.keras.layers.Conv3DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            output_padding=output_padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias
        )

    def call(self, inputs):
        return self.conv3d_transpose(inputs)

def my_model_function():
    return MyModel()

def GetInput():
    # From the issue, input shape is [3, 5, 1, 1, 1], dtype=float64,
    # minval=0, maxval=0 results in zeros, but likely the maxval 0 was a typo or to force zeros.
    # To generate a meaningful random input tensor we use minval=0, maxval=1.
    # Use float64 dtype to match issue.
    input_tensor = tf.random.uniform(
        shape=(3, 5, 1, 1, 1),
        minval=0,
        maxval=1,
        dtype=tf.float64
    )
    return input_tensor

