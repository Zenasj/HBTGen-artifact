# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assumed input shape with batch B, height H, width W, channels C

import tensorflow as tf
from tensorflow.keras import activations, initializers, regularizers, constraints

class MyModel(tf.keras.Model):
    def __init__(self,
                 input_channels=32,
                 output_channels=64,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 groups=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        A Keras Model implementing grouped convolution for 2D inputs,
        mimicking a grouped Conv2D layer by splitting input channels into groups,
        applying a standard Conv2D on each split, then concatenating results.

        Args mostly match tf.keras.layers.Conv2D parameters, but with explicit groups.

        Assumptions:
        - Input shape: (batch, height, width, channels)
        - input_channels must be divisible by groups
        - output_channels must be divisible by groups
        """
        super(MyModel, self).__init__()

        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be divisible by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be divisible by the value of groups.")

        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.groups = groups
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups

        # Create a Conv2D layer per group
        self.conv_list = []
        for i in range(groups):
            conv_layer = tf.keras.layers.Conv2D(
                filters=self.group_out_num,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                data_format=data_format,
                dilation_rate=dilation_rate,
                activation=activations.get(activation),
                use_bias=use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                activity_regularizer=self.activity_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                **kwargs)
            self.conv_list.append(conv_layer)

    def call(self, inputs, **kwargs):
        """
        Splits the input along the channel axis into groups,
        applies corresponding Conv2D layers, and concatenates outputs
        along the channel axis.
        """
        feature_map_list = []
        # Assuming channels last: inputs.shape = (B, H, W, C)
        for i in range(self.groups):
            # Slice input channels for the i-th group
            x_i = inputs[:, :, :, i * self.group_in_num: (i + 1) * self.group_in_num]
            x_i = self.conv_list[i](x_i)
            feature_map_list.append(x_i)
        # Concatenate outputs from each group on the channel axis
        out = tf.concat(feature_map_list, axis=-1)
        return out


def my_model_function():
    """
    Returns an instance of MyModel with default initialization.
    Default parameters assume input_channels=32, output_channels=64,
    kernel_size=3x3, groups=4 as an example.
    """
    return MyModel(
        input_channels=32,
        output_channels=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        activation='relu',
        groups=4,
        use_bias=True
    )


def GetInput():
    """
    Returns a random input tensor compatible with MyModel.
    Since my_model_function above expects input_channels=32,
    we create input tensor with shape (batch, height, width, 32).
    Use batch size = 2, height = 64, width = 64 as example.
    """
    batch_size = 2
    height = 64
    width = 64
    channels = 32  # must match input_channels in MyModel

    # Return float32 tensor with values between 0 and 1
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

