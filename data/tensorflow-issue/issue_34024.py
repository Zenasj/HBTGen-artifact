# tf.random.normal((B, H, W, C), dtype=tf.float32) ← Typical NHWC input tensor for 2D conv, e.g. (8, 32, 32, 16)

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
import tensorflow.keras.activations as activations
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.initializers as initializers
import tensorflow.keras.constraints as constraints
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn

class MyModel(tf.keras.Model):
    """
    Implements a Group Convolution 2D Layer based on TensorFlow internal Conv2D, 
    supporting group convolution by splitting input channels accordingly.

    This model implements the logic from the reported GitHub issue for group convolutions:
    - Input channel count must be divisible by groups.
    - Output filters must be divisible by groups.
    - Uses tf.nn.conv2d for group convolution (supported mainly on GPU).
    """

    def __init__(self, filters=16, kernel_size=(3,3), groups=4, strides=(1,1), padding='SAME',
                 data_format='NHWC', dilation_rate=(1,1), activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None):
        super().__init__()
        if filters % groups != 0:
            raise ValueError(f"Output filters ({filters}) must be divisible by groups ({groups})")

        self.filters = filters
        self.groups = groups
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = padding.upper()
        self.data_format = data_format
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.built = False  # mark for build later

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if conv_utils.normalize_data_format(self.data_format) == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        if input_dim % self.groups != 0:
            raise ValueError(f"Input channel dimension ({input_dim}) must be divisible by groups ({self.groups}).")

        # Kernel shape: (kernel_height, kernel_width, input_channels_per_group, filters)
        kernel_shape = self.kernel_size + (input_dim // self.groups, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        # Call tf.nn.conv2d with the parameters ensuring grouped convolution.
        outputs = tf.nn.conv2d(inputs,
                               filters=self.kernel,
                               strides=[1, self.strides[0], self.strides[1], 1] if self.data_format == 'NHWC' else [1,1,self.strides[0],self.strides[1]],
                               padding=self.padding,
                               data_format='NHWC' if self.data_format is None else self.data_format,
                               dilations=[1, self.dilation_rate[0], self.dilation_rate[1], 1] if self.data_format == 'NHWC' else [1,1,self.dilation_rate[0],self.dilation_rate[1]],
                               name=self.name)

        if self.use_bias:
            if self.data_format == 'channels_first':
                # NCHW data format bias add
                if len(outputs.shape) == 3:
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                # NHWC bias add (most common)
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'groups': self.groups,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config() if hasattr(super(), 'get_config') else {}
        # Return combined config as a dictionary
        return {**base_config, **config}

def my_model_function():
    # Return an instance of MyModel configured to test group conv
    # Default parameters: filters=16, kernel_size=(3,3), groups=4
    return MyModel(filters=16, kernel_size=(3,3), groups=4, padding='SAME')

def GetInput():
    # Return a random tensor input compatible with MyModel's input requirement:
    # Input shape NHWC: batch, height, width, channels
    # Channels must be divisible by groups=4; choosing channels=16 to match filters
    batch = 8
    height = 32
    width = 32
    channels = 16  # Must be divisible by group=4
    return tf.random.uniform(shape=(batch, height, width, channels), dtype=tf.float32)

