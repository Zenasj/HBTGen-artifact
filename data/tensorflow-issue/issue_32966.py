# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← assuming channels_last input format

import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.python.keras.utils import conv_utils, tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import keras_utils
from tensorflow.python.keras import backend
from tensorflow.python.framework import tensor_shape


class MyModel(tf.keras.Model):
    """
    This model implements a DepthwiseConv2D layer extended to support dilation_rate argument,
    with channels_last data format assumed for input and output.

    The implementation reflects the fixed DepthwiseConv2D class from TF issue #32966,
    adding dilation_rate support to the DepthwiseConv2D convolution,
    including build, call, compute_output_shape, and get_config methods.

    The forward pass uses backend.depthwise_conv2d with strides, padding, dilation_rate arguments,
    use_bias and activation support.
    """

    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format='channels_last',
                 activation=None,
                 use_bias=True,
                 dilation_rate=(1, 1),
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MyModel, self).__init__(**kwargs)
        # Parameters for the DepthwiseConv2D
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.depth_multiplier = depth_multiplier
        self.data_format = data_format
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')

        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Weights will be created on build()
        self.depthwise_kernel = None
        self.bias = None
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        # input_shape expected in format (batch, height, width, channels) or channels_first
        if len(input_shape) != 4:
            raise ValueError('Inputs to `MyModel` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`MyModel` should be defined. Found `None`.')

        input_dim = int(input_shape[channel_axis])
        
        depthwise_kernel_shape = (self.kernel_size[0],
                                 self.kernel_size[1],
                                 input_dim,
                                 self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            name='depthwise_kernel',
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint,
            trainable=True,
            dtype=self.dtype)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(input_dim * self.depth_multiplier,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})

        self.built = True

    def call(self, inputs):
        outputs = backend.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.use_bias:
            outputs = backend.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            batch = input_shape[0]
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        else:
            batch = input_shape[0]
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(
            rows,
            self.kernel_size[0],
            padding=self.padding,
            stride=self.strides[0],
            dilation=self.dilation_rate[0])
        cols = conv_utils.conv_output_length(
            cols,
            self.kernel_size[1],
            padding=self.padding,
            stride=self.strides[1],
            dilation=self.dilation_rate[1])

        if self.data_format == 'channels_first':
            return (batch, out_filters, rows, cols)
        else:
            return (batch, rows, cols, out_filters)

    def get_config(self):
        config = super(MyModel, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'depth_multiplier': self.depth_multiplier,
            'data_format': self.data_format,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'dilation_rate': self.dilation_rate,
            'depthwise_initializer': initializers.serialize(self.depthwise_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'depthwise_regularizer': regularizers.serialize(self.depthwise_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'depthwise_constraint': constraints.serialize(self.depthwise_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        })
        return config


def my_model_function():
    # Return an instance of MyModel with some typical parameters.
    # User can customize as needed.
    return MyModel(
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        depth_multiplier=1,
        data_format='channels_last',
        activation=None,
        use_bias=True,
        dilation_rate=(1, 1))


def GetInput():
    # Generate a random input tensor compatible with MyModel's expected input:
    # 4D tensor: (batch, height, width, channels)
    # Here, let's assume batch=2, height=32, width=32, channels=3 (e.g. RGB image-like)
    batch = 2
    height = 32
    width = 32
    channels = 3
    return tf.random.uniform((batch, height, width, channels), dtype=tf.float32)

