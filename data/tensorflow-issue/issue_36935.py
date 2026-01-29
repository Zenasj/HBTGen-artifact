# tf.random.uniform((B, L, C), dtype=tf.float32)  ← Assumed input shape in 'channels_last' format (batch, length, channels)

import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints, activations
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils, tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.keras import backend


class MyModel(tf.keras.Model):
  def __init__(self,
               kernel_size,
               strides=1,
               padding='valid',
               depth_multiplier=1,
               data_format='channels_last',
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(MyModel, self).__init__(**kwargs)

    # Store parameters
    self.kernel_size = (kernel_size,) if isinstance(kernel_size, int) else tuple(kernel_size)
    self.strides = (strides,) if isinstance(strides, int) else tuple(strides)
    self.padding = padding.lower()
    self.depth_multiplier = depth_multiplier
    self.data_format = data_format or 'channels_last'
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.depthwise_initializer = initializers.get(depthwise_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.depthwise_constraint = constraints.get(depthwise_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    # Will be set during build
    self.depthwise_kernel = None
    self.bias = None
    self.input_spec = None

  def build(self, input_shape):
    if len(input_shape) != 3:
      raise ValueError('Inputs to `MyModel` should have rank 3. '
                       'Received input shape: {}'.format(input_shape))

    if self.data_format == 'channels_last':
      channel_axis = 2
    elif self.data_format == 'channels_first':
      channel_axis = 1
    else:
      raise ValueError('Invalid data_format: ' + str(self.data_format))

    input_shape = tf.TensorShape(input_shape)
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs to `MyModel` '
                       'should be defined. Found `None`.')

    input_dim = int(input_shape[channel_axis])

    # Depthwise kernel shape: (kernel_size, input_channels, depth_multiplier)
    depthwise_kernel_shape = (self.kernel_size[0], input_dim, self.depth_multiplier)

    self.depthwise_kernel = self.add_weight(
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        name='depthwise_kernel',
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint,
        trainable=True,
        dtype=self.dtype)

    if self.use_bias:
      self.bias = self.add_weight(
          shape=(input_dim * self.depth_multiplier,),
          initializer=self.bias_initializer,
          name='bias',
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None

    # Set input spec to enforce rank and channels dimension
    self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})

    super(MyModel, self).build(input_shape)

  def call(self, inputs):
    # Pad input if causal padding requested (not implemented here, placeholder)
    if self.padding == 'causal':
      # This would require causal padding code, omitted for simplicity / not in original
      raise NotImplementedError('Causal padding is not implemented.')

    if self.data_format == 'channels_last':
      spatial_start_dim = 1  # length dim
    else:
      spatial_start_dim = 2  # length dim for channels_first

    # Expand dims to 4D to use backend.depthwise_conv2d (simulate 1D conv as 2D conv with width=1)
    inputs_expanded = tf.expand_dims(inputs, spatial_start_dim)
    kernel_expanded = tf.expand_dims(self.depthwise_kernel, 0)  # add dim for height=1

    # For strides and dilation_rate, replicate the 1D tuple to 2D
    strides_2d = self.strides * 2  # e.g. (stride,) -> (stride, stride)
    dilation_rate_2d = (1,) + (1,)  # For simplicity, no dilation implemented beyond (1,1)

    # Perform depthwise conv2d on expanded inputs
    outputs = backend.depthwise_conv2d(
        inputs_expanded,
        kernel_expanded,
        strides=strides_2d,
        padding=self.padding,
        data_format=self.data_format,
        dilation_rate=dilation_rate_2d)

    if self.use_bias:
      outputs = backend.bias_add(outputs, self.bias, data_format=self.data_format)

    # Squeeze back the extra dimension
    outputs = tf.squeeze(outputs, spatial_start_dim)

    if self.activation is not None:
      outputs = self.activation(outputs)

    return outputs

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if self.data_format == 'channels_first':
      length = input_shape[2]
      out_channels = input_shape[1] * self.depth_multiplier
    elif self.data_format == 'channels_last':
      length = input_shape[1]
      out_channels = input_shape[2] * self.depth_multiplier
    else:
      raise ValueError('Invalid data_format: ' + str(self.data_format))

    length = conv_utils.conv_output_length(length,
                                           self.kernel_size[0],
                                           self.padding,
                                           self.strides[0])
    if self.data_format == 'channels_first':
      return (input_shape[0], out_channels, length)
    else:
      return (input_shape[0], length, out_channels)

  def get_config(self):
    config = super(MyModel, self).get_config()
    config.update({
        'kernel_size': self.kernel_size[0],
        'strides': self.strides[0],
        'padding': self.padding,
        'depth_multiplier': self.depth_multiplier,
        'data_format': self.data_format,
        'activation': activations.serialize(self.activation),
        'use_bias': self.use_bias,
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
  # Return an instance of MyModel with some default parameters
  return MyModel(
      kernel_size=3,
      strides=1,
      padding='valid',
      depth_multiplier=1,
      data_format='channels_last',
      activation=None,
      use_bias=True
  )


def GetInput():
  # Generate a random input tensor matching expected input shape:
  # Assume batch=2, length=16, channels=4 (arbitrary but realistic values)
  batch_size = 2
  length = 16
  channels = 4
  # dtype float32 = most common for conv layers
  return tf.random.uniform((batch_size, length, channels), dtype=tf.float32)

