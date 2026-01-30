import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

kernel_size_0 = 3
kernel_size_1 = 3
kernel_size = [kernel_size_0,kernel_size_1, ]
strides_0 = 2
strides_1 = 2
strides = [
    strides_0,
    strides_1, ]
padding = "valid"
depth_multiplier = 1
data_format = None
dilation_rate_0 = 1
dilation_rate_1 = 1
dilation_rate = [dilation_rate_0,dilation_rate_1,]
activation = None
use_bias = False
depthwise_initializer = "glorot_uniform"
bias_initializer = "zeros"
depthwise_regularizer = None
bias_regularizer = None
activity_regularizer = None
depthwise_constraint = None
bias_constraint = None
__input___0_tensor = tf.random.uniform([2, 15, 1, 1], minval=0, maxval=0, dtype=tf.float64)
__input___0 = tf.identity(__input___0_tensor)
DepthwiseConv2D_class = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=strides, padding=padding, depth_multiplier=depth_multiplier, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, depthwise_initializer=depthwise_initializer, bias_initializer=bias_initializer, depthwise_regularizer=depthwise_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, depthwise_constraint=depthwise_constraint, bias_constraint=bias_constraint)

layer = DepthwiseConv2D_class
inputs = __input___0

with tf.GradientTape() as g:
    g.watch(inputs)
    res = layer(inputs)
grad = g.jacobian(res, inputs)