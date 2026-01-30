import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

filters = 2
kernel_size_0 = 3
kernel_size_1 = 3
kernel_size = [kernel_size_0,kernel_size_1, ]
strides_0 = 1
strides_1 = 1
strides = [
    strides_0,
    strides_1, ]
padding = "same"
output_padding = None
data_format = "channels_last"
dilation_rate_0 = 2
dilation_rate_1 = 2
dilation_rate = [dilation_rate_0,dilation_rate_1,]
activation = "linear"
use_bias = True
kernel_initializer = None
bias_initializer = None
kernel_regularizer = None
bias_regularizer = None
activity_regularizer = None
kernel_constraint = None
bias_constraint = None
__input___0_tensor = tf.random.uniform([1, 5, 6, 1], minval=-2, maxval=2, dtype=tf.float32)
__input___0 = tf.identity(__input___0_tensor)
Conv2DTranspose_class = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, output_padding=output_padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,  dtype=tf.float32)
layer = Conv2DTranspose_class
inputs = __input___0

with tf.GradientTape(persistent=True, ) as g:
    g.watch(inputs)
    res_backward = layer(inputs)
    grad_backward = g.jacobian(res_backward, res_backward)
    print("res_backward:", res_backward)
    print("grad_backward:", grad_backward)
tangents = tf.constant(1., dtype=tf.float32, shape=[1, 5, 6, 1])

with tf.autodiff.ForwardAccumulator(inputs, tangents) as acc:
    res_forward = layer(inputs)
    grad_jvp = acc.jvp(res_forward)
    print("res_forward:", res_forward)
    print("grad_forward", grad_jvp)