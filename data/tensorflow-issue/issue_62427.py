import random
from tensorflow import keras
from tensorflow.keras import layers

filters = 2
kernel_size_0 = 4
kernel_size_1 = 2
kernel_size = [kernel_size_0,kernel_size_1,]
strides_0 = 1
strides_1 = 1
strides = [strides_0,strides_1,]
padding = "valid"
data_format = "channels_first"
dilation_rate_0 = 1
dilation_rate_1 = 1
dilation_rate = [dilation_rate_0,dilation_rate_1,]
activation = "tanh"
recurrent_activation = "hard_sigmoid"
use_bias = False
kernel_initializer = None
recurrent_initializer = None
bias_initializer = "zeros"
unit_forget_bias = True
kernel_regularizer = None
recurrent_regularizer = None
bias_regularizer = None
activity_regularizer = None
kernel_constraint = None
recurrent_constraint = None
bias_constraint = None
return_sequences = False
return_state = False
go_backwards = True
stateful = False
dropout = 0.0
recurrent_dropout = 0.0
__input___0_tensor = tf.random.uniform([2, 2, 2, 5, 1], minval=1.7518786460769893, maxval=2.8945805380363145, dtype=tf.float32)
__input___0 = tf.identity(__input___0_tensor)
ConvLSTM2D_class = tf.keras.layers.ConvLSTM2D(filters, kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards, stateful=stateful, dropout=dropout, recurrent_dropout=recurrent_dropout)

layer = ConvLSTM2D_class
inputs = __input___0

with tf.GradientTape() as g:
    g.watch(inputs)
    res = layer(inputs)
print(res.shape)
grad = g.jacobian(res, inputs)  # Error

import tensorflow as tf
print(tf.__version__)
filters = 2

kernel_size_0 = 4  
kernel_size_1 = 2
kernel_size = [kernel_size_0, kernel_size_1]
strides_0 = 1
strides_1 = 1
strides = [strides_0, strides_1]
# Use 'same' padding ................
padding = "same"  # Changed from 'valid...............'
data_format = "channels_first"
dilation_rate_0 = 1
dilation_rate_1 = 1
dilation_rate = [dilation_rate_0, dilation_rate_1]
activation = "tanh"
recurrent_activation = "hard_sigmoid"
use_bias = False
kernel_initializer = None
recurrent_initializer = None
bias_initializer = "zeros"
unit_forget_bias = True
kernel_regularizer = None
recurrent_regularizer = None
bias_regularizer = None
activity_regularizer = None
kernel_constraint = None
recurrent_constraint = None
bias_constraint = None
return_sequences = True
return_state = False
go_backwards = True
stateful = False
dropout = 0.0
recurrent_dropout = 0.0
__input___0_tensor = tf.random.uniform([2, 2, 2, 5, 1], minval=1.7518786460769893, maxval=2.8945805380363145, dtype=tf.float32)
__input___0 = tf.identity(__input___0_tensor)
ConvLSTM2D_class = tf.keras.layers.ConvLSTM2D(filters, kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, recurrent_activation=recurrent_activation, use_bias=use_bias, kernel_initializer=kernel_initializer, recurrent_initializer=recurrent_initializer, bias_initializer=bias_initializer, unit_forget_bias=unit_forget_bias, kernel_regularizer=kernel_regularizer, recurrent_regularizer=recurrent_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, recurrent_constraint=recurrent_constraint, bias_constraint=bias_constraint, return_sequences=return_sequences, return_state=return_state, go_backwards=go_backwards, stateful=stateful, dropout=dropout, recurrent_dropout=recurrent_dropout)

layer = ConvLSTM2D_class
inputs = __input___0

with tf.GradientTape() as g:
    g.watch(inputs)
    res = layer(inputs)
print(res.shape)
grad = g.jacobian(res, inputs)

print(inputs.shape)
print(res.shape)
(2, 2, 2, 5, 1)
(2, 2, 2, 5, 1)