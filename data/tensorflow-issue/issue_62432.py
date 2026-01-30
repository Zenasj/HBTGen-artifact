import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
units = 0
activation = "sigmoid"
use_bias = True
kernel_initializer = "ones"
bias_initializer = "zeros"
kernel_regularizer = None
bias_regularizer = None
activity_regularizer = None
kernel_constraint = None
bias_constraint = None
__input___0_tensor = tf.random.uniform([50, 1], minval=1.0, maxval=3.0, dtype=tf.float64)
__input___0 = tf.identity(__input___0_tensor)
Dense_class = tf.keras.layers.Dense(units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)

layer = Dense_class
inputs = __input___0

with tf.GradientTape() as g:
    g.watch(inputs)
    res = layer(inputs)
print(res.shape)
grad = g.jacobian(res, inputs)  # Error
print(grad)