import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

filters = 2
kernel_size_0 = 3
kernel_size_1 = 3
kernel_size_2 = 3
kernel_size = [kernel_size_0, kernel_size_1,kernel_size_2, ]
strides_0 = 2
strides_1 = 2
strides_2 = 2
strides = [
    strides_0, strides_1,
    strides_2, ]
padding = "valid"
output_padding = None
data_format = "channels_last"
dilation_rate_0 = 1
dilation_rate_1 = 1
dilation_rate_2 = 1
dilation_rate = [dilation_rate_0,dilation_rate_1,dilation_rate_2,]
activation = "relu"
use_bias = False
kernel_initializer = None
bias_initializer = None
kernel_regularizer = None
bias_regularizer = None
activity_regularizer = None
kernel_constraint = None
bias_constraint = None
__input___0_tensor = tf.random.uniform([3, 5, 1, 1, 1], minval=0, maxval=0, dtype=tf.float64)
__input___0 = tf.identity(__input___0_tensor)
Conv3DTranspose_class = tf.keras.layers.Conv3DTranspose(filters, kernel_size, strides=strides, padding=padding, output_padding=output_padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)

layer = Conv3DTranspose_class
inputs = __input___0

r = Conv3DTranspose_class(inputs)
theoretical, numerical = tf.test.compute_gradient(Conv3DTranspose_class, [inputs])
print(theoretical)
print(numerical)

(array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]]),)
(array([[-0.05650689,  0.        ,  0.        , ...,  0.        ,
         0.        ,  0.        ],
       [-0.11539936,  0.        ,  0.        , ...,  0.        ,
         0.        ,  0.        ],
       [ 0.06827629,  0.        ,  0.        , ...,  0.        ,
         0.        ,  0.        ],
       ...,
       [ 0.        ,  0.        ,  0.        , ...,  0.        ,
         0.        , -0.02971415],
       [ 0.        ,  0.        ,  0.        , ...,  0.        ,
         0.        , -0.1242566 ],
       [ 0.        ,  0.        ,  0.        , ...,  0.        ,
         0.        ,  0.11878401]]),)