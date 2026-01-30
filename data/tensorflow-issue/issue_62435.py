import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

data_format = "channels_last"
keepdims = False
__input___0_tensor = tf.random.uniform([3, 5, 1, 1], minval=0, maxval=0, dtype=tf.float64)
__input___0 = tf.identity(__input___0_tensor)
GlobalMaxPooling2D_class = tf.keras.layers.GlobalMaxPooling2D(data_format=data_format, keepdims=keepdims)

layer = GlobalMaxPooling2D_class
inputs = __input___0

r = GlobalMaxPooling2D_class(inputs)
theoretical, numerical = tf.test.compute_gradient(GlobalMaxPooling2D_class, [inputs])
print(theoretical)
print(numerical)

(array([[0.2, 0.2, 0.2, 0.2, 0.2, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0.2, 0.2, 0.2, 0.2, 0.2, 0. , 0. , 0. ,
        0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.2, 0.2, 0.2,
        0.2, 0.2]]),)
(array([[0.5, 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
        0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 0.5, 0. , 0. , 0. ,
        0. , 0. ],
       [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5,
        0.5, 0.5]]),)

import tensorflow as tf

data_format = "channels_first"
keepdims = False
__input___0_tensor = tf.random.uniform([2, 4, 3, 1, 1], minval=0, maxval=0, dtype=tf.float64)
__input___0 = tf.identity(__input___0_tensor)
GlobalMaxPooling3D_class = tf.keras.layers.GlobalMaxPooling3D(data_format=data_format, keepdims=keepdims)

layer = GlobalMaxPooling3D_class
inputs = __input___0

r = GlobalMaxPooling3D_class(inputs)
theoretical, numerical = tf.test.compute_gradient(GlobalMaxPooling3D_class, [inputs])
print(theoretical)
print(numerical)

pool_size_0 = 2
pool_size = [pool_size_0, ]
strides_0 = 3
strides = [
    strides_0, ]
padding = "valid"
data_format = "channels_last"
__input___0_tensor = tf.random.uniform([3, 5, 4], minval=0.0, maxval=0.0, dtype=tf.float64)
__input___0 = tf.identity(__input___0_tensor)
MaxPooling1D_class = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format)

layer = MaxPooling1D_class
inputs = __input___0

r = MaxPooling1D_class(inputs)
theoretical, numerical = tf.test.compute_gradient(MaxPooling1D_class, [inputs])
print(theoretical)
print(numerical)