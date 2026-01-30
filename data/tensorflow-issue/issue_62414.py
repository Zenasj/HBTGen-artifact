import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
pool_size_0 = 3
pool_size_1 = 3
pool_size_2 = 3
pool_size = [pool_size_0,pool_size_1,pool_size_2,]
strides_0 = 0
strides_1 = 21.0
strides_2 = 1
strides = [strides_0,strides_1,strides_2,]
padding = "valid"
data_format = "channels_last"
__input___0_tensor = tf.random.uniform([1, 11, 1, 1, 1], minval=-1.0, maxval=0.7764022115238914, dtype=tf.float32)
__input___0 = tf.identity(__input___0_tensor)
AveragePooling3D_class = tf.keras.layers.AveragePooling3D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format,  dtype=tf.float64)
print(AveragePooling3D_class(__input___0))