from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

def res_net_block(shape):
    filters = shape[-1]

    inputs = tf.keras.layers.Input(shape)
    x = tf.keras.layers.Conv3D(filters=filters, kernel_size=3, padding='same')(inputs)
    outputs = x + inputs

    return tf.keras.Model(inputs, outputs)

def encoder(shape):
    kernel_size = 3
    strides = 2

    inputs = tf.keras.layers.Input(shape)
    outputs = res_net_block(inputs.shape[1:])(inputs)

    return tf.keras.Model(inputs, outputs)

shape = [256,256,128,1]
model = encoder(shape)
model.summary()

def res_net_block(shape):
    filters = shape[-1]

    inputs = tf.keras.layers.Input(shape)
    outputs = tf.keras.layers.Conv3D(filters=filters, kernel_size=3, padding='same')(inputs)

    return tf.keras.Model(inputs, outputs)

def encoder(shape):
    kernel_size = 3
    strides = 2

    inputs = tf.keras.layers.Input(shape)
    x = res_net_block(inputs.shape[1:])(inputs)
    outputs = x + inputs # This addition is not here in previous code, instead it's inside `res_net_block` function

    return tf.keras.Model(inputs, outputs)