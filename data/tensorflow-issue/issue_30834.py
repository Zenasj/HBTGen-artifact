from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
print(tf.__version__)
def model_float32():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, use_bias=False, input_shape=(10,)
        ,dtype=tf.float32))

    model.add(tf.keras.layers.GaussianNoise(0.0003))
    return model

testmodel_32 =model_float32()

import tensorflow as tf
print(tf.__version__)
def model_float64():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(10, use_bias=False, input_shape=(10,)
        ,dtype=tf.float64))

    model.add(tf.keras.layers.GaussianNoise(0.0003))
    return model

testmodel_64 =model_float64()