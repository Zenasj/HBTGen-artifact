from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='test')
class CustomLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs

wrapper = tf.keras.layers.Wrapper(CustomLayer())
config = wrapper.get_config()
tf.keras.layers.Wrapper.from_config(config)

import tensorflow as tf

dense = tf.keras.layers.Dense(32)
wrapper = tf.keras.layers.Wrapper(dense)
config = wrapper.get_config()
assert 'layer' in config, 'before wrapper instantiation'
tf.keras.layers.Wrapper.from_config(config)
assert 'layer' in config, 'after wrapper instantiation'