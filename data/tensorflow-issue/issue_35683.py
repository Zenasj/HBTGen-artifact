from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


class MyWrapper(tf.keras.layers.Wrapper):
    def call(self, inputs, *args, **kwargs):
        return self.layer(inputs, *args, **kwargs)


wrapper = MyWrapper(tf.keras.layers.Dense(1))
config = wrapper.get_config()
config_copy = config.copy()
assert config == config_copy

wrapper_from_config = MyWrapper.from_config(config)
new_config = wrapper.get_config()
assert new_config == config_copy
assert config == config_copy  # Fails! The 'layer' key has been popped from config