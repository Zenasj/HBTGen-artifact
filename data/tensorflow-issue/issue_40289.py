import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, scale_initializer=None, **kwargs):
        if scale_initializer is None:
            self.scale_initializer = tf.keras.initializers.Constant(0.5)
        else:
            self.scale_initializer = tf.keras.initializers.get(scale_initializer)
        super().__init__(**kwargs)

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, scale_initial_value=0.5, **kwargs):
        self.scale_initializer = tf.keras.initializers.Constant(scale_initial_value)
        super().__init__(**kwargs)

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, scale_initializer=0.5, **kwargs):
        self.scale_initializer = tf.keras.initializers.get(scale_initializer)
        super().__init__(**kwargs)