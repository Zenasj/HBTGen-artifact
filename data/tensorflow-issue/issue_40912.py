from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

class Lookup(tf.keras.layers.Layer):
    def __init__(self, depth: int = 3, **kwargs):
        super().__init__(**kwargs)

        self.depth = depth
        self.texture = None

    def build(self, input_shape):
        lookup_shape = input_shape[1:-1] + [self.depth]

        self.lookup_table = self.add_weight('lookup_table', lookup_shape, 'float32',
                                       initializer='random_normal',
                                       trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        entries = tf.gather_nd(params=self.lookup_table,
                                      indices=inputs,
                                      name='lookup_call')

        return entries

    def get_config(self):
        config = super().get_config()
        config.update({"depth": self.depth})
        return config
    
custom_objects = {"Lookup": Lookup}

lookup = Lookup()

x = tf.keras.Input(shape=(2,2), dtype='int32')
y = lookup(x)
model = tf.keras.Model(x, y)
model.compile()

model.save('/tmp/lookup_test')
loaded_model = tf.keras.models.load_model('/tmp/lookup_test', custom_objects=custom_objects)