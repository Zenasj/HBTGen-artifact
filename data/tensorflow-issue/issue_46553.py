from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf


def initializer(shape, dtype=None):
    return tf.reshape(tf.constant([str(x) for x in range(shape[0])]), shape)


class CustomLayer(tf.keras.layers.Layer):

    def __init__(self, length=10, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.length = length

    def build(self, input_shape=None):
        self.keys = self.add_weight(shape=(self.length,),
                                    name='keys',
                                    initializer=initializer,
                                    dtype=tf.string,
                                    trainable=False)

        self.built = True

    def call(self, x):
        return self.keys

    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({'length': self.length})
        return config


inputs = tf.keras.layers.Input(shape=())
outputs = CustomLayer(name='lookup')(inputs)
model = tf.keras.Model(inputs, outputs)
model.compile()

model.get_layer('lookup').set_weights(model.get_layer('lookup').get_weights())

import tensorflow as tf

class CustomLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)

    def build(self, input_shape=None):
        self.keys = tf.Variable(['foo'],
                                shape=(1,),
                                trainable=False,
                                dtype=tf.string)
        self.built = True

    def call(self, x):
        return x

inputs = tf.keras.layers.Input(shape=())
outputs = CustomLayer(name='lookup')(inputs)
model = tf.keras.Model(inputs, outputs)
model.compile(loss='mse')

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', restore_best_weights=True)
]

model.fit([1], [1], epochs=10, callbacks=callbacks)

dtype=dtypes_module.as_dtype(x.dtype).as_numpy_dtype