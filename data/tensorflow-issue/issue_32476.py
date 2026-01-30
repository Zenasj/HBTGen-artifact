from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

class Example(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs["dynamic"] = True
        super(Example, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        return [(None, 2)]

inp = tf.keras.layers.Input(batch_shape=(None, 1))
comp = Example()(inp)

model = tf.keras.models.Model(inputs=[inp], outputs=[comp])
model.summary()

import tensorflow as tf
import numpy as np

class Example(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        kwargs["dynamic"] = True
        super(Example, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, 2])

inp = tf.keras.layers.Input(batch_shape=(None, 1))
comp = Example()(inp)

model = tf.keras.models.Model(inputs=[inp], outputs=[comp])
model.summary()