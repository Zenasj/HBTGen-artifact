from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf


class CustomCell(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.state_size = [tf.TensorShape((16, 16, 1))]
        super(CustomCell, self).__init__(**kwargs)

    def call(self, inputs, states, **kwargs):
        output = states[0] + tf.nn.conv2d(inputs, tf.ones((3, 3, 1, 1)), (1, 1), "SAME")
        new_state = output
        return output, new_state


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.RNN(CustomCell(), batch_input_shape=(1, 1, 16, 16, 1)))

# Error here
model.predict(tf.ones(model.input_shape))

import tensorflow as tf


class CustomCell(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.state_size = [tf.TensorShape((16, 1))]
        super(CustomCell, self).__init__(**kwargs)

    def call(self, inputs, states, **kwargs):
        output = states[0] + tf.nn.conv1d(inputs, tf.ones((3, 1, 1)), (1,), "SAME")
        new_state = output
        return output, new_state


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.RNN(CustomCell(), batch_input_shape=(1, 1, 16, 1)))

# No error
model.predict(tf.ones(model.input_shape))