from tensorflow import keras
from tensorflow.keras import layers

python
import tensorflow as tf
import numpy as np


class TestCell(tf.keras.layers.Layer):
    state_size = 1
    output_size = 1

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.ones((batch_size, 1), dtype=dtype)

    def call(self, inputs, states):
        tf.assert_equal(states, 1.0)
        return inputs, states


layer = tf.keras.layers.RNN(TestCell(), stateful=True)

x = np.ones((1, 10, 1), dtype=np.float32)

layer(x)