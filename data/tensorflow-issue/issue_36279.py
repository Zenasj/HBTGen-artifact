from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
import sys


batch_size = 16
input_batch = np.ones((batch_size,)).astype(np.float32)


class TestSequence(tf.keras.utils.Sequence):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return input_batch, input_batch

# Dummy layer to show the tensor shape
class PrintLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        tf.print(inputs.shape, output_stream=sys.stderr)
        return inputs


# The result is the same if shape=() is used to specify scalar input
input_layer = tf.keras.layers.Input(
    name='the_input', batch_shape=(batch_size,), dtype='float32')
x = PrintLayer()(input_layer)
output_layer = tf.keras.layers.ReLU()(x)
model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])

# If run_eagerly=False then the shapes are correct
model.compile(loss='mse', run_eagerly=True)

model.fit(TestSequence(), epochs=3)