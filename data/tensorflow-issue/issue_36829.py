from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

# This script works if v2 is disabled, but fails if v2 is enabled.
tf.compat.v1.disable_v2_behavior()

# Create a very simple generator.
class Generator(tf.keras.utils.Sequence):
	def __len__(self):
		return 1

	def __getitem__(self, index):
		return np.zeros((1, 100, 100)), np.zeros((1, 2))
generator = Generator()

# Create a simple model with two outputs, one has a loss attached to it the other does not.
inputs = tf.keras.Input((100, 100))
flattened = tf.keras.layers.Flatten()(inputs)
output_1 = tf.keras.layers.Dense(2, name='output_for_loss')(flattened)
output_2 = tf.keras.layers.Reshape((2, -1), name='some_other_output')(flattened)
model = tf.keras.Model(inputs=inputs, outputs=[output_1, output_2])
model.compile(loss={'output_for_loss': tf.keras.losses.binary_crossentropy})

# Train using the generator.
model.fit(generator)