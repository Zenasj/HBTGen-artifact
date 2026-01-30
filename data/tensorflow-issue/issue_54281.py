import random
from tensorflow.keras import layers

import keras
import numpy as np
import tensorflow as tf


class CustomModel(keras.Model):
	def train_step(self, data):
		return {m.name: m.result() for m in self.metrics}


if __name__ == '__main__':
	# config
	# tf.compat.v1.enable_eager_execution()
	tf.compat.v1.disable_eager_execution()

	print("TensorFlow version: {}".format(tf.__version__))
	print("Eager execution: {}".format(tf.executing_eagerly()))

	# Construct and compile an instance of CustomModel
	inputs = keras.Input(shape=(32,))
	outputs = keras.layers.Dense(1)(inputs)
	model = CustomModel(inputs, outputs)
	model.compile(optimizer="adam", loss="mse", metrics=["mae"])

	# Just use `fit` as usual
	x = np.random.random((1000, 32))
	y = np.random.random((1000, 1))

	print(model.evaluate(x, y))
	model.fit(x, y, epochs=3)
	print(model.evaluate(x, y))