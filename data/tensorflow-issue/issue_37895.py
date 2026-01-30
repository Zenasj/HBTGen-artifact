import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

class CustomLayer(tf.keras.layers.Layer):
	def __init__(self):
		super().__init__()
		
	def call(self, inputs):
		tf.print('Running eagerly: ', tf.executing_eagerly())
		return inputs

if __name__ == "__main__" :
	data = np.random.random((16, 3)).astype(np.float32)

	inputs = tf.keras.Input(shape=(3,))
	outputs = tf.keras.layers.Dense(3)(inputs)
	outputs = CustomLayer()(outputs)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	
	model.compile(loss='mse', run_eagerly=True) # does not set model.run_eagerly to True, and model does not run eagerly
	# model.run_eagerly = True # sets model.run_eagerly to True, and model runs eagerly

	print('run_eagerly: ', model.run_eagerly)

	model.fit(x=data, y=data)