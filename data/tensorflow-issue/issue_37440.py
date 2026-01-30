import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Custom loss with a FOR loop. Raises an error in AutoGraph mode. 
# A similar FOR loop in a Keras model works as expected.
class CustomLoss(keras.losses.Loss):
	def call(self, y_true, y_pred):
		x = y_true + y_pred
		for i in tf.range(tf.shape(y_true)[0]): # The error is reaised here.
			x += 1
		return tf.reduce_mean(x)

if __name__ == "__main__" :
	data = np.random.random((1000, 3)).astype(np.float32)
	
	inputs = tf.keras.Input(shape=(1000,3))
	outputs = tf.keras.layers.Dense(3)(inputs)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	
	model.compile(loss=CustomLoss()) # does NOT work
	# model.compile(loss=CustomLoss(), run_eagerly = True) # works
	
	model.fit(x=data, y=data)

import numpy as np
import tensorflow as tf
from tensorflow import keras

class CustomMetric(keras.metrics.Metric):
	def __init__(self):
		super(CustomMetric, self).__init__()
		self._metric = self.add_weight(name='metric', initializer='zeros', shape=())
	
	def update_state(self, y_pred, y_true, sample_weights=None):
		batchSize = tf.shape(y_pred)[0]
		for b in range(batchSize):
			self._metric.assign_add(1.0)
	
	def result(self):
		return self._metric
	
	def reset_states(self):
		self._metric.assign(0.0)

if __name__ == "__main__" :
	data = np.random.random((1000, 3)).astype(np.float32)
	
	inputs = tf.keras.Input(shape=(1000, 3))
	outputs = tf.keras.layers.Dense(3)(inputs)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	
	model.compile(loss="mse", metrics=CustomMetric()) #Does NOT WORK
	# model.compile(loss="mse", metrics=CustomMetric(), run_eagerly=True) #WORKS
	
	model.fit(x=data, y=data)