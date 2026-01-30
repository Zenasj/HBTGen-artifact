import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Custom loss
class CustomLoss(keras.losses.Loss):
	def call(self, y_true, y_pred):
		print(tf.executing_eagerly())
		
		x = y_true + y_pred
		return tf.reduce_mean(x)

if __name__ == "__main__" :
	data = np.random.random((16, 1000, 3)).astype(np.float32)
	
	inputs = tf.keras.Input(shape=(1000,3))
	outputs = tf.keras.layers.Dense(3)(inputs)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	
	# tf.config.experimental_run_functions_eagerly(True) # runs custom loss eagerly
	# model.compile(loss=CustomLoss())
	
	model.compile(loss=CustomLoss(), run_eagerly = True) # does not run custom loss eagerly
	
	model.fit(x=data, y=data)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.enable_eager_execution()

# Custom loss
class CustomLoss(keras.losses.Loss):
	def call(self, y_true, y_pred):
		print(tf.executing_eagerly())

		x = y_true + y_pred
		return tf.reduce_mean(x)

if __name__ == "__main__" :
	data = np.random.random((16, 1000, 3)).astype(np.float32)
	
	inputs = tf.keras.Input(shape=(1000,3))
	outputs = tf.keras.layers.Dense(3)(inputs)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	
	# tf.config.experimental_run_functions_eagerly(True) # runs custom loss eagerly
	# model.compile(loss=CustomLoss())
	
	model.compile(loss=CustomLoss(), run_eagerly = True) # does not run custom loss eagerly
	
	model.fit(x=data, y=data)