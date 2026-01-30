from time import sleep

import os
import time
import gc
import numpy as np
import tensorflow.keras as k
import tensorflow as tf

if __name__ == '__main__':
	"""
	the Flatten and BN layer will cause memory leak, and performance problem
	memory usage keep increasing during the iteration, and never down after gc:
		Flatten + BN : mem 1.3GB, time 105s
		Flatten only: mem 1.3GB, time 105s
		BN only: mem 760MB, time 45s
		no Flatten or BN: mem 490M, time 21s
	"""
	model_file = 'l:/m'
	for i in range(100):
		if os.path.exists(model_file):
			model = k.models.load_model(model_file)
		else:
			model = k.models.Sequential([
				k.layers.Flatten(input_shape=(10,)), # this layer will cause HUGE memory leak and performance problem
				k.layers.Dense(100),
				k.layers.ELU(),
				k.layers.Dense(500),
				k.layers.BatchNormalization(), # this layer will cause memory leak
				k.layers.ELU(),
				k.layers.Dense(2, activation=tf.nn.softmax),
			])
			model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
			model.fit(np.ndarray((5000, 10), dtype=float), np.array([0, 1] * 2500), epochs=1, verbose=1)
			model.save(model_file)
		
		x = np.ndarray((1, 10), dtype=float)
		x.fill(1)
		print(i, model.predict(x))
	
	os.remove(model_file)
	gc.collect()
	sleep(100000)