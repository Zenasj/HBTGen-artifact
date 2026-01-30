from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

#!/usr/bin/python

import tensorflow as tf;
import numpy as np;

constant = tf.keras.initializers.constant;

inter_op_parallelism_threads = 1
# inter_op_parallelism_threads = 20

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=inter_op_parallelism_threads)
graph = tf.get_default_graph()
session = tf.Session(graph=graph, config=session_conf)
tf.keras.backend.set_session(session)

optimizer = tf.keras.optimizers.Nadam(lr=1.0, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

bias = constant(np.array([1.0]))
weights = constant(np.array([1.0]))
layer = tf.keras.layers.Dense(1, bias_initializer=bias, kernel_initializer=weights, input_shape=(1,))

model = tf.keras.Sequential()
model.add(layer)
model.compile(optimizer=optimizer, metrics=["accuracy"], loss=["mean_squared_error"])

input = np.array([1.0])
output = np.array([1.0])
model.train_on_batch(input, output)

w = layer.get_weights()
print(w)