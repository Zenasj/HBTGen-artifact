from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras
import numpy as np

print(tf.__version__)

train_input = np.array([ -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
train_truth = np.array([ -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

def build_keras_model():
	return keras.models.Sequential([
		keras.layers.Dense(units=1, input_shape=[1]),
	])

### train the model
train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

keras.backend.set_session(train_sess)

with train_graph.as_default():
	keras.backend.set_learning_phase(1)
	train_model = build_keras_model()

	tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
	train_sess.run(tf.global_variables_initializer())

	train_model.compile(
		optimizer='sgd',
		loss='mean_squared_error'
	)
	train_model.fit(train_input, train_truth, epochs=250)

	saver = tf.train.Saver()
	saver.save(train_sess, 'linear.ckpt')

import tensorflow as tf
from tensorflow import keras
import numpy as np

def build_keras_model():
	return keras.models.Sequential([
		keras.layers.Dense(units=1, input_shape=[1]),
	])

# eval
eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

keras.backend.set_session(eval_sess)

with eval_graph.as_default():
	keras.backend.set_learning_phase(0)
	eval_model = build_keras_model()

	tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)

	eval_graph_def = eval_graph.as_graph_def()
	saver = tf.train.Saver()
	saver.restore(eval_sess, 'linear.ckpt')

	frozen_graph_def = tf.graph_util.convert_variables_to_constants(
		eval_sess,
		eval_graph_def,
		[eval_model.output.op.name]
	)

	with open('frozen_model.pb', 'wb') as f:
		f.write(frozen_graph_def.SerializeToString())