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

import tensorflow as tf

graph_def_file = 'frozen_model.pb'
inputs = ["dense_input"]
outputs = ["dense/BiasAdd"]

sess = tf.Session()

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, inputs, outputs)
converter.inference_type = tf.lite.constants.FLOAT
input_arrays = converter.get_input_arrays()

converter.quantized_input_stats = {input_arrays[0]: (0., 1.)} # mean, std_dev

tflite_model = converter.convert()

open('py_converted_model.tflite', 'wb').write(tflite_model)

import tensorflow as tf

graph_def_file = 'frozen_model.pb'
inputs = ["dense_input"]
outputs = ["dense/BiasAdd"]

sess = tf.Session()

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, inputs, outputs)
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()

converter.allow_custom_ops = True
converter.quantized_input_stats = {input_arrays[0]: (0., 1.)} # mean, std_dev
converter.default_ranges_stats = (-128, 127)

tflite_model = converter.convert()

open('py_converted_model.tflite', 'wb').write(tflite_model)

import tensorflow as tf
from tensorflow import keras
import numpy as np

train_x = np.array([ [-4.0], [-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
train_y = np.array([ -9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0], dtype=np.float32)

test_x = np.array([[4.5]], dtype=np.float32)

def build_model():

	model = keras.models.Sequential()
	model.add(keras.layers.Dense(1, input_shape=(1,)))

	return model

###
### TRAINING
###

model = build_model()
model.compile(optimizer='sgd',
		  	  loss='mean_squared_error',
)

keras_file = 'linear.h5'
model.fit(train_x, train_y, epochs=150)
keras.models.save_model(model, keras_file)

###
### CONVERSION
###

if (tf.__version__ == '2.0.0-beta1'):
	converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(keras_file)
else:
	raise NotImplementedError('Converter not implemented for this tf version')

def representative_dataset_gen():
	for i in range(len(train_x)):
		yield [train_x[i:i+1]]

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_file_name = 'converted_linearmodel_tpu_' + str(tf.__version__) + '.tflite'

tflite_model = converter.convert()
open(tflite_file_name, 'wb').write(tflite_model)

###
### INFERENCE FOR TESTING
###

interpreter = tf.lite.Interpreter(model_path=tflite_file_name)
interpreter.allocate_tensors()

input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]

def quantize(real_value):
	std, mean = input_detail['quantization']
	return (real_value / std + mean).astype(np.uint8)

def dequantize(quant_value):
	std, mean = output_detail['quantization']
	return (std * (quant_value - mean)).astype(np.float32)

sample_input = quantize(test_x[0]).reshape(input_detail['shape'])

interpreter.set_tensor(input_detail['index'], sample_input)
interpreter.invoke()

pred_original_model = model.predict(test_x[0])
pred_quantized_model = interpreter.get_tensor(output_detail['index'])

print("Pred Original : ", pred_original_model)
print("Pred Quantized : ", dequantize(pred_quantized_model[0].reshape(output_detail['shape'])))

converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8