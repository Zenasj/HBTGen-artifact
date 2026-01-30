from tensorflow import keras
from tensorflow.keras import models

def build_mode(input_shape, 
			output_bias=None, 
			l1_factor=0, 
			l2_factor=0, 
			dropout_rate=0.2,
			alpha=1.0):
	if l1_factor > 0 and l2_factor == 0:
		kernel_regularizer = regularizers.L1(l1_factor)
	elif l2_factor > 0 and l1_factor == 0:
		kernel_regularizer = regularizers.L2(l2_factor)
	elif l1_factor > 0 and l2_factor > 0:
		kernel_regularizer = regularizers.L1L2(l1_factor, l2_factor)
	else:
		kernel_regularizer = None

	inputs = layers.Input(shape=(input_shape[0], input_shape[1], 3))
	channel_axis = -1
	kernel = 3
	activation = relu
	x = inputs
	x = layers.Conv2D(16, 
					kernel_size=3, 
					strides=(2,2), 
					padding='same', 
					use_bias=False, 
					kernel_regularizer=kernel_regularizer,
					name='Conv')(x)
	x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm')(x)
	x = activation(x)

	# stack
	def depth(d):
		return _depth(d * alpha)

	x = _inverted_res_block(x, 1, depth(16), 3, 2, relu, 0, kernel_regularizer)
	x = _inverted_res_block(x, 72. / 16, depth(24), 3, 2, relu, 1, kernel_regularizer)
	x = _inverted_res_block(x, 88. / 24, depth(24), 3, 1, relu, 2, kernel_regularizer)
	x = _inverted_res_block(x, 4, depth(40), kernel, 2, activation, 3, kernel_regularizer)
	x = _inverted_res_block(x, 6, depth(40), kernel, 1, activation, 4, kernel_regularizer)
	x = _inverted_res_block(x, 6, depth(40), kernel, 1, activation, 5, kernel_regularizer)
	x = _inverted_res_block(x, 6, depth(96), kernel, 2, activation, 8, kernel_regularizer)

	# final
	last_conv_ch = _depth(backend.int_shape(x)[channel_axis]*6)
	last_point_ch = 128#1024

	if alpha > 1.0:
		last_point_ch = _depth(last_point_ch * alpha)

	x = layers.Conv2D(last_conv_ch, kernel_size=1, padding='same', use_bias=False, name='Conv_1')(x)
	x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv_1/BatchNorm')(x)
	x = activation(x)
	
	x = layers.AveragePooling2D(7, 1)(x)

	x = layers.Conv2D(last_point_ch, kernel_size=1, padding='same', use_bias=True, name='Conv_2')(x)
	x = activation(x)

	if dropout_rate > 0:
	  x = layers.Dropout(dropout_rate)(x)

	x = layers.Conv2D(2, kernel_size=1, padding='same', name='Logits')(x)
	x = layers.Flatten()(x)
	prob = layers.Softmax()(x) 


	model = tf.keras.Model(inputs=inputs, outputs=prob)

	return model

import tensorflow_model_optimization as tfmot
import tensorflow as tf

model = build_model()
annotated_model = tf.keras.models.clone_model(model, clone_function=custom_quantization)

with tfmot.quantization.keras.quantize_scope():
	model = tfmot.quantization.keras.quantize_apply(annotated_model)

# standard training pipeline
# ...

with tfmot.quantization.keras.quantize_scope():
    model = tf.keras.models.load_model(model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(quantized_tflite_model)