from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow.contrib.lite as tflite

sess = tf.keras.backend.get_session()

# create model
x = x_input = tf.keras.layers.Input([10,10,3])
x = tf.keras.layers.Conv2D(1, 3)(x)
model = tf.keras.Model(x_input, x)

# add fake quantization nodes 
tf.contrib.quantize.create_eval_graph(sess.graph)

# simulate loading weights and quantization min/max values
sess.run(tf.global_variables_initializer())

# try to quantize
toco_converter = tflite.TFLiteConverter.from_session(sess, model.inputs, model.outputs)
toco_converter.post_training_quantize = True
toco_converter.inference_type = tflite.constants.QUANTIZED_UINT8
toco_converter.inference_input_type = tflite.constants.QUANTIZED_UINT8
toco_converter.quantized_input_stats = {model.inputs[0].name.split(':')[0]: (0., 255.)}
model_tflite_binary = toco_converter.convert()

with tf.summary.FileWriter('/tmp/conv') as fw:
    fw.add_graph(sess.graph)