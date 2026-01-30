import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

import os
import sys
import absl.logging as logging
import numpy as np
import tensorflow.compat.v1 as tf

np.set_printoptions(threshold=sys.maxsize)
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

converter = tf.lite.TFLiteConverter.from_saved_model('best_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.allow_custom_ops = True
quant_best_model = converter.convert()
with open('quant_best_model.tflite', 'wb') as w:
  w.write(quant_best_model)

import os
import sys
import absl.logging as logging
import numpy as np
import tensorflow.compat.v1 as tf

np.set_printoptions(threshold=sys.maxsize)
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

read_fingerprints = np.loadtxt('./testbench/data.csv', delimiter=',')

test_fingerprints = read_fingerprints.reshape(-1,16384).astype(np.float32)
print('type(test_fingerprints):', type(test_fingerprints))
print('shape(test_fingerprints):', test_fingerprints.shape)

train_dir = 'tcn/kws_7x36_1'

converter = tf.lite.TFLiteConverter.from_saved_model(train_dir + '/best_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.allow_custom_ops = True
quant_best_model = converter.convert()
with open(train_dir + '/quant_best_model.tflite', 'wb') as w:
  w.write(quant_best_model)


interpreter = tf.lite.Interpreter(model_path=os.path.join(train_dir + '/quant_best_model.tflite'))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
inputs = []
for s in range(len(input_details)):
  inputs.append(np.zeros(input_details[s]['shape'], dtype=np.float32))
  
model = tf.keras.models.load_model(train_dir + '/best_model', custom_objects={'tf': tf})

read_fingerprints = np.loadtxt('./testbench/data.csv', delimiter=',')
test_fingerprints = read_fingerprints.reshape(1,16384)
model.run_eagerly = True
print('tf.executing_eagerly:',tf.executing_eagerly())
print('model.run_eagerly:',model.run_eagerly)

interpreter.set_tensor(input_details[0]['index'], test_fingerprints.astype(np.float32))
interpreter.invoke()
out_tflite = interpreter.get_tensor(output_details[0]['index'])
print('out_tflite:',out_tflite)


intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(index=len(model.layers)-1).output)
intermediate_output = intermediate_layer_model(test_fingerprints)
print('intermediate_output:',intermediate_output)