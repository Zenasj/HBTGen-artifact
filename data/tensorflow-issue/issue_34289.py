from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

def build_keras_model():
  return keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10)
  ])

train_graph = tf.Graph()
train_sess = tf.Session(graph=train_graph)

keras.backend.set_session(train_sess)
with train_graph.as_default():
  train_model = build_keras_model()

  tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
  train_sess.run(tf.global_variables_initializer())

  train_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
  )
  train_model.fit(train_images, train_labels, epochs=5)

  saver = tf.train.Saver()
  saver.save(train_sess, 'checkpoints')

eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

keras.backend.set_session(eval_sess)

with eval_graph.as_default():
  keras.backend.set_learning_phase(0)
  eval_model = build_keras_model()
  tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
  eval_graph_def = eval_graph.as_graph_def()
  saver = tf.train.Saver()
  saver.restore(eval_sess, 'checkpoints')

  frozen_graph_def = tf.graph_util.convert_variables_to_constants(
    eval_sess,
    eval_graph_def,
    [eval_model.output.op.name]
  )
  with open('frozen_model.pb', 'wb') as f:
    f.write(frozen_graph_def.SerializeToString())

converter = tf.lite.TFLiteConverter.from_frozen_graph(
  'frozen_model.pb',
  ['flatten_input'],
  ['dense_1/BiasAdd'])
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0, 255)}  # mean, std_dev
tflite_model = converter.convert()
open('model.tflite', 'wb').write(tflite_model)

# Put link here or attach to the issue.