from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(shape=(1, 1,)))

cell = tf.keras.layers.GRUCell(10)

model.add(tf.keras.layers.RNN(cell))

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.allow_custom_ops = True # does the trick
#converter.experimental_new_converter = True # does not help

tflite_model = converter.convert()

import tensorflow as tf

model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(shape=(1, 1,)))

cell = tf.keras.layers.GRUCell(10)

model.add(tf.keras.layers.RNN(cell, unroll=True))

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

# for testing if operations are implemented by Tensorflow Lite
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
interpreter.invoke()

converter = TFLiteConverter.from_saved_model(model_dir)
converter.experimental_new_converter = False
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops=True
tflite_model = converter.convert()