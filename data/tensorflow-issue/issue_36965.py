import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as numpy

model = tf.keras.Sequential()

model.add(tf.keras.layers.Input(shape=(1, 1,)))

cell = tf.keras.layers.GRUCell(10)

model.add(tf.keras.layers.RNN(cell, unroll=True))

model.save("test_gru_cell.h5", save_format='h5')

def representative_dataset_gen():
    yield [numpy.random.uniform(low=-1, high=1, size=(1,1,1)).astype(numpy.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_dataset_gen
converter.experimental_new_converter = False

tflite_model = converter.convert()

open("test_gru_cell.tflite", 'wb').write(tflite_model)