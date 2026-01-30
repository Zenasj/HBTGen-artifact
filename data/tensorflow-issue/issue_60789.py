import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation=None),
  #  tf.keras.layers.ReLU(),
  tf.keras.layers.ELU(),
  tf.keras.layers.Dense(10, activation='softmax')
])

# full-integer quantization, simulate dataset with random input
def representative_data_gen():
  for input_value in [np.random.randn(1, 28, 28).astype(np.float32) for _ in range(10)]:
    yield [input_value] 

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = {tf.int8}

converter.inference_input_type = tf.int8 #tf.uint8
converter.inference_output_type = tf.int8 #tf.uint8

tflite_model_quant = converter.convert()
with open('basic_fullint_quant.tflite', 'wb') as f:
  f.write(tflite_model_quant)