import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

inputs = tf.keras.Input(shape=(1, 126))
outputs = tf.keras.layers.Reshape((-1, 21))(inputs)
#outputs = tf.reshape(inputs, (-1, 21))
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(optimizer='sgd', loss='mean_squared_error')

inputs = np.random.rand(126).reshape((1, 1, 126))
outputs = np.random.rand(126).reshape((1, 6, 21))

model.fit(x=inputs, y=outputs, epochs=1)

def representative_dataset_gen():
  for input in inputs:
    input = input.astype(np.float32)
    yield [input]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen

tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)