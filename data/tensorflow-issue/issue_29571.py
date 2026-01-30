import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
print(tf.version.GIT_VERSION, tf.version.VERSION)

channels = 64
# Slicing & quantization together results in an error
use_slice = True
quantize = True

input = tf.keras.layers.Input(shape=(channels))
x = input
x *= x
if use_slice:
    x = x[:, ::2]

model = tf.keras.Model(inputs=[input], outputs=[x])
model.summary()


def _gen_input(channels):
    return tf.constant(np.random.uniform(0, 1, size=(1, channels,)), dtype=tf.float32)

# Test normal tensorflow forward pass
model(_gen_input(channels))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
if quantize:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    for _ in range(100):
        yield [_gen_input(channels)]

converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], _gen_input(channels))
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])