import random
from tensorflow import keras
from tensorflow.keras import layers

import pathlib

import numpy as np
import tensorflow as tf

print(tf.__version__)

i = tf.keras.layers.Input(shape=(32, 32, 3))
o = tf.keras.layers.Conv2D(3, 3, dilation_rate=2)(i)
model = tf.keras.Model(inputs=[i], outputs=[o])

converter = tf.lite.TFLiteConverter.from_keras_model(model)

def representative_data_gen():
    yield [np.random.random((1, 32, 32, 3)).astype(np.float32)]


converter.representative_dataset = representative_data_gen
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_models_dir = pathlib.Path("/tmp/")
tflite_model_file = tflite_models_dir / "dilation_model.tflite"
tflite_model_file.write_bytes(tflite_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]

interpreter.set_tensor(input_details["index"], np.random.random((1, 32, 32, 3)).astype(np.float32))
interpreter.invoke()
output_details = interpreter.get_output_details()[0]
output = interpreter.get_tensor(output_details["index"])[0]