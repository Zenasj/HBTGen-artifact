import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import pathlib
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

print('\u2022 Using TensorFlow Version:', tf.__version__)

# Create a simple Keras model.
x = [-5, -1, 0, 1, 4]
y = [-10, -2, 0, 2, 8]

model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units = 1, input_shape=[1,1,1,1])
])
model.compile(optimizer='sgd',
              loss='mean_squared_error')

model.fit(x, y, epochs=200)

export_dir = 'saved_model/2'
tf.saved_model.save(model, export_dir)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
# converter.target_spec.supported_ops = [
#   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
# ]
tflite_model = converter.convert()

tflite_model_file = pathlib.Path('model.tflite')
tflite_model_file.write_bytes(tflite_model)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the TensorFlow Lite model on random input data.
input_shape = input_details[0]['shape']
inputs, outputs, outputsTFL = [], [], []
for _ in range(100):
    input_data = np.array(np.random.random_sample(input_shape)*100, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]['index'])
    
    # Test the TensorFlow model on random input data.
    tf_results = model(tf.constant(input_data))
    output_data = np.array(tf_results)
    
    inputs.append(input_data[0][0][0][0][0])
    outputs.append(output_data[0][0][0][0][0])
    outputsTFL.append(np.array(tflite_results)[0][0][0][0][0])

plt.plot(inputs, outputs, 'r')
plt.show()
plt.plot(inputs, outputsTFL, 'r')
plt.show()

import pathlib
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

print('\u2022 Using TensorFlow Version:', tf.__version__)

# Create a simple Keras model.
x = [-5, -1, 0, 1, 4]
y = [-10, -2, 0, 2, 8]

model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units = 1, input_shape=[1,1,1,1,1])
])
model.compile(optimizer='sgd',
              loss='mean_squared_error')

model.fit(x, y, epochs=200)

export_dir = 'saved_model/2'
tf.saved_model.save(model, export_dir)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()

tflite_model_file = pathlib.Path('model.tflite')
tflite_model_file.write_bytes(tflite_model)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the TensorFlow Lite model on random input data.
input_shape = input_details[0]['shape']
inputs, outputs, outputsTFL = [], [], []
for _ in range(100):
    input_data = np.array(np.random.random_sample(input_shape)*100, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]['index'])
    
    # Test the TensorFlow model on random input data.
    tf_results = model(tf.constant(input_data))
    output_data = np.array(tf_results)
    
    inputs.append(input_data[0][0][0][0][0][0])
    outputs.append(output_data[0][0][0][0][0][0])
    outputsTFL.append(np.array(tflite_results)[0][0][0][0][0][0])

plt.plot(inputs, outputs, 'r')
plt.show()
plt.plot(inputs, outputsTFL, 'r')
plt.show()

model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units = 1, input_shape=[1])
])

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]