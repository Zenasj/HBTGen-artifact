import random
from tensorflow import keras

import numpy as np
import tensorflow as tf

# Load the MobileNet tf.keras model (and set include_top=False).
model = tf.keras.applications.MobileNetV2(
    weights="imagenet", input_shape=(224, 224, 3), include_top=False)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the TensorFlow Lite model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
tflite_results = interpreter.get_tensor(output_details[0]['index'])

# Test the TensorFlow model on random input data.
tf_results = model(tf.constant(input_data))

# Compare the result.
for tf_result, tflite_result in zip(tf_results, tflite_results):
  np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)

import numpy as np
import tensorflow as tf

# Load the MobileNet tf.keras model (and set include_top=False).
model = tf.keras.applications.MobileNetV2(
    weights="imagenet", input_shape=(224, 224, 3), include_top=True)

for layer in model.layers:
    submodel = tf.keras.Model(inputs=model.inputs, outputs=layer.output)

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(submodel)
    tflite_model = converter.convert()

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the TensorFlow Lite model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    # Test the TensorFlow model on random input data.
    tf_results = submodel(tf.constant(input_data))

    # Compare the result.
    for tf_result, tflite_result in zip(tf_results, tflite_results):
      try:
          np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)
          print(f"EQUAL {layer.__class__.__name__}")
      except:
          print(f"NOT EQUAL {layer.__class__.__name__}")