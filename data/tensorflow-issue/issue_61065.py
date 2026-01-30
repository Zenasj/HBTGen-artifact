from tensorflow import keras

import tensorflow as tf
import numpy as np

x1 = tf.constant([1.], shape=[1,1])

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()

  def call(self, x1):
    x2 = tf.eye(1)
    x3 = tf.eye(2)
    x4 = tf.eye(1)
    return [x2, x3, x4]


m = Model()
expected_value = m(x1)

converter = tf.lite.TFLiteConverter.from_keras_model(m)
tflite_model = converter.convert()

def _evaluateTFLiteModel(tflite_model, input_data):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(input_data)):
        interpreter.set_tensor(input_details[i]['index'], input_data[i])

    interpreter.invoke()

    output_data = [interpreter.get_tensor(output_details[i]['index'])
                   for i in range(len(output_details))]
    return output_data

actual_value = _evaluateTFLiteModel(tflite_model,[x1])

#Outputs
print(f"Expected output_1: {expected_value[0].numpy()}")
print(f"Lite output_1: {actual_value[0]}")
print("-----------------------------------")
print(f"Expected output_2: {expected_value[1].numpy()}")
print(f"Lite output_2: {actual_value[1]}")
print("-----------------------------------")
print(f"Expected output_3: {expected_value[2].numpy()}")
print(f"Lite output_3: {actual_value[2]}")
#wrong order

tf.lite.experimental.Analyzer.analyze(model_content=tflite_model) #Output IR