import math
from tensorflow import keras

import tensorflow as tf

class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()

  def call(self, x):
    values, indices = tf.math.top_k(x, k=2, sorted=False)
    y = tf.slice(values, tf.constant([0, 0]), tf.constant([0, 1]))
    return y

# Initializing the model
m = Model()

# Inputs to the model
x = tf.constant([1., 2.], shape=[1, 2])

expected_value = m(x)
print(expected_value.numpy())


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


actual_value = _evaluateTFLiteModel(tflite_model,[x])
print('tflite model output:')
print(actual_value[0])