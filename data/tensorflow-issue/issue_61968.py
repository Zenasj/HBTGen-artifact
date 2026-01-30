import math
from tensorflow import keras

import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.b = tf.Variable(np.array([[1],[2]],dtype=np.float32))

  def call(self, x):
    x = tf.add(x,1)
    return tf.math.l2_normalize(tf.transpose(x))

# Initializing the model
m = Model()

# Call model
input_shape = [1, 1, 2]
x1 = tf.constant(1., shape=input_shape)
y1 = m(x1)
print('expected model output:')
print(y1)


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
print('tflite model output:')
print(actual_value[0])