import math
from tensorflow import keras

import tensorflow as tf

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__(name="model")
    self.w1 = tf.Variable([[0.], [0.5]])
    self.b1 = tf.Variable([-4.])
    self.r = tf.Variable([-7.])
    self.c = tf.Variable(1.)
    self.m1 = tf.Variable([-4.])
    self.m2 = tf.Variable([1.])

  def call(self, x):
    x = x + self.m1
    x2 = tf.math.multiply(x, self.r)
    x3 = tf.linalg.matmul(x2, self.w1)
    x4 = tf.math.add(x3, self.b1)
    x5 = tf.math.multiply(x4, self.r)
    x6 = tf.math.add(x5, self.m2)
    x7 = tf.math.multiply(x6, self.r)
    return x7

# Initializing the model
m = Model()

# Inputs to the model
x = tf.constant([[2., -3.]], shape=[1, 2], dtype=tf.float32)

# Call model
y = m(x)
print('expected model output:')
print(y)


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

py
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([[0.], [0.5]], dtype=torch.float32))
        self.b1 = nn.Parameter(torch.tensor([-4.], dtype=torch.float32))
        self.r = nn.Parameter(torch.tensor([-7.], dtype=torch.float32))
        self.c = nn.Parameter(torch.tensor(1., dtype=torch.float32))
        self.m1 = nn.Parameter(torch.tensor([-4.], dtype=torch.float32))
        self.m2 = nn.Parameter(torch.tensor([1.], dtype=torch.float32))

    def forward(self, x):
        x = x + self.m1
        x2 = x * self.r
        x3 = torch.matmul(x2, self.w1)
        x4 = x3 + self.b1
        x5 = x4 * self.r
        x6 = x5 + self.m2
        x7 = x6 * self.r
        return x7


m = Model()


x = torch.tensor([[2., -3.]], dtype=torch.float32)


y = m(x)
print('expected model output:')
print(y)





from ai_edge_torch import convert

# Convert the model to TFLite
tflite_model = convert(m.eval(),(x,))

tflite_model.export('simple_nn.tflite')

def load_tflite_model(model_path):
    with open(model_path, 'rb') as f:
        tflite_model = f.read()
    return tflite_model

# Load the TFLite model
tflite_model = load_tflite_model('simple_nn.tflite')


def _evaluateTFLiteModel(tflite_model, input_data):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(input_data)):
        interpreter.set_tensor(input_details[i]['index'], input_data[i].numpy())
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output_details[i]['index'])
                   for i in range(len(output_details))]
    return output_data

x = tf.constant([[2., -3.]], shape=[1, 2], dtype=tf.float32)
actual_value = _evaluateTFLiteModel(tflite_model, [x])
print('tflite model output:')
print(actual_value[0])