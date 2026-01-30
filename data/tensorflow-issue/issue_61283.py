from tensorflow import keras

import tensorflow as tf
import numpy as np
input_shape = [1, 2]
x1 = tf.keras.Input(shape=input_shape, dtype="float32")

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.w1 = tf.Variable([[3., 4.], [5., 6.]])
    self.b1 = tf.Variable([7., 8.])
  @tf.function(input_signature=[tf.TensorSpec(x1.shape, x1.dtype)])
  def call(self, x1):
    return tf.matmul(x1, self.w1) + self.b1

m = Model()
converter = tf.lite.TFLiteConverter.from_keras_model(m)
tflite_model = converter.convert()

def _evaluateTFLiteModel(tflite_model, input_data):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f'Keras input shape: {input_data[0].shape}') # print keras input shape
    print(f'Lite input shape: {input_details[0]["shape"]}') # print lite input shape
    
    for i in range(len(input_data)):
        interpreter.set_tensor(input_details[i]['index'], input_data[i])
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output_details[i]['index'])
                   for i in range(len(output_details))]
    return output_data

x = tf.constant([1., 2.], shape=input_shape)
actual_value = _evaluateTFLiteModel(tflite_model,[x])